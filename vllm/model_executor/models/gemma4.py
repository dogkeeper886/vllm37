# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#126): the Gemma 4 text model (gemma-4-26B-A4B: 30 layers, a
# dense MLP beside 128 experts, top 8) on the V0 engine, loaded from Google's
# GGUF. Follows upstream vLLM's gemma4.py and llama.cpp's gemma4 graph.
#
# Attention runs here, not in the V0 backends: 25 sliding-window layers
# (head size 256, 8 KV heads, window 1,024) and 5 full layers (head size 512,
# 2 KV heads, V = K, proportional RoPE). V0 allocates one KV-cache shape for
# all layers and its paged-attention decode has neither per-layer windows nor
# head size 512. Every layer's cache is the [2, blocks, block_size * E] tensor
# V0 allocates for the sliding layers' shape; a token's K (or V) takes its
# first num_kv_heads * head_dim of the E values at its slot, which a full
# layer's 2 x 512 fits (against 8 x 256, split alike by the dies). Prefill
# attends to the prompt in chunks of query rows; decode gathers each
# sequence's keys (the last 1,024 on sliding layers) from the cache.

from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import Attention
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

# Query rows x keys x heads per prefill attention call (#96's bound).
_MAX_SCORES = 1 << 26


def _text_config(config):
    return getattr(config, "text_config", config)


class Gemma4RotaryEmbedding(RotaryEmbedding):
    """Neox-style RoPE over the whole head; on full layers proportional:
    exponents over the head size, and only the first partial_rotary_factor of
    the pairs rotate (cos 1, sin 0 after them; llama.cpp's rope_freqs of 1
    and 1e30). As upstream vLLM's Gemma4RotaryEmbedding."""

    def __init__(self, head_dim: int, params: dict, max_positions: int,
                 dtype: torch.dtype):
        self.rope_angles = int(
            params.get("partial_rotary_factor", 1.0) * head_dim) // 2
        super().__init__(head_dim, head_dim, max_positions,
                         params["rope_theta"], True, dtype)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv = base**(-torch.arange(0, 2 * self.rope_angles, 2,
                                   dtype=torch.float) / self.head_size)
        return torch.cat(
            (inv, torch.zeros(self.head_size // 2 - self.rope_angles)))


_ROPES: dict[tuple, Gemma4RotaryEmbedding] = {}


def gemma4_rope(head_dim: int, params: dict, max_positions: int,
                dtype: torch.dtype) -> Gemma4RotaryEmbedding:
    """One RoPE module per layer type, shared by its layers (a cos / sin
    cache over max_positions for each)."""
    key = (head_dim, tuple(sorted(params.items())), max_positions, dtype)
    if key not in _ROPES:
        _ROPES[key] = Gemma4RotaryEmbedding(head_dim, params, max_positions,
                                            dtype)
    return _ROPES[key]


def _heads(norm: RMSNorm, x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """norm over each head of x [T, heads * head_dim]."""
    return norm(x.reshape(-1, head_dim)).view(x.shape)


def gemma4_prefill_attention(q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor,
                             window: Optional[int]) -> torch.Tensor:
    """One prompt's causal attention, scale 1.0: q [L, H, D], k / v
    [L, Hkv, D] -> [L, H, D] in FP32, each query seeing the last `window`
    positions (all when None), in chunks of query rows."""
    n, h, _ = q.shape
    groups = h // k.shape[1]
    q = q.float().transpose(0, 1)  # [H, L, D]
    k = k.float().repeat_interleave(groups, 1).transpose(0, 1)
    v = v.float().repeat_interleave(groups, 1).transpose(0, 1)
    # A chunk of rows sees at most n keys, or 2 * window with a window (rows
    # capped at the window).
    keys = n if window is None else min(n, 2 * window)
    rows = max(1, _MAX_SCORES // (h * keys))
    if window is not None:
        rows = min(rows, window)
    out = torch.empty_like(q)
    pos = torch.arange(n, device=q.device)
    for a in range(0, n, rows):
        b = min(n, a + rows)
        k0 = 0 if window is None else max(0, a - window + 1)
        qi, kj = pos[a:b, None], pos[None, k0:b]
        mask = kj <= qi
        if window is not None:
            mask &= kj > qi - window
        out[:, a:b] = F.scaled_dot_product_attention(q[:, a:b],
                                                     k[:, k0:b],
                                                     v[:, k0:b],
                                                     attn_mask=mask,
                                                     scale=1.0)
    return out.transpose(0, 1)


def gemma4_decode_slots(seq_lens: torch.Tensor, block_tables: torch.Tensor,
                       block_size: int, window: Optional[int],
                       max_seq_len: Optional[int] = None):
    """For one new token per sequence: the cache slots of its keys, the last
    `window` of its seq_lens[b] positions (all when None), as slots [B, W]
    and valid [B, W]. max_seq_len (the largest seq_lens, known on the host)
    avoids a host sync. The same for every layer of a type in a step."""
    lens = seq_lens.long()
    span = lens if window is None else lens.clamp(max=window)
    if max_seq_len is None:
        max_seq_len = int(lens.max())
    width = max_seq_len if window is None else min(max_seq_len, window)
    idx = torch.arange(width, device=lens.device)
    pos = (lens - span).unsqueeze(1) + idx  # [B, width]
    valid = idx < span.unsqueeze(1)
    pos = torch.where(valid, pos, torch.zeros_like(pos))
    slots = (block_tables.long().gather(1, pos // block_size) * block_size +
             pos % block_size)
    return slots, valid


def gemma4_decode_attention(q: torch.Tensor, kc: torch.Tensor,
                            vc: torch.Tensor, slots: torch.Tensor,
                            valid: torch.Tensor) -> torch.Tensor:
    """One new token per sequence, q [B, H, D], against the keys at its
    slots (gemma4_decode_slots) in the cache kc / vc [slots, Hkv * D]. Scale
    1.0, FP32."""
    b, h, d = q.shape
    width = slots.shape[1]
    hkv = kc.shape[-1] // d
    keys = kc[slots].view(b, width, hkv, d).float()
    vals = vc[slots].view(b, width, hkv, d).float()
    qg = q.float().view(b, hkv, h // hkv, d)
    scores = torch.einsum("bkgd,bnkd->bkgn", qg, keys)
    scores.masked_fill_(~valid.view(b, 1, 1, width), float("-inf"))
    out = torch.einsum("bkgn,bnkd->bkgd", scores.softmax(-1), vals)
    return out.reshape(b, h, d)


class Gemma4Attention(nn.Module):

    def __init__(self, config, layer_idx: int,
                 cache_config: Optional[CacheConfig],
                 quant_config: Optional[QuantizationConfig], prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        layer_type = config.gemma4_layer_types[layer_idx]
        self.sliding = layer_type == "sliding_attention"
        self.window = config.gemma4_sliding_window if self.sliding else None
        self.head_dim = (config.head_dim
                         if self.sliding else config.global_head_dim)
        total_kv = (config.num_key_value_heads
                    if self.sliding else config.num_global_key_value_heads)
        total_heads = config.num_attention_heads
        if total_heads % tp or total_kv % tp:
            raise ValueError(
                f"Gemma 4 on the K80: {total_heads} heads and {total_kv} KV "
                f"heads must divide by tensor_parallel_size {tp}")
        self.num_heads = total_heads // tp
        self.num_kv_heads = total_kv // tp
        self.k_eq_v = not self.sliding and config.attention_k_eq_v
        self.eps = config.rms_norm_eps
        hidden = config.hidden_size
        # k_eq_v layers ship no V weight; K's is loaded into the V shard too
        # (as upstream), and V is then normalized without K's norm and RoPE.
        self.qkv_proj = QKVParallelLinear(hidden,
                                          self.head_dim,
                                          total_heads,
                                          total_kv,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.o_proj = RowParallelLinear(total_heads * self.head_dim,
                                        hidden,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.q_norm = RMSNorm(self.head_dim, self.eps)
        self.k_norm = RMSNorm(self.head_dim, self.eps)
        self.v_norm = RMSNorm(self.head_dim, self.eps, has_weight=False)
        vllm_config = get_current_vllm_config()
        self.rope = gemma4_rope(self.head_dim,
                                config.rope_parameters[layer_type],
                                vllm_config.model_config.max_model_len,
                                vllm_config.model_config.dtype)
        # Registered so that V0 allocates and binds this layer's KV cache; its
        # forward is not used (see the file comment).
        self.attn = Attention(total_heads // tp,
                              config.head_dim,
                              1.0,
                              num_kv_heads=config.num_key_value_heads // tp,
                              cache_config=cache_config,
                              prefix=f"{prefix}.attn")
        self.kv_width = self.num_kv_heads * self.head_dim
        if self.kv_width > self.attn.num_kv_heads * config.head_dim:
            raise ValueError(
                "Gemma 4 on the K80: a full-attention layer's KV does not fit "
                "the sliding layers' cache slot")

    def _cache(self):
        """(K, V) as [slots, kv_width] views of this layer's cache, and the
        block size; (None, None, 0) before the cache exists (profiling)."""
        kv = self.attn.kv_cache[get_forward_context().virtual_engine]
        if kv is None or kv.numel() == 0:
            return None, None, 0
        e = self.attn.num_kv_heads * self.attn.head_size
        return (kv[0].view(-1, e)[:, :self.kv_width],
                kv[1].view(-1, e)[:, :self.kv_width], kv.shape[-1] // e)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        t = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)[0]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = _heads(self.q_norm, q, self.head_dim)
        k = _heads(self.k_norm, k, self.head_dim)
        v = _heads(self.v_norm, v, self.head_dim)
        q, k = self.rope(positions, q, k)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)

        md = get_forward_context().attn_metadata
        kc, vc, block_size = self._cache()
        if kc is not None and md is not None:
            # Every slot is valid once the cache exists: V0 pads slots (-1)
            # only for CUDA graphs, the profile run (no cache) and its own
            # sliding window, none of which applies here. No mask, so no
            # host sync per layer.
            slots = md.slot_mapping.long()
            kc.index_copy_(0, slots, k.reshape(t, -1).to(kc.dtype))
            vc.index_copy_(0, slots, v.reshape(t, -1).to(vc.dtype))
        out = torch.empty(t,
                          self.num_heads,
                          self.head_dim,
                          dtype=hidden_states.dtype,
                          device=hidden_states.device)
        num_prefill = md.num_prefill_tokens if md is not None else t
        start = 0
        prefill_lens = (md.seq_lens[:md.num_prefills]
                        if md is not None else [t])
        for n in prefill_lens:
            out[start:start + n] = gemma4_prefill_attention(
                q[start:start + n], k[start:start + n], v[start:start + n],
                self.window)
            start += n
        if md is not None and md.num_decode_tokens > 0:
            assert kc is not None
            p = md.num_prefills
            # Computed once per step for each window (layer type).
            per_step = md.__dict__.setdefault("_gemma4_decode_slots", {})
            if self.window not in per_step:
                per_step[self.window] = gemma4_decode_slots(
                    md.seq_lens_tensor[p:], md.block_tables[p:], block_size,
                    self.window, md.max_decode_seq_len)
            out[num_prefill:] = gemma4_decode_attention(
                q[num_prefill:], kc, vc, *per_step[self.window])
        return self.o_proj(out.view(t, -1))[0]


class Gemma4MLP(nn.Module):

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(config.intermediate_size,
                                           config.hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.act = GeluAndMul(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act(self.gate_up_proj(x)[0]))[0]


class Gemma4Router(nn.Module):
    """Logits from rms_norm(x) / sqrt(hidden) * scale; softmax over all
    experts, top k, renormalized, times each expert's output scale."""

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            has_weight=False)
        self.root = config.hidden_size**-0.5
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
        self.proj = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=torch.float32)

    def forward(self, x):
        h = self.norm(x).float() * (self.root * self.scale.float())
        return self.proj(h)[0]

    def routing_function(self, hidden_states, gating_output, topk,
                         renormalize):
        probs = gating_output.float().softmax(-1)
        weights, ids = torch.topk(probs, topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.per_expert_scale.float()[ids]
        return weights, ids.to(torch.int32)


class Gemma4DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx, cache_config, quant_config,
                 prefix):
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = Gemma4Attention(config, layer_idx, cache_config,
                                         quant_config, f"{prefix}.self_attn")
        self.mlp = Gemma4MLP(config, quant_config, f"{prefix}.mlp")
        self.router = Gemma4Router(config)
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=self.router.routing_function,
            activation="gelu_tanh")
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm",
                     "post_feedforward_layernorm_1",
                     "pre_feedforward_layernorm_2",
                     "post_feedforward_layernorm_2"):
            setattr(self, name, RMSNorm(config.hidden_size, eps))
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(self, positions, hidden_states):
        residual = hidden_states
        h = self.self_attn(positions, self.input_layernorm(hidden_states))
        h = self.post_attention_layernorm(h) + residual
        residual = h
        dense = self.post_feedforward_layernorm_1(
            self.mlp(self.pre_feedforward_layernorm(h)))
        experts = self.experts(self.pre_feedforward_layernorm_2(h),
                               self.router(h))
        experts = self.post_feedforward_layernorm_2(experts)
        h = self.post_feedforward_layernorm(dense + experts) + residual
        return h * self.layer_scalar.to(h.dtype)


class Gemma4Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        self.config = config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma4DecoderLayer(
                config, int(prefix.rsplit(".", 1)[1]), cache_config,
                quant_config, prefix),
            prefix=f"{prefix}.layers")
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.normalizer = config.hidden_size**0.5
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def forward(self, input_ids, positions, intermediate_tensors,
                inputs_embeds=None):
        if get_pp_group().is_first_rank:
            h = (inputs_embeds if inputs_embeds is not None else
                 self.embed_tokens(input_ids) * self.normalizer)
        else:
            h = intermediate_tensors["hidden_states"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            h = layer(positions, h)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": h})
        return self.norm(h)


class Gemma4ForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        self.config = config
        self.model = Gemma4Model(vllm_config=vllm_config, prefix="model")
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix="lm_head")
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata
                       ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Names from gemma4_gguf_name_map (the GGUF loader)."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        cfg = self.config
        for name, w in weights:
            targets = [(name, None)]
            if name.startswith("model.embed_tokens."):  # tied LM head
                targets.append((name.replace("model.embed_tokens.",
                                             "lm_head."), None))
            for src, dst, shard in _STACKED:
                if src in name:
                    targets = [(name.replace(src, dst), shard)]
                    layer = int(name.split(".")[2])
                    if (shard == "k" and cfg.attention_k_eq_v
                            and cfg.gemma4_layer_types[layer]
                            == "full_attention"):
                        targets.append((name.replace(src, dst), "v"))
                    break
            for target, shard in targets:
                if ".experts." in target:
                    self._load_expert(params, target, w, loaded)
                    continue
                if target not in params or is_pp_missing_parameter(
                        target, self):
                    continue
                param = params[target]
                if shard is not None:
                    param.weight_loader(param, w, shard)
                    loaded.add(target)
                    continue
                if (not isinstance(param, nn.parameter.UninitializedParameter)
                        and w.shape != param.shape
                        and w.numel() == param.numel()):
                    w = w.view(param.shape)  # e.g. layer_scalar [1]
                getattr(param, "weight_loader", default_weight_loader)(param,
                                                                       w)
                loaded.add(target)
        return loaded

    def _load_expert(self, params, name, w, loaded):
        """experts.gate_up (merged [E, 2I, ...]) and experts.down."""
        prefix, kind = name.split(".experts.")
        part, suffix = kind.split(".", 1)  # gate_up | down, qweight[_type]
        if is_pp_missing_parameter(f"{prefix}.experts.w2_{suffix}", self):
            return
        if part == "down":
            shards = [("w2", w)]
        elif suffix.endswith("_type"):
            shards = [("w1", w), ("w3", w)]
        else:
            half = w.shape[1] // 2
            shards = [("w1", w[:, :half]), ("w3", w[:, half:])]
        target = f"{prefix}.experts.{'w2' if part == 'down' else 'w13'}_" \
                 f"{suffix}"
        if target not in params:
            return
        param = params[target]
        for shard_id, t in shards:
            param.weight_loader(param,
                                t,
                                target,
                                shard_id=shard_id,
                                expert_id=0)
        loaded.add(target)


# (name from gemma4_gguf_name_map, fused parameter, shard)
_STACKED = [
    (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"),
    (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
    (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
    (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
    (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
]


def gemma4_gguf_name_map(config) -> dict[str, str]:
    """GGUF tensor names (llama.cpp's gemma4) -> this model's names."""
    m = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
    }
    per_layer = {
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm",
        "attn_norm": "input_layernorm",
        "post_attention_norm": "post_attention_layernorm",
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffw_norm": "post_feedforward_layernorm",
        "post_ffw_norm_1": "post_feedforward_layernorm_1",
        "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
        "post_ffw_norm_2": "post_feedforward_layernorm_2",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate_inp": "router.proj",
        "ffn_gate_up_exps": "experts.gate_up",
        "ffn_down_exps": "experts.down",
    }
    for i in range(_text_config(config).num_hidden_layers):
        for g, h in per_layer.items():
            m[f"blk.{i}.{g}.weight"] = f"model.layers.{i}.{h}.weight"
        m[f"blk.{i}.ffn_gate_inp.scale"] = f"model.layers.{i}.router.scale"
        m[f"blk.{i}.ffn_down_exps.scale"] = (
            f"model.layers.{i}.router.per_expert_scale")
        m[f"blk.{i}.layer_output_scale.weight"] = (
            f"model.layers.{i}.layer_scalar")
    return m

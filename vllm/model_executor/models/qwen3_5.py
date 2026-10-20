# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#127): Qwen3.5 / Qwen3.6 / Qwen3.8 (model_type qwen3_5 and
# qwen3_5_moe), text only, on the V0 engine, after upstream vLLM's
# qwen3_5.py and qwen3_next.py and transformers' Qwen3-Next.
#
# Three of every four layers are Gated DeltaNet linear attention: projections
# to q, k, v (one fused tensor), the output gate z, and per value head b and
# a; a causal convolution (width 4, SiLU) over q, k, v; the delta rule with a
# float32 state per sequence (k80_gated_delta.py, no Triton); an RMSNorm gated
# by SiLU(z); the output projection. Its conv and recurrent states live in a
# V0 constant-size cache, one slot per sequence. Every fourth layer is full
# attention: head 256, partial RoPE (a quarter of the head), q and k
# RMSNorm, and the output gated by sigmoid of a second half of q_proj. The
# dense MLP is SwiGLU; qwen3_5_moe uses Qwen2-MoE's sparse block (router, top
# k experts, a sigmoid-gated shared expert). The vision tower and the MTP
# head are not loaded.

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.k80_gated_delta import (
    causal_conv1d_prefill, causal_conv1d_step, chunk_gated_delta_rule, gating,
    l2norm, recurrent_step, rms_norm_gated)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.constant_size_cache import ConstantSizeCache
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from .interfaces import HasInnerState, IsHybrid
from .qwen2 import Qwen2MLP
from .qwen2_moe import Qwen2MoeSparseMoeBlock
from .utils import extract_layer_index, maybe_prefix


class GatedDeltaCache(ConstantSizeCache):
    """One slot per running sequence: per Gated DeltaNet layer a conv state
    [C, W - 1] in the model dtype and a recurrent state [H, K, V] in
    float32."""

    def __init__(self, vllm_config: VllmConfig, dtype: torch.dtype,
                 num_layers: int, conv_shape: tuple[int, int],
                 state_shape: tuple[int, int, int]):
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        if not vllm_config.model_config.enforce_eager:
            max_batch_size = vllm_config.pad_for_cudagraph(max_batch_size)
        super().__init__(max_batch_size)
        self.conv = torch.zeros((num_layers, max_batch_size) + conv_shape,
                                dtype=dtype,
                                device="cuda")
        self.state = torch.zeros((num_layers, max_batch_size) + state_shape,
                                 dtype=torch.float32,
                                 device="cuda")

    @property
    def cache(self):
        return (self.conv, self.state)

    def _copy_cache(self, from_index: int, to_index: int):
        for t in self.cache:
            t[:, to_index].copy_(t[:, from_index], non_blocking=True)


def _rank_slice(loaded: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """The rows of this rank's share of each part of a tensor whose dim 0 is
    the parts in sizes, one after another."""
    tp, rank = (get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank())
    parts, start = [], 0
    for size in sizes:
        share = size // tp
        parts.append(loaded.narrow(0, start + rank * share, share))
        start += size
    return torch.cat(parts)


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(self, config, quant_config: Optional[QuantizationConfig],
                 prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        hidden = config.hidden_size
        nk, nv = config.linear_num_key_heads, config.linear_num_value_heads
        self.head_k = config.linear_key_head_dim
        self.head_v = config.linear_value_head_dim
        if nk % tp or nv % tp:
            raise ValueError(
                f"Gated DeltaNet: {nk} key and {nv} value heads do not split "
                f"across {tp} dies")
        key_dim, value_dim = nk * self.head_k, nv * self.head_v
        self.nk, self.nv = nk // tp, nv // tp
        self.sizes = [key_dim, key_dim, value_dim]
        self.eps = config.rms_norm_eps
        self.in_proj_qkv = MergedColumnParallelLinear(
            hidden,
            self.sizes,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkv")
        self.in_proj_z = ColumnParallelLinear(hidden,
                                              value_dim,
                                              bias=False,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.in_proj_z")
        self.in_proj_b = ColumnParallelLinear(hidden,
                                              nv,
                                              bias=False,
                                              prefix=f"{prefix}.in_proj_b")
        self.in_proj_a = ColumnParallelLinear(hidden,
                                              nv,
                                              bias=False,
                                              prefix=f"{prefix}.in_proj_a")
        self.out_proj = RowParallelLinear(value_dim,
                                          hidden,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")
        width = config.linear_conv_kernel_dim
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(
            torch.empty(sum(self.sizes) // tp, width), requires_grad=False)
        sizes = self.sizes
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                lambda p, w: p.data.copy_(_rank_slice(w.squeeze(1), sizes))
            })
        self.A_log = nn.Parameter(torch.empty(self.nv, dtype=torch.float32),
                                  requires_grad=False)
        self.dt_bias = nn.Parameter(torch.empty(self.nv), requires_grad=False)
        for p in (self.A_log, self.dt_bias):
            set_weight_attrs(
                p, {
                    "weight_loader":
                    lambda p, w: p.data.copy_(_rank_slice(w, [nv]))
                })
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.empty(self.head_v,
                                                    dtype=torch.float32),
                                        requires_grad=False)
        set_weight_attrs(self.norm.weight,
                         {"weight_loader": default_weight_loader})

    def forward(self, x: torch.Tensor, conv: torch.Tensor,
                state: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """x [T, hidden]; conv [slots, C, W - 1] and state [slots, H, K, V],
        this layer's caches; slots: each sequence's cache slot, prefills
        first."""
        md = get_forward_context().attn_metadata
        T = x.shape[0]
        qkv, _ = self.in_proj_qkv(x)
        z, _ = self.in_proj_z(x)
        b, _ = self.in_proj_b(x)
        a, _ = self.in_proj_a(x)
        g, beta = gating(self.A_log, a, b, self.dt_bias)
        weight = self.conv1d.weight
        out = torch.empty(T, self.nv, self.head_v, dtype=torch.float32,
                          device=x.device)
        if md is None:  # a run without a cache, as the memory profile
            prefill_lens, num_prefill_tokens = [T], T
        else:
            prefill_lens = md.seq_lens[:md.num_prefills]
            num_prefill_tokens = md.num_prefill_tokens
        start = 0
        for i, n in enumerate(prefill_lens):
            end = start + n
            slot = int(slots[i]) if slots.numel() else -1
            tmp = torch.empty(weight.shape[0], weight.shape[1] - 1,
                              dtype=x.dtype, device=x.device)
            mixed = causal_conv1d_prefill(qkv[start:end], weight,
                                          conv[slot] if slot >= 0 else tmp)
            q, k, v = self._heads(mixed)
            o, s = chunk_gated_delta_rule(q, k, v, g[start:end],
                                          beta[start:end])
            if slot >= 0:
                state[slot] = s
            out[start:end] = o
            start = end
        assert start == num_prefill_tokens, (start, num_prefill_tokens)
        if T > num_prefill_tokens:
            idx = slots[len(prefill_lens):].long()
            # A padding slot (PAD_SLOT_ID, -1) reads slot 0 and is not
            # written back.
            keep = idx >= 0
            conv_d, state_d = conv[idx.clamp(min=0)], state[idx.clamp(min=0)]
            mixed = causal_conv1d_step(qkv[num_prefill_tokens:], weight,
                                       conv_d)
            q, k, v = self._heads(mixed)
            out[num_prefill_tokens:] = recurrent_step(
                q, k, v, g[num_prefill_tokens:], beta[num_prefill_tokens:],
                state_d)
            conv[idx[keep]] = conv_d[keep]
            state[idx[keep]] = state_d[keep]
        o = rms_norm_gated(out, z.view(T, self.nv, self.head_v),
                           self.norm.weight, self.eps)
        o, _ = self.out_proj(o.reshape(T, -1))
        return o

    def _heads(self, mixed: torch.Tensor):
        """q, k (L2-normalised, repeated to the value heads) and v from the
        conv output [T, C]."""
        kd = self.nk * self.head_k
        q, k, v = mixed.split([kd, kd, self.nv * self.head_v], dim=-1)
        rep = self.nv // self.nk
        q = l2norm(q.float().view(-1, self.nk, self.head_k))
        k = l2norm(k.float().view(-1, self.nk, self.head_k))
        if rep > 1:
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        return q, k, v.float().view(-1, self.nv, self.head_v)


class Qwen3_5Attention(nn.Module):
    """Full attention with an output gate: q_proj holds, per head, the query
    and then the gate."""

    def __init__(self, config, cache_config, quant_config, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads * 2,  # query and gate per head
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(config.num_attention_heads *
                                        self.head_dim,
                                        config.hidden_size,
                                        bias=config.attention_bias,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * config.partial_rotary_factor),
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True)
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.head_dim**-0.5,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        qg, k, v = qkv.split([2 * q_size, kv_size, kv_size], dim=-1)
        q, gate = qg.view(T, self.num_heads, 2 * self.head_dim).chunk(2, -1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(T, q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(T, kv_size)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o * torch.sigmoid(gate.reshape(T, q_size))
        o, _ = self.o_proj(o)
        return o


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, layer_type: str,
                 prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config, quant_config, prefix=f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen3_5Attention(config,
                                              vllm_config.cache_config,
                                              quant_config,
                                              prefix=f"{prefix}.self_attn")
        if config.num_experts:
            self.mlp = Qwen2MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen2MLP(config.hidden_size,
                                config.intermediate_size,
                                config.hidden_act,
                                quant_config=quant_config,
                                prefix=f"{prefix}.mlp")
        self.input_layernorm = GemmaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, cache):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, *cache)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3_5Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(vllm_config,
                                config.layer_types[i],
                                prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, caches):
        conv, state, slots = caches
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        n = 0
        for layer in self.layers:
            cache = None
            if layer.layer_type == "linear_attention":
                cache = (conv[n], state[n], slots)
                n += 1
            hidden_states, residual = layer(positions, hidden_states,
                                            residual, cache)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """Qwen3.5 / 3.6 / 3.8 text, dense or MoE. Also serves the
    ...ForConditionalGeneration checkpoints, without the vision tower."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = Qwen3_5Model(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix,
                                                      "model.language_model"))
        self.tie = getattr(hf_config, "tie_word_embeddings",
                           getattr(config, "tie_word_embeddings", False))
        if self.tie:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          prefix=maybe_prefix(
                                              prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.cache: Optional[GatedDeltaCache] = None

    def _new_cache(self) -> GatedDeltaCache:
        c = self.config
        tp = get_tensor_model_parallel_world_size()
        conv_dim = (2 * c.linear_num_key_heads * c.linear_key_head_dim +
                    c.linear_num_value_heads * c.linear_value_head_dim) // tp
        return GatedDeltaCache(
            self.vllm_config, self.model.embed_tokens.weight.dtype,
            sum(t == "linear_attention" for t in c.layer_types),
            (conv_dim, c.linear_conv_kernel_dim - 1),
            (c.linear_num_value_heads // tp, c.linear_key_head_dim,
             c.linear_value_head_dim))

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if self.cache is None:
            self.cache = self._new_cache()
        (conv, state), slots = self.cache.current_run_tensors(**kwargs)
        return self.model(input_ids, positions, (conv, state, slots))

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params = []
        if self.config.num_experts:
            from vllm.model_executor.layers.fused_moe import FusedMoE
            expert_params = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts)
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        unplaced: list[str] = []
        for name, weight in weights:
            if name.startswith("mtp.") or ".visual." in name or \
                    name.startswith("visual."):
                continue
            name = name.replace("model.language_model.", "model.")
            if name.startswith("lm_head.") and self.tie:
                continue
            if name.endswith(("k_scale", "v_scale")):
                name = maybe_remap_kv_scale_name(name, params)
                if name is None:
                    continue
            if self._load_expert(name, weight, params, expert_params,
                                 loaded, unplaced):
                continue
            for param_name, ckpt_name, shard in stacked:
                if ckpt_name not in name or ".mlp.experts." in name:
                    continue
                name = name.replace(ckpt_name, param_name)
                if name not in params:
                    unplaced.append(name)
                    break
                params[name].weight_loader(params[name], weight, shard)
                loaded.add(name)
                break
            else:
                if name not in params:
                    unplaced.append(name)
                    continue
                param = params[name]
                getattr(param, "weight_loader",
                        default_weight_loader)(param, weight)
                loaded.add(name)
        if unplaced:
            raise ValueError(
                f"Qwen3.5: {len(unplaced)} checkpoint tensors match no "
                f"parameter, such as {unplaced[:4]}")
        return loaded

    def _load_expert(self, name, weight, params, expert_params, loaded,
                     unplaced) -> bool:
        if ".mlp.experts." not in name:
            return False
        for param_name, weight_name, expert_id, shard_id in expert_params:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            if name in params:
                param = params[name]
                param.weight_loader(param,
                                    weight,
                                    name,
                                    shard_id=shard_id,
                                    expert_id=expert_id)
                loaded.add(name)
            else:
                unplaced.append(name)
            return True
        unplaced.append(name)
        return True

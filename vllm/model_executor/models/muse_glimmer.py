# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#130): Muse-Glimmer (model_type muse_glimmer), text only, from the
# official GGUF (arch muse-glimmer), after upstream vLLM's muse_glimmer.py and
# ollama37's model/models/museglimmer.
#
# Per layer: RMSNorm, attention, RMSNorm (post_norm_eps), residual; RMSNorm,
# SwiGLU MLP, RMSNorm (post_norm_eps), residual. Attention: head 128, q and k
# RMSNorm per head, an output gate sigmoid(attn_gate(layer input)), and an
# iRoPE layout: sliding-window layers (2,048) use RoPE, full-attention layers
# none. The token embeddings get a weightless RMSNorm; the logits are scaled
# by logit_scale, then soft-capped.
#
# Weights as the GGUF stores them: the norms' Gemma-style +1 is baked into
# their weights, the q-norm weights carry the query pre-scale, and q and k are
# permuted for interleaved (non-NeoX) RoPE, as llama.cpp converts them.

from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)


def _text_config(config):
    return getattr(config, "text_config", config)


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


class Norm(nn.Module):
    """RMSNorm in float32 times the stored weight (no offset)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (_rms(x, self.eps) * self.weight.float()).to(x.dtype)


class MuseGlimmerAttention(nn.Module):

    def __init__(self, config, layer: int, cache_config, quant_config,
                 prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(config.hidden_size,
                                          self.head_dim,
                                          config.num_attention_heads,
                                          config.num_key_value_heads,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(config.num_attention_heads *
                                        self.head_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.gate_proj = ColumnParallelLinear(config.hidden_size,
                                              config.num_attention_heads *
                                              self.head_dim,
                                              bias=False,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.gate_proj")
        self.q_norm = Norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Norm(self.head_dim, config.rms_norm_eps)
        sliding = config.layer_types[layer] == "sliding_attention"
        self.rotary_emb = get_rope(self.head_dim,
                                   rotary_dim=self.head_dim,
                                   max_position=config.max_position_embeddings,
                                   base=config.rope_theta,
                                   is_neox_style=False) if sliding else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window
            if sliding else None,
            prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(-1, self.q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(-1, self.kv_size)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        gate, _ = self.gate_proj(x)
        o, _ = self.o_proj(o * torch.sigmoid(gate))
        return o


class MuseGlimmerMLP(nn.Module):

    def __init__(self, config, quant_config, prefix: str):
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
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gate_up_proj(x)
        h, _ = self.down_proj(self.act_fn(h))
        return h


class MuseGlimmerDecoderLayer(nn.Module):

    def __init__(self, config, layer: int, cache_config, quant_config,
                 prefix: str):
        super().__init__()
        self.self_attn = MuseGlimmerAttention(config, layer, cache_config,
                                              quant_config,
                                              f"{prefix}.self_attn")
        self.mlp = MuseGlimmerMLP(config, quant_config, f"{prefix}.mlp")
        eps, post = config.rms_norm_eps, config.post_norm_eps
        self.input_layernorm = Norm(config.hidden_size, eps)
        self.post_attention_layernorm = Norm(config.hidden_size, post)
        self.pre_feedforward_layernorm = Norm(config.hidden_size, eps)
        self.post_feedforward_layernorm = Norm(config.hidden_size, post)

    def forward(self, positions: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        h = h + self.post_attention_layernorm(
            self.self_attn(positions, self.input_layernorm(h)))
        return h + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(h)))


class MuseGlimmerModel(nn.Module):

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
            lambda prefix: MuseGlimmerDecoderLayer(
                config, int(prefix.rsplit(".", 1)[1]), cache_config,
                quant_config, prefix),
            prefix=f"{prefix}.layers")
        if get_pp_group().is_last_rank:
            self.norm = Norm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def forward(self, input_ids, positions, intermediate_tensors,
                inputs_embeds=None):
        if get_pp_group().is_first_rank:
            h = (self.embed_tokens(input_ids)
                 if inputs_embeds is None else inputs_embeds)
            h = _rms(h, self.config.rms_norm_eps).to(h.dtype)
        else:
            h = intermediate_tensors["hidden_states"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            h = layer(positions, h)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": h})
        return self.norm(h)


class MuseGlimmerForCausalLM(nn.Module, SupportsPP):
    """Muse-Glimmer text from its GGUF; also registered for
    MuseGlimmerForConditionalGeneration, without the vision tower."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        self.config = config
        self.model = MuseGlimmerModel(vllm_config=vllm_config, prefix="model")
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix="lm_head")
        else:
            self.lm_head = PPMissingLayer()
        # logits * s, then cap * tanh(. / cap), as soft_cap = cap / s and then
        # scale = s, the order LogitsProcessor applies them in.
        scale = config.output_multiplier
        cap = config.final_logit_softcapping
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            scale=scale,
            soft_cap=None if cap is None else cap / scale)
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
        """Names from muse_glimmer_gguf_name_map (the GGUF loader)."""
        stacked = [(".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
                   (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
                   (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
                   (".mlp.gate_up_proj", ".mlp.gate_proj", 0),
                   (".mlp.gate_up_proj", ".mlp.up_proj", 1)]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            for target, source, shard in stacked:
                if source not in name:
                    continue
                target = name.replace(source, target)
                if target in params and not is_pp_missing_parameter(
                        target, self):
                    params[target].weight_loader(params[target], weight,
                                                 shard)
                    loaded.add(target)
                break
            else:
                if name in params and not is_pp_missing_parameter(name, self):
                    param = params[name]
                    getattr(param, "weight_loader",
                            default_weight_loader)(param, weight)
                    loaded.add(name)
        return loaded


def muse_glimmer_gguf_name_map(config) -> dict[str, str]:
    """GGUF tensor names (arch muse-glimmer) -> this model's names."""
    m = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    per_layer = {
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "attn_gate": "self_attn.gate_proj",
        "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm",
        "attn_norm": "input_layernorm",
        "post_attention_norm": "post_attention_layernorm",
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffw_norm": "post_feedforward_layernorm",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    for i in range(_text_config(config).num_hidden_layers):
        for g, h in per_layer.items():
            m[f"blk.{i}.{g}.weight"] = f"model.layers.{i}.{h}.weight"
    return m

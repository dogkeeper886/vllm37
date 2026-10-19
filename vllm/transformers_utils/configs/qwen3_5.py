# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#127): the Qwen3.5 family's configs (model_type "qwen3_5" and
# "qwen3_5_moe", Qwen3.5 / 3.6 / 3.8), which the fork's transformers (4.57)
# predates. Only the text model is served; the vision config is kept as a
# plain dict.
#
# layers_block_type maps layer_types onto the names V0 counts layers by:
# "full_attention" layers take a KV cache ("attention"), "linear_attention"
# (Gated DeltaNet) layers take a recurrent state ("mamba").

from transformers import PretrainedConfig

_BLOCK_TYPE = {"full_attention": "attention", "linear_attention": "mamba"}


class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"

    def __init__(self, **kwargs):
        rope = kwargs.pop("rope_parameters", None) or {}
        self.rope_parameters = rope
        self.rope_theta = rope.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = rope.get("partial_rotary_factor", 0.25)
        for key, default in (("head_dim", 256), ("rms_norm_eps", 1e-6),
                             ("attn_output_gate", True),
                             ("linear_conv_kernel_dim", 4),
                             ("linear_key_head_dim", 128),
                             ("linear_value_head_dim", 128),
                             ("full_attention_interval", 4),
                             ("hidden_act", "silu"),
                             ("attention_bias", False)):
            setattr(self, key, kwargs.pop(key, default))
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads",
                    "linear_num_key_heads", "linear_num_value_heads",
                    "vocab_size", "max_position_embeddings", "layer_types",
                    "num_experts", "num_experts_per_tok",
                    "moe_intermediate_size",
                    "shared_expert_intermediate_size"):
            setattr(self, key, kwargs.pop(key, None))
        # Top-k routing weights renormalised, as upstream's default.
        self.norm_topk_prob = kwargs.pop("norm_topk_prob", True)
        if self.layer_types is None and self.num_hidden_layers:
            n = self.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % n == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__init__(**kwargs)

    @property
    def layers_block_type(self) -> list[str]:
        return [_BLOCK_TYPE[t] for t in self.layer_types]


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_moe_text"


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"text_config": Qwen3_5TextConfig}
    _text_config_class = Qwen3_5TextConfig

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = self._text_config_class(**text_config)
        self.text_config = text_config or self._text_config_class()
        self.vision_config = vision_config
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config

    @property
    def layers_block_type(self) -> list[str]:
        return self.text_config.layers_block_type


class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type = "qwen3_5_moe"
    sub_configs = {"text_config": Qwen3_5MoeTextConfig}
    _text_config_class = Qwen3_5MoeTextConfig

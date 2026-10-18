# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#126): the Gemma 4 config (model_type "gemma4"), which the fork's
# transformers (4.57) predates. Only the text model is served; the vision and
# audio configs are kept as plain dicts.
#
# The attention pattern is stored as gemma4_layer_types and
# gemma4_sliding_window, not layer_types and sliding_window: V0 treats a
# config with both "sliding_attention" and "full_attention" layers as
# interleaved and, on XFormers, drops the window and caps the context at it
# (1,024 tokens). The fork's Gemma 4 model applies each layer's window in its
# own attention instead.

from transformers import PretrainedConfig


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"

    def __init__(self, **kwargs):
        self.gemma4_layer_types = kwargs.pop("layer_types", None)
        self.gemma4_sliding_window = kwargs.pop("sliding_window", None)
        for key, default in (("head_dim", 256), ("num_key_value_heads", 8),
                             ("num_attention_heads", 16),
                             ("rms_norm_eps", 1e-6),
                             ("final_logit_softcapping", None),
                             ("attention_k_eq_v", False),
                             ("num_kv_shared_layers", 0),
                             ("hidden_size_per_layer_input", 0),
                             ("rope_parameters", {})):
            setattr(self, key, kwargs.pop(key, default))
        self.global_head_dim = kwargs.pop("global_head_dim", self.head_dim)
        self.num_global_key_value_heads = kwargs.pop(
            "num_global_key_value_heads", self.num_key_value_heads)
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "vocab_size", "max_position_embeddings",
                    "hidden_activation", "num_experts", "top_k_experts",
                    "moe_intermediate_size"):
            setattr(self, key, kwargs.pop(key, None))
        self.sliding_window = None
        super().__init__(**kwargs)


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"
    sub_configs = {"text_config": Gemma4TextConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Gemma4TextConfig(**text_config)
        self.text_config = text_config or Gemma4TextConfig()
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config

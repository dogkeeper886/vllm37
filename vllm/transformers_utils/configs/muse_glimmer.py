# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#130): the Muse-Glimmer config (model_type "muse_glimmer"), which
# the fork's transformers (4.57) predates. Only the text model is served; the
# vision config is kept as a plain dict.

from transformers import PretrainedConfig


class MuseGlimmerTextConfig(PretrainedConfig):
    model_type = "muse_glimmer_text"

    def __init__(self, **kwargs):
        rope = kwargs.pop("rope_parameters", None) or {}
        self.rope_parameters = rope
        self.rope_theta = rope.get("rope_theta", 500_000.0)
        for key, default in (("head_dim", 128), ("rms_norm_eps", 1e-5),
                             ("post_norm_eps", 1e-8),
                             ("final_logit_softcapping", None),
                             ("output_multiplier", 1.0),
                             ("qk_scale_factor", None),
                             ("sliding_window", None),
                             ("hidden_activation", "silu"),
                             ("attention_bias", False)):
            setattr(self, key, kwargs.pop(key, default))
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads",
                    "vocab_size", "max_position_embeddings", "layer_types",
                    "layer_rope_theta"):
            setattr(self, key, kwargs.pop(key, None))
        super().__init__(**kwargs)


class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"
    sub_configs = {"text_config": MuseGlimmerTextConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = MuseGlimmerTextConfig(**text_config)
        self.text_config = text_config or MuseGlimmerTextConfig()
        self.vision_config = vision_config
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config

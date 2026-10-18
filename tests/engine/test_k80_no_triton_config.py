# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#123): without Triton, the V0 engine turns off chunked prefill
and prefix caching, whose prefill kernel (context_attention_fwd) is Triton,
and lets one step hold a whole prompt.

Run on a host without Triton: pytest tests/engine/test_k80_no_triton_config.py
"""
import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(HAS_TRITON,
                                reason="checks the config without Triton")

MODEL = "Qwen/Qwen3-0.6B"
CASES = [
    # above 32K, vLLM turns chunked prefill on by default
    dict(max_model_len=40960),
    dict(max_model_len=8192, enable_prefix_caching=True),
    dict(max_model_len=8192,
         enable_chunked_prefill=True,
         max_num_batched_tokens=1024),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_prefix_prefill_off_without_triton(case):
    kwargs = CASES[case]
    config = EngineArgs(model=MODEL, **kwargs).create_engine_config()
    assert not config.scheduler_config.chunked_prefill_enabled
    assert not config.cache_config.enable_prefix_caching
    assert (config.scheduler_config.max_num_batched_tokens
            >= kwargs["max_model_len"])


def test_short_context_unchanged():
    """At 32K and below, chunked prefill is off already and the step size
    stays as set."""
    config = EngineArgs(model=MODEL,
                        max_model_len=4096,
                        max_num_batched_tokens=8192).create_engine_config()
    assert not config.scheduler_config.chunked_prefill_enabled
    assert config.scheduler_config.max_num_batched_tokens == 8192

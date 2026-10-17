# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#86): guided decoding on the V0 engine (xgrammar through
vllm/engine/k80_guided_decoding.py). Every guided reply must match its
constraint, for one sample and for two.

Run on a Kepler GPU: pytest tests/entrypoints/llm/test_k80_guided_decoding.py
"""
import json
import re

import pytest
import torch

from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    on_kepler)
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()),
    reason="the fork's V0 guided decoding is tested on Kepler")

MODEL = "Qwen/Qwen3-0.6B"
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        }
    },
    "required": ["name", "age"],
}
_llm = None


def llm():
    global _llm
    if _llm is None:
        from vllm import LLM
        _llm = LLM(model=MODEL,
                   dtype="half",
                   enforce_eager=True,
                   max_model_len=2048,
                   gpu_memory_utilization=0.6,
                   seed=0)
    return _llm


def generate(prompt, guided, n):
    params = SamplingParams(temperature=0 if n == 1 else 0.8,
                            n=n,
                            max_tokens=80,
                            guided_decoding=guided)
    return [o.text for o in llm().generate([prompt], params)[0].outputs]


def _json_person(text):
    value = json.loads(text)
    return isinstance(value["name"], str) and isinstance(value["age"], int)


CASES = [
    ("Give a JSON object with a name and an age for a fictional person:",
     GuidedDecodingParams(json=SCHEMA), _json_person),
    ("The phone number is", GuidedDecodingParams(regex=r"\d{3}-\d{4}"),
     lambda t: re.fullmatch(r"\d{3}-\d{4}", t) is not None),
    ("Is the sky blue? Answer:", GuidedDecodingParams(choice=["yes", "no"]),
     lambda t: t in ("yes", "no")),
    ("Return any JSON object:", GuidedDecodingParams(json_object=True),
     lambda t: isinstance(json.loads(t), dict)),
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("n", [1, 2])
def test_guided_reply_matches(case, n):
    prompt, guided, check = CASES[case]
    texts = generate(prompt, guided, n)
    assert len(texts) == n
    for text in texts:
        assert check(text), text


def test_other_backend_is_refused():
    guided = GuidedDecodingParams(json=SCHEMA, backend="outlines")
    try:
        generate("x", guided, 1)
    except ValueError as e:
        assert "xgrammar" in str(e)
    else:
        raise AssertionError("the outlines backend was accepted")

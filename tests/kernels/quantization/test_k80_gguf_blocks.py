# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#132): a tensor-parallel split that cuts a GGUF block is refused
at load time, for dense row-parallel layers and MoE w2, with a message that
names the width, the block and a die count that works.

Run: pytest tests/kernels/quantization/test_k80_gguf_blocks.py
"""
from types import SimpleNamespace

import gguf
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

import vllm.distributed
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.k80_gguf import (
    K80GGUFLinearMethod, K80GGUFMoEMethod, k80_gguf_check_blocks)

Q4_K = int(WeightType.Q4_K)
Q6_K = int(WeightType.Q6_K)
Q4_0 = int(WeightType.Q4_0)


def die_bytes(k, tp, qtype):
    """The bytes of one die's share of a row of k features, as the GGUF
    loader splits a row-parallel weight: the row's bytes // tp."""
    block, size = gguf.GGML_QUANT_SIZES[WeightType(qtype)]
    return k // block * size // tp


@pytest.fixture
def dies(monkeypatch):

    def set_dies(n):
        monkeypatch.setattr(vllm.distributed,
                            "get_tensor_model_parallel_world_size", lambda: n)

    return set_dies


# The study models' Q4_K_M files at TP=4 (#132): Muse-Glimmer-30B's
# down_proj (19,968 features) and Ornith-1.5-35B-A3B's expert down_proj
# (512).
@pytest.mark.parametrize("k, qtype", [(19968, Q4_K), (19968, Q6_K),
                                      (512, Q4_K)])
def test_cut_block_refused(dies, k, qtype):
    dies(4)
    with pytest.raises(ValueError) as e:
        k80_gguf_check_blocks(die_bytes(k, 4, qtype), qtype, k, "down_proj")
    msg = str(e.value)
    assert f"{k} input features" in msg and "blocks of 256" in msg
    assert f"a die's {k // 4} features cut a block" in msg
    assert "tensor_parallel_size 2" in msg


@pytest.mark.parametrize("k, tp, qtype", [(19968, 2, Q4_K), (512, 2, Q4_K),
                                          (19968, 4, Q4_0), (3072, 4, Q4_K),
                                          (512, 4, int(WeightType.F16))])
def test_whole_blocks_pass(dies, k, tp, qtype):
    """Shares of whole blocks load: TP=2 for both study models, blocks of 32
    at TP=4, Qwen3-0.6B's 3,072 at TP=4, and unquantized tensors."""
    dies(tp)
    k80_gguf_check_blocks(die_bytes(k, tp, qtype), qtype, k, "down_proj")


def test_dense_layer_refused_at_load(dies):
    """K80GGUFLinearMethod refuses the shard before it repacks anything."""
    dies(4)
    raw = torch.zeros(8, die_bytes(19968, 4, Q4_K), dtype=torch.uint8)
    layer = SimpleNamespace(
        qweight=SimpleNamespace(shard_id=None, data=raw),
        qweight_type=SimpleNamespace(weight_type=Q4_K, device="cpu"),
        input_size=19968,
        prefix="model.layers.0.mlp.down_proj")
    with pytest.raises(ValueError, match="model.layers.0.mlp.down_proj has "
                       "19968 input features"):
        K80GGUFLinearMethod(GGUFConfig()).process_weights_after_loading(layer)


def test_moe_w2_refused_at_load(dies):
    """K80GGUFMoEMethod refuses a die's share of w2 that cuts a block."""
    dies(4)
    hidden, inter = 256, 512
    w13 = torch.zeros(2, 2 * inter // 4, die_bytes(hidden, 1, Q4_K),
                      dtype=torch.uint8)
    w2 = torch.zeros(2, hidden, die_bytes(inter, 4, Q4_K), dtype=torch.uint8)
    layer = SimpleNamespace(
        w13_qweight=SimpleNamespace(data=w13),
        w13_qweight_type=SimpleNamespace(weight_type=Q4_K, device="cpu"),
        w2_qweight=SimpleNamespace(data=w2),
        w2_qweight_type=SimpleNamespace(weight_type=Q4_K, device="cpu"),
        layer_name="model.layers.0.mlp.experts")
    with pytest.raises(ValueError, match="model.layers.0.mlp.experts.w2 has "
                       "512 input features, 2 Q4_K blocks of 256"):
        K80GGUFMoEMethod(GGUFConfig()).process_weights_after_loading(layer)

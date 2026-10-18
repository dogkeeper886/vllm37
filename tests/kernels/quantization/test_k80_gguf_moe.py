# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#131): GGUF MoE experts on Kepler. Each type's repack into the
K80 int8 layout against the gguf package's dequantize, and K80GGUFMoEMethod's
experts (int4 / int8 kernels, or dequantized) against a float64 reference.

Run on a Kepler GPU: pytest tests/kernels/quantization/test_k80_gguf_moe.py
"""
import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gguf import GGMLQuantizationType as WeightType

from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    k80_int4_dequantize, on_kepler)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()
         and hasattr(torch.ops._C, "k80_int4_moe_gemm")),
    reason="K80 MoE kernels run on Kepler only")

# Byte offsets of each type's fp16 fields within a block.
F16_FIELDS = {
    WeightType.Q4_0: [0], WeightType.Q4_1: [0, 2], WeightType.Q4_K: [0, 2],
    WeightType.Q5_0: [0], WeightType.Q5_1: [0, 2], WeightType.Q5_K: [0, 2],
    WeightType.Q8_0: [0], WeightType.Q6_K: [-2], WeightType.Q3_K: [-2],
}
INT8_TYPES = [WeightType.Q8_0, WeightType.Q5_0, WeightType.Q5_1,
              WeightType.Q5_K, WeightType.Q6_K]


def blocks(qtype, rows, k, seed=0):
    """Random GGUF rows [rows, bytes] of `qtype` with finite fp16 fields."""
    rng = np.random.default_rng(seed)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, (rows, k // block, size), dtype=np.uint8)
    for at in F16_FIELDS[qtype]:
        f16 = rng.uniform(1e-3, 1e-2, raw.shape[:2]).astype(np.float16)
        raw[..., at:at + 2 if at != -2 else None] = f16[..., None].view(
            np.uint8)
    return raw.reshape(rows, -1)


def ref(raw, qtype):
    shape = raw.shape
    flat = raw.reshape(-1, shape[-1])
    w = gguf.quants.dequantize(flat, qtype)
    return torch.from_numpy(w.reshape(*shape[:-1], -1)).cuda().double()


@pytest.mark.parametrize("qtype", INT8_TYPES)
def test_int8_repack_matches_gguf(qtype):
    from vllm.model_executor.layers.quantization.k80_gguf import (
        k80_gguf_moe_int8)
    rows, k = 64, 1024
    raw = blocks(qtype, rows, k)
    q, s, z = k80_gguf_moe_int8(torch.from_numpy(raw).cuda(), qtype)
    group = 16 if qtype == WeightType.Q6_K else 32
    assert q.shape == (rows, k // 4) and s.shape == (rows, k // group)
    w = k80_int4_dequantize(q, s, z, torch.float32, bits=8).double()
    expected = ref(raw, qtype)
    if qtype in (WeightType.Q5_1, WeightType.Q5_K):
        # (q - m / d) * d against d * q - m: one FP32 rounding
        err = (w - expected).abs().max() / expected.abs().max()
        assert err.item() < 1e-6
    else:  # (q - zero) * scale with an integer zero, as gguf computes it
        assert torch.equal(w, expected)


class Layer(torch.nn.Module):

    def __init__(self, w13, w13_type, w2, w2_type):
        super().__init__()
        for name, raw, qtype in (("w13", w13, w13_type), ("w2", w2, w2_type)):
            self.register_parameter(
                f"{name}_qweight",
                torch.nn.Parameter(torch.from_numpy(raw).cuda(),
                                   requires_grad=False))
            t = torch.nn.Parameter(torch.tensor([int(qtype)],
                                                dtype=torch.uint8).cuda(),
                                   requires_grad=False)
            t.weight_type = int(qtype)
            self.register_parameter(f"{name}_qweight_type", t)


def moe64(x, w13, w2, topk_weights, topk_ids):
    x = x.double()
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float64,
                      device=x.device)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = topk_ids[t, k].item()
            gate, up = (w13[e] @ x[t]).chunk(2)
            out[t] += topk_weights[t, k].double() * (w2[e] @
                                                     (F.silu(gate) * up))
    return out


# (w13 type, w2 type): Q4_0 as gemma-4's QAT GGUF; Q4_K + Q6_K as a Q4_K_M
# file; mixed int8 types; Q3_K takes the dequantized path.
PAIRS = [(WeightType.Q4_0, WeightType.Q4_0), (WeightType.Q4_K, WeightType.Q6_K),
         (WeightType.Q8_0, WeightType.Q5_K), (WeightType.Q5_0, WeightType.Q5_1),
         (WeightType.Q3_K, WeightType.Q4_0)]


@pytest.mark.parametrize("pair", range(len(PAIRS)))
@pytest.mark.parametrize("tokens", [37, 300])  # pair mode, block mode
def test_gguf_moe_experts(pair, tokens):
    from vllm.model_executor.layers.fused_moe.k80_moe import (
        k80_fused_experts, k80_moe_experts)
    from vllm.model_executor.layers.quantization.k80_gguf import (
        K80GGUFMoEMethod)
    w13_type, w2_type = PAIRS[pair]
    e, hidden, inter, topk = 8, 512, 256, 2
    w13 = np.stack([blocks(w13_type, 2 * inter, hidden, seed=i)
                    for i in range(e)])
    w2 = np.stack([blocks(w2_type, hidden, inter, seed=100 + i)
                   for i in range(e)])
    layer = Layer(w13, w13_type, w2, w2_type)
    method = K80GGUFMoEMethod.__new__(K80GGUFMoEMethod)
    method.params_dtype = torch.float32
    method.process_weights_after_loading(layer)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", generator=g)
    ids = torch.stack([
        torch.randperm(e, device="cuda", generator=g)[:topk]
        for _ in range(tokens)
    ]).to(torch.int32)
    w = torch.rand(tokens, topk, device="cuda", generator=g)
    w = w / w.sum(-1, keepdim=True)
    if method.dense:
        assert w13_type == WeightType.Q3_K
        y = k80_fused_experts(x, layer.k80_w13_weight, layer.k80_w2_weight, w,
                              ids)
    else:
        y = k80_moe_experts(
            x, (layer.k80_w13_qweight, layer.k80_w13_scales,
                layer.k80_w13_zeros),
            (layer.k80_w2_qweight, layer.k80_w2_scales, layer.k80_w2_zeros),
            layer.k80_bits, w, ids, w2_bits=layer.k80_w2_bits)
    expected = moe64(x, ref(w13, w13_type), ref(w2, w2_type), w, ids)
    err = ((y.double() - expected).norm() / expected.norm()).item()
    assert err < 1e-5, err


def test_split_block_is_refused():
    """A die's share of w2 that cuts a block fails at load time (#132)."""
    from vllm.model_executor.layers.quantization.k80_gguf import (
        K80GGUFMoEMethod)
    w13 = np.stack([blocks(WeightType.Q4_K, 512, 512) for _ in range(2)])
    w2 = np.stack([blocks(WeightType.Q4_K, 512, 256) for _ in range(2)])
    layer = Layer(w13, WeightType.Q4_K, w2[..., :72], WeightType.Q4_K)
    method = K80GGUFMoEMethod.__new__(K80GGUFMoEMethod)
    method.params_dtype = torch.float32
    try:
        method.process_weights_after_loading(layer)
    except ValueError as err:
        assert "Q4_K" in str(err)
    else:
        raise AssertionError("a split Q4_K block was accepted")

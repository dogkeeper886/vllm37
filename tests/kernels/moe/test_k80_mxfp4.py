# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#128): MXFP4 experts on Kepler, against a float64 reference:
k80_mxfp4_moe_gemm (pair and block mode) and k80_mxfp4_experts (gpt-oss's
biases and swiglu_oai).

Run on a Kepler GPU: pytest tests/kernels/moe/test_k80_mxfp4.py
"""
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.k80_moe import (
    K80_MOE_BLOCK, K80_MOE_PAIR_SLOTS, k80_mxfp4_experts)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    on_kepler)
from vllm.model_executor.layers.quantization.mxfp4 import K80Mxfp4MoEMethod

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()
         and hasattr(torch.ops._C, "k80_mxfp4_moe_gemm")),
    reason="K80 MXFP4 kernels run on Kepler only")

DTYPES = [torch.float32, torch.float16]
# E2M1 code -> value, bit 3 the sign (the OCP MX spec, transformers'
# FP4_VALUES).
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-v for v in E2M1]
# (experts, N, K): gpt-oss-20b per die at TP=2 (gate_up and down), and K a
# multiple of 32 only.
SHAPES = [(32, 2880, 2880), (32, 2880, 1440), (8, 72, 96)]


def make(e, n, k, seed=0, exps=(120, 131)):
    """The codes as gpt-oss stores them (uint8 [E, N, K / 2], byte j: value
    2j low, 2j + 1 high) seen as int32, and E8M0 scales in exps."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8,
                          device="cuda", generator=g)
    scales = torch.randint(*exps, (e, n, k // 32), dtype=torch.uint8,
                           device="cuda", generator=g)
    return codes.view(torch.int32), scales


def weight64(qweight, scales):
    """[E, N, K] float64."""
    b = qweight.view(torch.uint8).long()
    lut = torch.tensor(E2M1, dtype=torch.float64, device=b.device)
    v = torch.stack((lut[b & 15], lut[b >> 4]), -1).flatten(-2)
    e, n, k = v.shape
    s = torch.exp2(scales.double() - 127)
    return (v.view(e, n, k // 32, 32) * s.unsqueeze(-1)).view(e, n, k)


def route(t, experts, topk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack([
        torch.randperm(experts, device="cuda", generator=g)[:topk]
        for _ in range(t)
    ]).to(torch.int32)
    w = torch.rand(t, topk, device="cuda", generator=g)
    return w / w.sum(-1, keepdim=True), ids


def swiglu_oai64(h):
    gate, up = h[..., ::2].clamp(max=7.0), h[..., 1::2].clamp(-7.0, 7.0)
    return (up + 1) * gate * torch.sigmoid(gate * 1.702)


def moe64(x, w13, b13, w2, b2, topk_weights, topk_ids):
    """float64 reference of gpt-oss's experts: w13 [E, 2I, H] with gate and
    up interleaved, w2 [E, H, I], biases [E, N]."""
    x = x.double()
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float64,
                      device=x.device)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = topk_ids[t, k].item()
            h = w13[e] @ x[t] + b13[e].double()
            out[t] += topk_weights[t, k].double() * (
                w2[e] @ swiglu_oai64(h) + b2[e].double())
    return out


def rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("blocks", [False, True])
def test_every_code(dtype, blocks):
    """Each of the 16 codes, one per input, at the scales the format can take
    from 2^-20 to 2^20: the kernel decodes each value exactly."""
    k = 32
    codes = torch.arange(16, dtype=torch.uint8, device="cuda")
    packed = (codes[0::2] | (codes[1::2] << 4)).repeat(2)  # 32 values
    exps = torch.arange(107, 148, dtype=torch.uint8, device="cuda")
    n = exps.numel()
    qweight = packed.expand(n, -1).contiguous().view(torch.int32)[None]
    scales = exps.view(1, n, 1).contiguous()
    w = weight64(qweight, scales)[0]
    for i in range(k):
        x = torch.zeros(1, k, dtype=dtype, device="cuda")
        x[0, i] = 1
        experts = torch.zeros(1, dtype=torch.int32, device="cuda")
        if blocks:
            ids = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
            slots, experts, padded = moe_align_block_size(ids,
                                                          K80_MOE_BLOCK,
                                                          1,
                                                          pad_sorted_ids=True)
        else:
            slots = padded = None
        y = ops.k80_mxfp4_moe_gemm(x, qweight, scales, experts, slots, padded,
                                   1, 1)
        ref = w[:, i]
        if dtype == torch.float16:  # the output rounds to half
            ref = ref.half().double()
        torch.testing.assert_close(y[0].double(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tokens", [1, 4, 37])
@pytest.mark.parametrize("blocks", [False, True])
def test_mxfp4_moe_gemm(dtype, shape, tokens, blocks):
    e, n, k = shape
    topk = min(4, e)
    qweight, scales = make(e, n, k)
    w = weight64(qweight, scales)
    _, ids = route(tokens, e, topk)
    x = torch.randn(tokens, k, device="cuda").to(dtype)
    num_slots = ids.numel()
    if blocks:
        slots, experts, padded = moe_align_block_size(ids,
                                                      K80_MOE_BLOCK,
                                                      e,
                                                      pad_sorted_ids=True)
    else:
        experts, slots, padded = ids.flatten(), None, None
    y = ops.k80_mxfp4_moe_gemm(x, qweight, scales, experts, slots, padded,
                               num_slots, topk)
    assert y.shape == (num_slots, n) and y.dtype == dtype
    flat = ids.flatten().tolist()
    ref = torch.stack(
        [w[e] @ x[i // topk].double() for i, e in enumerate(flat)])
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    err = (y.double() - ref).norm(dim=1) / ref.norm(dim=1)
    assert err.max().item() < tol


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens", [1, 4, K80_MOE_PAIR_SLOTS // 4,
                                    K80_MOE_PAIR_SLOTS // 4 + 1, 100])
def test_k80_mxfp4_experts(dtype, tokens):
    """gpt-oss-20b's expert shapes per die at TP=2, top 4 of 32."""
    e, hidden, inter, topk = 32, 2880, 1440, 4
    # Scales of 2^-9 .. 2^-1, as gpt-oss's are, keep FP16 in range.
    w13 = make(e, 2 * inter, hidden, seed=1, exps=(118, 127))
    w2 = make(e, hidden, inter, seed=2, exps=(118, 127))
    g = torch.Generator(device="cuda").manual_seed(3)
    b13 = torch.randn(e, 2 * inter, device="cuda", generator=g).to(dtype)
    b2 = torch.randn(e, hidden, device="cuda", generator=g).to(dtype)
    topk_weights, topk_ids = route(tokens, e, topk, seed=4)
    x = torch.randn(tokens, hidden, device="cuda", generator=g).to(dtype)
    y = k80_mxfp4_experts(x, w13, w2, topk_weights, topk_ids, "swiglu_oai",
                          b13, b2)
    assert y.shape == x.shape and y.dtype == dtype
    w13_64 = weight64(*w13)
    h = torch.einsum("enk,tk->ten", w13_64, x.double())
    # swiglu_oai's clamps take effect.
    assert (h[..., ::2] > 7).any() and (h[..., 1::2].abs() > 7).any()
    ref = moe64(x, w13_64, b13, weight64(*w2), b2, topk_weights, topk_ids)
    assert rel(y, ref) < (1e-5 if dtype == torch.float32 else 3e-3)


def test_k80_mxfp4_experts_chunks(monkeypatch):
    """A batch of more than K80_MOE_CHUNK_SLOTS slots goes in chunks of
    tokens, with the same result."""
    from vllm.model_executor.layers.fused_moe import k80_moe
    e, hidden, inter, topk, tokens = 8, 256, 128, 4, 50
    w13 = make(e, 2 * inter, hidden, seed=7, exps=(118, 127))
    w2 = make(e, hidden, inter, seed=8, exps=(118, 127))
    b13 = torch.randn(e, 2 * inter, device="cuda")
    b2 = torch.randn(e, hidden, device="cuda")
    topk_weights, topk_ids = route(tokens, e, topk, seed=9)
    x = torch.randn(tokens, hidden, device="cuda")
    whole = k80_mxfp4_experts(x, w13, w2, topk_weights, topk_ids,
                              "swiglu_oai", b13, b2)
    monkeypatch.setattr(k80_moe, "K80_MOE_CHUNK_SLOTS", 4 * 16)
    chunked = k80_mxfp4_experts(x, w13, w2, topk_weights, topk_ids,
                                "swiglu_oai", b13, b2)
    torch.testing.assert_close(chunked, whole, rtol=0, atol=0)
    ref = moe64(x, weight64(*w13), b13, weight64(*w2), b2, topk_weights,
                topk_ids)
    assert rel(chunked, ref) < 1e-5


def test_split_block_refused():
    """A die's share of K that cuts an MXFP4 block of 32 is refused."""
    method = K80Mxfp4MoEMethod(None)
    layer = torch.nn.Module()
    with pytest.raises(ValueError, match="blocks of 32"):
        method.create_weights(layer, 32, 2880, 728, torch.float32)
    method.create_weights(layer, 4, 64, 96, torch.float32)
    assert layer.w13_weight.shape == (4, 192, 32)
    assert layer.w2_weight_scale.shape == (4, 64, 3)

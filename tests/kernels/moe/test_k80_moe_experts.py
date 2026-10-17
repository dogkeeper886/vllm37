# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#119): MoE experts on Kepler without Triton, against a float64
reference: k80_int4_moe_gemm (pair and block mode), k80_moe_experts (int4 and
int8 experts) and k80_fused_experts (FP16 / FP32 experts).

Run on a Kepler GPU: pytest tests/kernels/moe/test_k80_moe_experts.py
"""
import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.k80_moe import (
    K80_MOE_BLOCK, K80_MOE_PAIR_SLOTS, k80_check_groups, k80_fused_experts,
    k80_moe_experts, k80_repack_wna16)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    on_kepler)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()
         and hasattr(torch.ops._C, "k80_int4_moe_gemm")),
    reason="K80 MoE kernels run on Kepler only")

DTYPES = [torch.float32, torch.float16]
BITS = [4, 8]
# (experts, N, K, group size): Qwen3-30B-A3B per die at TP=2 (gate_up and
# down), K a multiple of 16 only, and of 8 only.
SHAPES = [(128, 768, 2048, 128), (128, 2048, 384, 128), (16, 520, 1040, 16),
          (8, 264, 776, 8)]


def make(e, n, k, gs, bits, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qweight = torch.randint(-2**31,
                            2**31 - 1, (e, n, k * bits // 32),
                            dtype=torch.int32,
                            device="cuda",
                            generator=g)
    scales = torch.rand(e, n, k // gs, device="cuda", generator=g) * 0.01 + 1e-3
    zeros = torch.randint(0, 2**bits, (e, n, k // gs), device="cuda",
                          generator=g).float()
    return qweight, scales, zeros


def weight64(qweight, scales, zeros, bits):
    """[E, N, K] float64."""
    e, n = qweight.shape[:2]
    b = qweight.view(torch.uint8)
    if bits == 4:
        b = torch.stack((b & 15, b >> 4), -1)
    w = b.reshape(e, n, scales.shape[2], -1)
    return ((w.double() - zeros.double().unsqueeze(-1)) *
            scales.double().unsqueeze(-1)).reshape(e, n, -1)


def route(t, experts, topk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack([
        torch.randperm(experts, device="cuda", generator=g)[:topk]
        for _ in range(t)
    ]).to(torch.int32)
    w = torch.rand(t, topk, device="cuda", generator=g)
    return w / w.sum(-1, keepdim=True), ids


def moe64(x, w13, w2, topk_weights, topk_ids, activation="silu"):
    """float64 reference: w13 [E, 2I, H], w2 [E, H, I]."""
    x = x.double()
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float64,
                      device=x.device)
    act = F.silu if activation == "silu" else F.gelu
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = topk_ids[t, k].item()
            gate, up = (w13[e].double() @ x[t]).chunk(2)
            out[t] += topk_weights[t, k].double() * (w2[e].double() @
                                                     (act(gate) * up))
    return out


def rel(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tokens", [1, 4, 37])
@pytest.mark.parametrize("blocks", [False, True])
def test_moe_gemm(bits, dtype, shape, tokens, blocks):
    e, n, k, gs = shape
    topk = min(8, e)
    qweight, scales, zeros = make(e, n, k, gs, bits)
    w = weight64(qweight, scales, zeros, bits)
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
    y = ops.k80_int4_moe_gemm(x, qweight, scales, zeros, experts, slots,
                              padded, num_slots, topk, bits)
    assert y.shape == (num_slots, n) and y.dtype == dtype
    flat = ids.flatten().long()
    ref = torch.einsum("snk,sk->sn", w[flat],
                       x.double().repeat_interleave(topk, 0))
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    err = (y.double() - ref).norm(dim=1) / ref.norm(dim=1)
    assert err.max().item() < tol


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens", [1, 4, K80_MOE_PAIR_SLOTS // 8,
                                    K80_MOE_PAIR_SLOTS // 8 + 1, 100])
def test_k80_moe_experts(bits, dtype, tokens):
    e, hidden, inter, gs, topk = 32, 1024, 256, 128, 8
    w13 = make(e, 2 * inter, hidden, gs, bits, seed=1)
    w2 = make(e, hidden, inter, gs, bits, seed=2)
    topk_weights, topk_ids = route(tokens, e, topk, seed=3)
    x = torch.randn(tokens, hidden, device="cuda").to(dtype)
    y = k80_moe_experts(x, w13, w2, bits, topk_weights, topk_ids)
    assert y.shape == x.shape and y.dtype == dtype
    ref = moe64(x, weight64(*w13, bits), weight64(*w2, bits), topk_weights,
                topk_ids)
    assert rel(y, ref) < (1e-5 if dtype == torch.float32 else 3e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_k80_moe_experts_weight_on_input(dtype):
    """apply_router_weight_on_input (top-1): the weight scales the input."""
    e, hidden, inter, gs = 16, 512, 256, 64
    w13 = make(e, 2 * inter, hidden, gs, 4, seed=4)
    w2 = make(e, hidden, inter, gs, 4, seed=5)
    topk_weights, topk_ids = route(5, e, 1, seed=6)
    topk_weights = topk_weights * 0.7
    x = torch.randn(5, hidden, device="cuda").to(dtype)
    y = k80_moe_experts(x, w13, w2, 4, topk_weights, topk_ids,
                        apply_router_weight_on_input=True)
    w13_64, w2_64 = weight64(*w13, 4), weight64(*w2, 4)
    ref = moe64(x * topk_weights.to(dtype), w13_64, w2_64,
                torch.ones_like(topk_weights), topk_ids)
    assert rel(y, ref) < (1e-5 if dtype == torch.float32 else 3e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("tokens", [1, K80_MOE_PAIR_SLOTS // 8,
                                    K80_MOE_PAIR_SLOTS // 8 + 1, 100])
def test_k80_fused_experts(dtype, activation, tokens):
    e, hidden, inter, topk = 32, 1024, 512, 8
    g = torch.Generator(device="cuda").manual_seed(7)
    w1 = (torch.randn(e, 2 * inter, hidden, device="cuda", generator=g) /
          hidden**0.5).to(dtype)
    w2 = (torch.randn(e, hidden, inter, device="cuda", generator=g) /
          inter**0.5).to(dtype)
    topk_weights, topk_ids = route(tokens, e, topk, seed=8)
    x = torch.randn(tokens, hidden, device="cuda").to(dtype)
    ref = moe64(x, w1, w2, topk_weights, topk_ids, activation)
    y = k80_fused_experts(x.clone(), w1, w2, topk_weights, topk_ids,
                          activation=activation)
    assert y.shape == x.shape and y.dtype == dtype
    assert rel(y, ref) < (1e-5 if dtype == torch.float32 else 3e-3)
    # inplace=True, as UnquantizedFusedMoEMethod calls it
    xi = x.clone()
    yi = k80_fused_experts(xi, w1, w2, topk_weights, topk_ids, inplace=True,
                           activation=activation)
    assert yi.data_ptr() == xi.data_ptr() and torch.equal(yi, y)


@pytest.mark.parametrize("bits", BITS)
def test_repack_wna16(bits):
    """compressed-tensors / GPTQ MoE layout [E, K / P, N] to the K80 one."""
    e, n, k, gs = 4, 96, 256, 64
    qweight, scales, _ = make(e, n, k, gs, bits)
    q, s, z = k80_repack_wna16(qweight.transpose(1, 2).contiguous(),
                               scales.transpose(1, 2).half(), bits,
                               2**(bits - 1))
    assert torch.equal(q, qweight)
    assert torch.equal(s, scales.half().float())
    assert torch.all(z == 2**(bits - 1)) and z.shape == s.shape


@pytest.mark.parametrize("tokens", [4, K80_MOE_PAIR_SLOTS // 8 + 1])
@pytest.mark.parametrize("quantized", [True, False])
def test_cuda_graph(tokens, quantized):
    """Both slot modes need no host sync: a captured step replays with new
    inputs and routing (FP16 above K80_MOE_PAIR_SLOTS stays on the slot
    kernel while capturing)."""
    e, hidden, inter, gs, topk = 32, 512, 256, 128, 8
    if quantized:
        w13 = make(e, 2 * inter, hidden, gs, 4, seed=9)
        w2 = make(e, hidden, inter, gs, 4, seed=10)

        def step(x, w, ids):
            return k80_moe_experts(x, w13, w2, 4, w, ids)
    else:
        g = torch.Generator(device="cuda").manual_seed(9)
        w1 = (torch.randn(e, 2 * inter, hidden, device="cuda", generator=g) /
              hidden**0.5).half()
        w2 = (torch.randn(e, hidden, inter, device="cuda", generator=g) /
              inter**0.5).half()

        def step(x, w, ids):
            return k80_fused_experts(x, w1, w2, w, ids)

    x = torch.randn(tokens, hidden, device="cuda").half()
    w, ids = route(tokens, e, topk, seed=11)
    step(x, w, ids)  # warm-up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(x, w, ids)
    w_new, ids_new = route(tokens, e, topk, seed=12)
    x.copy_(torch.randn_like(x))
    w.copy_(w_new)
    ids.copy_(ids_new)
    graph.replay()
    torch.cuda.synchronize()
    expected = step(x, w, ids)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


def test_check_groups():
    """A die count that splits a quantization group is refused."""
    k80_check_groups(384, 128)  # Qwen3-30B-A3B at TP=2
    k80_check_groups(192, -1)  # channelwise
    try:
        k80_check_groups(192, 128)  # Qwen3-30B-A3B at TP=4
    except ValueError as e:
        assert "groups of 128" in str(e)
    else:
        raise AssertionError("192 features in groups of 128 passed")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#107, #111, #109): GGUF tensors on Kepler (repacked into the K80
int4 or int8 layout, or read as stored by the K80 GGUF kernels), against the
gguf package's dequantize.

Run on a Kepler GPU: pytest tests/kernels/quantization/test_k80_gguf.py
"""
import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    k80_int4_dequantize, on_kepler)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()),
    reason="the K80 GGUF path runs on Kepler only")

ROWS, K = 96, 1024


def blocks(qtype, rows=ROWS, k=K, seed=0, flat=False):
    """Random GGUF rows [rows, bytes] of `qtype` with finite fp16 fields;
    `flat` zeroes some Q4_K sub-block scales (groups of -m only)."""
    rng = np.random.default_rng(seed)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, (rows, k // block, size), dtype=np.uint8)
    halves = {WeightType.Q4_0: 1, WeightType.Q4_1: 2, WeightType.Q4_K: 2,
              WeightType.Q6_K: 0, WeightType.Q8_0: 1}[qtype]
    for h in range(halves):
        raw[..., 2 * h:2 * h + 2] = (rng.uniform(1e-4, 1e-2, raw.shape[:2])
                                     .astype(np.float16)[..., None]
                                     .view(np.uint8))
    if qtype == WeightType.Q6_K:  # d is the last 2 bytes
        raw[..., -2:] = (rng.uniform(1e-4, 1e-2, raw.shape[:2])
                         .astype(np.float16)[..., None].view(np.uint8))
    if flat:
        raw[::3, :, 4:8] &= 0xC0  # 6-bit scales of sub-blocks 0-3 set to 0
    return raw.reshape(rows, -1)


def ref(raw, qtype):
    return torch.from_numpy(gguf.quants.dequantize(raw, qtype)).cuda()


@pytest.mark.parametrize("qtype",
                         [WeightType.Q4_0, WeightType.Q4_1, WeightType.Q4_K])
@pytest.mark.parametrize("flat", [False, True])
def test_repack_matches_gguf(qtype, flat):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.k80_gguf import (
        k80_gguf_int4)
    if flat and qtype != WeightType.Q4_K:
        pytest.skip("only Q4_K has sub-block scales")
    raw = blocks(qtype, flat=flat)
    q, s, z = k80_gguf_int4(torch.from_numpy(raw).cuda(), qtype)
    assert q.shape == (ROWS, K // 8) and s.shape == (ROWS, K // 32)
    w = k80_int4_dequantize(q, s, z, torch.float32)
    expected = ref(raw, qtype)
    if qtype == WeightType.Q4_0:  # (q - 8) * d, no rounding to differ
        assert torch.equal(w, expected)
    else:  # (q - m / d) * d against d * q - m: one FP32 rounding
        err = (w - expected).abs().max() / expected.abs().max()
        assert err.item() < 1e-6
    if hasattr(torch.ops._C, "k80_int4_dequantize"):
        out = torch.empty_like(w)
        ops.k80_int4_dequantize(out, q, s, z)
        assert torch.equal(out, w)


def test_q8_0_repack_matches_gguf():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.k80_gguf import (
        k80_gguf_int8)
    raw = blocks(WeightType.Q8_0)
    q, s, z = k80_gguf_int8(torch.from_numpy(raw).cuda(), WeightType.Q8_0)
    assert q.shape == (ROWS, K // 4) and s.shape == (ROWS, K // 32)
    w = k80_int4_dequantize(q, s, z, torch.float32, 8)
    # (q + 128 - 128) * d: no rounding to differ
    assert torch.equal(w, ref(raw, WeightType.Q8_0))
    if hasattr(torch.ops._C, "k80_int4_dequantize"):
        out = torch.empty_like(w)
        ops.k80_int4_dequantize(out, q, s, z, 8)
        assert torch.equal(out, w)


def _layer_of(raw, qtype):
    """A GGUF layer with one unsharded tensor, as the loader leaves it."""
    layer = torch.nn.Module()
    qweight = torch.nn.Parameter(torch.from_numpy(raw).cuda(),
                                 requires_grad=False)
    qweight.shard_id, qweight.data_container = [], []
    layer.qweight = qweight
    qweight_type = torch.nn.Parameter(torch.zeros(1,
                                                  dtype=torch.uint8,
                                                  device="cuda"),
                                      requires_grad=False)
    qweight_type.weight_type = int(qtype)
    layer.qweight_type = qweight_type
    return layer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [1, 48, 49])
def test_q8_0_linear_and_embedding(dtype, m):
    """A Q8_0 layer and a Q8_0 embedding through the K80 int8 layout."""
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.k80_gguf import (
        K80GGUFEmbeddingMethod, K80GGUFLinearMethod)
    raw = blocks(WeightType.Q8_0, rows=96, seed=7)
    method = K80GGUFLinearMethod(GGUFConfig())
    method.params_dtype = dtype
    layer = _layer_of(raw, WeightType.Q8_0)
    method.process_weights_after_loading(layer)
    assert [kind for kind, _ in layer.k80_gguf_parts] == [("int", 8)]
    x = torch.randn(m, K, device="cuda").to(dtype)
    y = method.apply(layer, x)
    expected = x.double() @ ref(raw, WeightType.Q8_0).double().t()
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    assert ((y.double() - expected).norm() / expected.norm()).item() < tol

    raw = blocks(WeightType.Q8_0, rows=500, seed=8)
    emb = K80GGUFEmbeddingMethod(GGUFConfig())
    emb.params_dtype = torch.float32
    layer = _layer_of(raw, WeightType.Q8_0)
    emb.process_weights_after_loading(layer)
    ids = torch.randint(0, 500, (m, ), device="cuda")
    assert torch.equal(emb.embedding(layer, ids),
                       ref(raw, WeightType.Q8_0)[ids])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [1, 5, 300])
def test_linear_mixed_shards(dtype, m):
    """A fused QKV layer whose shards are Q4_K, Q6_K and F16 (as in
    ollama's Q4_K_M files): one int4 piece and one dense piece."""
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.k80_gguf import (
        K80GGUFLinearMethod)
    shards = {"q": (WeightType.Q4_K, 0), "k": (WeightType.Q6_K, 1),
              "v": (WeightType.F16, 2)}
    raws, dense = {}, []
    for sid, (qtype, seed) in shards.items():
        if qtype == WeightType.F16:
            w = torch.randn(64, K, generator=torch.Generator().manual_seed(
                seed)).half() * 0.01
            raws[sid] = w.cuda()
            dense.append(w.float().cuda())
        else:
            raw = blocks(qtype, rows=64, seed=seed)
            raws[sid] = torch.from_numpy(raw).cuda()
            dense.append(ref(raw, qtype))
    method = K80GGUFLinearMethod(GGUFConfig())
    method.params_dtype = dtype
    layer = torch.nn.Module()
    qweight = torch.nn.Parameter(torch.empty(0, device="cuda"),
                                 requires_grad=False)
    # Loaded in file order (k, q, v), as the GGUF loader does.
    qweight.shard_id = ["k", "q", "v"]
    qweight.data_container = [raws["k"], raws["q"], raws["v"]]
    qweight.shard_id_map = {"k": 0, "q": 1, "v": 2}
    layer.qweight = qweight
    qweight_type = torch.nn.Parameter(torch.zeros(3, dtype=torch.uint8,
                                                  device="cuda"),
                                      requires_grad=False)
    qweight_type.shard_weight_type = {s: int(t) for s, (t, _) in
                                      shards.items()}
    layer.qweight_type = qweight_type
    method.process_weights_after_loading(layer)
    kinds = [kind[0] for kind, _ in layer.k80_gguf_parts]
    if hasattr(torch.ops._C, "k80_gguf_gemm"):  # Q4_K and Q6_K as stored
        assert kinds == ["gguf", "gguf", "dense"]
    else:
        assert kinds == ["int", "dense"]
    x = torch.randn(m, K, device="cuda").to(dtype)
    bias = torch.randn(192, device="cuda").to(dtype)
    y = method.apply(layer, x, bias)
    w = torch.cat(dense).double()
    expected = x.double() @ w.t() + bias.double()
    tol = 1e-5 if dtype == torch.float32 else 3e-3
    assert y.shape == (m, 192)
    assert ((y.double() - expected).norm() / expected.norm()).item() < tol


NATIVE = [WeightType.Q4_K, WeightType.Q6_K]
needs_native = pytest.mark.skipif(
    not hasattr(torch.ops._C, "k80_gguf_gemm"),
    reason="needs the K80 GGUF kernels (csrc/quantization/k80/gguf.cu)")


@needs_native
@pytest.mark.parametrize("qtype", NATIVE)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_dequantize_bit_identical(qtype, dtype):
    from vllm import _custom_ops as ops
    raw = blocks(qtype, rows=40, k=2048, seed=3)
    out = torch.empty(40, 2048, dtype=dtype, device="cuda")
    ops.k80_gguf_dequantize(out, torch.from_numpy(raw).cuda(), int(qtype))
    expected = ref(raw, qtype).to(dtype)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(out.view(bits), expected.view(bits))


@needs_native
@pytest.mark.parametrize("qtype", NATIVE)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [1, 2, 3, 8, 9, 16])
def test_native_gemm(qtype, dtype, m):
    from vllm import _custom_ops as ops
    raw = blocks(qtype, rows=72, k=1536, seed=4)
    w = ref(raw, qtype).double()
    x = torch.randn(m, 1536, device="cuda").to(dtype)
    y = ops.k80_gguf_gemm(x, torch.from_numpy(raw).cuda(), int(qtype))
    assert y.shape == (m, 72) and y.dtype == dtype
    expected = x.double() @ w.t()
    err = (y.double() - expected).norm(dim=1) / expected.norm(dim=1)
    assert err.max().item() < (1e-5 if dtype == torch.float32 else 1e-3)


@needs_native
@pytest.mark.parametrize("m", [1, 16, 17, 300])
def test_native_linear_and_embedding(m):
    """A Q4_K layer through K80GGUFLinearMethod on both sides of
    K80_GGUF_GEMM_MAX_ROWS, and a Q6_K embedding lookup."""
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.k80_gguf import (
        K80GGUFEmbeddingMethod, K80GGUFLinearMethod)
    raw = blocks(WeightType.Q4_K, rows=96, k=K, seed=5)
    method = K80GGUFLinearMethod(GGUFConfig())
    method.params_dtype = torch.float16
    layer = _layer_of(raw, WeightType.Q4_K)
    method.process_weights_after_loading(layer)
    x = torch.randn(m, K, device="cuda").half()
    y = method.apply(layer, x)
    expected = x.double() @ ref(raw, WeightType.Q4_K).double().t()
    assert ((y.double() - expected).norm() / expected.norm()).item() < 2e-3

    raw = blocks(WeightType.Q6_K, rows=500, k=K, seed=6)
    emb = K80GGUFEmbeddingMethod(GGUFConfig())
    emb.params_dtype = torch.float32
    layer = _layer_of(raw, WeightType.Q6_K)
    emb.process_weights_after_loading(layer)
    ids = torch.randint(0, 500, (m, ), device="cuda")
    assert torch.equal(emb.embedding(layer, ids),
                       ref(raw, WeightType.Q6_K)[ids])

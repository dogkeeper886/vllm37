# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#74): the K80 int4 CUDA kernels against the PyTorch reference
of #73 (k80_int4_dequantize) and a float64 product.

Run on a Kepler GPU: pytest tests/kernels/quantization/test_k80_int4.py
"""
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    K80_INT4_GEMM_MAX_ROWS, k80_int4_dequantize, on_kepler)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()
         and hasattr(torch.ops._C, "k80_int4_gemm")),
    reason="K80 int4 kernels run on Kepler only")

DTYPES = [torch.float32, torch.float16]
# (N, K, group size): K a multiple of 32 (4 words per load), of 16 only (2)
# and of 8 only (1), and one group per row.
SHAPES = [(1024, 4096, 128), (4096, 1024, 64), (11008, 4096, 128),
          (520, 1040, 16), (264, 776, 8), (512, 3072, 3072)]


def make(n, k, gs, zeros="int", seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qweight = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32,
                            device="cuda", generator=g)
    scales = torch.rand(n, k // gs, device="cuda", generator=g) * 0.01 + 1e-3
    z = torch.randint(0, 16, (n, k // gs), device="cuda", generator=g).float()
    if zeros == "fraction":
        z = z + torch.rand(z.shape, device="cuda", generator=g)
    elif zeros == "large":
        z = z * 300 - 2000
    return qweight, scales, z


def weight64(qweight, scales, zeros):
    n, k = qweight.shape[0], qweight.shape[1] * 8
    b = qweight.view(torch.uint8)
    w = torch.stack((b & 15, b >> 4), -1).view(n, scales.shape[1], -1)
    return ((w.double() - zeros.double().unsqueeze(-1)) *
            scales.double().unsqueeze(-1)).view(n, k)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("zeros", ["int", "fraction", "large"])
def test_dequantize_bit_identical(dtype, shape, zeros):
    n, k, gs = shape
    qweight, scales, z = make(n, k, gs, zeros)
    out = torch.empty(n, k, dtype=dtype, device="cuda")
    ops.k80_int4_dequantize(out, qweight, scales, z)
    ref = k80_int4_dequantize(qweight, scales, z, dtype)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(out.view(bits), ref.view(bits))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 13, 48])
def test_gemm(dtype, shape, m):
    n, k, gs = shape
    qweight, scales, z = make(n, k, gs)
    w = weight64(qweight, scales, z)
    x = torch.randn(m, k, device="cuda").to(dtype)
    y = ops.k80_int4_gemm(x, qweight, scales, z)
    assert y.shape == (m, n) and y.dtype == dtype
    ref = x.double() @ w.t()
    # FP32 accumulation; FP16 adds the rounding of the output.
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    err = (y.double() - ref).norm(dim=1) / ref.norm(dim=1)
    assert err.max().item() < tol


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [1, 7, K80_INT4_GEMM_MAX_ROWS,
                               K80_INT4_GEMM_MAX_ROWS + 1, 300])
def test_linear_kernel(dtype, m):
    """K80Int4LinearKernel on the kernels vs on the reference path."""
    from vllm.model_executor.layers.quantization.kernels.mixed_precision import (  # noqa: E501
        MPLinearLayerConfig, choose_mp_linear_kernel)
    from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                               PackedvLLMParameter)
    from vllm.scalar_type import scalar_types
    n, k, gs = 1536, 1024, 128
    qweight, scales, z = make(n, k, gs)
    c = MPLinearLayerConfig((k, n), (k, n), scalar_types.uint4b8, dtype, gs,
                            False, False)
    kernel_type = choose_mp_linear_kernel(c)
    assert kernel_type.__name__ == "K80Int4LinearKernel"
    layer = torch.nn.Module()
    # GPTQ checkpoint layout: [K / 8, N] packed along K.
    layer.register_parameter("qweight", PackedvLLMParameter(
        data=qweight.t().contiguous(), input_dim=0, output_dim=1,
        packed_dim=0, packed_factor=8, weight_loader=lambda *a: None))
    layer.register_parameter("scales", GroupQuantScaleParameter(
        data=scales.t().contiguous(), output_dim=1, input_dim=0,
        weight_loader=lambda *a: None))
    kernel = kernel_type(c, w_q_param_name="qweight", w_s_param_name="scales",
                         w_zp_param_name="qzeros", w_gidx_param_name="g_idx")
    kernel.process_weights_after_loading(layer)
    assert kernel._ops is not None
    x = torch.randn(m, k, device="cuda").to(dtype)
    bias = torch.randn(n, device="cuda").to(dtype)
    y = kernel.apply_weights(layer, x, bias)
    w = weight64(layer.k80_qweight, layer.k80_scales, layer.k80_zeros)
    ref = x.double() @ w.t() + bias.double()
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    assert ((y.double() - ref).norm() / ref.norm()).item() < tol
    kernel._ops = None  # the PyTorch reference path
    y_ref = kernel.apply_weights(layer, x, bias)
    assert ((y.double() - y_ref.double()).norm() /
            ref.norm()).item() < 2 * tol

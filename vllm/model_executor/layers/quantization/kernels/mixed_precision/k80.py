# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#73): weight-only int4 on Kepler (sm_37). Every other mixed-
# precision kernel is built on an instruction the K80 lacks (half2 math, LOP3,
# ldmatrix, mma.sync, cp.async, dp4a), so int4 checkpoints are repacked at load
# time into one K80 layout:
#   qweight  int32 [N, K / 8]  8 int4 values per word along K, value i of a
#                              word in bits 4i..4i+3 (GPTQ and compressed-
#                              tensors already pack this way; AWQ, packed
#                              along N, is unpacked and packed again)
#   scales   fp32  [N, K / group_size]
#   zeros    fp32  [N, K / group_size]  w = (q - zero) * scale
#   perm     int64 [K] or absent        act-order (g_idx) resolved at load by
#                                       permuting K, so groups are contiguous
# Forward (#74), with the K80 CUDA kernels in csrc/quantization/k80/int4.cu:
#   up to K80_INT4_GEMM_MAX_ROWS rows (decode)  k80_int4_gemm, which reads the
#       packed words and multiply-adds in FP32 (no dequantized copy)
#   more rows (prefill)  k80_int4_dequantize into the activation dtype, then
#       the unquantized GEMM (cuBLAS)
# k80_int4_dequantize() below is the PyTorch reference: the path when the
# kernels are missing or VLLM_K80_INT4_KERNEL=0, and the kernels' test oracle.

from typing import Optional

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)

# The quantization methods whose int4 layers reach this kernel.
K80_INT4_METHODS = ("compressed-tensors", "gptq_marlin", "awq_marlin")
# AWQ packs along N; value i of a word is feature AWQ_PACK_ORDER[i] of its 8.
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
# Above this many rows dequantize + cuBLAS beats k80_int4_gemm: on a K80 die
# the crossover is between 48 and 64 rows for 1024-11008 features (#74).
K80_INT4_GEMM_MAX_ROWS = 48


def on_kepler() -> bool:
    """True on a CUDA GPU below compute capability 5.0, such as the K80."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major < 5


def k80_int4_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                        zeros: torch.Tensor,
                        dtype: torch.dtype) -> torch.Tensor:
    """The [N, K] weight of a layer in the K80 int4 layout, in `dtype`."""
    n = qweight.shape[0]
    k = qweight.shape[1] * 8
    groups = scales.shape[1]
    # Little-endian: byte j of a row holds values 2j (low nibble) and 2j + 1.
    b = qweight.view(torch.uint8)
    w = torch.stack((b & 15, b >> 4), dim=-1).view(n, groups, k // groups)
    w = w.to(dtype)
    w.sub_(zeros.to(dtype).unsqueeze(-1)).mul_(scales.to(dtype).unsqueeze(-1))
    return w.view(n, k)


def k80_int4_ops(group_size: int):
    """vllm._custom_ops when the K80 int4 kernels can serve this group size,
    else None (the PyTorch reference path). Logs which, once."""
    from vllm import _custom_ops as ops
    use = (envs.VLLM_K80_INT4_KERNEL and hasattr(torch.ops._C, "k80_int4_gemm")
           and group_size % 8 == 0)
    logger.info_once(
        "K80 int4 layers use %s", "the K80 CUDA kernels" if use else
        "the PyTorch reference path (slow; VLLM_K80_INT4_KERNEL=0, a build "
        "without csrc/quantization/k80, or a group size that is not a "
        "multiple of 8)")
    return ops if use else None


def k80_int4_matmul(layer: torch.nn.Module, x_2d: torch.Tensor,
                    qweight: torch.Tensor, scales: torch.Tensor,
                    zeros: torch.Tensor, ops, gemm,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x_2d [M, K] times the transpose of an [N, K] weight in the K80 int4
    layout. `ops` comes from k80_int4_ops(); `gemm` is the unquantized GEMM."""
    if ops is None:
        weight = k80_int4_dequantize(qweight, scales, zeros, x_2d.dtype)
        return gemm(layer, x_2d, weight, bias)
    if x_2d.shape[0] <= K80_INT4_GEMM_MAX_ROWS:
        x_2d = x_2d.contiguous()
        if x_2d.data_ptr() % 16:
            x_2d = x_2d.clone()
        out = ops.k80_int4_gemm(x_2d, qweight, scales, zeros)
        if bias is not None:
            out.add_(bias)
        return out
    weight = torch.empty((qweight.shape[0], qweight.shape[1] * 8),
                         dtype=x_2d.dtype,
                         device=x_2d.device)
    ops.k80_int4_dequantize(weight, qweight, scales, zeros)
    return gemm(layer, x_2d, weight, bias)


def k80_int4_rows(qweight: torch.Tensor, scales: torch.Tensor,
                  zeros: torch.Tensor, rows: torch.Tensor, ops,
                  dtype: torch.dtype) -> torch.Tensor:
    """Rows `rows` of an [N, K] weight in the K80 int4 layout, in `dtype` (an
    embedding lookup)."""
    q = qweight.index_select(0, rows)
    s = scales.index_select(0, rows)
    z = zeros.index_select(0, rows)
    if ops is None:
        return k80_int4_dequantize(q, s, z, dtype)
    out = torch.empty((q.shape[0], q.shape[1] * 8),
                      dtype=dtype,
                      device=q.device)
    ops.k80_int4_dequantize(out, q, s, z)
    return out


def _rows_out(t: torch.Tensor, output_dim: int) -> torch.Tensor:
    """A 2D tensor with the output features as rows."""
    return t if output_dim == 0 else t.t()


def _unpack(t: torch.Tensor, packed_dim: int,
            order: Optional[list[int]]) -> torch.Tensor:
    """int4 values of a packed tensor; value i of a word is feature order[i]
    of its 8 (in sequence when order is None)."""
    values = unpack_quantized_values_into_int32(t, scalar_types.uint4,
                                                packed_dim)
    if order is None:
        return values
    values = values.movedim(packed_dim, -1)
    shape = values.shape
    inverse = torch.argsort(torch.tensor(order)).to(values.device)
    values = values.reshape(*shape[:-1], -1, 8).index_select(-1, inverse)
    return values.reshape(shape).movedim(-1, packed_dim)


class K80Int4LinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]
    # Which feature of its 8 each packed value holds, when not in sequence
    # (AWQ sets [0, 2, 4, 6, 1, 3, 5, 7]).
    pack_order: Optional[list[int]] = None

    @classmethod
    def get_min_capability(cls) -> int:
        return 30

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        if not on_kepler():
            return False, "only for compute capability below 5.0 (K80)"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, (f"quant type {c.weight_type} not supported, "
                           f"supported types are {cls.SUPPORTED_QUANT_TYPES}")
        if c.act_type not in (torch.float32, torch.float16):
            return False, "activations must be float32 or float16"
        k, n = c.partition_weight_shape
        group_size = k if c.group_size == -1 else c.group_size
        if k % 8 != 0 or k % group_size != 0:
            return False, (f"input features per die ({k}) must divide by 8 "
                           f"and by the group size ({group_size})")
        if c.zero_points and n % 8 != 0:
            return False, (f"output features per die ({n}) must divide by 8 "
                           "to unpack the zero points")
        if c.has_g_idx and k != c.full_weight_shape[0]:
            return False, ("act-order (g_idx) is not supported when the "
                           "input features are split across dies")
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        k, n = c.partition_weight_shape
        group_size = k if c.group_size == -1 else c.group_size
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        if w_q.packed_dim == w_q.input_dim and self.pack_order is None:
            qweight = _rows_out(w_q.data, w_q.output_dim)
        else:  # AWQ: packed along N, interleaved
            values = _rows_out(
                _unpack(w_q.data, w_q.packed_dim, self.pack_order),
                w_q.output_dim)
            qweight = pack_quantized_values_into_int32(values,
                                                       c.weight_type,
                                                       packed_dim=1)
        scales = _rows_out(w_s.data, getattr(w_s, "output_dim", 0))
        scales = scales.to(torch.float32).contiguous()
        if c.zero_points:
            assert w_zp is not None
            zeros = _rows_out(
                _unpack(w_zp.data, w_zp.packed_dim, self.pack_order),
                w_zp.output_dim)
        else:
            zeros = torch.full_like(scales, c.weight_type.bias)
        zeros = zeros.to(torch.float32).contiguous()

        perm = None
        if c.has_g_idx:
            assert w_gidx is not None
            g_idx = w_gidx.data
            perm = torch.argsort(g_idx, stable=True)
            expected = torch.arange(k, device=g_idx.device) // group_size
            if not torch.equal(g_idx[perm].long(), expected):
                raise ValueError("act-order groups must each hold "
                                 f"{group_size} input features")
            values = unpack_quantized_values_into_int32(qweight,
                                                        c.weight_type,
                                                        packed_dim=1)
            qweight = pack_quantized_values_into_int32(values[:, perm],
                                                       c.weight_type,
                                                       packed_dim=1)
        assert scales.shape == (n, k // group_size), scales.shape

        # Replace the checkpoint's tensors so their memory is freed.
        for name in (self.w_q_name, self.w_s_name, self.w_zp_name,
                     self.w_gidx_name):
            if name is not None and hasattr(layer, name):
                delattr(layer, name)
        layer.k80_qweight = torch.nn.Parameter(qweight.contiguous(),
                                               requires_grad=False)
        layer.k80_scales = torch.nn.Parameter(scales, requires_grad=False)
        layer.k80_zeros = torch.nn.Parameter(zeros, requires_grad=False)
        layer.k80_perm = perm
        self._gemm = dispatch_unquantized_gemm()
        self._ops = k80_int4_ops(group_size)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        k, n = self.config.partition_weight_shape
        x_2d = x.reshape(-1, k)
        if layer.k80_perm is not None:
            x_2d = x_2d.index_select(1, layer.k80_perm)
        out = k80_int4_matmul(layer, x_2d, layer.k80_qweight,
                              layer.k80_scales, layer.k80_zeros, self._ops,
                              self._gemm, bias)
        return out.reshape(x.shape[:-1] + (n, ))

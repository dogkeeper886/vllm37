# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#107): GGUF checkpoints on Kepler (sm_37). The GGUF CUDA kernels
# need dp4a and half2 math, which the K80 lacks, so each GGUF tensor is
# converted at load time:
#   Q4_0, Q4_1, Q4_K  repacked into the K80 int4 layout (kernels/
#       mixed_precision/k80.py) with one fp32 scale and zero per 32 weights,
#       and run on the K80 int4 kernels (#74)
#   any other type    dequantized to the model dtype (Q6_K, Q8_0, ...) or kept
#       (F16, BF16, F32), and run as an unquantized layer
# A fused layer (QKV, gate-up) whose shards have different types runs as one
# piece per run of shards that share a path.

from typing import Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm.model_executor.layers.quantization.gguf import (GGUFConfig,
                                                         GGUFLinearMethod)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    k80_int4_matmul, k80_int4_ops, k80_int4_rows)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

K80_GGUF_INT4_TYPES = (WeightType.Q4_0, WeightType.Q4_1, WeightType.Q4_K)
K80_GGUF_GROUP_SIZE = 32
# Rows decoded at a time, to bound the load-time scratch memory.
_CHUNK_ROWS = 4096


def _f16(b: torch.Tensor) -> torch.Tensor:
    """fp16 values stored as uint8 pairs in the last dim, as fp32."""
    return b.contiguous().view(torch.float16).squeeze(-1).float()


def _affine(q: torch.Tensor, d: torch.Tensor, m: torch.Tensor):
    """Groups with w = d * q - m, as (q, scale, zero) with w = (q - zero) *
    scale. A group with d = 0 holds -m everywhere: q = 0, scale m, zero 1."""
    flat = d == 0
    q = q.masked_fill(flat.unsqueeze(-1), 0)
    scale = torch.where(flat, m, d)
    zero = torch.where(flat, torch.ones_like(m), m / torch.where(flat, 1, d))
    return q, scale, zero


def _decode_q4(raw: torch.Tensor, qtype: int):
    """GGUF Q4_0 / Q4_1 / Q4_K rows [R, bytes] -> int4 values uint8
    [R, K / 32, 32], fp32 scales and zeros [R, K / 32], following ggml's
    dequantize_row_q4_0 / q4_1 / q4_K."""
    rows = raw.shape[0]
    if qtype == WeightType.Q4_0:  # d, 16 bytes: w = (q - 8) * d
        blk = raw.view(rows, -1, 18)
        qs = blk[..., 2:]
        q = torch.cat((qs & 15, qs >> 4), dim=-1)
        scale = _f16(blk[..., 0:2])
        return q, scale, torch.full_like(scale, 8.0)
    if qtype == WeightType.Q4_1:  # d, m, 16 bytes: w = q * d + m
        blk = raw.view(rows, -1, 20)
        qs = blk[..., 4:]
        q = torch.cat((qs & 15, qs >> 4), dim=-1)
        return _affine(q, _f16(blk[..., 0:2]), -_f16(blk[..., 2:4]))
    assert qtype == WeightType.Q4_K
    # d, dmin, 12 bytes of 6-bit scales and mins for 8 sub-blocks of 32,
    # 128 bytes: sub-blocks 2c and 2c + 1 are the low and high nibbles of
    # bytes 32c..32c + 31. w = d * sc * q - dmin * m.
    blk = raw.view(rows, -1, 144)
    nb = blk.shape[1]
    s = blk[..., 4:16].int()
    lo, mid, hi = s[..., 0:4], s[..., 4:8], s[..., 8:12]
    # ggml get_scale_min_k4
    sc = torch.cat((lo & 63, (hi & 15) | ((lo >> 6) << 4)), dim=-1)
    mn = torch.cat((mid & 63, (hi >> 4) | ((mid >> 6) << 4)), dim=-1)
    qs = blk[..., 16:].reshape(rows, nb, 4, 32)
    q = torch.stack((qs & 15, qs >> 4), dim=3).reshape(rows, nb * 8, 32)
    d = (_f16(blk[..., 0:2]).unsqueeze(-1) * sc).reshape(rows, nb * 8)
    m = (_f16(blk[..., 2:4]).unsqueeze(-1) * mn).reshape(rows, nb * 8)
    return _affine(q, d, m)


def _pack(q: torch.Tensor) -> torch.Tensor:
    """uint8 values [R, K] -> int32 [R, K / 8], value i of a word in bits
    4i..4i + 3 (the K80 int4 layout)."""
    v = q.view(q.shape[0], -1, 8).int()
    word = v[..., 0]
    for i in range(1, 8):
        word = word | (v[..., i] << (4 * i))
    return word.contiguous()  # the slices' strides carry over


def k80_gguf_int4(raw: torch.Tensor, qtype: int):
    """A GGUF Q4 tensor [N, bytes] in the K80 int4 layout: qweight int32
    [N, K / 8], scales and zeros fp32 [N, K / 32]."""
    qweight, scales, zeros = [], [], []
    for start in range(0, raw.shape[0], _CHUNK_ROWS):
        q, s, z = _decode_q4(raw[start:start + _CHUNK_ROWS], qtype)
        qweight.append(_pack(q.reshape(q.shape[0], -1)))
        scales.append(s)
        zeros.append(z)
    return (torch.cat(qweight).contiguous(), torch.cat(scales).contiguous(),
            torch.cat(zeros).contiguous())


def k80_gguf_dense(raw: torch.Tensor, qtype: int,
                   dtype: torch.dtype) -> torch.Tensor:
    """Any GGUF tensor [N, bytes] (or an unquantized [N, K]) as [N, K] in
    `dtype`, dequantized on the CPU by the gguf package."""
    if qtype in (WeightType.F32, WeightType.F16, WeightType.BF16):
        return raw.to(dtype)
    values = gguf.quants.dequantize(raw.cpu().numpy(), WeightType(qtype))
    return torch.from_numpy(values).to(device=raw.device, dtype=dtype)


def _shards(layer: torch.nn.Module) -> list[tuple[torch.Tensor, int]]:
    """The layer's GGUF tensors and types, in output-feature order."""
    qweight = layer.qweight
    if not qweight.shard_id:
        return [(qweight.data, layer.qweight_type.weight_type)]
    ids = qweight.shard_id
    ids = [i for i in ("q", "k", "v") if i in ids] if "q" in ids else sorted(
        ids)
    types = layer.qweight_type.shard_weight_type
    return [(qweight.data_container[qweight.shard_id_map[i]], types[i])
            for i in ids]


class K80GGUFLinearMethod(GGUFLinearMethod):
    """GGUF linear layers on Kepler, through the K80 int4 layout."""

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # The loader moves parameters to the GPU for this call; a fused
        # layer's shards are in a list it does not move.
        device = layer.qweight_type.device
        parts = []
        for raw, qtype in _shards(layer):
            raw = raw.to(device)
            int4 = qtype in K80_GGUF_INT4_TYPES
            tensors = (k80_gguf_int4(raw, qtype) if int4 else
                       (k80_gguf_dense(raw, qtype, self.params_dtype), ))
            if parts and parts[-1][0] == int4:
                parts[-1] = (int4,
                             tuple(
                                 torch.cat((a, b))
                                 for a, b in zip(parts[-1][1], tensors)))
            else:
                parts.append((int4, tensors))
        layer.qweight.data_container.clear()
        del layer.qweight
        layer.k80_gguf_parts = parts
        self._gemm = dispatch_unquantized_gemm()
        self._ops = k80_int4_ops(K80_GGUF_GROUP_SIZE) if any(
            int4 for int4, _ in parts) else None

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        outs = []
        for int4, tensors in layer.k80_gguf_parts:
            if int4:
                outs.append(
                    k80_int4_matmul(layer, x_2d, *tensors, self._ops,
                                    self._gemm))
            else:
                outs.append(self._gemm(layer, x_2d, tensors[0], None))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (out.shape[-1], ))


class K80GGUFEmbeddingMethod(K80GGUFLinearMethod):
    """GGUF embeddings (and a tied LM head) on Kepler."""

    def embedding(self, layer: torch.nn.Module,
                  x: torch.Tensor) -> torch.Tensor:
        (int4, tensors), = layer.k80_gguf_parts
        rows = x.flatten()
        if int4:
            out = k80_int4_rows(*tensors, rows, self._ops, self.params_dtype)
        else:
            out = tensors[0].index_select(0, rows)
        return out.view(*x.shape, out.shape[-1])


def k80_gguf_quant_method(config: GGUFConfig, layer: torch.nn.Module):
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding)
    if isinstance(layer, LinearBase):
        return K80GGUFLinearMethod(config)
    if isinstance(layer, VocabParallelEmbedding):
        return K80GGUFEmbeddingMethod(config)
    if isinstance(layer, FusedMoE):
        raise ValueError("GGUF MoE layers are not supported on Kepler (K80)")
    return None

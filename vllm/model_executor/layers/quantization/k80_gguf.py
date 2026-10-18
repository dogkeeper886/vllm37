# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#107, #111, #109): GGUF checkpoints on Kepler (sm_37). The GGUF
# CUDA kernels need dp4a and half2 math, which the K80 lacks, so each GGUF
# tensor takes one of these paths at load time:
#   Q4_K, Q6_K        kept as stored and run on the K80 GGUF kernels
#       (csrc/quantization/k80/gguf.cu, #111): the fused product for up to
#       K80_GGUF_GEMM_MAX_ROWS rows, else dequantize + an FP32 GEMM
#   Q4_0, Q4_1 (and Q4_K without the GGUF kernels)  repacked into the K80
#       int4 layout (kernels/mixed_precision/k80.py) with one fp32 scale and
#       zero per 32 weights, and run on the K80 int4 kernels (#74)
#   Q8_0              repacked into the K80 int8 layout (q + 128, zero 128)
#       and run on the same kernels (#109)
#   any other type    dequantized to the model dtype (Q5_K, ..., and Q6_K
#       without the GGUF kernels) or kept (F16, BF16, F32), and run as an
#       unquantized layer
# A fused layer (QKV, gate-up) whose shards have different types runs as one
# piece per run of shards that share a path.
# MoE experts (#131) run on the K80 MoE kernels (k80_moe_experts), each
# expert's blocks repacked exactly:
#   Q4_0, Q4_1, Q4_K          int4, a scale and zero per 32 weights
#   Q8_0, Q5_0, Q5_1, Q5_K    int8, a scale and zero per 32 weights
#   Q6_K                      int8, a scale and zero per 16 weights
# and any other type dequantized to the model dtype (k80_fused_experts).

from typing import Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.gguf import (GGUFConfig,
                                                         GGUFLinearMethod,
                                                         GGUFMoEMethod)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    k80_int4_matmul, k80_int4_ops, k80_int4_rows)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

logger = init_logger(__name__)

K80_GGUF_INT4_TYPES = (WeightType.Q4_0, WeightType.Q4_1, WeightType.Q4_K)
K80_GGUF_INT8_TYPES = (WeightType.Q8_0, )
K80_GGUF_GROUP_SIZE = 32
K80_GGUF_MOE_INT8_TYPES = (WeightType.Q8_0, WeightType.Q5_0, WeightType.Q5_1,
                           WeightType.Q5_K, WeightType.Q6_K)
# Read as stored by the K80 GGUF kernels.
K80_GGUF_NATIVE_TYPES = (WeightType.Q4_K, WeightType.Q6_K)
# Above this many rows dequantize + FP32 GEMM beats k80_gguf_gemm: on a K80
# die the crossover is between 16 and 32 rows (#111).
K80_GGUF_GEMM_MAX_ROWS = 16
# Rows decoded at a time, to bound the load-time scratch memory.
_CHUNK_ROWS = 4096
# FP32 values dequantized at a time above K80_GGUF_GEMM_MAX_ROWS (256 MB), so
# a large LM head never needs a full FP32 copy.
_GEMM_CHUNK_ELEMS = 1 << 26


def k80_gguf_ops():
    """vllm._custom_ops when the K80 GGUF kernels are built and enabled,
    else None (Q4_K and Q6_K take the #107 paths). Logs which, once."""
    from vllm import _custom_ops as ops
    use = envs.VLLM_K80_INT4_KERNEL and hasattr(torch.ops._C,
                                                 "k80_gguf_gemm")
    logger.info_once(
        "K80 GGUF Q4_K and Q6_K tensors %s", "stay as stored (K80 GGUF "
        "kernels)" if use else "are converted (no K80 GGUF kernels, or "
        "VLLM_K80_INT4_KERNEL=0)")
    return ops if use else None


def k80_gguf_check_blocks(row_bytes: int, qtype: int, k: Optional[int],
                          what: str) -> None:
    """K80 fork (#132): tensor parallelism splits a row-parallel layer's
    stored bytes along K, k features in all (None: from row_bytes), so each
    die's share must hold whole GGUF blocks; a share that cuts one gives the
    die bytes of a block without its scales, and wrong weights. Refused at
    load time, naming a die count that keeps the blocks whole."""
    block, size = gguf.GGML_QUANT_SIZES[WeightType(qtype)]
    if row_bytes % size == 0:
        return
    from vllm.distributed import get_tensor_model_parallel_world_size
    try:
        tp = get_tensor_model_parallel_world_size()
    except AssertionError:  # no parallel group: a layer built on its own
        tp = 1
    if k is None:
        k = round(row_bytes * tp / size) * block
    blocks = k // block
    fits = max(d for d in range(1, tp + 1) if tp % d == 0 and blocks % d == 0)
    raise ValueError(
        f"K80 GGUF: {what} has {k} input features, {blocks} "
        f"{WeightType(qtype).name} blocks of {block}; split across "
        f"{tp} dies, a die's {k / tp:g} features cut a block (#132). Use "
        f"tensor_parallel_size {fits}, or a GGUF file whose types have blocks "
        "of 32 (Q4_0, Q4_1, Q8_0).")


def _gguf_k(raw: torch.Tensor, qtype: int) -> int:
    block, size = gguf.GGML_QUANT_SIZES[WeightType(qtype)]
    return raw.shape[1] // size * block


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


def _bits(t: torch.Tensor, n: int, step: int = 1) -> torch.Tensor:
    """[..., B] -> [..., n, B]: field i of each element is bits
    step*i .. step*i + step - 1."""
    shifts = torch.arange(0, n * step, step, device=t.device, dtype=t.dtype)
    return (t.unsqueeze(-2) >> shifts.view(n, 1)) & ((1 << step) - 1)


def _decode_q8(raw: torch.Tensor, qtype: int):
    """GGUF Q8_0 / Q5_0 / Q5_1 / Q5_K / Q6_K rows [R, bytes] -> byte values
    uint8 [R, G, K / G], fp32 scales and zeros [R, G] with w = (q - zero) *
    scale: groups of 32, or 16 for Q6_K. Follows the gguf package's
    dequantize_blocks for each type."""
    rows = raw.shape[0]
    if qtype == WeightType.Q8_0:  # d, 32 int8: w = d * q
        blk = raw.view(rows, -1, 34)
        scale = _f16(blk[..., 0:2])
        return blk[..., 2:] ^ 0x80, scale, torch.full_like(scale, 128.0)
    if qtype in (WeightType.Q5_0, WeightType.Q5_1):
        # Q5_0: d, qh, 16 bytes: w = (q - 16) * d. Q5_1: d, m, qh, 16 bytes:
        # w = q * d + m. Bit i of the 32-bit qh is the fifth bit of value i.
        lead = 2 if qtype == WeightType.Q5_0 else 4
        blk = raw.view(rows, -1, lead + 20)
        qh = blk[..., lead:lead + 4].contiguous().view(torch.int32)
        hi = _bits(qh, 32).squeeze(-1).to(torch.uint8)
        qs = blk[..., lead + 4:]
        q = torch.cat((qs & 15, qs >> 4), dim=-1) | (hi << 4)
        d = _f16(blk[..., 0:2])
        if qtype == WeightType.Q5_0:
            return q, d, torch.full_like(d, 16.0)
        return _affine(q, d, -_f16(blk[..., 2:4]))
    if qtype == WeightType.Q5_K:
        # d, dmin, 12 bytes of scales and mins (as Q4_K), 32 bytes of fifth
        # bits (bit j of byte l: sub-block j, value l), 128 bytes of nibbles
        # (as Q4_K). w = d * sc * q - dmin * m.
        blk = raw.view(rows, -1, 176)
        nb = blk.shape[1]
        s = blk[..., 4:16].int()
        lo, mid, hi = s[..., 0:4], s[..., 4:8], s[..., 8:12]
        sc = torch.cat((lo & 63, (hi & 15) | ((lo >> 6) << 4)), dim=-1)
        mn = torch.cat((mid & 63, (hi >> 4) | ((mid >> 6) << 4)), dim=-1)
        qs = blk[..., 48:].reshape(rows, nb, 4, 32)
        q = torch.stack((qs & 15, qs >> 4), dim=3).reshape(rows, nb * 8, 32)
        q = q | (_bits(blk[..., 16:48], 8).reshape(rows, nb * 8, 32) << 4)
        d = (_f16(blk[..., 0:2]).unsqueeze(-1) * sc).reshape(rows, nb * 8)
        m = (_f16(blk[..., 2:4]).unsqueeze(-1) * mn).reshape(rows, nb * 8)
        return _affine(q, d, m)
    assert qtype == WeightType.Q6_K
    # 128 bytes of low nibbles, 64 bytes of high 2-bit pairs, 16 int8
    # scales, d. w = d * sc * (q - 32), one scale per 16 values.
    blk = raw.view(rows, -1, 210)
    nb = blk.shape[1]
    ql = blk[..., :128].reshape(rows, nb, 2, 1, 64)
    ql = torch.cat((ql & 15, ql >> 4), dim=3).reshape(rows, nb * 8, 32)
    qh = _bits(blk[..., 128:192].reshape(rows, nb, 2, 32), 4, 2)
    q = (ql | (qh.reshape(rows, nb * 8, 32) << 4)).reshape(rows, nb * 16, 16)
    sc = blk[..., 192:208].contiguous().view(torch.int8).float()
    scale = (_f16(blk[..., 208:210]).unsqueeze(-1) * sc).reshape(rows, nb * 16)
    return q, scale, torch.full_like(scale, 32.0)


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


def k80_gguf_int8(raw: torch.Tensor, qtype: int):
    """A GGUF Q8_0 tensor [N, bytes] in the K80 int8 layout: qweight int32
    [N, K / 4], scales and zeros fp32 [N, K / 32]. Q8_0 blocks are d and 32
    int8 q, w = d * q; q + 128 (the byte with its top bit flipped) is stored
    with zero 128, following ggml's dequantize_row_q8_0."""
    assert qtype == WeightType.Q8_0
    qweight, scales = [], []
    for start in range(0, raw.shape[0], _CHUNK_ROWS):
        blk = raw[start:start + _CHUNK_ROWS]
        blk = blk.view(blk.shape[0], -1, 34)
        q = (blk[..., 2:] ^ 0x80).reshape(blk.shape[0], -1)
        qweight.append(q.contiguous().view(torch.int32))
        scales.append(_f16(blk[..., 0:2]))
    scales = torch.cat(scales).contiguous()
    return (torch.cat(qweight).contiguous(), scales,
            torch.full_like(scales, 128.0))


def k80_gguf_moe_int8(raw: torch.Tensor, qtype: int):
    """A GGUF tensor [N, bytes] of a K80_GGUF_MOE_INT8_TYPES type in the K80
    int8 layout: qweight int32 [N, K / 4], scales and zeros fp32 [N, G]."""
    q, s, z = _decode_q8(raw, qtype)
    return (q.reshape(q.shape[0], -1).contiguous().view(torch.int32),
            s.contiguous(), z.contiguous())


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
    """GGUF linear layers on Kepler: each piece of a layer is
    ("gguf", qtype): (raw, ), ("int", bits): (qweight, scales, zeros) or
    ("dense", ): (weight, )."""

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # The loader moves parameters to the GPU for this call; a fused
        # layer's shards are in a list it does not move.
        device = layer.qweight_type.device
        self._gguf_ops = k80_gguf_ops()
        parts = []
        for raw, qtype in _shards(layer):
            k80_gguf_check_blocks(
                raw.shape[1], qtype,
                getattr(layer, "input_size", _gguf_k(raw, qtype)),
                getattr(layer, "prefix", "") or type(layer).__name__)
            raw = raw.to(device)
            if self._gguf_ops is not None and qtype in K80_GGUF_NATIVE_TYPES:
                kind, tensors = ("gguf", qtype), (raw.contiguous(), )
            elif qtype in K80_GGUF_INT4_TYPES:
                kind, tensors = ("int", 4), k80_gguf_int4(raw, qtype)
            elif qtype in K80_GGUF_INT8_TYPES:
                kind, tensors = ("int", 8), k80_gguf_int8(raw, qtype)
            else:
                kind = ("dense", )
                tensors = (k80_gguf_dense(raw, qtype, self.params_dtype), )
            if parts and parts[-1][0] == kind:
                parts[-1] = (kind,
                             tuple(
                                 torch.cat((a, b))
                                 for a, b in zip(parts[-1][1], tensors)))
            else:
                parts.append((kind, tensors))
        layer.qweight.data_container.clear()
        del layer.qweight
        layer.k80_gguf_parts = parts
        self._gemm = dispatch_unquantized_gemm()
        self._ops = k80_int4_ops(K80_GGUF_GROUP_SIZE) if any(
            kind[0] == "int" for kind, _ in parts) else None

    def _gguf_matmul(self, layer: torch.nn.Module, x_2d: torch.Tensor,
                     raw: torch.Tensor, qtype: int) -> torch.Tensor:
        ops = self._gguf_ops
        if x_2d.shape[0] <= K80_GGUF_GEMM_MAX_ROWS:
            return ops.k80_gguf_gemm(x_2d.contiguous(), raw, qtype)
        # Dequantize (bit for bit the gguf package's FP32) and multiply in
        # FP32, which the K80 runs at twice its FP16 rate, a chunk of output
        # features at a time.
        k = _gguf_k(raw, qtype)
        step = max(16, _GEMM_CHUNK_ELEMS // k // 16 * 16)
        xf = x_2d.float()
        outs = []
        for n0 in range(0, raw.shape[0], step):
            chunk = raw[n0:n0 + step]
            w = torch.empty((chunk.shape[0], k),
                            dtype=torch.float32,
                            device=raw.device)
            ops.k80_gguf_dequantize(w, chunk, qtype)
            outs.append(self._gemm(layer, xf, w, None))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out.to(x_2d.dtype)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        outs = []
        for kind, tensors in layer.k80_gguf_parts:
            if kind[0] == "gguf":
                outs.append(
                    self._gguf_matmul(layer, x_2d, tensors[0], kind[1]))
            elif kind[0] == "int":
                outs.append(
                    k80_int4_matmul(layer, x_2d, *tensors, self._ops,
                                    self._gemm, None, kind[1]))
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
        (kind, tensors), = layer.k80_gguf_parts
        rows = x.flatten()
        if kind[0] == "gguf":
            raw = tensors[0].index_select(0, rows)
            out = torch.empty((rows.shape[0], _gguf_k(raw, kind[1])),
                              dtype=self.params_dtype,
                              device=raw.device)
            self._gguf_ops.k80_gguf_dequantize(out, raw, kind[1])
        elif kind[0] == "int":
            out = k80_int4_rows(*tensors, rows, self._ops, self.params_dtype,
                                kind[1])
        else:
            out = tensors[0].index_select(0, rows)
        return out.view(*x.shape, out.shape[-1])


def _moe_bits(qtype: int) -> Optional[int]:
    if qtype in K80_GGUF_INT4_TYPES:
        return 4
    return 8 if qtype in K80_GGUF_MOE_INT8_TYPES else None


class K80GGUFMoEMethod(GGUFMoEMethod):
    """GGUF MoE experts on Kepler (#131): w13 and w2 repacked into the K80
    int4 / int8 MoE layout (k80_moe_load's parameters), or both dequantized
    when either has another type."""

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        self.params_dtype = params_dtype
        super().create_weights(layer, num_experts, hidden_size,
                               intermediate_size_per_partition, params_dtype,
                               **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        device = layer.w13_qweight_type.device
        stacks = {}
        for name in ("w13", "w2"):
            raw = getattr(layer, f"{name}_qweight").data.to(device)
            qtype = getattr(layer, f"{name}_qweight_type").weight_type
            # w2 is split along K across dies; w13's K is the hidden size.
            k80_gguf_check_blocks(raw.shape[-1], qtype, None,
                                  f"{getattr(layer, 'layer_name', 'MoE')}."
                                  f"{name}")
            stacks[name] = (raw, qtype)
            delattr(layer, f"{name}_qweight")
        self.dense = any(
            _moe_bits(qtype) is None for _, qtype in stacks.values())
        for name, (raw, qtype) in stacks.items():
            if self.dense:
                w = torch.stack([
                    k80_gguf_dense(raw[e], qtype, self.params_dtype)
                    for e in range(raw.shape[0])
                ])
                layer.register_parameter(
                    f"k80_{name}_weight",
                    torch.nn.Parameter(w, requires_grad=False))
                continue
            bits = _moe_bits(qtype)
            repack = k80_gguf_int4 if bits == 4 else k80_gguf_moe_int8
            parts = [repack(raw[e], qtype) for e in range(raw.shape[0])]
            for i, kind in enumerate(("qweight", "scales", "zeros")):
                t = torch.stack([p[i] for p in parts])
                layer.register_parameter(
                    f"k80_{name}_{kind}",
                    torch.nn.Parameter(t, requires_grad=False))
            setattr(layer, "k80_bits" if name == "w13" else "k80_w2_bits",
                    bits)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              router_logits: torch.Tensor, top_k: int, renormalize: bool,
              **kwargs) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.k80_moe import (
            k80_fused_experts, k80_moe_apply)
        if not self.dense:
            return k80_moe_apply(layer, x, router_logits, top_k, renormalize,
                                 **kwargs)
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        if kwargs.get("enable_eplb"):
            raise NotImplementedError(
                "K80 MoE experts: no expert load balancing")
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=kwargs.get("use_grouped_topk", False),
            top_k=top_k,
            renormalize=renormalize,
            topk_group=kwargs.get("topk_group"),
            num_expert_group=kwargs.get("num_expert_group"),
            custom_routing_function=kwargs.get("custom_routing_function"),
            scoring_func=kwargs.get("scoring_func", "softmax"),
            e_score_correction_bias=kwargs.get("e_score_correction_bias"))
        return k80_fused_experts(
            x,
            layer.k80_w13_weight,
            layer.k80_w2_weight,
            topk_weights,
            topk_ids,
            activation=kwargs.get("activation", "silu"),
            apply_router_weight_on_input=kwargs.get(
                "apply_router_weight_on_input", False),
            global_num_experts=kwargs.get("global_num_experts", -1),
            expert_map=kwargs.get("expert_map"))


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
        return K80GGUFMoEMethod(config)
    return None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#119): MoE experts on Kepler (sm_37), where fused_experts and the
# quantized MoE methods cannot run (their products are Triton kernels, and
# Triton does not support Kepler). Routing stays vLLM's (fused_topk on the
# _moe_C ops, #85); a token's top-k experts become top-k slots, slot
# s = token * top_k + k.
#   int4 / int8 experts (k80_moe_experts): k80_int4_moe_gemm, the K80 int4
#       kernel with weights [E, N, K / P] in the K80Int4LinearKernel layout.
#       Up to K80_MOE_PAIR_SLOTS slots, one block row per slot;
#       above, the slots grouped by expert in blocks of 8
#       (moe_align_block_size), so a block reads its expert's weights once.
#       Neither syncs with the host, so CUDA graphs can capture both.
#   MXFP4 experts (k80_mxfp4_experts, #128): k80_mxfp4_moe_gemm, the same
#       product on gpt-oss's E2M1 codes and E8M0 scales as stored, with the
#       experts' biases.
#   FP16 / FP32 experts (k80_fused_experts): up to K80_MOE_PAIR_SLOTS slots,
#       or while a CUDA graph is captured, k80_moe_dense_gemm, one block row
#       per slot reading its expert's weights in place (no host sync); above,
#       one cuBLAS product per expert that has tokens.
# Each then applies silu_and_mul, gelu_and_mul or swiglu_oai between the two
# products and adds the slots of a token with its routing weights (moe_sum).

from typing import Optional

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)

# Slots up to which each slot gets its own block row; above, slots are
# grouped by expert. Qwen3-30B-A3B int4 per die at TP=2 (128 experts, top 8),
# one layer on a K80 die: pair mode 0.17 / 0.52 / 2.0 / 8.6 ms for 8 / 32 /
# 128 / 512 slots, block mode 0.63 / 1.9 / 5.2 / 8.7 ms (#119).
K80_MOE_PAIR_SLOTS = 512
# Slots per block of the grouped int4/int8 product (kMoeBlock in int4.cu).
K80_MOE_BLOCK = 8
# Slots per call of k80_mxfp4_experts: a longer batch goes in chunks of
# tokens, so its per-slot buffers stay small. gpt-oss-20b's 8,192-token
# profile run (32,768 slots) took about 1.7 GiB in them and ran out of memory
# on a die at TP=2 (#128).
K80_MOE_CHUNK_SLOTS = 4096


def _act_and_mul(h: torch.Tensor, activation: str) -> torch.Tensor:
    out = h.new_empty((h.shape[0], h.shape[1] // 2))
    if activation == "silu":
        torch.ops._C.silu_and_mul(out, h)
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(out, h)
    elif activation == "gelu_tanh":  # Gemma 4 experts (#126)
        torch.ops._C.gelu_tanh_and_mul(out, h)
    elif activation == "swiglu_oai":  # gpt-oss experts (#128)
        # Gate and up interleaved, as fused_experts' swiglu_oai reads them.
        gate = h[:, ::2].clamp(max=7.0)
        up = h[:, 1::2].clamp(min=-7.0, max=7.0)
        torch.mul(gate, torch.sigmoid(gate * 1.702), out=out)
        out.mul_(up + 1)
    else:
        raise NotImplementedError(
            f"K80 MoE experts support silu, gelu, gelu_tanh and swiglu_oai, "
            f"not {activation}")
    return out


def _inputs(x: torch.Tensor, topk_weights: torch.Tensor,
            apply_router_weight_on_input: bool) -> tuple[torch.Tensor, int]:
    """The rows the first product reads and the slots per row."""
    if not apply_router_weight_on_input:
        return x.contiguous(), topk_weights.shape[1]
    # The routing weight scales the input; one row per slot.
    rows = x.unsqueeze(1) * topk_weights.to(x.dtype).unsqueeze(-1)
    return rows.reshape(-1, x.shape[1]).contiguous(), 1


def _combine(y: torch.Tensor, topk_weights: torch.Tensor,
             apply_router_weight_on_input: bool) -> torch.Tensor:
    """[T, H]: the top-k slots of each token, added with their weights."""
    t, topk = topk_weights.shape
    y = y.view(t, topk, -1)
    if not apply_router_weight_on_input:
        y.mul_(topk_weights.to(y.dtype).unsqueeze(-1))
    if topk == 1:
        return y.view(t, -1)
    out = y.new_empty((t, y.shape[-1]))
    ops.moe_sum(y, out)
    return out


def _check(expert_map: Optional[torch.Tensor], **unsupported) -> None:
    if expert_map is not None:
        raise NotImplementedError("K80 MoE experts: no expert parallelism")
    used = [k for k, v in unsupported.items() if v not in (None, False)]
    if used:
        raise NotImplementedError(f"K80 MoE experts: {', '.join(used)} not "
                                  "supported")


def _slots(
    topk_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(experts, slots, padded) of the K80 MoE products: one expert per slot
    up to K80_MOE_PAIR_SLOTS slots, else the slots grouped by expert."""
    if topk_ids.numel() <= K80_MOE_PAIR_SLOTS:
        return topk_ids.to(torch.int32).flatten(), None, None
    slots, experts, padded = moe_align_block_size(topk_ids,
                                                  K80_MOE_BLOCK,
                                                  num_experts,
                                                  pad_sorted_ids=True)
    return experts, slots, padded


def k80_moe_experts(x: torch.Tensor,
                    w13: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    w2: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    bits: int,
                    topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor,
                    activation: str = "silu",
                    apply_router_weight_on_input: bool = False,
                    global_num_experts: int = -1,
                    expert_map: Optional[torch.Tensor] = None,
                    w2_bits: Optional[int] = None) -> torch.Tensor:
    """MoE experts with int4 or int8 weights in the K80 layout: w13 and w2
    are (qweight [E, N, K / P], scales [E, N, G], zeros [E, N, G]); w2 has
    w2_bits when it differs (a GGUF Q4_K_M checkpoint pairs Q4_K and Q6_K,
    #131)."""
    _check(expert_map)
    w2_bits = bits if w2_bits is None else w2_bits
    num_experts = w13[0].shape[0]
    rows, per_row = _inputs(x, topk_weights, apply_router_weight_on_input)
    num_slots = topk_ids.numel()
    experts, slots, padded = _slots(topk_ids, num_experts)
    h = ops.k80_int4_moe_gemm(rows, *w13, experts, slots, padded, num_slots,
                              per_row, bits)
    a = _act_and_mul(h, activation)
    y = ops.k80_int4_moe_gemm(a, *w2, experts, slots, padded, num_slots, 1,
                              w2_bits)
    return _combine(y, topk_weights, apply_router_weight_on_input)


def k80_mxfp4_experts(x: torch.Tensor,
                      w13: tuple[torch.Tensor, torch.Tensor],
                      w2: tuple[torch.Tensor, torch.Tensor],
                      topk_weights: torch.Tensor,
                      topk_ids: torch.Tensor,
                      activation: str = "swiglu_oai",
                      w13_bias: Optional[torch.Tensor] = None,
                      w2_bias: Optional[torch.Tensor] = None,
                      apply_router_weight_on_input: bool = False,
                      expert_map: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
    """MoE experts with MXFP4 weights as gpt-oss stores them: w13 and w2 are
    (qweight int32 [E, N, K / 8], scales uint8 [E, N, K / 32]), and the
    biases [E, N] are added to each product."""
    _check(expert_map)
    step = max(1, K80_MOE_CHUNK_SLOTS // topk_ids.shape[1])
    if topk_ids.shape[0] > step:
        out = x.new_empty((x.shape[0], w2[0].shape[1]))
        for a in range(0, topk_ids.shape[0], step):
            b = a + step
            out[a:b] = k80_mxfp4_experts(x[a:b], w13, w2, topk_weights[a:b],
                                         topk_ids[a:b], activation, w13_bias,
                                         w2_bias,
                                         apply_router_weight_on_input,
                                         expert_map)
        return out
    rows, per_row = _inputs(x, topk_weights, apply_router_weight_on_input)
    num_slots = topk_ids.numel()
    experts, slots, padded = _slots(topk_ids, w13[0].shape[0])
    ids = topk_ids.flatten().long()
    h = ops.k80_mxfp4_moe_gemm(rows, *w13, experts, slots, padded, num_slots,
                               per_row)
    if w13_bias is not None:
        h += w13_bias[ids]
    y = ops.k80_mxfp4_moe_gemm(_act_and_mul(h, activation), *w2, experts,
                               slots, padded, num_slots, 1)
    if w2_bias is not None:
        y += w2_bias[ids]
    return _combine(y, topk_weights, apply_router_weight_on_input)


def k80_fused_experts(hidden_states: torch.Tensor,
                      w1: torch.Tensor,
                      w2: torch.Tensor,
                      topk_weights: torch.Tensor,
                      topk_ids: torch.Tensor,
                      inplace: bool = False,
                      activation: str = "silu",
                      apply_router_weight_on_input: bool = False,
                      global_num_experts: int = -1,
                      expert_map: Optional[torch.Tensor] = None,
                      w1_bias: Optional[torch.Tensor] = None,
                      w2_bias: Optional[torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
    """fused_experts for FP16 / FP32 experts (w1 [E, 2I, H], w2 [E, H, I])
    on Kepler: k80_moe_dense_gemm for a few slots, cuBLAS for many."""
    _check(expert_map, w1_bias=w1_bias, w2_bias=w2_bias, **kwargs)
    rows, per_row = _inputs(hidden_states, topk_weights,
                            apply_router_weight_on_input)
    ids = topk_ids.flatten().long()
    num_slots = ids.numel()
    if (num_slots <= K80_MOE_PAIR_SLOTS
            or torch.cuda.is_current_stream_capturing()):
        experts = topk_ids.to(torch.int32).flatten()
        h = ops.k80_moe_dense_gemm(rows, w1, experts, per_row)
        y = ops.k80_moe_dense_gemm(_act_and_mul(h, activation), w2, experts,
                                   1)
    else:
        # One product per expert that has tokens (a host sync).
        src = torch.arange(num_slots, device=ids.device) // per_row
        y = hidden_states.new_empty((num_slots, w2.shape[1]))
        order = torch.argsort(ids, stable=True)
        counts = torch.bincount(ids, minlength=w1.shape[0]).tolist()
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            idx = order[start:start + n]
            start += n
            h = F.linear(rows.index_select(0, src[idx]), w1[e])
            y[idx] = F.linear(_act_and_mul(h, activation), w2[e])
    out = _combine(y, topk_weights, apply_router_weight_on_input)
    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out


def k80_repack_wna16(qweight: torch.Tensor, scales: torch.Tensor, bits: int,
                     zero: float) -> tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor]:
    """The K80 layout of MoE weights packed along K, as compressed-tensors
    and GPTQ load them: qweight int32 [E, K / P, N] and scales [E, G, N] give
    qweight [E, N, K / P], scales and zeros fp32 [E, N, G]. Value i of a word
    is in bits (bits * i) and up in both layouts."""
    assert qweight.dtype == torch.int32 and bits in (4, 8)
    q = qweight.transpose(1, 2).contiguous()
    s = scales.transpose(1, 2).to(torch.float32).contiguous()
    return q, s, torch.full_like(s, zero)


def k80_check_groups(intermediate_size_per_partition: int,
                     group_size: int) -> None:
    """The down product's K, split across dies, must hold whole groups: the
    loader gives each die only the groups that start in its part, so a split
    group takes a wrong scale (Qwen3-30B-A3B at TP=4: 192 of 768 features
    per die, groups of 128; perplexity 7.87 against 7.57 at TP=2)."""
    if group_size > 0 and intermediate_size_per_partition % group_size:
        raise ValueError(
            f"K80 MoE experts: {intermediate_size_per_partition} intermediate "
            f"features per die do not divide into groups of {group_size}; "
            "use fewer dies (tensor_parallel_size)")


def k80_moe_load(layer: torch.nn.Module, w13: tuple[str, str],
                 w2: tuple[str, str], bits: int, zero: float,
                 drop: tuple[str, ...]) -> None:
    """Replace a quantized MoE layer's (qweight, scales) parameters, named in
    w13 and w2, with k80_w13_* and k80_w2_* in the K80 layout, and drop the
    names in drop, so the checkpoint's tensors are freed."""
    for prefix, (q_name, s_name) in (("k80_w13", w13), ("k80_w2", w2)):
        q, s, z = k80_repack_wna16(getattr(layer, q_name).data,
                                   getattr(layer, s_name).data, bits, zero)
        delattr(layer, q_name)
        delattr(layer, s_name)
        for name, t in (("qweight", q), ("scales", s), ("zeros", z)):
            layer.register_parameter(f"{prefix}_{name}",
                                     torch.nn.Parameter(t, requires_grad=False))
    for name in drop:
        if hasattr(layer, name):
            delattr(layer, name)
    layer.k80_bits = bits


def k80_moe_apply(layer: torch.nn.Module,
                  x: torch.Tensor,
                  router_logits: torch.Tensor,
                  top_k: int,
                  renormalize: bool,
                  use_grouped_topk: bool = False,
                  topk_group: Optional[int] = None,
                  num_expert_group: Optional[int] = None,
                  global_num_experts: int = -1,
                  expert_map: Optional[torch.Tensor] = None,
                  custom_routing_function=None,
                  scoring_func: str = "softmax",
                  e_score_correction_bias: Optional[torch.Tensor] = None,
                  apply_router_weight_on_input: bool = False,
                  activation: str = "silu",
                  enable_eplb: bool = False,
                  **kwargs) -> torch.Tensor:
    """FusedMoEMethodBase.apply for a layer that k80_moe_load repacked."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    if enable_eplb:
        raise NotImplementedError("K80 MoE experts: no expert load balancing")
    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias)
    return k80_moe_experts(
        x, (layer.k80_w13_qweight, layer.k80_w13_scales, layer.k80_w13_zeros),
        (layer.k80_w2_qweight, layer.k80_w2_scales, layer.k80_w2_zeros),
        layer.k80_bits, topk_weights, topk_ids, activation,
        apply_router_weight_on_input, global_num_experts, expert_map,
        getattr(layer, "k80_w2_bits", None))

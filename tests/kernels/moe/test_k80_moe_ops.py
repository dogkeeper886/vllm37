# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#85): the MoE routing ops of vllm._moe_C load and run on Kepler.

The legacy build leaves out the permute/unpermute kernels (CUTLASS,
cuda_fp8.h), and the extension must still load with the ops it has.

Run on a Kepler GPU: pytest tests/kernels/moe/test_k80_moe_ops.py
"""
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    on_kepler)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()),
    reason="the legacy build's MoE ops run on Kepler only")

# (experts, top k): Mixtral 8/2, Qwen1.5-MoE 60/4 (not a power of two, so
# the kernel takes its workspace path), OLMoE 64/8, Qwen3-MoE 128/8, and
# 512 experts (past the 256 of the fused path).
ROUTES = [(8, 2), (60, 4), (64, 8), (128, 8), (512, 10)]
TOKENS = [1, 7, 256]


def test_moe_ops_load():
    assert ops.supports_moe_ops
    assert not torch.ops._moe_C.moe_permute_unpermute_supported()
    assert not hasattr(torch.ops._moe_C, "moe_permute")


@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_topk_softmax(m, route, index_dtype):
    experts, topk = route
    torch.manual_seed(m * experts + topk)
    gating = torch.randn(m, experts, device="cuda", dtype=torch.float32)
    weights = torch.empty(m, topk, device="cuda", dtype=torch.float32)
    ids = torch.empty(m, topk, device="cuda", dtype=index_dtype)
    rows = torch.empty(m, topk, device="cuda", dtype=torch.int32)

    ops.topk_softmax(weights, ids, rows, gating)

    ref_weights, ref_ids = torch.softmax(gating, dim=-1).topk(topk, dim=-1)
    torch.testing.assert_close(weights, ref_weights, atol=1e-5, rtol=1e-5)
    assert torch.equal(ids.long(), ref_ids)
    ref_rows = (torch.arange(topk, device="cuda") * m).unsqueeze(0) + \
        torch.arange(m, device="cuda").unsqueeze(1)
    assert torch.equal(rows.long(), ref_rows)


@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("block_size", [16, 64])
def test_moe_align_block_size(m, route, block_size):
    experts, topk = route
    torch.manual_seed(m + experts + block_size)
    topk_ids = torch.stack([
        torch.randperm(experts, device="cuda")[:topk] for _ in range(m)
    ]).to(torch.int32)
    numel = m * topk
    max_padded = numel + experts * (block_size - 1)
    sorted_ids = torch.empty(max_padded, device="cuda", dtype=torch.int32)
    expert_ids = torch.empty(max_padded // block_size + 1,
                             device="cuda",
                             dtype=torch.int32)
    num_post_pad = torch.empty(1, device="cuda", dtype=torch.int32)

    ops.moe_align_block_size(topk_ids, experts, block_size, sorted_ids,
                             expert_ids, num_post_pad)

    counts = torch.bincount(topk_ids.flatten().long(), minlength=experts)
    padded = (counts + block_size - 1) // block_size * block_size
    assert num_post_pad.item() == padded.sum().item()
    flat = topk_ids.flatten()
    start = 0
    for e in range(experts):
        n = padded[e].item()
        if n == 0:
            continue
        blocks = expert_ids[start // block_size:(start + n) // block_size]
        assert torch.all(blocks == e)
        slots = sorted_ids[start:start + n]
        real = slots[slots < numel].long()
        assert sorted(real.tolist()) == \
            torch.nonzero(flat == e).flatten().tolist()
        assert torch.all(slots[real.numel():] == numel)
        start += n


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("topk", [2, 8])
def test_moe_sum(dtype, topk):
    x = torch.randn(33, topk, 1024, device="cuda", dtype=dtype)
    out = torch.empty(33, 1024, device="cuda", dtype=dtype)
    ops.moe_sum(x, out)
    torch.testing.assert_close(out, x.float().sum(dim=1).to(dtype),
                               atol=2e-2,
                               rtol=1e-3)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#96): Torch SDPA prefill in chunks of query rows gives the
attention of the whole prompt (and, in FP32, a float64 reference): causal and
not, GQA, ALiBi, and several prompts packed in one batch.

Run: pytest tests/kernels/attention/test_k80_sdpa_chunks.py
"""
import types

import pytest
import torch

from vllm.attention.backends import torch_sdpa
from vllm.attention.backends.abstract import AttentionType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# (query heads, kv heads, head size)
HEADS = [(16, 8, 128), (8, 8, 64)]


def reference(q, k, v, scale, causal, slopes):
    """[len, H, D] float64 attention of one prompt."""
    q, k, v = (t.double().transpose(0, 1) for t in (q, k, v))  # [H, len, D]
    reps = q.shape[0] // k.shape[0]
    k = k.repeat_interleave(reps, 0)
    v = v.repeat_interleave(reps, 0)
    s = (q @ k.transpose(1, 2)) * scale
    n = s.shape[-1]
    pos = torch.arange(n, device=q.device)
    if slopes is not None:
        s = s + slopes.double().to(q.device)[:, None, None] * (
            pos[None, None, :] - pos[None, :, None]).double()
    if causal:
        s = s.masked_fill(pos[None, None, :] > pos[None, :, None],
                          float("-inf"))
    return (s.softmax(-1) @ v).transpose(0, 1)


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("kind", [AttentionType.DECODER, AttentionType.ENCODER])
def test_chunks_match_whole_prompt(heads, dtype, alibi, kind):
    num_heads, num_kv_heads, head_size = heads
    slopes = [2**(-(i + 1) / 2) for i in range(num_heads)] if alibi else None
    impl = torch_sdpa.TorchSDPAImpl(num_heads, head_size, head_size**-0.5,
                                    num_kv_heads, slopes, None, "auto")
    lens = [300, 7, 129]
    g = torch.Generator(device=DEVICE).manual_seed(0)
    total = sum(lens)
    q = torch.randn(total, num_heads, head_size, device=DEVICE,
                    generator=g).to(dtype)
    k = torch.randn(total, num_kv_heads, head_size, device=DEVICE,
                    generator=g).to(dtype)
    v = torch.randn(total, num_kv_heads, head_size, device=DEVICE,
                    generator=g).to(dtype)
    meta = types.SimpleNamespace(seq_lens=lens, encoder_seq_lens=lens)
    causal = kind == AttentionType.DECODER
    old = torch_sdpa.K80_SDPA_MAX_SCORES
    try:
        torch_sdpa.K80_SDPA_MAX_SCORES = 1 << 40
        whole = impl._run_sdpa_forward(q, k, v, meta, kind)
        # 50 query rows per chunk for the 300-token prompt at 16 heads.
        torch_sdpa.K80_SDPA_MAX_SCORES = num_heads * 300 * 50
        chunked = impl._run_sdpa_forward(q, k, v, meta, kind)
    finally:
        torch_sdpa.K80_SDPA_MAX_SCORES = old
    # Chunks change only the rounding of the output.
    tol = 2e-6 if dtype == torch.float32 else 5e-4
    assert (chunked.double() - whole.double()).abs().max().item() < tol
    if dtype == torch.float16:
        return  # FP16 rounds the ALiBi bias (up to 0.7 x 300) coarsely
    start = 0
    for n in lens:
        sl = slice(start, start + n)
        ref = reference(q[sl], k[sl], v[sl], impl.scale, causal,
                        impl.alibi_slopes)
        err = (chunked[sl].double() - ref).abs().max().item()
        assert err < 5e-5, (n, err)
        start += n

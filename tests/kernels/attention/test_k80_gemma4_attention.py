# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#126): the Gemma 4 model's own attention, RoPE and routing
against float64 references: prefill in chunks with and without a sliding
window, decode from a paged cache, proportional RoPE, and the router's
softmax-top-k-renormalize-scale.

Run: pytest tests/kernels/attention/test_k80_gemma4_attention.py
"""
import pytest
import torch

from vllm.model_executor.models import gemma4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reference(q, k, v, window):
    """[L, H, D] float64 causal attention, scale 1.0."""
    n, h, _ = q.shape
    g = h // k.shape[1]
    q, k, v = (t.double().transpose(0, 1) for t in (q, k, v))
    k, v = k.repeat_interleave(g, 0), v.repeat_interleave(g, 0)
    s = q @ k.transpose(1, 2)
    i = torch.arange(n, device=q.device)
    mask = i[None, :] <= i[:, None]
    if window is not None:
        mask &= i[None, :] > i[:, None] - window
    s = s.masked_fill(~mask, float("-inf"))
    return (s.softmax(-1) @ v).transpose(0, 1)


@pytest.mark.parametrize("window", [None, 64])
def test_prefill_matches_reference(window):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    n, h, hkv, d = 300, 8, 2, 64
    q, k, v = (torch.randn(n, x, d, device=DEVICE, generator=g)
               for x in (h, hkv, hkv))
    old = gemma4._MAX_SCORES
    try:
        gemma4._MAX_SCORES = h * 300 * 40  # several chunks of rows
        out = gemma4.gemma4_prefill_attention(q, k, v, window)
    finally:
        gemma4._MAX_SCORES = old
    err = (out.double() - reference(q, k, v, window)).abs().max().item()
    assert err < 1e-4, err


@pytest.mark.parametrize("window", [None, 64])
def test_decode_matches_reference(window):
    """Each sequence's tokens scattered over shuffled blocks of a cache; the
    last token attends to its keys there."""
    g = torch.Generator(device=DEVICE).manual_seed(1)
    h, hkv, d, bs, lens = 8, 2, 64, 16, [5, 70, 200]
    blocks_per = [(n + bs - 1) // bs for n in lens]
    perm = torch.randperm(sum(blocks_per) + 3, device=DEVICE, generator=g)
    kc = torch.zeros(perm.numel() * bs, hkv * d, device=DEVICE)
    vc = torch.zeros_like(kc)
    tables = torch.zeros(len(lens), max(blocks_per), dtype=torch.int32,
                         device=DEVICE)
    qs, expected, used = [], [], 0
    for s, n in enumerate(lens):
        tables[s, :blocks_per[s]] = perm[used:used + blocks_per[s]]
        used += blocks_per[s]
        q, k, v = (torch.randn(n, x, d, device=DEVICE, generator=g)
                   for x in (h, hkv, hkv))
        p = torch.arange(n, device=DEVICE)
        slots = tables[s].long()[p // bs] * bs + p % bs
        kc[slots], vc[slots] = k.reshape(n, -1), v.reshape(n, -1)
        qs.append(q[-1])
        expected.append(reference(q, k, v, window)[-1])
    slots, valid = gemma4.gemma4_decode_slots(
        torch.tensor(lens, device=DEVICE), tables, bs, window)
    out = gemma4.gemma4_decode_attention(torch.stack(qs), kc, vc, slots,
                                         valid)
    err = (out.double() - torch.stack(expected)).abs().max().item()
    assert err < 1e-4, err


def test_proportional_rope():
    """Full layers: head 512, a quarter of the pairs rotate, frequencies
    over the head size (llama.cpp: rope_freqs 1 for 64 pairs, 1e30 after)."""
    d, base = 512, 1e6
    rope = gemma4.Gemma4RotaryEmbedding(d, {
        "partial_rotary_factor": 0.25,
        "rope_theta": base
    }, 8192, torch.float32).to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(2)
    x = torch.randn(5, 2, d, device=DEVICE, generator=g, dtype=torch.float64)
    pos = torch.tensor([0, 1, 7, 100, 4000], device=DEVICE)
    q, _ = rope.forward_native(pos, x.float().reshape(5, -1).clone(),
                               x.float().reshape(5, -1).clone())
    out = q.view(5, 2, d).double()
    ff = torch.cat((torch.ones(64), torch.full((192, ), 1e30))).double()
    theta = pos.double()[:, None] * base**(
        -torch.arange(0, d, 2).double() / d) / ff.to(DEVICE)
    cos, sin = theta.cos()[:, None], theta.sin()[:, None]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    expected = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)
    # FP32 angles, as V0's RoPE cache: the error grows with the position
    # (about 2e-4 at 4,000).
    assert (out - expected).abs().max().item() < 5e-4
    assert torch.equal(out[..., 64:256], x[..., 64:256].float().double())


def test_router_routing():
    """Softmax over all experts, top k, renormalize, times the expert's
    output scale."""

    class Cfg:
        rms_norm_eps, hidden_size, num_experts = 1e-6, 32, 16

    router = gemma4.Gemma4Router.__new__(gemma4.Gemma4Router)
    torch.nn.Module.__init__(router)
    router.per_expert_scale = torch.nn.Parameter(
        torch.linspace(0.5, 1.5, Cfg.num_experts))
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(7, Cfg.num_experts, generator=g)
    w, ids = router.routing_function(None, logits, 4, True)
    p = logits.double().softmax(-1)
    top = p.topk(4, -1)
    expected = (top.values / top.values.sum(-1, keepdim=True) *
                router.per_expert_scale.double()[top.indices])
    assert torch.equal(ids.long(), top.indices)
    assert (w.double() - expected).abs().max().item() < 1e-6

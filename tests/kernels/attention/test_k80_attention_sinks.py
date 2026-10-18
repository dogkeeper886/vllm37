# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#129): attention sinks and per-layer sliding windows on the V0
attention path, against a float64 reference: the paged decode kernels (v1
and v2), XFormers prefill and Torch SDPA prefill. Windows are those of the
2026 study models: 128 (gpt-oss), 1,024 (Gemma 4) and 2,048 (Muse-Glimmer).

Run on a Kepler GPU: pytest tests/kernels/attention/test_k80_attention_sinks.py
"""
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    on_kepler)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()),
    reason="the K80 attention path is tested on Kepler")

DTYPES = [torch.float32, torch.float16]
WINDOWS = [None, 128, 1024, 2048]
BLOCK = 16
HEAD = 64  # gpt-oss


def attend64(q, k, v, q_pos, window, sinks, scale):
    """float64 attention of queries q [M, H, D] at positions q_pos [M] over
    keys k, v [L, KV, D] at positions 0 .. L - 1, causal. On the CPU: these
    float64 products gave a different result run to run on the K80."""
    device = q.device
    q, k, v, q_pos = q.cpu(), k.cpu(), v.cpu(), q_pos.cpu()
    sinks = None if sinks is None else sinks.cpu()
    h, kv = q.shape[1], k.shape[1]
    k = k.double().repeat_interleave(h // kv, 1)
    v = v.double().repeat_interleave(h // kv, 1)
    s = torch.einsum("mhd,lhd->hml", q.double(), k) * scale
    pos = torch.arange(k.shape[0], device=q.device)
    masked = pos[None, :] > q_pos[:, None]
    if window is not None:
        masked |= pos[None, :] <= q_pos[:, None] - window
    s = s.masked_fill(masked, float("-inf"))
    if sinks is not None:
        sink = sinks.double().view(h, 1, 1).expand(h, s.shape[1], 1)
        p = torch.softmax(torch.cat((s, sink), -1), -1)[..., :-1]
    else:
        p = torch.softmax(s, -1)
    return torch.einsum("hml,lhd->mhd", p, v).to(device)


def make_sinks(h, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # gpt-oss's sinks are logits of about the scores' size.
    return torch.randn(h, device="cuda", generator=g) * 2


def paged_cache(seq_lens, kv, dtype, seed=0):
    """A random paged KV cache in the layout the kernels read, each sequence
    on shuffled blocks, and its keys and values per sequence [L, KV, D]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    nblocks = [(n + BLOCK - 1) // BLOCK for n in seq_lens]
    total = sum(nblocks)
    key_cache = torch.randn(total, kv, HEAD // x, BLOCK, x, device="cuda",
                            generator=g).to(dtype)
    value_cache = torch.randn(total, kv, HEAD, BLOCK, device="cuda",
                              generator=g).to(dtype)
    order = torch.randperm(total, device="cuda", generator=g).to(torch.int32)
    table = torch.zeros(len(seq_lens), max(nblocks), dtype=torch.int32,
                        device="cuda")
    keys, values, start = [], [], 0
    for i, (n, b) in enumerate(zip(seq_lens, nblocks)):
        blocks = order[start:start + b]
        start += b
        table[i, :b] = blocks
        kb = key_cache[blocks.long()].permute(0, 3, 1, 2, 4)
        keys.append(kb.reshape(b * BLOCK, kv, HEAD)[:n])
        vb = value_cache[blocks.long()].permute(0, 3, 1, 2)
        values.append(vb.reshape(b * BLOCK, kv, HEAD)[:n])
    return key_cache, value_cache, table, keys, values


def decode(version, q, key_cache, value_cache, table, seq_lens, kv, scale,
           sinks, window):
    out = torch.empty_like(q)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    one = torch.tensor(1.0, device="cuda")
    args = (key_cache, value_cache, kv, scale, table, lens, BLOCK,
            max(seq_lens), None, "auto", one, one)
    extra = dict(sinks=sinks, sliding_window=window or 0)
    if version == 1:
        ops.paged_attention_v1(out, q, *args, **extra)
    else:
        parts = (max(seq_lens) + 511) // 512
        tmp = torch.empty(q.shape[0], q.shape[1], parts, HEAD, dtype=q.dtype,
                          device="cuda")
        sums = torch.empty(q.shape[0], q.shape[1], parts, device="cuda")
        ops.paged_attention_v2(out, sums, torch.empty_like(sums), tmp, q,
                               *args, **extra)
    return out


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("use_sinks", [False, True])
def test_paged_decode(version, dtype, window, use_sinks):
    """gpt-oss's heads per die at TP=2 (32 queries, 4 KV heads), sequences
    shorter than, at and well past each window, and window edges inside a
    block."""
    h, kv, scale = 32, 4, HEAD**-0.5
    seq_lens = [1, 37, 128, 129, 1000, 1024, 1031, 2048, 2049, 3001]
    key_cache, value_cache, table, keys, values = paged_cache(
        seq_lens, kv, dtype)
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(len(seq_lens), h, HEAD, device="cuda",
                    generator=g).to(dtype)
    sinks = make_sinks(h, 2) if use_sinks else None
    out = decode(version, q, key_cache, value_cache, table, seq_lens, kv,
                 scale, sinks, window)
    tol = 2e-5 if dtype == torch.float32 else 2e-3
    for i, n in enumerate(seq_lens):
        pos = torch.tensor([n - 1], device="cuda")
        ref = attend64(q[i:i + 1], keys[i], values[i], pos, window, sinks,
                       scale)
        err = ((out[i].double() - ref[0]).norm() / ref[0].norm()).item()
        assert err < tol, (n, err)


def test_paged_decode_defaults_unchanged():
    """Without sinks or a window, both kernels give what they gave before:
    plain causal attention over the whole sequence."""
    h, kv, scale = 8, 8, HEAD**-0.5
    seq_lens = [5, 700, 1500]
    key_cache, value_cache, table, keys, values = paged_cache(
        seq_lens, kv, torch.float32, seed=3)
    q = torch.randn(len(seq_lens), h, HEAD, device="cuda")
    for version in (1, 2):
        out = decode(version, q, key_cache, value_cache, table, seq_lens, kv,
                     scale, None, None)
        for i, n in enumerate(seq_lens):
            ref = attend64(q[i:i + 1], keys[i], values[i],
                           torch.tensor([n - 1], device="cuda"), None, None,
                           scale)
            torch.testing.assert_close(out[i].double(), ref[0], rtol=2e-5,
                                       atol=2e-5)


def prefill_inputs(seq_lens, h, kv, dtype, seed=4):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = sum(seq_lens)
    q = torch.randn(t, h, HEAD, device="cuda", generator=g).to(dtype)
    k = torch.randn(t, kv, HEAD, device="cuda", generator=g).to(dtype)
    v = torch.randn(t, kv, HEAD, device="cuda", generator=g).to(dtype)
    return q, k, v


def prefill64(q, k, v, seq_lens, window, sinks, scale):
    out, start = [], 0
    for n in seq_lens:
        sl = slice(start, start + n)
        out.append(
            attend64(q[sl], k[sl], v[sl], torch.arange(n, device="cuda"),
                     window, sinks, scale))
        start += n
    return torch.cat(out)


def check_prefill(out, ref, dtype):
    tol = 2e-5 if dtype == torch.float32 else 3e-3
    err = ((out.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()
    assert err < tol, err


def xformers_impl(h, kv, window, sinks):
    from vllm.attention.backends.xformers import XFormersImpl
    return XFormersImpl(h, HEAD, HEAD**-0.5, kv, None, window, "auto",
                        sinks=sinks)


def sdpa_impl(h, kv, window, sinks):
    from vllm.attention.backends.torch_sdpa import TorchSDPAImpl
    return TorchSDPAImpl(h, HEAD, HEAD**-0.5, kv, None, window, "auto",
                         sinks=sinks)


SEQ_LENS = [3, 200, 2500]


@pytest.mark.parametrize("backend", ["xformers", "sdpa"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("use_sinks", [False, True])
def test_prefill(backend, dtype, window, use_sinks):
    h, kv = 8, 2
    sinks = make_sinks(h, 5) if use_sinks else None
    q, k, v = prefill_inputs(SEQ_LENS, h, kv, dtype)
    meta = SimpleNamespace(seq_lens=SEQ_LENS, attn_bias=None)
    if backend == "xformers":
        out = xformers_impl(h, kv, window, sinks).\
            _run_memory_efficient_xformers_forward(q, k, v, meta)
    else:
        out = sdpa_impl(h, kv, window, sinks)._run_sdpa_forward(q, k, v, meta)
    assert out.dtype == dtype
    check_prefill(out, prefill64(q, k, v, SEQ_LENS, window, sinks,
                                 HEAD**-0.5), dtype)


def test_sdpa_prefill_chunks(monkeypatch):
    """A prompt split into query-row chunks (#96) keeps each row's window and
    sink."""
    from vllm.attention.backends import torch_sdpa
    monkeypatch.setattr(torch_sdpa, "K80_SDPA_MAX_SCORES", 8 * 300 * 2500)
    h, kv, window = 8, 2, 1024
    sinks = make_sinks(h, 6)
    q, k, v = prefill_inputs([2500], h, kv, torch.float32, seed=7)
    meta = SimpleNamespace(seq_lens=[2500], attn_bias=None)
    for s in (None, sinks):
        out = sdpa_impl(h, kv, window, s)._run_sdpa_forward(q, k, v, meta)
        check_prefill(out, prefill64(q, k, v, [2500], window, s, HEAD**-0.5),
                      torch.float32)


def test_xformers_layers_keep_their_windows():
    """Layers of an interleaved model share one metadata object: a full
    layer after a sliding one still attends to every earlier key."""
    h, kv = 8, 2
    q, k, v = prefill_inputs(SEQ_LENS, h, kv, torch.float32, seed=8)
    meta = SimpleNamespace(seq_lens=SEQ_LENS, attn_bias=None)
    windows: list[Optional[int]] = [128, None, 128]
    for window in windows:
        out = xformers_impl(h, kv, window, None).\
            _run_memory_efficient_xformers_forward(q, k, v, meta)
        check_prefill(out, prefill64(q, k, v, SEQ_LENS, window, None,
                                     HEAD**-0.5), torch.float32)

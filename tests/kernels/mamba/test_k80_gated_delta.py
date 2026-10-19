# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K80 fork (#127): Gated DeltaNet without Triton (k80_gated_delta.py),
against a float64 per-token reference: the chunked prefill, the decode step,
prefill then decode, and the causal convolution.

Run on a Kepler GPU: pytest tests/kernels/mamba/test_k80_gated_delta.py
"""
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.k80_gated_delta import (
    causal_conv1d_prefill, causal_conv1d_step, chunk_gated_delta_rule, gating,
    l2norm, recurrent_step)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.k80 import (  # noqa: E501
    on_kepler)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and on_kepler()),
    reason="the K80 Gated DeltaNet ops are tested on Kepler")

K = V = 128  # every Qwen3.5 / 3.6 / 3.8 config


def inputs(t, h, seed=0):
    """q, k (L2-normalised), v, g and beta as the model makes them, with
    Qwen3.5-0.8B's spread of A_log and dt_bias."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = l2norm(torch.randn(t, h, K, device="cuda", generator=gen))
    k = l2norm(torch.randn(t, h, K, device="cuda", generator=gen))
    v = torch.randn(t, h, V, device="cuda", generator=gen)
    A_log = torch.empty(h, device="cuda").uniform_(-6.5, 0.3, generator=gen)
    dt_bias = torch.empty(h, device="cuda").uniform_(-13, 8, generator=gen)
    a = torch.randn(t, h, device="cuda", generator=gen)
    b = torch.randn(t, h, device="cuda", generator=gen)
    g, beta = gating(A_log, a, b, dt_bias)
    return q, k, v, g, beta


def delta64(q, k, v, g, beta, state=None):
    """float64 per-token reference on the CPU: o [T, H, V], state [H, K, V]."""
    q, k, v, g, beta = (x.double().cpu() for x in (q, k, v, g, beta))
    t, h, _ = q.shape
    s = torch.zeros(h, K, V, dtype=torch.float64) if state is None else \
        state.double().cpu().clone()
    out = torch.empty(t, h, V, dtype=torch.float64)
    for i in range(t):
        s = s * g[i].exp()[:, None, None]
        kv = (k[i][:, :, None] * s).sum(1)
        s = s + k[i][:, :, None] * ((v[i] - kv) * beta[i][:, None])[:, None, :]
        out[i] = (q[i][:, :, None] * K**-0.5 * s).sum(1)
    return out, s


def rel(x, ref):
    return ((x.double().cpu() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("t", [1, 37, 64, 65, 300])
@pytest.mark.parametrize("initial", [False, True])
def test_chunk_prefill(t, initial):
    """16 value heads, as Qwen3.5-0.8B per die; prompts shorter than, equal
    to and longer than a chunk."""
    h = 16
    q, k, v, g, beta = inputs(t, h)
    s0 = torch.randn(h, K, V, device="cuda") * 0.1 if initial else None
    out, state = chunk_gated_delta_rule(q, k, v, g, beta, s0)
    ref, ref_state = delta64(q, k, v, g, beta, s0)
    assert out.dtype == torch.float32 and out.shape == (t, h, V)
    assert rel(out, ref) < 2e-5
    assert rel(state, ref_state) < 2e-5


def test_recurrent_step():
    """A batch of 3 sequences decoding 20 tokens, state kept in float32."""
    b, h, t = 3, 32, 20
    seqs = [inputs(t, h, seed=10 + i) for i in range(b)]
    state = torch.zeros(b, h, K, V, device="cuda")
    outs = []
    for i in range(t):
        step = [torch.stack([s[j][i] for s in seqs]) for j in range(5)]
        outs.append(recurrent_step(*step, state))
    for n, s in enumerate(seqs):
        ref, ref_state = delta64(*s)
        out = torch.stack([o[n] for o in outs])
        assert rel(out, ref) < 2e-5
        assert rel(state[n], ref_state) < 2e-5


def test_prefill_then_decode():
    """A 150-token prompt, then 30 decode steps from its final state."""
    h, t0, t1 = 16, 150, 30
    q, k, v, g, beta = inputs(t0 + t1, h, seed=3)
    out0, state = chunk_gated_delta_rule(q[:t0], k[:t0], v[:t0], g[:t0],
                                         beta[:t0])
    state = state.unsqueeze(0)
    outs = [
        recurrent_step(q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1],
                       beta[i:i + 1], state)[0] for i in range(t0, t0 + t1)
    ]
    out = torch.cat((out0, torch.stack(outs)))
    ref, _ = delta64(q, k, v, g, beta)
    assert rel(out, ref) < 2e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_causal_conv1d(dtype):
    """Qwen3.5-0.8B's 6,144 conv channels, width 4: a 50-token prompt, then
    10 decode steps, against one float64 convolution over all 60."""
    c, w, t0, t1 = 6144, 4, 50, 10
    gen = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(t0 + t1, c, device="cuda", generator=gen).to(dtype)
    weight = (torch.randn(c, w, device="cuda", generator=gen) * 0.5).to(dtype)
    state = torch.empty(c, w - 1, device="cuda", dtype=dtype)
    out0 = causal_conv1d_prefill(x[:t0], weight, state)
    state = state.unsqueeze(0)
    outs = [causal_conv1d_step(x[i:i + 1], weight, state)[0]
            for i in range(t0, t0 + t1)]
    out = torch.cat((out0, torch.stack(outs)))
    xr = F.pad(x.double().cpu().t(), (w - 1, 0)).unsqueeze(0)
    ref = F.silu(F.conv1d(xr, weight.double().cpu().unsqueeze(1),
                          groups=c))[0].t()
    assert out.dtype == dtype
    assert rel(out, ref) < (1e-6 if dtype == torch.float32 else 2e-3)

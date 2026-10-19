# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# K80 fork (#127): Gated DeltaNet, the linear attention of Qwen3.5 / Qwen3.6
# / Qwen3.8 (and Qwen3-Next), on Kepler (sm_37). Upstream's kernels for it are
# Triton (FLA), FlashInfer or a BF16 sm_80 CUDA kernel, none of which the K80
# runs; these are torch ops in float32, after transformers' reference
# implementation (torch_chunk_gated_delta_rule, torch_causal_conv1d_update).
#
# Per value head, with q and k L2-normalised and q scaled by K^-0.5, the state
# S (K x V, float32) follows
#   S <- exp(g_t) S;  S <- S + k_t (beta_t (v_t - S^T k_t))^T;  o_t = S^T q_t
#   chunk_gated_delta_rule   a prompt in chunks of 64 tokens: within a chunk
#       the delta rule is a unit lower-triangular solve (cuBLAS trsm), across
#       chunks one state update per chunk
#   recurrent_step           one decode token for a batch of sequences
# and the short causal convolution in front of it (width 4, SiLU):
#   causal_conv1d_prefill    a prompt, from a zero state
#   causal_conv1d_step       one decode token, updating the state

from typing import Optional

import torch
import torch.nn.functional as F

CHUNK = 64


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x / sqrt(sum x^2 + eps) over the last dim, as FLA's l2norm."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
           dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The decay g = -exp(A_log) softplus(a + dt_bias) and the write strength
    beta = sigmoid(b), float32, shaped as a and b ([T, H])."""
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, b.float().sigmoid()


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk: int = CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence: q, k [T, H, K], v [T, H, V] (q and k L2-normalised,
    already repeated to the value heads), g, beta [T, H]. Returns o [T, H, V]
    and the final state [H, K, V], both float32."""
    T, H, K = k.shape
    V = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    # [H, N, C, D] and [H, N, C]
    q, k, v = (F.pad(x.float(), (0, 0, 0, 0, 0, pad)).transpose(0, 1).reshape(
        H, -1, chunk, x.shape[-1]) for x in (q, k, v))
    g, beta = (F.pad(x.float(), (0, 0, 0, pad)).transpose(0, 1).reshape(
        H, -1, chunk) for x in (g, beta))
    q = q * K**-0.5
    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    g = g.cumsum(dim=-1)
    # decay[i, j] = exp(g_i - g_j) for j <= i, else 0
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    lower = torch.ones(chunk, chunk, dtype=torch.bool,
                       device=q.device).tril()
    decay = diff.masked_fill(~lower, float("-inf")).exp()
    # T = (I + A)^-1, A strictly lower with A[i, j] = (k_beta_i . k_j) decay
    a = (k_beta @ k.transpose(-1, -2) * decay).masked_fill(
        ~lower.tril(-1), 0)
    eye = torch.eye(chunk, dtype=a.dtype, device=a.device)
    t = torch.linalg.solve_triangular(a + eye,
                                      eye.expand_as(a),
                                      upper=False,
                                      unitriangular=True)
    u = t @ v_beta
    w = t @ (k_beta * g.exp().unsqueeze(-1))
    state = (torch.zeros(H, K, V, dtype=torch.float32, device=q.device)
             if initial_state is None else initial_state.float().clone())
    out = torch.empty_like(u)
    for n in range(q.shape[1]):
        q_n, k_n, g_n = q[:, n], k[:, n], g[:, n]
        new = u[:, n] - w[:, n] @ state
        inner = (q_n @ k_n.transpose(-1, -2)) * decay[:, n]
        out[:, n] = (q_n * g_n.exp().unsqueeze(-1)) @ state + inner @ new
        last = g_n[:, -1:]
        state = state * last.exp().unsqueeze(-1) + (
            k_n * (last - g_n).exp().unsqueeze(-1)).transpose(-1, -2) @ new
    out = out.reshape(H, -1, V)[:, :T].transpose(0, 1)
    return out, state


def recurrent_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   g: torch.Tensor, beta: torch.Tensor,
                   state: torch.Tensor) -> torch.Tensor:
    """One token per sequence: q, k [B, H, K], v [B, H, V], g, beta [B, H],
    state [B, H, K, V] float32, updated in place. Returns o [B, H, V]."""
    q = q.float() * q.shape[-1]**-0.5
    k = k.float()
    state.mul_(g.exp()[..., None, None])
    kv = (k.unsqueeze(-1) * state).sum(dim=-2)
    delta = (v.float() - kv) * beta.unsqueeze(-1)
    state.add_(k.unsqueeze(-1) * delta.unsqueeze(-2))
    return (q.unsqueeze(-1) * state).sum(dim=-2)


def causal_conv1d_prefill(x: torch.Tensor, weight: torch.Tensor,
                          state: torch.Tensor) -> torch.Tensor:
    """One sequence from a zero state: x [T, C], weight [C, W]. Writes the
    last W - 1 inputs into state [C, W - 1] and returns SiLU(conv) [T, C]."""
    width = weight.shape[-1]
    xt = F.pad(x.transpose(0, 1), (width - 1, 0))  # [C, W - 1 + T]
    state.copy_(xt[:, -(width - 1):])
    out = F.conv1d(xt.unsqueeze(0).to(weight.dtype), weight.unsqueeze(1),
                   groups=x.shape[1])[0]
    return F.silu(out).transpose(0, 1).to(x.dtype)


def causal_conv1d_step(x: torch.Tensor, weight: torch.Tensor,
                       state: torch.Tensor) -> torch.Tensor:
    """One token per sequence: x [B, C], state [B, C, W - 1], updated in
    place. Returns SiLU(conv) [B, C]."""
    window = torch.cat((state, x.unsqueeze(-1).to(state.dtype)), dim=-1)
    state.copy_(window[..., 1:])
    out = (window.to(weight.dtype) * weight).sum(dim=-1)
    return F.silu(out).to(x.dtype)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, times weight, times SiLU(gate): the norm
    before the output projection (norm_before_gate)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * weight.float() * F.silu(gate.float())).to(gate.dtype)

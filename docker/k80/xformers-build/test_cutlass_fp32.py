# SPDX-License-Identifier: BSD-3-Clause
#
# Runtime smoke test for XFormers cutlassF on K80. k80-runtime.yml runs it
# inside the runtime image before the server takes the GPU.
#
# Allocates small fp32
# q/k/v tensors on cuda:0, calls xformers.ops.memory_efficient_attention, and
# compares against a reference computed manually with PyTorch eager ops.
# Reports PASS/FAIL based on max absolute / max relative error.
#
# This covers the full attention algorithm (Q·K^T → softmax → ·V) on sm_37,
# beyond the GEMM that cutlass-repro/ checks.
#
# Hardware safety: TP=1-equivalent, single die, batch=1, small tensors.
# Total VRAM: ~16 MB. No NCCL.

import sys
import torch


def reference_attention(q, k, v):
    """
    Reference: scaled_dot_product_attention computed manually with eager ops.

    Shapes follow xformers convention:
        q, k, v: (B, M, H, K) where M is seq_len, H is num_heads, K is head_dim
    Returns:
        out: (B, M, H, K)

    Simple non-causal attention without mask. Computed entirely in fp32 to
    minimize floating-point noise relative to the kernel under test.
    """
    # Permute (B, M, H, K) -> (B, H, M, K)
    qt = q.permute(0, 2, 1, 3)
    kt = k.permute(0, 2, 1, 3)
    vt = v.permute(0, 2, 1, 3)

    # scale = 1/sqrt(head_dim)
    scale = float(qt.size(-1)) ** -0.5

    # scores: (B, H, M_q, M_k)
    scores = torch.matmul(qt, kt.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)

    # out: (B, H, M_q, K) -> (B, M_q, H, K)
    out = torch.matmul(attn, vt).permute(0, 2, 1, 3).contiguous()
    return out


def main():
    print("=== XFormers cutlassF fp32 smoke test ===")

    # Device + props
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 2

    dev = torch.device("cuda:0")
    prop = torch.cuda.get_device_properties(dev)
    print(f"device       : {prop.name} (cc {prop.major}.{prop.minor})")

    # Imports (after CUDA is up; xformers may probe device on import)
    import xformers
    import xformers.ops
    from xformers.ops.fmha import memory_efficient_attention_forward
    from xformers.ops.fmha.common import Inputs
    print(f"xformers     : {xformers.__version__}")
    print(f"location     : {xformers.__file__}")

    # Confirm cutlassF is available and accepts our K80
    from xformers.ops.fmha.cutlass import FwOp as CutlassFwOp
    print(f"cutlass FwOp : CUDA_MINIMUM_COMPUTE_CAPABILITY = "
          f"{CutlassFwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY}")

    # Problem shape — small enough that the smoke runs in milliseconds.
    # Dims chosen to match common attention configs (head_dim=64, num_heads=8).
    B, M, H, K = 1, 128, 8, 64
    print(f"shape        : B={B} M={M} H={H} K={K}, fp32")

    # Deterministic inputs for reproducibility across runs.
    torch.manual_seed(20260426)
    q = torch.randn(B, M, H, K, dtype=torch.float32, device=dev)
    k = torch.randn(B, M, H, K, dtype=torch.float32, device=dev)
    v = torch.randn(B, M, H, K, dtype=torch.float32, device=dev)

    # Reference, in fp32 on the same device.
    out_ref = reference_attention(q, k, v)

    # Pre-flight: confirm cutlassF supports() returns True for these inputs.
    # If False here, the dispatcher would skip to a different op or raise.
    inp = Inputs(query=q, key=k, value=v, attn_bias=None, p=0.0, scale=None)
    supports_reason = CutlassFwOp.not_supported_reasons(inp)
    if supports_reason:
        print(f"FAIL: cutlass.FwOp does not support these inputs: {supports_reason}")
        return 3
    print(f"cutlass FwOp.supports(inputs) = True")

    # Use the forward-only API to surgically dispatch through cutlassF —
    # explicit, no ambiguity about backward op resolution. This is the
    # correct form per xformers v0.0.23 ops/fmha/__init__.py:
    #     def memory_efficient_attention_forward(..., op: Optional[Type[AttentionFwOpBase]])
    try:
        out_xfm = memory_efficient_attention_forward(q, k, v, op=CutlassFwOp)
    except Exception as e:
        print(f"FAIL: memory_efficient_attention_forward raised: "
              f"{type(e).__name__}: {e}")
        return 3

    # Sanity check shape
    if out_xfm.shape != out_ref.shape:
        print(f"FAIL: shape mismatch — xformers {tuple(out_xfm.shape)} vs "
              f"reference {tuple(out_ref.shape)}")
        return 4

    # Compare element-wise.
    abs_diff = (out_xfm - out_ref).abs()
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())

    denom = torch.maximum(out_xfm.abs(), out_ref.abs())
    valid = denom > 1e-3   # avoid divide-by-tiny noise
    rel_diff = torch.where(valid, abs_diff / denom, torch.zeros_like(abs_diff))
    max_rel = float(rel_diff.max())

    # Tolerance budget: cutlass attention sums over M*K products in fp32;
    # softmax adds another small noise term. With M=128, K=64, fp32 eps ~6e-8,
    # expected relative error is ~M*K*eps = ~5e-4 worst case.
    # Use 1e-3 as the headline tolerance — comfortable headroom.
    rel_tolerance = 1e-3

    print(f"max abs err  : {max_abs:.3e}")
    print(f"mean abs err : {mean_abs:.3e}")
    print(f"max rel err  : {max_rel:.3e} (tolerance {rel_tolerance:.0e})")

    if max_rel <= rel_tolerance:
        print("RESULT       : PASS")
        return 0
    else:
        print("RESULT       : FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())

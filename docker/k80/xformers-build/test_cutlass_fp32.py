# SPDX-License-Identifier: BSD-3-Clause

import sys
import torch


def reference_attention(q, k, v):
    qt = q.permute(0, 2, 1, 3)
    kt = k.permute(0, 2, 1, 3)
    vt = v.permute(0, 2, 1, 3)

    scale = float(qt.size(-1)) ** -0.5

    scores = torch.matmul(qt, kt.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, vt).permute(0, 2, 1, 3).contiguous()
    return out


def main():
    print("=== XFormers cutlassF fp32 smoke test ===")

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 2

    # cuda:N, N a die index from 0
    dev = torch.device("cuda:0")
    prop = torch.cuda.get_device_properties(dev)
    print(f"device       : {prop.name} (cc {prop.major}.{prop.minor})")

    import xformers
    import xformers.ops
    from xformers.ops.fmha import memory_efficient_attention_forward
    from xformers.ops.fmha.common import Inputs
    print(f"xformers     : {xformers.__version__}")
    print(f"location     : {xformers.__file__}")

    from xformers.ops.fmha.cutlass import FwOp as CutlassFwOp
    print(f"cutlass FwOp : CUDA_MINIMUM_COMPUTE_CAPABILITY = "
          f"{CutlassFwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY}")

    # whole numbers from 1, K up to 65536
    B, M, H, K = 1, 128, 8, 64
    print(f"shape        : B={B} M={M} H={H} K={K}, fp32")

    # an integer from -2**63 to 2**64 - 1
    torch.manual_seed(20260426)
    q = torch.randn(B, M, H, K, dtype=torch.float32, device=dev)
    k = torch.randn(B, M, H, K, dtype=torch.float32, device=dev)
    v = torch.randn(B, M, H, K, dtype=torch.float32, device=dev)

    out_ref = reference_attention(q, k, v)

    inp = Inputs(query=q, key=k, value=v, attn_bias=None, p=0.0, scale=None)
    supports_reason = CutlassFwOp.not_supported_reasons(inp)
    if supports_reason:
        print(f"FAIL: cutlass.FwOp does not support these inputs: {supports_reason}")
        return 3
    print(f"cutlass FwOp.supports(inputs) = True")

    try:
        out_xfm = memory_efficient_attention_forward(q, k, v, op=CutlassFwOp)
    except Exception as e:
        print(f"FAIL: memory_efficient_attention_forward raised: "
              f"{type(e).__name__}: {e}")
        return 3

    if out_xfm.shape != out_ref.shape:
        print(f"FAIL: shape mismatch — xformers {tuple(out_xfm.shape)} vs "
              f"reference {tuple(out_ref.shape)}")
        return 4

    abs_diff = (out_xfm - out_ref).abs()
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())

    denom = torch.maximum(out_xfm.abs(), out_ref.abs())
    valid = denom > 1e-3
    rel_diff = torch.where(valid, abs_diff / denom, torch.zeros_like(abs_diff))
    max_rel = float(rel_diff.max())

    # a relative error above 0
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

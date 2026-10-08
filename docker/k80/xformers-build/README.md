# XFormers sm_37 build (Story #33)

End-to-end build of XFormers v0.0.23 from source, against patched CUTLASS, targeting Tesla K80 (sm_37). Closes Phase 2 Story [#33][issue-33] of the K80 attention-port epic [#12][epic].

[issue-33]: https://github.com/dogkeeper886/vllm37/issues/33
[epic]: https://github.com/dogkeeper886/vllm37/issues/12

## What this proves

When the workflow runs green:

1. **NVCC + the K80 builder image's CUDA 11.4 toolchain accept** `TORCH_CUDA_ARCH_LIST=3.7` against XFormers v0.0.23 with our patches applied.
2. **The XFormers Python sm_37 generate_kernels patch** (Story [#31][s31]) actually causes the autogen to materialize sm_37 kernel instantiations, and **the CUTLASS sm_37 trait patch** (Story [#25][s25]) lets those kernels compile via CUTLASS's SIMT path (verified bit-exact by Phase 1 Story [#29][s29]).
3. **The CC gate patch** (Story [#32][s32]) lowers `AttentionOpBase.CUDA_MINIMUM_COMPUTE_CAPABILITY` from `(5, 0)` to `(3, 7)` so the dispatcher accepts K80 at runtime.
4. **The resulting xformers installs cleanly** into the K80 builder image's Python 3.10 + PyTorch environment (2.0.1 when written; 2.4.1 since #84).

[s25]: https://github.com/dogkeeper886/vllm37/issues/25
[s29]: https://github.com/dogkeeper886/vllm37/issues/29
[s31]: https://github.com/dogkeeper886/vllm37/issues/31
[s32]: https://github.com/dogkeeper886/vllm37/issues/32

## Files

| File | Purpose |
|---|---|
| `build.sh` | Clones XFormers v0.0.23 (with submodules), applies our XFormers + CUTLASS patches, builds + installs via `pip install`, verifies the import. |
| `test_cutlass_fp32.py` | Story #34 — runtime smoke test. Calls `xformers.ops.memory_efficient_attention` with fp32 inputs forced through `cutlass.FwOp` on K80, compares element-wise against a manual softmax(QK^T)V reference. |
| `README.md` | This file. |

## Running it

### In CI (recommended)

The `k80-xformers-build` workflow runs the full pipeline on the self-hosted K80 runner.

```bash
gh workflow run k80-xformers-build.yml --repo dogkeeper886/vllm37
```

**Expected runtime:** 30–60 minutes. XFormers v0.0.23's autogen produces a lot of kernels, and adding `sm_37` adds ~20% more compilation. NVCC compilation on the K80 runner's CPUs is the bottleneck (the build itself does not touch the GPU).

### Manually inside the K80 builder image

```bash
docker run --rm \
    -v "$(pwd)/docker/k80/xformers-build:/build:ro" \
    -v "$(pwd)/docker/k80/xformers-patches:/xfmr-patches:ro" \
    -v "$(pwd)/docker/k80/cutlass-patches:/cutlass-patches:ro" \
    vllm37-builder:latest \
    bash -c "PATCH_DIR=/xfmr-patches CUTLASS_PATCH_DIR=/cutlass-patches /build/build.sh"
```

GPU access is **not** required for the build itself — `pip install` only invokes nvcc, not the GPU.

### What success looks like

The verification step at the end of `build.sh`:

```
xformers version: 0.0.23
xformers location: /usr/local/lib/python3.10/site-packages/xformers/__init__.py
xformers._C imported OK from: ...
AttentionOpBase.CUDA_MINIMUM_COMPUTE_CAPABILITY = (3, 7)
cutlass.FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY = (3, 7)
RESULT: PASS — xformers built and imports cleanly
```

## What this does NOT do

- **Does not integrate with vLLM's main build.** Story [#41][s41] (Phase 4) wires XFormers into vLLM's backend dispatcher.
- **Does not benchmark.** Story 5.x.
- **Does not run XFormers' full pytest suite.** The smoke test (`test_cutlass_fp32.py`, Story #34) is a hand-rolled minimal proof. Running `xformers/tests/test_mem_eff_attention.py` end-to-end is plausible additional coverage but typically reports many skips for fp16/bf16/sm_70+ paths that don't apply on K80; treat as informational if it ever lands.

[s34]: https://github.com/dogkeeper886/vllm37/issues/34
[s41]: https://github.com/dogkeeper886/vllm37/issues/41

## Hardware safety notes

- The build itself is **CPU-only** — nvcc compilation. No GPU access needed during `pip install`.
- The verification step does `import xformers._C` which loads the .so but doesn't launch any kernel. Memory-safe, no compute, no thermal load.
- The whole flow is "always fine" timing per the saved hardware-risky-ops rule.

## Known caveats

- **XFormers v0.0.23 predates PyTorch 2.4.** v0.0.23's `requirements.txt` says `torch >= 1.12`. It builds against PyTorch 2.4.1 and passes the GPU kernel test (#84); if a later PyTorch breaks an import inside XFormers, that is a downstream patch to add.
- **XFormers v0.0.23 was the last tag with the cutlass mem-eff backend in-tree AND torch>=1.12.** See [`xformers-patches/README.md`](../xformers-patches/README.md) for the full pin rationale.
- **`XFORMERS_DISABLE_TRITON=1`** is set during build to skip the Triton-based ops, which require sm_70+ for tensor cores anyway. Triton-on-Kepler is out of scope for this fork.

## Related

- [`docker/k80/xformers-patches/`](../xformers-patches/) — the XFormers source patches this build script applies
- [`docker/k80/cutlass-patches/`](../cutlass-patches/) — the CUTLASS patches applied to the bundled submodule
- [`docker/k80/cutlass-repro/`](../cutlass-repro/) — Phase 1's standalone CUTLASS GEMM verifier (bit-exact PASS on K80)

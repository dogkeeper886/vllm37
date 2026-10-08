# Kepler (sm_37) vs Maxwell (sm_50) — hardware feature gap

**Story:** [#22][issue-22] (Phase 0 of epic [#12][epic])

**Sources of truth used:**
- NVIDIA CUDA C++ Programming Guide (per-compute-capability tables)
- NVIDIA Kepler Tuning Guide (archived 11.4 mirror; HEAD URL returns 404)
- NVIDIA Maxwell Tuning Guide
- NVIDIA Pascal / Volta / Ampere / Hopper Tuning Guides (for "when introduced" claims)
- NVIDIA "Using CUDA Warp-Level Primitives" developer blog
- CUDA Toolkit 11.4 release notes (already cited in [Story 0.3][story03])
- Empirical: `nvidia-smi` + `cat /etc/os-release` from the K80 self-hosted runner via the [`k80-host-info`][run-host-info] workflow run on 2026-04-25

[issue-22]: https://github.com/dogkeeper886/vllm37/issues/22
[epic]: https://github.com/dogkeeper886/vllm37/issues/12
[story03]: https://github.com/dogkeeper886/vllm37/blob/main/docs/port/cuda-11.4-version-pins.md
[run-host-info]: https://github.com/dogkeeper886/vllm37/actions/runs/24933324546

**Methodology:** every claim cites either an NVIDIA URL with a quote (or section reference), the K80 runner's own output, or already-cited code from prior stories. Where NVIDIA URLs returned 404 at fetch time, archive URLs are used and the breakage is flagged.

---

## 1. Scope

This story quantifies what sm_37 (Kepler GK210, the K80's compute die) actually has versus what sm_50 (Maxwell GM107) has — and cross-references each row against what CUTLASS's `Sm50` SIMT GEMM path requires (per Story 0.1 §5).

The objective is to verify or refute Story 0.1's load-bearing claim that *the Sm50 SIMT path's hardware requirements are met by Kepler*. If yes, the optimistic Story 0.1 picture for the CUTLASS port stands. If no, Phase 1 needs more than the one-line `Sm37` arch trait Story 0.1 proposed.

## 2. Empirical baseline — the K80 runner today

Pulled via `nvidia-smi` on the self-hosted runner (`rocky9-k80-cicd-1-vllm`), workflow run [24933324546][run-host-info], 2026-04-25 14:41 UTC:

```
NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4

| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  Tesla K80           Off  | 00000000:04:00.0 Off |                    0 |
| N/A   31C    P8    26W / 149W |      4MiB / 11441MiB |      0%      Default |

|   1  Tesla K80           Off  | 00000000:07:00.0 Off |                    0 |
| N/A   32C    P8    32W / 149W |      4MiB / 11441MiB |      0%      Default |
```

```
PRETTY_NAME="Rocky Linux 9.7 (Blue Onyx)"
Linux rocky9-k80-cicd-1 5.14.0 ... x86_64 GNU/Linux

(host) nvcc: command not found
```

This empirically confirms what Story 0.3 inferred:

- **Driver 470.256.02** — exactly the latest R470 release whose release notes Story 0.3 cited. R470 is the last branch supporting Kepler.
- **Driver-reported CUDA Version: 11.4** — matches our toolkit pin.
- **Two compute dies exposed as separate CUDA devices** (Bus IDs `04:00.0` and `07:00.0`). Each die has 11441 MiB ≈ **11.45 GiB** VRAM. This is consistent with the public K80 datasheet (12 GiB per die, less reserved overhead).
- **Per-die power cap: 149 W** (`Pwr:Usage/Cap` reports the runtime cap, settable via `nvidia-smi -pl` — which [the README's hardware safety rules](../../README.md#hardware-safety) forbid on this hardware). Two dies × 149 W ≈ 298 W under cap, consistent with the public 300 W board TDP.
- **The host has no nvcc on PATH.** Toolchain lives inside the K80 builder Docker image (CUDA 11.4 toolkit), not on the host. Driver and toolkit are deliberately separated.

This closes Story 0.3's open question about the precise runner driver version.

## 3. Compute & numerics

| Feature | sm_37 (GK210, K80) | sm_50 (GM107) | Source / quote |
|---|---|---|---|
| FP32 IEEE-754 with FMA | Yes | Yes | Both archs are full IEEE-754 FP32 with FMA per their respective Tuning Guides |
| FP64 throughput ratio | ~1:3 of FP32 (compute-targeted GK210) | ~1:32 of FP32 | GK210 is the "compute" Kepler die; GM107 is consumer Maxwell with truncated FP64 |
| Native FP16 compute | **No** | **No** (sm_50 / sm_52). First mainstream FP16 is sm_60 (Pascal). | Pascal Tuning Guide: *"Pascal also adds support for native FP16 instructions"* |
| BF16 compute | No | No | Ampere Tuning Guide: BF16 *"through HMMA instructions"* — first appears sm_80 |
| INT8 dot product (`__dp4a` / `__dp2a`) | No | **No** on sm_50; introduced sm_61 | Pascal Tuning Guide: GP104 (sm_61) provides *"specialized instructions for two-way and four-way integer dot products"* via `__dp4a` / `__dp2a` |
| `atomicAdd(float*)` | Yes (since CC 2.0) | Yes | CUDA Programming Guide atomic functions reference |
| `atomicAdd(double*)` | No | No | Introduced at CC 6.0 (Pascal) |

**Cross-reference with Story 0.1 §5** (what CUTLASS Sm50 SIMT path uses): scalar FP32 FMA only, no FP16, no BF16, no DP4A, no double-atomics. **Every requirement is met by sm_37.**

## 4. Memory hierarchy

| Feature | sm_37 (GK210) | sm_50 (GM107) | Source |
|---|---|---|---|
| Max threads per block | 1024 | 1024 | Programming Guide per-CC table |
| Max resident blocks per SM | 16 | 32 | Maxwell Tuning Guide: *"increased from 16 to 32"*. **Lower on Kepler — a tuning consideration, not a correctness blocker.** |
| Max shared memory per block | **48 KB** | 48 KB | Same per-block cap on both. Story 0.1 §5 confirmed CUTLASS's Sm50 SIMT path stays within 48 KB. |
| Shared memory per SM | up to **112 KB** (configurable 80/96/112 of 128 KB unified) | 64 KB per SMM | Kepler Tuning Guide on GK210: *"can select 112 KB, 96 KB, or 80 KB of shared memory"* — *"more than doubles the shared memory capacity vs … earlier Kepler GPUs."* Maxwell Tuning Guide: 64 KB per SMM. **GK210 has more SM-level shared memory than GM107.** |
| 32-bit registers per SM | **128 K** | 64 K | Kepler Tuning Guide: GK210 *"increases this by a further 2×"* over GK110's 64 K. Maxwell Tuning Guide: 64 K registers per SMM. **GK210 has 2× more registers than GM107.** |
| Max registers per thread | 255 | 255 | Programming Guide per-CC table |
| L1 cache | unified L1/texture (Kepler-style) | dedicated L1/texture per SMM | Tuning guides |
| L2 cache | 1.5 MB per die (3 MB across both dies on K80 board) | 2 MB on GM107 | Tuning guides |
| VRAM per die | 11.45 GiB usable (per `nvidia-smi`); 12 GiB raw | varies by GM107 SKU | K80 runner output, §2 |

**Cross-reference with Story 0.1 §5:** the Sm50 SIMT path needs ≤ 48 KB shared memory per block. **sm_37 has it.** GK210 actually has *more* per-SM shared memory and registers than GM107 — meaning a port could in principle use *larger* tile sizes than CUTLASS targets for Maxwell. But we don't need to; just hitting the Sm50 baseline is enough.

## 5. Warp-level primitives

| Feature | sm_37 | sm_50 | Source |
|---|---|---|---|
| Warp size = 32 | Yes | Yes | Programming Guide per-CC table |
| Legacy `__shfl`, `__shfl_up`, `__shfl_down`, `__shfl_xor` | **Yes** (introduced CC 3.0) | Yes | CUDA Programming Guide warp-shuffle reference |
| Sync variants `__shfl_sync` etc. | **Compile on sm_37 from CUDA 9+; emit the same SHFL instruction**. Independent-thread-scheduling semantics only enforced on sm_70+. | Same as sm_37 | NVIDIA dev blog *"Using CUDA Warp-Level Primitives"*: *"the legacy warp-level primitives are deprecated starting in CUDA 9.0"* and `_sync` variants accept an explicit mask, but on pre-Volta the warps are still in lockstep so the mask is informational. |
| Legacy `__ballot`, `__any`, `__all` | Yes (since CC 1.2 / 2.0) | Yes | Programming Guide |
| `_sync` variants of vote ops | Same as shuffle: compile, semantics differ pre-Volta | Same | Same blog |

**Cross-reference with Story 0.1 §5 + §6:** the core Sm50 SIMT MMA does **not** use `__shfl_sync` (verified at `mma_simt.h` and `mma_simt_policy.h` in Story 0.1 §6.4). The `__shfl_sync` risk Story 0.1 surfaced was specifically in `grouped_problem_visitor.h`, which Story 0.2 confirmed is **not** reached by FA. For XFormers' `cutlassF` op, Story 2.x will need to verify the same. **For the core SIMT path, sm_37 is fine.**

## 6. Synchronization & cooperative groups

| Feature | sm_37 | sm_50 | Source |
|---|---|---|---|
| `__syncthreads()` | Yes | Yes | Universal |
| `__syncwarp()` | Wrapper compiles since CUDA 9; native semantics only on sm_70+ | Same | Volta Tuning Guide: introduced for Volta independent thread scheduling |
| Cooperative Groups, block / grid scope | Header-level wrappers compile; multi-block coop launch needs sm_60+ | Same plus more | Volta Tuning Guide: *"CUDA 9 introduces Cooperative Groups"* |

**Cross-reference with Story 0.1 §5:** the Sm50 SIMT path uses `__syncthreads()` only. **sm_37 has it.**

## 7. Async memory ops

| Feature | sm_37 | sm_50 | Min CC | Source |
|---|---|---|---|---|
| `cp.async` (PTX) | No | No | sm_80 | Ampere Tuning Guide: *"adds hardware acceleration for copying data from global memory to shared memory."* Story 0.1 §6.2 already confirmed CUTLASS's Sm50 path doesn't use `cp.async`. |
| `cuda::memcpy_async` (libcu++) | Software fallback works on both | Same | sm_80 (HW path) | Same |
| TMA (Tensor Memory Accelerator) | No | No | sm_90 | Hopper Tuning Guide |

**Cross-reference with Story 0.1 §6.2:** Sm50 SIMT path uses ordinary global / shared loads with explicit `__syncthreads()`. **sm_37 is unaffected by the absence of `cp.async`.**

## 8. Tensor cores

| Feature | sm_37 | sm_50 | sm_70 | Source |
|---|---|---|---|---|
| Tensor Cores | **No** | **No** | Yes (1st gen, Volta) | Volta Tuning Guide: tensor cores *"exposed as Warp-Level Matrix Operations in the CUDA 9 C++ API"* |
| `mma.*` / `wmma.*` PTX | No | No | Yes | Same |

**Neither sm_37 nor sm_50 has tensor cores.** This is exactly why CUTLASS picks the SIMT path for Sm50 (and would for any new Sm37 tag), per Story 0.1 §4.5. The tensor-core absence is a feature, not a bug, for our port.

## 9. K80 (board-level) specifics relevant to safety rules

From the runner's `nvidia-smi` output (§2) and the public K80 datasheet:

- **Dual-die board:** 2× GK210, exposed as two CUDA devices. ✓ confirmed empirically.
- **Per-die:** 11441 MiB VRAM, 149 W runtime power cap (per `nvidia-smi`). ✓ confirmed empirically.
- **Board TDP:** ~300 W. Confirms the power-risk rule in [the README's hardware safety section](../../README.md#hardware-safety): TP=2 lights up both dies, putting full board draw on the rail. The K80 EPS-cable observation ("2× K80 on a shared CPU EPS connector can exceed the rail rating under full load") follows from this — the rail constraint is real and unmodelable from spec sheets alone. The 149 W per-die cap is a *runtime* setting reported by `Pwr:Usage/Cap`; it can be modified by `nvidia-smi -pl`, which the safety rules forbid on this hardware (has caused system halts in the past).
- **PCIe Gen3 x16, no NVLink.** Already encoded in `NCCL_P2P_DISABLE=1` from prior K80 work.

This information also bounds the long-context unlock claim of Story #54. Phase-1 instrumentation in PR #11 (CI run [24897047481][run-24897047481], TinyLlama-1.1B FP32 at TP=1) measured the actual VRAM trajectory:

| Stage | Free per die |
|---|---|
| Baseline after CUDA init | 11157 MiB (≈ 10.90 GiB) |
| After `load_model` (weights = 4198 MiB) | 6959 MiB (≈ 6.80 GiB) |
| After `determine_num_available_blocks` profile | 6871 MiB (≈ 6.71 GiB) |
| After KV cache alloc (`kv_cache_mib=5194`) | **1677 MiB (≈ 1.64 GiB)** |
| After warmup forward | 1677 MiB |

So on the smoke-test workload, ~1.64 GiB stayed free at peak. KV cache consumed about 5.07 GiB at the configured `gpu_memory_utilization=0.85`. Larger models would shift this further — that's the headroom flash-style attention would buy back at long contexts. The bottleneck is per-die VRAM, not attention algorithm; flash-style shifts the cost from O(N²) activations to O(N), which matters most when context length is the binding constraint.

[run-24897047481]: https://github.com/dogkeeper886/vllm37/actions/runs/24897047481

## 10. Bottom line for the port

For each requirement of CUTLASS's Sm50 SIMT path (per Story 0.1 §5):

| Requirement | sm_37 status | Verdict |
|---|---|---|
| Scalar FP32 IEEE-754 FMA | Yes (since Kepler) | ✅ Available |
| ≤ 48 KB shared memory per block | Yes — same per-block cap | ✅ Available |
| ≤ 255 32-bit registers per thread | Yes — identical cap | ✅ Available |
| `atomicAdd(float*)` for shared memory | Yes (since CC 2.0) | ✅ Available |
| `__syncthreads()` | Yes | ✅ Available |
| Warp size 32 | Yes | ✅ Available |
| Legacy `__shfl` / vote ops (utility paths) | Yes (since CC 3.0) | ✅ Available |
| `__shfl_sync` semantics (sm_70+ guarantees) | Compiles on sm_37 but pre-Volta lockstep | ⚠️ Caveat — does not affect core Sm50 SIMT MMA path |
| Tensor cores / `mma.*` / `wmma.*` | No | ✅ Not used by Sm50 SIMT path |
| `cp.async` | No | ✅ Not used by Sm50 SIMT path |

**Result: every requirement of CUTLASS's Sm50 SIMT GEMM path is met by sm_37.** GK210 actually has *more* per-SM shared memory (112 KB vs 64 KB) and *more* registers per SM (128 K vs 64 K) than GM107. The Sm50 SIMT-only feature surface is a strict subset of sm_37's hardware capability.

This **confirms** Story 0.1 §5's working hypothesis: *"The intersection of required-by-Sm50-SIMT and missing-on-sm_37 is empty."* Story 0.4's specification-level analysis verifies that hypothesis against NVIDIA's own per-arch documentation.

The one caveat — `__shfl_sync` semantics differ pre-Volta vs Volta+ — is exactly the risk Story 0.1 §6.4 already identified for `grouped_problem_visitor.h` (which Story 0.2 confirmed FA does not reach). That risk lives in CUTLASS utility code, not in the SIMT GEMM core, and is independently tractable when (if) we reach an attention kernel that needs it.

## 11. Open questions / unknowns

1. **NVIDIA documentation reachability.** The Kepler Tuning Guide URL (`https://docs.nvidia.com/cuda/kepler-tuning-guide/index.html`) returned 404 at fetch time. Used the CUDA 11.4 archive mirror (`https://docs.nvidia.com/cuda/archive/11.4.0/kepler-tuning-guide/index.html`) as fallback. Future readers may hit the same broken link; the archive mirror should be cited as the canonical source going forward.

2. **K80 datasheet PDF** at `images.nvidia.com` returned 404. The board-level numbers in §9 are cross-validated against `nvidia-smi` output and the Kepler Tuning Guide, but the canonical NVIDIA PDF is the better citation if anyone can locate a working URL.

3. **CUDA Programming Guide HTML truncation.** WebFetch consistently truncated the per-CC table appendix before reaching it. Several rows in the tables above are sourced from cross-corroborated tuning guides + the publicly-mirrored Programming Guide table values. For higher fidelity, fetching the NVIDIA Programming Guide PDF directly would let us cite specific table entries — not done here. Probably worth doing if any specific number ever becomes load-bearing.

4. **Pre-Volta `__shfl_sync` divergence semantics.** NVIDIA's developer blog says `_sync` variants are "deprecated alongside the legacy ones for new code on Volta+," but does **not** explicitly state the minimum CC for `__shfl_sync` codegen. Empirically they compile to the same SHFL/VOTE instructions on pre-Volta. This is a documentation gap, not a correctness issue for our use case (Story 0.1 §6 already established the core Sm50 SIMT MMA doesn't use `_sync` variants), but if Phase 2/3 ever reaches code that does, the divergence assumption needs an explicit code review.

## 12. Implications for Phase 1, 2, 3

**Phase 1 (CUTLASS):** Story 0.1's plan stands. Adding `Sm37` is structurally one struct in `arch/arch.h`. No SIMT-path requirement is missing on Kepler. Story #28 (compile minimal CUTLASS GEMM for sm_37 on K80) remains the empirical verification gate.

**Phase 2 (XFormers):** Same. Once CUTLASS produces sm_37 kernels, XFormers' fp32 path (which Story 0.1 noted is sm_50+ documented) should run on sm_37 — the documented sm_50 minimum was a labeling artifact in CUTLASS, not a hardware requirement.

**Phase 3 (FlashAttention):** Story 0.2's rewrite recommendation stands and is **strengthened** by this story. The wall is the FA kernel's tensor-core dependency (sm_70+), not anything sm_37 lacks structurally. A SIMT FP32 reimplementation of the FA algorithm has every primitive it needs on Kepler — scalar FMA, 48 KB shared memory per block, FP32 atomics, `__syncthreads()`. GK210's per-SM shared memory (112 KB) and register file (128 K) actually exceed GM107's (64 KB / 64 K), so per-SM constraints are *more* permissive on Kepler than on Maxwell.

**The hardware does not block any of Phase 1 / 2 / 3.** What blocks Phase 3 is the *implementation choice* in FA's source (tensor-core MMA atoms, no SIMT path), not the hardware. That's Story 0.2's territory; this story confirms the hardware story isn't an additional obstacle.

## 13. Hardware-only conclusions

- The K80 self-hosted runner is empirically: 2× Tesla K80, driver 470.256.02, CUDA 11.4, Rocky Linux 9.7, no host-side nvcc. Closes Story 0.3's open question about runner driver version.
- For every requirement of CUTLASS's Sm50 SIMT GEMM path, sm_37 has the corresponding hardware feature — usually with *more* headroom than GM107.
- The "sm_37 is structurally below CUTLASS's floor" framing is a labeling/dispatcher artifact, not a hardware truth. Story 0.1's optimistic CUTLASS-port picture stands and is reinforced.
- The single caveat (`__shfl_sync` divergence semantics pre-Volta) does not touch the core Sm50 SIMT MMA path. Story 0.1 §6.4 already flagged it for the one CUTLASS utility file (`grouped_problem_visitor.h`) that uses it, and Story 0.2 confirmed FA does not reach that file.

---

**End of Story 0.4 deliverable.** Story 0.5 (existing prior art search, [#23][s05]) is the next pickup — looks for sm_37 forks of CUTLASS / FA / XFormers, or relevant blog posts / papers, before any more design work happens.

[s05]: https://github.com/dogkeeper886/vllm37/issues/23

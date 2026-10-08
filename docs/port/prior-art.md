# Existing prior art on Kepler / sm_37 LLM inference

**Story:** [#23][issue-23] (Phase 0 of epic [#12][epic])
**Methodology:** every project / paper / post cited by URL. Negative results are documented explicitly so future researchers don't repeat the same searches expecting different results.

[issue-23]: https://github.com/dogkeeper886/vllm37/issues/23
[epic]: https://github.com/dogkeeper886/vllm37/issues/12

---

## 1. Scope

This story answers Story 0.5's question: has anyone already done the work we're about to do? The deliverable is a directory of every relevant project, plus an honest summary of which negative results are real (vs. an artifact of bad search queries).

Findings cluster into:

- **Tier 1: direct prior art** for the three target libraries (CUTLASS / FlashAttention / XFormers) and vLLM
- **Tier 2: adjacent prior art** — different stacks (llama.cpp, Ollama) that solved related problems on K80
- **Tier 3: academic / research** — papers and blog posts

## 2. Tier 1: Direct prior art (CUTLASS / FA / XFormers / vLLM)

### 2.1 CUTLASS — no public Sm37 fork exists

Searched:
- `gh search prs --repo NVIDIA/cutlass "Sm37"` → 0 hits
- `gh search prs --repo NVIDIA/cutlass "Kepler"` → 0 hits
- `gh search issues --repo NVIDIA/cutlass "Sm37"` → 0 hits
- `gh search issues --repo NVIDIA/cutlass "Kepler"` → 1 hit ([issue #342](https://github.com/NVIDIA/cutlass/issues/342)), unrelated MAGMA question

**Result: no public CUTLASS fork adds an `arch::Sm37` struct.** Story 0.1's "structurally trivial one-line addition" claim survives because nobody has needed it before — CUTLASS users with sm < 50 hardware have either kept legacy CUTLASS pre-3.0 or moved to a different library entirely.

This is greenfield work. We are first.

### 2.2 FlashAttention — no Kepler/sm_37 forks; even Turing remains unsupported upstream

Searched:
- Forks of `Dao-AILab/flash-attention` filtered by Kepler/K80/sm_37/sm_35 keywords → 0 matches
- `gh search code "sm_37" --owner Dao-AILab` → 0 hits
- The closest existing arch-named fork via the public forks listing is `farnghwai/flash-attention-2080ti` (Turing sm_75) — not relevant to us

Open upstream issues that confirm the gap:
- [Dao-AILab/flash-attention#1342](https://github.com/Dao-AILab/flash-attention/issues/1342) — "Assistance on implementing Flash Attention 2 for Turing." Still OPEN.
- [Dao-AILab/flash-attention#1608](https://github.com/Dao-AILab/flash-attention/issues/1608) — same theme.

**The only sm < 80 FA work in the public ecosystem is** [`ssiu/flash-attention-turing`](https://github.com/ssiu/flash-attention-turing):

- What it actually does: hand-rolled FA-2 forward+backward kernels for Turing sm_75. Supports head-dim 64/128, causal, GQA, varlen. No dropout, no KV cache.
- Reusable for sm_37? **Indirectly.** It targets sm_75, which has tensor cores. Its core MMA strategy will not transplant to Kepler. But the *scaffolding* (varlen layout, causal masking, online-softmax statistic chaining) is informative reference material for our Story 3.2b rewrite.
- Status: active (last commit 2026-03-23, 92 stars, 10 forks).
- License: not declared in repo metadata. Treat as all-rights-reserved until confirmed; don't copy code without resolving this.

### 2.3 XFormers — no Kepler forks

Searched:
- `gh search repos "xformers Kepler"` → 0 hits
- `gh search repos "xformers K80"` → 0 hits
- `gh search repos "xformers sm_37"` → 0 hits

XFormers' own [`README.md`](https://github.com/facebookresearch/xformers/blob/main/README.md) recommends `TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"` — sm < 60 is not even in the recommended set, let alone supported.

**No xformers Kepler fork exists.**

### 2.4 vLLM siblings

This fork (`dogkeeper886/vllm37`) appears to be the only public **Kepler** vLLM fork. Searched:

- `gh search repos vllm-kepler` / `vllm-K80` / `vllm-sm37` / `vllm-sm_37` / `vllm-pre-Volta` → 0 hits
- `gh search code "compute_37" --owner vllm-project` → 0 hits

There is one **Pascal** sibling worth referencing:

- [`ampir-nn/vllm-pascal`](https://github.com/ampir-nn/vllm-pascal) — Pascal (sm_60/61) wheels for vLLM 0.10–0.13 against CUDA 12.6. MIT, last push 2026-01-03.
  - Reusable for us? **Indirectly.** They build against a much newer toolchain than we can (CUDA 12.6 vs our pinned 11.4). The lesson is the **packaging/distribution model**: custom torch wheel + custom triton wheel + prebuilt distribution. Our K80 builder image already follows a similar pattern; cross-pollinating Dockerfiles or CI structure may be valuable.

## 3. Tier 2: Adjacent prior art

### 3.1 fcheung2888/llama.cpp-kepler — direct, ~80% transferable

URL: https://github.com/fcheung2888/llama.cpp-kepler

What it does: patch set + Python applier (`kepler-patch-all.py`) that makes upstream llama.cpp build for **sm_30** (Kepler GK104 / K2000M class) under CUDA 10.2 + GCC 8 + driver 470. Verified by inspecting the script: **10 distinct files patched** across the ggml-cuda tree and build system —

- 8 ggml-cuda source files: `binbcast.cu`, `common.cuh`, `fattn-common.cuh`, `fattn-vec.cuh`, `ggml-cuda.cu`, `softmax.cu`, `vendors/cuda.h`, plus `ggml/src/ggml-cuda/CMakeLists.txt`
- 2 build-system files: `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`

Patch types (via 27 `patch_file` / `patch_file_regex` invocations across 12 logical sections in the script):

- Shims `__shfl_sync` → `__shfl` (matches Story 0.4's notes about Volta+ semantics not being needed)
- Stubs bf16 types (Story 0.4 confirmed Kepler has no native bf16)
- Polyfills `std::is_same_v` and other C++17 features back to C++14
- Rewrites C++17 fold expressions and structured bindings
- Guards `cooperative_groups/reduce.h` and `cg::this_grid()`
- **Disables `GGML_CUDA_FA` (Flash Attention) entirely** — sm_30 lacks FP16 / tensor cores, so they fall back to the standard attention path

**Reusable for sm_37? Directly.** sm_37 is sm_30's bigger cousin (same Kepler family, GK210 vs GK1xx). Almost every patch is exactly what we need; only the disabled-bf16 stub may be unnecessary on sm_37 if we ever want to use FP32 throughout (which we do; see the README). The "disables FA" note is the **practical baseline** — it confirms what every other Kepler user has done is run without flash attention. (At the time of writing our K80 fork used Torch SDPA only; it now uses patched xformers `cutlassF`.)

Status: active, last push 2026-03-29. License: not declared (upstream llama.cpp is MIT — derivative).

### 3.2 The user's own related project — dogkeeper886/ollama37

URL: https://github.com/dogkeeper886/ollama37

This is the user's own K80-targeting Ollama fork (mentioned in study plan issue [#2](https://github.com/dogkeeper886/vllm37/issues/2)). Not external prior art, but listed here for completeness because:

- It already pins **the same toolchain we use** (CUDA 11.4 / GCC 10 / driver 470, on Rocky Linux). The K80 builder image in `docker/k80/` here is direct kin.
- It already has a working CI flow on K80 hardware (the same self-hosted runner backing this fork).
- The "dual-judge" eval framework referenced in study plan #2 lives in this repo and is the upstream of what Phase 5 Story #51 (numerical quality) will reuse.

Status: active, 18 stars, 3 forks, last push 2026-04-18. Forked from `ollama/ollama`.

There is also a `EnlistedGhost/ollama37` (created 2026-04-10, identical description but no GitHub fork-relationship metadata, 0 stars / 0 forks). Relationship to `dogkeeper886/ollama37` is unclear from metadata alone — possibly an independent re-creation, possibly a copy with the remote renamed. **Not treating as a separate prior-art source.**

### 3.3 neverclover/ollama-on-k80

URL: https://github.com/neverclover/ollama-on-k80

What it does: a smaller Ollama K80 fork (Ollama v0.13.5 patch set, CUDA 11). README is minimal. 6 stars.

Reusable? **Indirect.** Overlaps with the user's own `ollama37` repo. Useful as a second confirmation that "K80 + Ollama" is a tractable problem in the wild.

Status: active, last push 2025-12-31. License: not stated.

### 3.4 cmarshall108/neuronet — different shape, not a port target

URL: https://github.com/cmarshall108/neuronet

What it does: a from-scratch C++ tensor library with a CUDA backend "optimized for older NVIDIA GPUs (Tesla K80)." PyTorch-shaped API but is its own implementation, not a port.

Reusable? **No, but informative.** The cost of switching to a different framework is higher than fixing vLLM. The existence is, however, a useful confirmation that "FP32-only K80 still fits an LLM-shaped workload" — someone built one, it works.

Status: last push 2025-03-10. License: not declared.

### 3.5 systemSpiderm/gemm-cuda — usable reference SIMT kernel

URL: https://github.com/systemSpiderm/gemm-cuda

What it does: hand-tuned shared-memory GEMM for K80 (FP64). 32×32 block tiling, beats cuBLAS on small matrices (<4096²). Educational repo, Chinese-language README.

Reusable? **Indirect.** Confirms Story 0.4's working hypothesis (48 KB SMEM budget is enough for SIMT GEMM on K80; tile sizes that fit). Worth reading before writing our own Phase 1 Story #28 reproducer kernel. Not a drop-in.

Status: last push 2025-04-06. License: not declared.

### 3.6 PyTorch-for-Kepler community wheels

Three wheel-distribution projects exist:

| Repo | Targets | Toolchain | Last push | License |
|---|---|---|---|---|
| [`xiaoran007/Pytorch-for-Kepler`](https://github.com/xiaoran007/Pytorch-for-Kepler) | sm_35 + **sm_37** | varies, up to torch 2.5 | 2025-04-02 | BSD-3 |
| [`neverclover/PyTorch2.x-wheels-for-k80`](https://github.com/neverclover/PyTorch2.x-wheels-for-k80) | sm_37 | torch 2.x + CUDA 11.8 + Python 3.12 | 2025-12 | not stated |
| [`jeremistderechte/PyTorch-Kepler-Wheels`](https://github.com/jeremistderechte/PyTorch-Kepler-Wheels) | sm_35 only (K20/K40, **not K80**) | older | 2024-09 | none |

Reusable for us? **Indirect.** Our K80 fork is pinned to **torch 2.0.1 + CUDA 11.4** (`docker/k80/constraints.txt`). These projects build against newer toolchains, so the wheels themselves are not drop-in. But:

- `xiaoran007/Pytorch-for-Kepler` is the most useful playbook (BSD-3 licensed; sm_37 explicitly supported; documented build process).
- They demonstrate that newer torch *can* be coaxed onto K80 — useful if we ever want to upgrade the torch pin within the CUDA 11.4 / driver R470 envelope.

### 3.7 llama.cpp upstream issue/PR landscape

Not a fork, but the upstream project's issue tracker contains real K80 datapoints:

- [`ggml-org/llama.cpp#12140`](https://github.com/ggml-org/llama.cpp/issues/12140) — "Feature Request: Enable cuda 11.4 and cuda arch 3.7." Closed stale 2025-04-23 without merge. The thread documents a successful build at `-DCMAKE_CUDA_ARCHITECTURES='52;61;70;75;37'` with CUDA 11.4 + GCC 10 + driver 470 (matches our toolchain) and reports ~5–6 T/s on a single K80 die for DeepSeek-R1-Distill-Qwen-7B-F16. **This sets a throughput envelope expectation for our fork.**
- [`ggml-org/llama.cpp#18743`](https://github.com/ggml-org/llama.cpp/issues/18743) — "no kernel image is available for execution on the device" for sm_35 K40 on a Maxwell-also build. **Failure mode our smoke test must catch.** Build success ≠ runtime success.
- [`LostRuins/koboldcpp#1409`](https://github.com/LostRuins/koboldcpp/issues/1409) — same K80 "no kernel image" failure (closed 2025-05-24). Confirms the symptom is general, not specific to one project.

No upstream llama.cpp **PR** has merged sm_37 support; all known support is via downstream patches (§3.1 above).

### 3.8 bitsandbytes / GPTQ / AWQ on Kepler — confirmed gap

[`bitsandbytes-foundation/bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) officially requires sm_60+ in recent releases. Searched: no public backport to sm_37/sm_50 found. No ready-made Kepler kernel exists. `docs/port/model-family-requirements.md` Part 9 later showed weight-only int4 is feasible with a new FP32 kernel (#73, #74).

## 4. Tier 3: Academic / research

### 4.1 SparkAttention (Volta-specific)

[arXiv 2502.12784](https://arxiv.org/abs/2502.12784) — *"SparkAttention: High-Performance Multi-Head Attention for Large Models on Volta GPU Architecture"* (submitted Feb 2025; published in *CCF Transactions on High Performance Computing*, 2025). Volta-specific (sm_70), uses Tensor Core Units. Does not discuss Kepler. Verified directly via WebFetch of the arXiv abstract page.

Why it matters to us: the paper exists *because* even Volta hit the upstream-FA gap. The structural argument ("modern attention libraries ignore older architectures, custom rewrite needed") parallels our case. **The kernels themselves don't transfer (Volta tensor cores are required), but the existence of two precedent rewrites — Volta (SparkAttention) and Turing (ssiu/flash-attention-turing) — triple-validates Story 0.2's "rewrite, not port" recommendation when extended to Kepler.**

### 4.2 Other arXiv searches

Searched: "Kepler attention kernel", "K80 transformer inference", "pre-Volta SIMT attention", "sm_37 GEMM optimization."

No paper found that targets Kepler / sm_37 specifically for transformer / attention inference. The "Transformer Based Linear Attention with Optimized GPU Kernel Implementation" ([arXiv 2510.21956](https://arxiv.org/abs/2510.21956)) targets modern GPUs only.

### 4.3 NVIDIA developer blog

`developer.nvidia.com/blog/?s=kepler` — no posts about modern ML on Kepler. Last Kepler-tagged content predates the transformer era.

NVIDIA forum reference: [forum thread on K80 + R470 driver lifecycle](https://forums.developer.nvidia.com/t/tesla-k80-and-discontinued-driver-support-for-kepler/193903) — confirms Story 0.3's R470 conclusion. CUDA 11.x is the last toolkit; math libs (cuBLAS, cuDNN) stop shipping for sm_30/32 at CUDA 11.5 — sm_37 happens to survive longer than its sm_30 siblings within CUDA 11.x.

## 5. Negative results (search terms that returned nothing useful)

Don't repeat these searches expecting new results — they reflect a structural absence, not stale indexing:

| Query | Result |
|---|---|
| `gh search repos "flash-attention kepler"` / `"flash-attention K80"` / `"flash-attention sm_37"` | 0 hits |
| `gh search repos "cutlass Kepler"` / `"cutlass sm_37"` / `"cutlass Sm37"` | 0 hits |
| `gh search repos "xformers Kepler"` / `"xformers K80"` | 0 hits |
| `gh search repos "vllm Kepler"` / `"vllm K80"` / `"vllm sm_37"` / `"vllm sm37"` / `"vllm pre-Volta"` | 0 hits (only `ampir-nn/vllm-pascal` sibling) |
| `gh search prs --repo NVIDIA/cutlass "Sm37"` / `"Kepler"` | 0 hits |
| `gh search code "sm_37" --owner Dao-AILab` | 0 hits |
| `gh search code "compute_37" --owner vllm-project` | 0 hits |
| `gh search repos "Pascal flash-attention"` / `"Maxwell flash-attention"` / `"Volta flash-attention"` | 0 hits |
| arXiv: "Kepler attention kernel" / "sm_37 attention" | 0 directly relevant papers |
| NVIDIA dev blog Kepler search | 0 ML-relevant posts |

## 6. Bottom-line table

| Source | Tier | Status | Reusable for our port? | Notes |
|---|---|---|---|---|
| ssiu/flash-attention-turing | 1 | Active 2026-03 | Indirect (sm_75, tensor cores) | Scaffolding reference for Story 3.2b |
| Dao-AILab/flash-attention forks | 1 | None Kepler | No | Issues #1342/#1608 confirm even Turing unsupported |
| NVIDIA/cutlass (forks) | 1 | None Kepler | No | Greenfield — we are first |
| facebookresearch/xformers (forks) | 1 | None Kepler | No | sm < 60 not even in upstream's recommended arch list |
| ampir-nn/vllm-pascal | 1 | Active 2026-01, MIT | Indirect | Wheel-distribution playbook |
| **fcheung2888/llama.cpp-kepler** | 2 | **Active 2026-03** | **Direct** | sm_30 patch set; warp shim, bf16 stub, C++17→14, FA disable — same envelope as our K80 fork |
| dogkeeper886/ollama37 | (own) | Active 2026-04, MIT | Direct (own) | Same toolchain pin (CUDA 11.4 / GCC 10 / R470). Source of dual-judge harness for Phase 5 |
| neverclover/ollama-on-k80 | 2 | Active 2025-12 | Indirect | Second K80 Ollama data point |
| cmarshall108/neuronet | 2 | Active 2025-03 | No | Reimplementation; framework switch costs more than fixing vLLM |
| systemSpiderm/gemm-cuda | 2 | Active 2025-04 | Indirect | Reference K80 SMEM-tiled SIMT GEMM kernel |
| xiaoran007/Pytorch-for-Kepler | 2 | Active 2025-04, BSD-3 | Indirect | sm_37 torch wheel build playbook |
| llama.cpp issue #12140 | 2 | Closed stale | Reference only | Validates toolchain + ~5–6 T/s envelope on K80 |
| llama.cpp #18743 / koboldcpp #1409 | 2 | Closed | Reference (failure mode) | "No kernel image" runtime crash — smoke test must catch |
| SparkAttention | 3 | arXiv 2025 | No | sm_70 tensor cores; precedent for "rewrite needed" argument |
| Academic Kepler attention | 3 | None | — | Confirmed gap |

## 7. Implications for the epic

### Phase 1 (CUTLASS Sm37 trait)

No prior art; we are first. Story 0.1's "structurally trivial" claim survives unchallenged — the absence isn't because someone tried and failed, it's because no one outside our specific use case has needed it.

### Phase 2 (XFormers patches + Sm37 GEMM)

`systemSpiderm/gemm-cuda` is a usable reference kernel. Worth reading before writing our own Phase 1 Story #28 reproducer (FP32 SGEMM compiles + runs on K80). Confirms 48 KB SMEM budget and tile sizes that fit Kepler.

### Phase 3 (FlashAttention rewrite for sm_37)

Story 0.2's "rewrite, not port" stands and is **triple-validated**:

1. `ssiu/flash-attention-turing` exists *because* even Turing required a rewrite.
2. SparkAttention exists *because* even Volta required a rewrite.
3. `fcheung2888/llama.cpp-kepler` **disables** flash attention (`GGML_CUDA_FA=OFF`) and runs without it. That's the practical baseline today.

This is meaningful: the active-Kepler-LLM-inference community's working answer to "FA on Kepler" is "don't try, run without it." Our Phase 3 ambition is approximately two hardware generations more aggressive than any active community effort. **That doesn't mean don't do it** — but Story 0.6 should weight Story #54's expected payoff (long-context unlock from O(N) memory) carefully against the engineering cost.

### Phase 4 (vLLM integration / packaging)

`ampir-nn/vllm-pascal` and `xiaoran007/Pytorch-for-Kepler` are both reference packaging playbooks. The user's own `dogkeeper886/ollama37` already uses our exact toolchain — its Dockerfile, CI workflow, and judge harness are likely worth direct reuse rather than parallel reinvention.

### Phase 5 (validation)

`dogkeeper886/ollama37`'s dual-judge eval framework (referenced in study plan #2) is the upstream of Phase 5 Story #51. Confirmed alive and on the same toolchain.

### New risk surfaced (across all phases)

Multiple K80 users hit "no kernel image is available for execution on the device" runtime crashes on builds that *succeeded* at compile time (llama.cpp #18743, koboldcpp #1409). Build success ≠ runtime success on this hardware. **The k80-runtime smoke test must explicitly catch this failure mode**, not just verify the build passes. Phase 1 Story #28's acceptance criterion already requires runtime execution on K80, so this is implicitly covered — but worth flagging explicitly so it doesn't get traded away during Phase 1 implementation.

## 8. Open questions / unknowns

1. **Relationship between `dogkeeper886/ollama37` and `EnlistedGhost/ollama37`.** Both have the identical description "Tesla K80 Compatible Ollama Fork." `dogkeeper886/ollama37` was created 2025-04-03 (older, 18 stars, 3 forks); `EnlistedGhost/ollama37` was created 2026-04-10 (no fork-relationship in metadata, 0 stars). Could be an independent re-creation, a copy with the remote renamed, or unrelated. Not load-bearing for this story but the user may want to be aware.

2. **License clarification for `ssiu/flash-attention-turing`.** The repo has no declared license. If we ever want to draw from its scaffolding (varlen layout, masking patterns), we need to either get explicit permission from the author or clean-room reimplement.

3. **Coordination with `fcheung2888/llama.cpp-kepler`.** That patch set is the closest direct cousin to our work. A friendly issue-thread discussion may surface lessons we don't see from the README alone — patches the author considered and rejected, performance traps, etc. Out of scope for this story; flagging as a potential courtesy outreach.

## 9. Conclusions

**No one has ported FlashAttention, XFormers, or CUTLASS to sm_37.** The gap is real, structural, and acknowledged by the upstream maintainers (FA #1342/#1608) — not an oversight. We are starting from zero on the kernel side.

Two adjacent fork families exist and are directly informative:

- `fcheung2888/llama.cpp-kepler` (sm_30 patch set) shows the C++/CUDA mechanical work — `__shfl_sync` shims, bf16 stubs, C++17 polyfills — that any Kepler port must do
- `dogkeeper886/ollama37` (the user's own work) already pins our exact toolchain and has working K80 CI

The epic's plan from Phase 0 stories 0.1–0.4 is correct in scope. **No finding from this prior-art search invalidates Phase 1, 2, 3, or 4.** Story 0.6 should ratify the recommendations from each prior story with the additional context that:

- Phase 1 (CUTLASS Sm37) is greenfield — no upstream precedent to crib from, no upstream PR to rebase against
- Phase 3 (FA rewrite) is two arch generations beyond the most aggressive active community work (the Turing fork) — a project-lifetime risk worth surfacing
- The K80 ecosystem's working answer to "FA on Kepler" is currently "run without FA" — our Phase 3 ambition is a step beyond that

---

**End of Story 0.5 deliverable.** Story 0.6 (consolidate findings + GO/NO-GO, [#24][s06]) is the next pickup — the gate where all of Phase 0's recon outputs get synthesized into a final feasibility recommendation.

[s06]: https://github.com/dogkeeper886/vllm37/issues/24

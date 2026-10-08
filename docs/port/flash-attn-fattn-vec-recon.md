# FlashAttention rewrite — fattn-vec reference recon

**Phase 3 starting point.** Story 0.2 (`flash-attn-mma.md`) recommended a rewrite over a port of upstream FlashAttention because every code path in `Dao-AILab/flash-attention` requires tensor-core MMA atoms (`SM75_*`, `SM80_*`), which sm_37 cannot run. That recommendation stands. What changes Phase 3's calculus is new evidence that *the rewrite does not have to start from a blank page*: a working K80 FlashAttention kernel exists in another framework and has been empirically validated.

This doc is the recon for adopting that kernel as the structural reference for vLLM's K80-native FA backend.

## TL;DR

- `ggml`'s `fattn-vec.cuh` (vendored into `ollama37`) is a hand-written, FP32-fallback-capable FlashAttention-family vector kernel that compiles for sm_37 with CUDA 11.4 and produces bit-exact output vs the non-FA reference path on K80 (gemma3:4b, [run 24960034243](https://github.com/dogkeeper886/ollama37/actions/runs/24960034243)).
- The kernel uses no tensor cores, no MMA, no `cp.async`, no GMMA/TMA. Just warp-cooperative loads, online softmax, FP32 accumulators on K80.
- For vLLM's K80 fork (FP32-only per the [README](../../README.md#how-vllm37-runs-on-a-k80)), most of `fattn-vec`'s complexity (Q4_0/Q8_0 K-V quant, FP16 fast paths, ALiBi, logit softcap) can drop in v1.
- The work is not algorithm design — it's interface translation: replace `ggml_tensor` I/O with vLLM's paged-KV + cu_seqlens + Torch tensor convention.
- This re-frames Phase 3 from a research project into a porting/adaptation project with a known-good algorithmic reference.

## ⚠ Scope honesty: Phase 3 vs Phase 2 on K80

A reader of an earlier draft caught the over-framing. **XFormers' `cutlassF` (the kernel landed in Phase 2) is itself memory-efficient attention** — same Rabe-Staats / FlashAttention-family algorithm: online softmax fused with the matmul, tile-by-tile KV loading, **O(N) attention memory not O(N²)**. The "FlashAttention" name is Tri Dao's branding of an algorithmic class that `cutlassF` already belongs to.

That means most of the wins commonly attributed to "adding FlashAttention" are already shipped via Phase 2:

| Potential benefit of Phase 3 over Phase 2 | Real on K80? |
|---|---|
| O(N) attention memory (long context) | **No — already shipped in Phase 2** |
| Tensor-core speedup | No — K80 has none |
| FA-2 warp specialization | Marginal at best — K80 lacks the right intrinsics |
| Causal-mask fusion | Marginal constant-factor speedup |
| `VLLM_ATTENTION_BACKEND=FLASH_ATTN` selectable | Cosmetic only |

**Realistic Phase 3 win on K80 is a single-digit-percent speedup at most.** Not a memory or capability unlock. Before committing to Phase 3 we should measure the actual ceiling Phase 2 reaches today (§6.0). Most of this doc was written before that re-framing — read sections 3–5 as "what porting *would* entail if we decide to do it," not as motivation.

## 1. What changed since Story 0.2

Story 0.2 surveyed `Dao-AILab/flash-attention` (the upstream PyPI package vLLM consumes via `vllm-flash-attn`) and concluded:

> No code path exists for sm_37. ... CUTLASS's existing SIMT atom is scalar 1×1×1 FMA — wrong shape for FA's tile structure. ... Any rewrite starts from the algorithm description (online softmax + tiling) and a sm_37-friendly kernel scaffold.

Story 0.2 also mentioned (§10, candidate strategies):

> Hand-written CUDA, FP32 SIMT. Write a `flash_fwd_sm37.cu` that follows FA's algorithm but uses scalar `__fma_rn` and standard `__syncthreads()`. Tile sizes would be smaller because K80 has 48 KB shared memory ...

What Story 0.2 did **not** know: the `ggml-cuda` backend in llama.cpp already implements exactly this — a hand-written FA kernel that uses scalar FMA and works on sm_37. The kernel has been part of `ollama37`'s vendored `ggml` since at least 2026-04 (issue [#108][o-108]) and was promoted to default-on after empirical validation in PR [#117][o-117].

[o-108]: https://github.com/dogkeeper886/ollama37/issues/108
[o-117]: https://github.com/dogkeeper886/ollama37/pull/117

## 2. Evidence — fattn-vec on K80

Source of truth in this repo's sister project: `/home/jack/src/ollama37/ml/backend/ggml/ggml/src/ggml-cuda/`.

### 2.1 K80 build gate
`ml/device.go:430-447` (ollama37) — `FlashAttentionSupported`:

```go
supportsFA := gpu.Library == "cpu" ||
    gpu.Name == "Metal" || gpu.Library == "Metal" ||
    (gpu.Library == "CUDA" && gpu.ComputeMajor >= 7 && !(gpu.ComputeMajor == 7 && gpu.ComputeMinor == 2)) ||
    // ollama37: K80 (compute 3.7) uses fattn-vec, empirically validated
    // against gemma3:4b in 2026-04 (issue #108). Output bit-exact match
    // with the non-FA path; unlocks Q8_0 KV cache quant for ~47% memory
    // reduction.
    (gpu.Library == "CUDA" && gpu.ComputeMajor == 3 && gpu.ComputeMinor == 7) ||
    gpu.Library == "ROCm"
```

K80 was added unconditionally to the FA-supported list in PR #117 after Phase 2 validation. The comment cites the validation provenance.

### 2.2 Empirical validation

[Workflow run 24960034243][o-run] (`test-fa-k80.yml`, default-on FA):

[o-run]: https://github.com/dogkeeper886/ollama37/actions/runs/24960034243

```
Outputs match exactly — FA produces identical result to non-FA baseline
```

Both runs (FA-off baseline + FA-on) produced identical token sequences on `gemma3:4b` for the prompt "What is the capital of France?":

```json
"context": [105,2364,107,3689,563,506,5279,529,7001,236881,25685,528,886,3658,236761,
            106,107,105,4368,107,50429]
"response": "Paris"
```

KV cache 254 MiB (FA-off baseline) → 135 MiB with Q8_0 KV quant + FA on, per PR #117. ~47% memory reduction. No CUBLAS errors, no kernel crashes.

The bit-exactness is significant: it means online-softmax accumulation order (the canonical floating-point divergence point in FA implementations) preserves enough precision in the FP32 fallback path that the K80 result is observationally indistinguishable from the non-FA reference. We do not need to litigate numerical correctness in Phase 3 — ollama37 already settled it on a real model.

## 3. fattn-vec algorithm summary

Source: `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh` (591 lines).

### 3.1 Kernel shape

Single global function `flash_attn_ext_vec`, templated on:
- `D` — head size (instantiated for 64, 128, 256)
- `ncols` — number of Q columns processed per block
- `type_K`, `type_V` — `GGML_TYPE_F16` / `GGML_TYPE_Q4_0` / `Q4_1` / `Q5_0` / `Q5_1` / `Q8_0`
- `use_logit_softcap` — bool

Launch shape: `nthreads = 128` (line 4-11), one warp dispatched per `D/2` portion of head dim. Grid `gridDim.z = sequence * num_heads`; sequence and head decoded from `blockIdx.z` (line 97-99).

### 3.2 K80 path (FP32 fallback)

The kernel branches on a compile-time macro `FAST_FP16_AVAILABLE` (set for sm_53+):

```cpp
#ifdef FAST_FP16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif
```

On sm_37, `FAST_FP16_AVAILABLE` is undefined → all accumulators (V·KQ output, shared-mem KQ scores, online softmax `KQ_max[]` / `KQ_sum[]` running stats) are `float`. The dequantize-V helper (line 89-93) is also chosen against `float` rather than `half`.

This is the path PR #117 productized. K/V tensors stored in F16 are dequantized to FP32 *at use*, accumulated in FP32, written back as FP32 result.

### 3.3 Online softmax structure

Standard FlashAttention-1 algorithm:

```cpp
float KQ_max[ncols];     // running max per query column
float KQ_sum[ncols];     // running denominator per query column
// VKQ[ncols][...] running output accumulator
```

Per K-block tile:
1. Load K block into shared memory (`KQ[]`).
2. Compute `Q · K_block` scores into `KQ[]`.
3. Update online softmax stats: rescale `VKQ` and `KQ_sum` by `exp(old_max - new_max)`.
4. Load V block, accumulate `softmax(KQ_block) · V_block` into `VKQ`.

No ring-buffering of K/V into shared memory in the way upstream FA does — `fattn-vec` is the *vector* (single-token-decode-friendly) variant. The *tile* variant (`fattn-tile.cu`) is the prefill-friendly cousin and has tensor-core paths that we'd skip on K80; tile FP32 fallback exists too but is less polished.

### 3.4 No tensor cores anywhere

No `mma.*`, no `wmma.*`, no `ldmatrix`, no `cp.async`. Just `__syncthreads()`, scalar FMA, and warp-cooperative loops. Compiles cleanly with `-arch=sm_37 -DCUDA_ARCH=370`.

## 4. The interface gap — ggml vs vLLM

This is where the rewrite work actually lives. The kernel signature differs significantly between ggml-shape and vLLM-shape.

### 4.1 ggml-shape (input to fattn-vec)

```cpp
flash_attn_ext_vec(
    const char * Q, const char * K, const char * V,
    const char * mask,        // half-precision precomputed bias
    const char * sinks,       // attention-sinks array (optional)
    const int  * KV_max,      // per-sequence KV length
    float      * dst,         // output [packed]
    float2     * dst_meta,    // for split-kv combine
    const float scale, max_bias, m0, m1,
    const uint32_t n_head_log2, const float logit_softcap,
    // strides:
    int32_t ne00, ne01, ne02, ne03,
            nb01, nb02, nb03,
    int32_t ne10, ne11, ne12, ne13,
            nb11, nb12, int64_t nb13,
            nb21, nb22, int64_t nb23,
    int32_t ne31, ne32, ne33,
            nb31, nb32, int64_t nb33);
```

Q/K/V are flat tensors with explicit per-axis byte strides (`nb*`). Sequences are encoded into the grid Z dim (`sequence * num_heads`) and decoded from `blockIdx.z`. KV is contiguous, not paged.

### 4.2 vLLM-shape (input expected by `vllm/attention/backends/flash_attn.py`)

The vLLM FA backend's `forward()` (at `flash_attn.py:663`) calls into the kernel via two entry points:

**Prefill / variable-length:** `flash_attn_varlen_func`:
```python
flash_attn_varlen_func(
    q=query,                         # [num_tokens, num_heads, head_size]
    k=key, v=value,                  # [num_tokens, num_kv_heads, head_size] OR paged
    cu_seqlens_q=...,                # cumulative seq starts (varlen packing)
    cu_seqlens_k=...,
    max_seqlen_q=..., max_seqlen_k=...,
    softmax_scale=..., causal=True,
    window_size=..., alibi_slopes=...,
    softcap=..., out=prefill_output,
    block_table=...,                 # [batch, max_blocks_per_seq] — paged KV indirection
    ...)
```

**Decoding (single token):** `flash_attn_with_kvcache`:
```python
flash_attn_with_kvcache(
    q=decode_query.unsqueeze(1),     # [num_tokens, 1, num_heads, head_size]
    k_cache=key_cache,               # [num_blocks, block_size, num_kv_heads, head_size]
    v_cache=value_cache,
    block_table=block_tables_arg,
    cache_seqlens=seq_lens_arg,
    ...)
```

The differences:
1. **Paged KV cache.** vLLM's K and V live in fixed-size blocks (typically 16 tokens); a per-sequence `block_table` indirects logical position → physical block. ggml's kernel reads K/V as one contiguous tensor.
2. **Packed token layout (`cu_seqlens`).** vLLM concatenates all sequences into one tensor of shape `[total_tokens, ...]` and uses cumulative-sum offsets to mark sequence boundaries. ggml encodes batch in the launch grid.
3. **Mask shape.** vLLM passes `causal=True` + sliding-window flags and constructs mask logic inside the kernel. ggml passes a precomputed `mask` tensor (or `nullptr`).
4. **Output shape.** vLLM writes back into `out` of shape `[num_tokens, num_heads, head_size]`. fattn-vec writes `dst` plus a `dst_meta` for the split-KV combine path (used when KV is too large for one block).

### 4.3 What this means

The algorithm is the same. The kernel's compute body — load tile, score, online-softmax, accumulate, write — is reusable. What changes:

| Concern | ggml/fattn-vec | vLLM expectation | Adaptation work |
|---|---|---|---|
| Q layout | strided contig | `[total_tokens, num_heads, head_size]` packed | substitute index expr |
| K/V layout | strided contig | paged `[num_blocks, block_size, num_kv_heads, head_size]` | swap inner-loop K/V index for `block_table[b][block_idx] * block_size + offset` |
| Sequence boundaries | grid Z indexes seq | `cu_seqlens_q/k` arrays | rewrite seq lookup |
| Mask | precomputed half tensor | `causal` flag + windowed | drop mask param, add causal cutoff in score loop |
| Output | `dst` + `dst_meta` for combine | `out` only (combine-free for prefill, varlen-aware for decode) | fold combine into kernel for vLLM split-kv case OR start with non-split version |

Single biggest item: paged-KV indirection. Affects every K/V load in the inner loop. The `fattn-stream-k` family in ggml (used for the multi-block split path) might be a closer structural match than `fattn-vec` since stream-k already handles non-contiguous KV chunks. Worth a separate look.

## 5. Simplifications under vLLM K80 fork's FP32-only constraint

The vLLM K80 fork serves FP32 models (about 2B parameters per die; [README](../../README.md#what-runs-today)). This drops a substantial chunk of fattn-vec's complexity:

| fattn-vec feature | Used in v1 vLLM port? | Reason |
|---|---|---|
| FAST_FP16_AVAILABLE branch | **no** — only the FP32 fallback path | K80 doesn't have it; vLLM K80 fork is FP32 anyway |
| Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 K-V quant | **no** — F32 K/V only | vLLM's K80 fork has no quantized kernels yet; see `docs/port/model-family-requirements.md` Part 9 |
| `q8_1` Q-side quant block (line 139-186) | **no** — Q is F32 | as above |
| ALiBi slopes | **no** in v1 | TinyLlama / gpt-oss / gemma3 use rope/no-bias; defer |
| Logit softcap | **no** in v1 | Gemma-2/3 specific; defer |
| Attention sinks | **no** in v1 | Recent Mistral/gpt-oss feature; defer |
| Split-KV combine (`dst_meta`) | **probably no** in v1 | vLLM's prefill is single-block already; revisit if max-len needs it |
| Sliding window | **no** in v1 | Defer |

What's left is the kernel core: fp32 Q · fp32 K_block, online-softmax accumulate, write fp32 out. That's a much smaller kernel than the templated fattn-vec, which spends most of its lines on quant/dequant fanout and feature-flag branches.

## 6. Proposed Phase 3 story breakdown

This replaces the placeholder Phase 3 stories created at portal time (#36, #37, #38, #39, #40). The new outline:

### Story 3.0 — measure Phase 2's actual ceiling FIRST (gating)

**Before any Phase 3 kernel work starts, establish what Phase 2 already delivers.** Without this measurement, Phase 3's benefit case is hand-waved.

Concrete deliverable: a `k80-context-stress.yml` workflow run that:
- Boots the current Phase 2 image (XFormers cutlassF backend) on K80
- Sweeps `--max-model-len` upward from 2048 (current default in `docker-compose.yml`) — try 4K, 8K, 16K — with TinyLlama-1.1B
- For each setting, measures: peak GPU memory, time-to-first-token, tokens/sec at decode, max successful context length before OOM or kernel failure
- Captures the same `[k80-trace]` evidence pattern as Phase 2's existing smoke test

If Phase 2 already hits ≥8K cleanly: Phase 3 has no headline prize on K80. Either close Phase 3 entirely or re-scope to a specific narrow target (e.g., "measure cutlassF vs ggml-fattn-vec speed delta on K80 — kernel comparison only, no integration").

If Phase 2 hits a wall before useful context: Story 3.0 surfaces the *specific* wall (memory? kernel-shape constraint? max-seqlen hardcode somewhere?) and Phase 3's later stories target that wall instead of "implement FA from scratch."

**This story is gating.** Stories 3.1–3.5 below should not start until 3.0 produces numbers.

### Story 3.1 — recon (this doc)
Capture the algorithm, evidence, and interface gap. Closes when this file lands.

### Story 3.2 — minimal vLLM-shaped FA kernel for K80
Write a new `csrc/k80/flash_attn_sm37.cu` (or under `vllm/k80/`) that:
- Takes vLLM's paged-KV inputs (q, k_cache, v_cache, block_table, cu_seqlens, scale, causal, out)
- Implements FP32-only forward pass for D=64, 128 (head sizes used by TinyLlama / Gemma)
- Borrows the online-softmax structure from fattn-vec
- Produces bit-exact output vs Phase 2's XFormers cutlassF on a fixed (q, k, v) input
- No prefill/decode split optimization — single kernel for both paths in v1
- Pure CPU-build verifiable (compile test) before runtime test

Acceptance gate: `make verify-flash-attn-sm37` analogous to the existing CUTLASS / xformers verifiers.

### Story 3.3 — vLLM backend wiring
Add `vllm/attention/backends/flash_attn_k80.py`:
- Class `FlashAttnK80Backend(AttentionBackend)` mirroring the XFormers-on-K80 dispatch pattern from PR #66
- Calls Story 3.2's kernel
- Modify `vllm/platforms/cuda.py:368` (the K80 gate added in PR #66) to prefer `FlashAttnK80Backend` when `VLLM_ATTENTION_BACKEND=FLASH_ATTN` and the kernel is built; fall back to XFormers otherwise.

### Story 3.4 — runtime smoke + golden
Mirror Phase 2's k80-runtime workflow:
- Trigger `k80-runtime.yml` with `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
- TinyLlama golden assertion identical to the cutlassF baseline (Phase 2 established it)
- Memory diff vs cutlassF (KV cache size — should not regress; FA's win is long-context, not 2K)

### Story 3.5 — comparative validation (only if 3.0 found a wall)
**Conditional on Story 3.0's findings.** If Phase 2 already hits acceptable long context, this story does not exist. If 3.0 found a specific wall, this story re-runs the same context sweep on the Phase 3 backend and confirms the wall is gone. This is a comparative measurement, not a prize.

### Out of scope for Phase 3
- Q4/Q5/Q8 K-V quant (separate phase if/when vLLM K80 fork ever does quant)
- ALiBi, softcap, sliding window, attention sinks (model-specific add-ons)
- Backward pass / training (vLLM is inference only)
- TP>1 (gated by hardware-risky-ops rule; tackle after TP=1 lands)

## 7. Open questions / things this recon did NOT settle

1. **Paged-KV in fattn-vec.** Is the ggml stream-k variant a better starting point than fattn-vec for the paged case? Worth reading `fattn-tile.cu` and the stream-k path before Story 3.2 commits to a kernel scaffold.
2. **Head size coverage.** TinyLlama is D=64; gemma3:4b is D=256. fattn-vec instantiates 64/128/256. vLLM Phase 3 should probably mirror that — at least 64 and 128 in v1.
3. **Sequence length ceilings.** K80 has 48 KB shared memory. fattn-vec's per-block shared usage scales with `ncols * D` and `nthreads * V_cols_per_iter * D`. For D=128, ncols=8, that's already tight. Must measure under `nvcc --ptxas-options=-v` before committing tile sizes.
4. **License compatibility.** ggml is MIT (per llama.cpp). vLLM is Apache 2.0. Permissive both ways — adapting ggml code or borrowing structural patterns is fine, but the v1 kernel should be re-implemented rather than copy-pasted, with attribution to ggml in the source comment.
5. **Numerical equivalence with cutlassF.** Phase 2's XFormers cutlassF is the established baseline (Story #34 verified bit-exactness vs manual softmax(QK^T)V). Story 3.4 should establish the same against XFormers, not against ggml — so the vLLM verification chain stays self-contained.

## 8. Working chain after Phase 3 completes

```
vLLM dispatcher (cuda.py:368)
  → VLLM_ATTENTION_BACKEND=FLASH_ATTN selected
    → FlashAttnK80Backend.forward()
      → Story 3.2 kernel (fattn-vec-derived, vLLM-shape, FP32-only)
        → online softmax + tile-loaded paged-KV (algorithm from ggml)
          → scalar FMA on sm_37 (no MMA, no FP16 fast paths)
            → Tesla K80 hardware
```

This sits alongside the Phase 2 chain (XFormers / cutlassF) as a *second* backend on K80, not a replacement. The XFormers backend remains the cutlass mem-eff path; FlashAttn-K80 is the new long-context path.

## 9. References

### vLLM K80 fork (this repo)
- `docs/port/flash-attn-mma.md` — Story 0.2's original recon (pre-fattn-vec discovery)
- `vllm/attention/backends/flash_attn.py:663` — vLLM FA backend forward signature
- `vllm/platforms/cuda.py:368` — Kepler dispatcher gate (added PR #66)
- `README.md` — FP32 serving
- `docker/k80/cutlass-patches/` and `xformers-patches/` — Phase 1 + 2 patch precedents
- Epic [#12][epic], Phase 3 portal [#16][phase3-portal]

### ollama37 (sister project)
- `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh` — kernel source
- `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh` — shared helpers (online softmax helpers, dequant)
- `ml/backend/ggml/ggml/src/ggml-cuda/fattn.cu:117-180` — dispatcher (FATTN_VEC_CASE macros)
- `ml/device.go:430-447` — Go-side FA gate (K80 added in PR #117)
- `docs/research/k80-fa-model-coverage.md` — model-by-model FA coverage audit
- `docs/traces/qwen35-flash-attention-gate.md` — trace of the per-model deny list
- PR [#117][o-117] — productize K80 FA support
- Issue [#108][o-108] — FA validation phase
- [Run 24960034243][o-run] — bit-exact validation

[epic]: https://github.com/dogkeeper886/vllm37/issues/12
[phase3-portal]: https://github.com/dogkeeper886/vllm37/issues/16

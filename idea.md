# Model targets for the K80 fork

**Format for now: HF safetensors (BF16), served as FP32** with `--dtype float32`.
Weight-only int4 checkpoints (GPTQ, AWQ, compressed-tensors w4a16, GGUF Q4) wait for
#73 (load and repack to a K80 int4 layout) and #74 (fused K80 int4 kernel).
Why, with measurements: `docs/port/model-family-requirements.md` Part 9.

Models listed from the Hugging Face Hub on 2026-10-07 (`hf models ls`, sorted by
downloads, ≤40B params). Sizes are computed from safetensors parameter counts,
not measured.

## The constraint: the fork's base is mid-2025

This fork's vLLM is ~v0.10 (Aug 2025) with transformers 4.55–4.57 and torch 2.4.1.
A model loads only if the base knows its architecture. That splits current
models into three tiers:

| Tier | Meaning | Families |
|---|---|---|
| **A: loads today** | Architecture is in `vllm/model_executor/models/registry.py` | Granite 4.1/4.2 (dense), Qwen3 incl. 2507, Hunyuan dense, ERNIE 4.5, Gemma 3 |
| **B: needs a backport** | Plain attention (full + sliding window), but the base lacks the model file and the newer transformers config | Gemma 4, Olmo 3, Ministral 3, SmolLM3 |
| **C: blocked on sm_37** | Hybrid layers whose vLLM kernels are Triton or Mamba CUDA; Triton has no Kepler target | Qwen3.5/3.6/3.8 (Gated DeltaNet, 3 of 4 layers linear), Nemotron 3 (Mamba-2), Granite 4.0-h (Mamba), LFM2.5 (conv), gpt-oss (MXFP4 MoE) |

## Memory budget

| Setup | Usable weights + KV (at `GPU_MEM_UTIL=0.85`) | Power |
|---|---|---|
| 1 die (TP=1) | ~9.7 GB | safe |
| 1 board (TP=2, GPU0+GPU1) | ~19 GB | safe, tested |
| 4 dies (TP=4) | ~39 GB | ⚠ power-risky, unvalidated |

KV cache is FP32: bytes per token = layers × 2 × kv_heads × head_dim × 4.

## Priority: safetensors models (FP32)

| Pri | Model | HF repo | Released | Params | Context | FP32 weights | KV / token | Fits | Tier |
|---|---|---|---|---|---|---|---|---|---|
| **P1** | Qwen3-0.6B | `Qwen/Qwen3-0.6B` | 2025-04 | 0.75B | 40K | 3.0 GB | 224 KiB | 1 die | A — CI model, replaces TinyLlama (2K cap, #72 / #48) |
| **P2** | Granite 4.2 3B | `ibm-granite/granite-4.2-3b` | 2026-08 | 3.66B | 128K | 14.6 GB | 160 KiB | TP=2 | A — newest model that loads today |
| **P2** | Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | 2025-08 | 4.02B | 256K | 16.1 GB | 288 KiB | TP=2 | A — 3.7M downloads |
| P2 | Granite 4.1 3B | `ibm-granite/granite-4.1-3b` | 2026-04 | 3.40B | 128K | 13.6 GB | 160 KiB | TP=2 | A |
| P2 | Hunyuan 1.8B Instruct | `tencent/Hunyuan-1.8B-Instruct` | 2025-07 | 1.79B | 256K | 7.2 GB | 128 KiB | 1 die | A |
| P2 | Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 2025-04 | 2.03B | 40K | 8.1 GB | 224 KiB | 1 die (tight) | A |
| P3 | Ministral 3 3B Instruct | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 2025-12 | 4.25B | 256K | 17 GB | — | TP=2 | B — `ministral3` text model + Pixtral vision |
| P3 | SmolLM3 3B | `HuggingFaceTB/SmolLM3-3B` | 2025-07 | 3.08B | 64K | 12.3 GB | — | TP=2 | B |
| P3 | Gemma 4 E2B it | `google/gemma-4-E2B-it` | 2026-03 | 5.12B | 128K | 20.5 GB | — | TP=4 ⚠ | B — any-to-any, per-layer embeddings |
| defer | Granite 4.2 8B | `ibm-granite/granite-4.2-8b` | 2026-08 | 8.79B | 128K | 35 GB | 320 KiB | TP=4 ⚠ | A |
| defer | Hunyuan 7B Instruct | `tencent/Hunyuan-7B-Instruct` | 2025-07 | 7.5B | 32K | 30 GB | — | TP=4 ⚠ | A |
| defer | Olmo 3 7B Instruct | `allenai/Olmo-3-7B-Instruct` | 2025-11 | 7.3B | 64K | 29 GB | — | TP=4 ⚠ | B |
| defer | Gemma 4 E4B it | `google/gemma-4-E4B-it` | 2026-03 | 8.0B | 128K | 32 GB | — | TP=4 ⚠ | B |
| — | Gemma 4 12B it | `google/gemma-4-12B-it` | 2026-05 | 12B | — | 48 GB | — | no fit | B; GGUF only |
| ✗ | Qwen3.5 0.8B / 2B / 4B / 9B | `Qwen/Qwen3.5-*` | 2026-02 | 0.87–9.65B | 256K | 3.5–38.6 GB | — | — | C — Gated DeltaNet |
| ✗ | Nemotron 3 Nano 4B | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 2026-03 | 3.97B | — | 15.9 GB | — | — | C — Mamba-2 |
| ✗ | LFM2.5 1.2B / 2.6B | `LiquidAI/LFM2.5-*` | 2026-01/07 | 1.17–2.7B | 128K | 4.7–10.8 GB | — | — | C — conv layers |
| ✗ | gpt-oss 20B | `openai/gpt-oss-20b` | 2025-08 | 20.9B | — | 84 GB | — | — | C — MXFP4 MoE |

Tier A means the architecture is registered; loading each model on K80 is still untested.
All repos above are ungated except Gemma (license click-through).

## Later: GGUF sizes

| Model | Q8_0 | Q4_K_M |
|---|---|---|
| Qwen3-0.6B | 0.8 GB | 0.5 GB |
| Qwen3-1.7B | 2.2 | 1.2 |
| Hunyuan 1.8B | 1.9 | 1.1 |
| Granite 4.2 3B | 3.9 | 2.2 |
| Qwen3-4B-2507 | 4.3 | 2.4 |
| Olmo 3 7B (B) | 7.8 → 1 die (tight) | 4.4 → 1 die |
| Hunyuan 7B | 8.0 → TP=2 | 4.5 → 1 die |
| Granite 4.2 8B | 9.3 → TP=2 | 5.3 → 1 die |
| Gemma 4 12B (B) | 12.7 → TP=2 | 7.2 → 1 die |

#73's reference path dequantizes to FP32 in memory, which saves memory but not bandwidth.
#74 unpacks int4 in registers and multiply-adds in FP32; a first prototype is 2–3× faster
than FP32 cuBLAS for decode (Part 9). vLLM loads single-file GGUF only.

## Formats on K80

| Format | Quant types | K80 status |
|---|---|---|
| HF safetensors (BF16/FP16) | none; upcast to FP32 | **works today** (tier A) |
| GGUF | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, IQ4_XS | blocked: kernels need `__dp4a` / half math; Q4 planned via #73, #74 |
| GPTQ / AWQ / compressed-tensors w4a16 | int4 | blocked: kernels need half2 math, tensor cores, `cp.async`; planned via #73, #74 |
| bitsandbytes | nf4, int8 | blocked: sm_60+, no backport |
| FP8 / NVFP4 / MXFP4 | fp8, fp4 | impossible: needs sm_89+ hardware |

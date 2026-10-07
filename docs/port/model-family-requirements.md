# Model families on Tesla K80: formats, code paths, requirements

Study date 2026-10-07. Every requirement below cites one of:
- `file:line` in **official vLLM** `/home/jack/src/vllm` at commit `3ca00a8261` (2026-10-07, ~v0.31.1rc0)
- `vllm37:file:line` in **this fork**
- a field in a vendor `config.json` on the Hugging Face Hub
- a URL
- a measurement on this host, marked **[measured]**

Anything not proven is marked UNVERIFIED. Parts 2–7 are the subagent reports, merged unedited except for heading levels.

## 1. Summary

### 1.1 Target hardware (this host)

| Item | Value | Source |
|---|---|---|
| GPU | 2× Tesla K80 = 4× GK210, **sm_37**, 11,441 MiB each | `nvidia-smi` [measured] |
| Topology | GPU0–GPU1 PIX, GPU2–GPU3 PIX, cross-board PHB; no NVLink | `nvidia-smi topo -m` [measured] |
| Driver | 470.256.02 → highest CUDA runtime **11.4** | `nvidia-smi` [measured] |
| CUDA toolkit | 11.4.4; nvcc offers `-std` up to **c++17** | `/usr/local/cuda-11.4/version.json`, `nvcc --help` [measured] |
| Host compiler | GCC **10.5.0** | `gcc --version` [measured] |
| FP16 arithmetic intrinsics | Not available: `__hmul`, `__hadd`, `__hmul2` … are defined only for `__CUDA_ARCH__ >= 530` | `/usr/local/cuda-11.4/include/cuda_fp16.hpp:1626-2397` |
| cuBLAS FP16 GEMM | `cublasGemmEx` (any FP16 combo) and `cublasHgemm` → **CUBLAS_STATUS_ARCH_MISMATCH**. `cublasSgemmEx` with FP16 A/B and FP16 or FP32 C → **SUCCESS**, correct result | 256×256 test on GPU0 [measured] |
| PyTorch 2.0.1 FP16 matmul on sm_37 | `gemm<at::Half>` calls `cublasSgemmEx` when `prop->major < 5`. Batched FP16 GEMM loops over that `gemm` | pytorch `v2.0.1` `aten/src/ATen/cuda/CUDABlas.cpp:420,449` and `:267,274-283` |
| Not on sm_37 | BF16, `__dp4a` (sm_61), tensor cores / `mma.sync` (sm_70), `ldmatrix` (sm_75), `cp.async` (sm_80) | `docs/port/kepler-vs-maxwell.md` |

**Consequence for dtype.** FP16 *storage* with FP32 *compute* works at the cuBLAS level on this card [measured], and PyTorch 2.0.1 routes FP16 matmuls to that path. This halves weight memory compared with FP32. It does not make custom CUDA kernels that use half arithmetic intrinsics work: those fail to compile for sm_37. Every family below would need its fork kernels checked for FP16 support before `--dtype half` can be trusted.

### 1.2 Official vLLM cannot run on K80, so every path goes through the fork

Detail and evidence: Part 2, rows R1–R27.

| Blocker | Required | K80 / this host | Evidence | Fallback |
|---|---|---|---|---|
| Triton on every step (slot mapping, KV-block zeroing, attention, Model Runner V2) | Triton 3.7.1: CC ≥ 8.0, bundled ptxas 12.8 | sm_37; CUDA 12 cannot target sm_37 | `vllm/v1/worker/block_table.py:201-228,398-548`; `vllm/v1/worker/utils.py:59`; `requirements/test/cuda.txt:1351`; Triton README release/3.7.x | None |
| Decoder attention backend | FLASH_ATTN CC ≥ 8.0; FLASHINFER 8.0–12.1; TRITON_ATTN and FLEX_ATTENTION generate Triton | No backend is valid, so the selector raises ValueError | `flash_attn.py:451-452`; `flashinfer.py:514-522`; `vllm/platforms/cuda.py:510-514` | None. xformers and CUDA paged attention were removed; TORCH_SDPA is for vision encoders only (`registry.py:70`) |
| `_C_stable_libtorch` build | arch ≥ 7.0; C++20; GCC ≥ 11.3; torch ≥ 2.11 stable ABI | 3.7 filtered out, then FATAL_ERROR; nvcc 11.4 = c++17; GCC 10.5 | `CMakeLists.txt:17-18,22-28,116-132,231-239,1232-1237` | None |
| Extension import | Built `_C_stable_libtorch` | Not buildable | `vllm/platforms/cuda.py:23` unconditional import | None |
| PyTorch | 2.13.0 (+cu130); wheels CUDA 12.6 / 13.x only | Driver 470 is below the 525 / 580 minimums | `requirements/cuda.txt`; pytorch RELEASE.md | None |
| Unguarded half intrinsics in core kernels | sm_53 | sm_37 | `csrc/libtorch_stable/cuda_vec_utils.cuh:350-358`; `csrc/custom_collective_common.cuh:105-108` | Source patch needed |
| torch.compile / Inductor | Triton, CC ≥ 7.0 | sm_37 | `vllm/config/vllm.py:1827-1831` | `--enforce-eager` (`vllm/config/vllm.py:1746-1748`), except helpers at `logprobs.py:10` and `vocab_parallel_embedding.py:175` |
| Auto dtype | — | Below CC 6.0 only `float32` | `vllm/platforms/cuda.py:255-264` | `--dtype half` is not blocked (`vllm/config/model.py:2467-2470`) |

**The fork avoids each blocker** with an older base (vLLM ~v0.10, v0 engine), torch 2.0.1 built for CUDA 11.4 with sm_37, xformers `cutlassF` patched for sm_37 (Phase 2, PR #66), and the v0 CUDA paged-attention kernels. The fork's paged attention supports head sizes `[32, 64, 80, 96, 112, 120, 128, 192, 256]` (`vllm37:vllm/attention/ops/paged_attn.py:40`), so **head_dim 512 (Gemma 4 full layers) is unsupported**. Its v0 paged decode has no sliding-window argument (`vllm37:vllm/attention/ops/paged_attn.py:90`, Part 6). With `VLLM_ATTENTION_BACKEND=XFORMERS` (the compose default), models with interleaved sliding and full layers have sliding windows disabled and `max_model_len` capped to the window size (`vllm37:vllm/config/__init__.py:718-731`).

Porting rule that follows: **a model family is usable on K80 only if its fork model file runs every layer through torch, cuBLAS, the fork's sm_37 attention, or sm_37-safe `_C` ops. Triton anywhere on the path is a blocker.**

### 1.3 Family matrix

FP32 sizes are weights only. The KV cache comes on top. One die has ~9.7 GB usable at `GPU_MEM_UTIL=0.85`, one board (TP=2) ~19 GB, four dies (TP=4, power-risky) ~39 GB.

| Family | Official formats | Layer types that need special kernels | Kernels with no non-Triton GPU path (official) | In the fork | FP32 size of usable sizes | Porting work for K80 | Part |
|---|---|---|---|---|---|---|---|
| **Qwen3 dense** (incl. 4B-2507) | BF16, FP8, AWQ, GPTQ-Int8, GGUF, MLX | none | none (model ops) | `qwen3.py`, registry `:129` | 0.6B 3.0 GB, 1.7B 8.1 GB (1 die); 4B 16.1 GB (TP=2) | **None model-specific** | 3 |
| **Qwen3.5 / 3.6 / 3.8** | BF16, FP8, GPTQ-Int4; all include a vision tower | Gated DeltaNet (3 of 4 layers), causal conv1d, MRoPE, gated RMSNorm, head_dim 256 | causal_conv1d, chunked gated-delta-rule prefill, DeltaNet decode (CUDA decode needs BF16 and CC ≥ 8.0) | none | 0.8B 3.5 GB, 2B 9.1 GB (1 die); 4B 18.6 GB (TP=2) | FP32 DeltaNet recurrence + chunked prefill + conv1d kernels; v0 hybrid state cache; model glue; native MRoPE / gated norm | 3 |
| **Gemma 3** | BF16, QAT q4_0 GGUF, QAT "unquantized" BF16 | sliding window (pattern 6) | none (model ops) | `gemma3.py`, `gemma3_mm.py` | 270m 1.1 GB, 1b 4.0 GB (1 die); 4b 17.2 GB (TP=2) | With the fork's XFORMERS backend, interleaved sliding models are capped to the window: **512 tokens** for 270m/1b, 1024 for 4b+ (`vllm37:vllm/config/__init__.py:718-731`). Longer context needs per-layer sliding windows in the fork's prefill and paged decode | 4 |
| **Gemma 4** | BF16, QAT GGUF, compressed-tensors w4a16, mobile-ct, NVFP4 (NVIDIA) | sliding (head_dim 256) + full (head_dim **512**), per-layer embeddings, KV sharing, K=V, 26B MoE | official forces TRITON_ATTN without FA4 (`models/config.py:276-283`); 26B MoE routing and experts Triton-only on CUDA | none | E2B 20.5 GB (TP=4 ⚠); larger: no fit | New model file; head_dim 512 attention; mixed 256/512 KV cache; KV sharing in v0; sliding-window decode | 4 |
| **Granite 4.1 / 4.2 dense**; Granite 4.0 micro, 1b, 350m (all-attention) | BF16, FP8 / NVFP4 / MXFP4 compressed-tensors | none (muP multipliers only) | none (model ops) | `granite.py`, registry `:81` | 4.1-3b 13.6 GB, 4.2-3b 14.6 GB (TP=2); 8b 35 GB (TP=4 ⚠) | **None model-specific** | 5 |
| **Granite 4.0-h** (h-tiny, h-small, h-micro, h-1b) | BF16, FP8 | Mamba-2 (36 of 40 layers), MoE (tiny/small) | causal_conv1d, SSD scan, decode state update, gated norm (n_groups=1), MoE experts | `granitemoehybrid.py`, but Mamba ops are Triton and `MambaMixer2.forward_native` is a stub (`vllm37:mamba_mixer2.py:422-430`) | — | torch or CUDA SSD scan + state update; rewire torch conv / gated norm; torch MoE expert loop | 5 |
| **Nemotron 3 / Nemotron-H** | BF16, FP8, NVFP4 (ModelOpt) | Mamba-2, relu² MLP, MoE (30B) | same Mamba-2 set as Granite-h; MoE experts | `nemotron_h.py` without MoE layer type `"E"` (`vllm37:nemotron_h.py:290-294`) | Nano-4B 15.9 GB (TP=2) | Same Mamba-2 work as Granite-h; MoE out of scope (30B ≈ 63 GB FP16) | 5 |
| **Ministral 3 / Magistral / Devstral Small 2** | **FP8 by default**; -BF16 variants (3B/8B/14B); Mistral-native consolidated; GGUF | yarn RoPE + `llama_4_scaling_beta`, Pixtral vision | none (model ops); FP8 needs CC ≥ 7.5 even via Marlin (`fp8.py:146-147`, `marlin_utils_fp8.py:30-31`) | `mistral3.py`, `pixtral.py`; no `Ministral3ForCausalLM`; no llama-4 scaling | 3B 17 GB (TP=2) | Use -BF16 repos; ~10-line long-context scaling port; registry entry; Devstral 2 needs offline FP8→FP16 dequant | 6 |
| **Olmo 3** | BF16 | sliding window 4096 (3 of 4 layers), yarn on full layers, QK-norm | official serves it only via the Transformers backend, which needs transformers ≥ 5.16.1 (`registry.py:717`) | none (`olmo2.py` is close) | 7B 29 GB (TP=4 ⚠) | New model file from `olmo2.py` + per-layer windows + two RoPEs; sliding-window decode beyond 4096 | 6 |
| **LFM2 / LFM2.5** dense | BF16, GGUF, ONNX, MLX | short conv (kernel 3) in ~2 of 3 layers | causal_conv1d (Triton on GPU: `short_conv.py:370-371`) | none | 1.2B 4.7 GB (1 die); 2.6B 10.8 GB (TP=2) | Route to existing torch conv fallbacks (`ops/cpu/causal_conv1d.py:13,119`); port ShortConv + conv-state cache to v0; no prefix caching | 7 |
| **gpt-oss** | MXFP4 experts + BF16 rest; no official BF16 | MoE (32 experts), attention sinks, sliding 128 | MXFP4 MoE (min capability 80, `mxfp4.py:67-77`); sinks only in TRITON_ATTN below CC 9.0 | `gpt_oss.py` (MXFP4 needs 90; no sinks in v0 backends) | 20B ≈ 84 GB | **Out of scope** (memory + MoE + sinks) | 7 |
| **Hunyuan dense** | BF16, FP8, GPTQ-Int4, AWQ-Int4 | dynamic NTK-alpha RoPE, QK-norm after RoPE | none (official uses the Transformers backend, `registry.py:710`) | `hunyuan_v1.py` dense class `:932`, registry `:88` | 1.8B 7.2 GB (1 die); 7B 30 GB (TP=4 ⚠) | **None model-specific** (BF16 only; all quantized repos fail the gate) | 8 |
| **ERNIE 4.5** | BF16 | 0.3B: none; 21B-A3B: MoE | 21B MoE: experts Triton or FlashInfer only (`oracle/unquantized.py:63-73`) | `ernie45.py`, `ernie45_moe.py`, registry `:63-64` | 0.3B 1.4 GB (1 die) | 0.3B: none; 21B: out of scope | 8 |
| **SmolLM3** | BF16 | NoPE every 4th layer | none (Transformers backend, `registry.py:720`) | none | 3B 12.3 GB (TP=2) | ~20-line `smollm3.py` from `llama.py` with a skip-RoPE flag | 8 |

### 1.4 Quantized formats: every official release is refused on sm_37

Official vLLM enforces `get_min_capability()` at `vllm/config/vllm.py:943`.

| Format (seen in these families) | vLLM method | Min capability | Evidence |
|---|---|---|---|
| FP8 (Qwen, Ministral default, Granite, Hunyuan) | `Fp8Config` | 75 (Marlin FP8 fallback also 75) | `fp8.py:146`; `marlin_utils_fp8.py:30-31` |
| FP8 (Nemotron, ModelOpt) | ModelOpt FP8 | 80 | `modelopt.py:437` |
| AWQ int4 (Qwen, Hunyuan) | AWQ | 75 | `auto_awq.py:233` |
| GPTQ (Qwen, Hunyuan) | GPTQ | 60; Exllama uses half2 (sm_53); Hunyuan's `desc_act=true` is rejected on any GPU | `auto_gptq.py:177-183`; `gptq_utils.py:32-37` |
| compressed-tensors w4a16 (Gemma 4 QAT) | compressed-tensors | 70 (config) / 75 (scheme) | Part 4 |
| NVFP4 (NVIDIA re-releases, Granite) | ModelOpt / compressed-tensors | 75 | Parts 4, 5 |
| MXFP4 (gpt-oss, Granite) | `Mxfp4Config` | 80 | `mxfp4.py:67-77` |
| GGUF | moved out of official main (vllm-gguf-plugin); fork's GGUF needs 60 | — | Parts 4, 7; `vllm37:gguf.py:43`; issues #73, #74 |

**So on K80 the format today is BF16 safetensors, loaded as FP32 (or FP16 storage, if the fork's kernels allow it; see 1.1).** The refusal is about kernels, not the formats themselves: Part 9 gives the exact instruction each kernel needs and a measured K80 int4 workaround.

### 1.5 Ranking by porting cost

1. **No model work; shared fork runtime only:** Qwen3 dense, Granite 4.1/4.2 dense (and the all-attention Granite 4.0 models), Hunyuan dense, ERNIE 4.5 0.3B. Gemma 3 too, but capped at its 512/1024-token sliding window.
2. **Small model-file work:** SmolLM3 (NoPE flag), Ministral 3 BF16 (registry + long-context scaling), Olmo 3 (from `olmo2.py`; the same XFORMERS cap limits it to 4096 tokens).
3. **New kernels (FP32, non-Triton):** LFM2.5 (short conv: torch code exists), Qwen3.5 family (Gated DeltaNet), Granite-h and Nemotron (Mamba-2 SSD scan), Gemma 4 (head_dim 512, KV sharing).
4. **Out of scope on 4× 12 GB:** gpt-oss-20b, ERNIE 21B-A3B, Nemotron 30B-A3B, Gemma 4 26B/31B, anything above ~8B in FP32.

### 1.6 Open items across reports

- **FP16 storage in the fork:** cuBLAS works [measured]. Still unverified: the fork's custom kernels, xformers `cutlassF` (the fork targets its FP32 path), and FP16 overflow for Gemma (official blocks FP16 for gemma3 at `vllm/config/model.py:2369-2374`) and possibly Granite's muP multipliers.
- **NCCL:** whether a Kepler-capable NCCL suits TP beyond what the fork already runs (TP=2 tested over SHM).
- **transformers version in the fork image** for newer configs (`ministral3`, `olmo3`, `smollm3`, `gemma4`): the fork pins `transformers >= 4.55`; the builder image was not inspected here.
- Per-report UNVERIFIED items are listed in each part.

## Parts

2. Shared runtime (official vLLM)
3. Qwen3 dense; Qwen3.5 / 3.6 / 3.8
4. Gemma 4; Gemma 3
5. Granite 4.x dense; Granite 4.0 hybrid; Nemotron 3 / Nemotron-H
6. Ministral 3 / Magistral / Devstral Small 2; Olmo 3
7. LFM2 / LFM2.5; gpt-oss
8. Hunyuan dense; ERNIE 4.5; SmolLM3
9. Why quantized kernels fail on K80, and the software workaround (with measurements)


## 2. Shared runtime (official vLLM 3ca00a8261)


Scope: plain dense decoder (Qwen3ForCausalLM, `vllm/model_executor/models/qwen3.py`) served with `vllm serve` on CUDA. All paths are relative to `/home/jack/src/vllm`. "K80" means GK210, sm_37, driver 470.256.02 (CUDA runtime 11.4 max), toolkit 11.4.4, GCC 10.5. These local facts were checked on this host: `nvidia-smi -L` lists 4x "Tesla K80". `/usr/local/cuda-11.4/bin/nvcc --help` offers only `c++03/11/14/17`. `gcc --version` reports 10.5.0. Torch is not installed on this host.

### Call path

1. CLI: `vllm/entrypoints/cli/serve.py:156` `uvloop.run(run_server(args))`, then `vllm/entrypoints/launchers/api_server/entry.py:163` `run_server` and `:70` `build_async_engine_client_from_engine_args`, which builds `AsyncLLM` (`entry.py:85`).
2. Platform detection: `vllm/platforms/__init__.py:57-105` `cuda_platform_plugin()` uses NVML `nvmlDeviceGetCount() > 0` and returns `vllm.platforms.cuda.CudaPlatform`. Importing `vllm/platforms/cuda.py` runs `import vllm._C_stable_libtorch` unconditionally at module top (`cuda.py:23`).
3. Config and dtype resolution: `vllm/config/model.py:2394-2435` `_resolve_auto_dtype` uses `current_platform.supported_dtypes` (`cuda.py:255-265`). Compilation defaults are set at `vllm/config/vllm.py:1827-1847`.
4. Engine (V1 only): `vllm/v1/engine/async_llm.py:85` `AsyncLLM`, `:185` `EngineCoreClient.make_async_mp_client`, `vllm/v1/engine/core.py:111` `EngineCore`, `:140` `executor_class(vllm_config)`, `:255` `_initialize_kv_caches`, `:630` `step`. The executor is chosen at `vllm/v1/executor/abstract.py:65` `Executor.get_class` (mp/uni/ray).
5. Worker: `vllm/v1/worker/gpu_worker.py:188` `Worker`, `:419` `init_device` (`:486` `check_if_supports_dtype`, `:492` `init_worker_distributed_environment`), `:544` `_make_model_runner` (V2 runner `vllm/v1/worker/gpu/model_runner.py:188` or V1 runner `vllm/v1/worker/gpu_model_runner.py`), `:568` `load_model`, `:601` `determine_available_memory`, `:870` `compile_or_warm_up_model`, `:1264` `execute_model`.
6. Attention backend selection: `vllm/v1/attention/selector.py:105` `get_attn_backend`, `:215` `current_platform.get_attn_backend_cls`, `vllm/platforms/cuda.py:438-536` `get_attn_backend_cls`, `:375-415` `get_valid_backends`, `:83-179` `_get_backend_priorities`, `vllm/v1/attention/backend.py:300-337` `validate_configuration`.
7. Model forward (Qwen3): RMSNorm `vllm/model_executor/layers/layernorm.py:96-114`, which goes to IR op `rms_norm` with impls `vllm_c` (`vllm/kernels/vllm_c.py:23-43` -> `torch.ops._C.rms_norm`) or `native`. RoPE `vllm/model_executor/layers/rotary_embedding/base.py:221-250` -> `ops.rotary_embedding` (custom C). SiluAndMul `vllm/model_executor/layers/activation.py:138-146` -> `torch.ops._C.silu_and_mul`. Linear layers go through `vllm/model_executor/layers/utils.py:616-628` `dispatch_unquantized_gemm`, which picks `default_unquantized_gemm` (torch / cuBLAS) on CUDA. Attention uses `vllm/v1/attention/backends/triton_attn.py:581` `forward`, `:699` `unified_attention` (Triton) and `:834` `triton_reshape_and_cache_flash` (Triton).
8. Per-step input prep: `vllm/v1/worker/block_table.py:201-228` `compute_slot_mapping` calls `_COMPUTE_SLOT_MAPPING_KERNEL` (`@triton.jit`, `:398-548`). `vllm/v1/worker/utils.py:59` `_zero_kv_blocks_kernel` is `@triton.jit`.
9. Sampling: V1 runner `vllm/v1/worker/gpu_model_runner.py:569` `Sampler(...)` -> `vllm/v1/sample/ops/topk_topp_sampler.py:205-260` (FlashInfer / Triton / native). V2 runner `vllm/v1/worker/gpu/sample/sampler.py:346` `gumbel_sample` (`vllm/v1/worker/gpu/sample/gumbel.py:21+`, `@triton.jit`).
10. TP communication: `vllm/distributed/device_communicators/cuda_communicator.py:59-160` (PyNccl, custom all-reduce, symm-mem, FlashInfer AR).

### Requirements

| # | Stage | Requirement | Minimum needed | K80 (sm_37, CUDA 11.4) | Evidence | Fallback in code | Fallback works on K80? |
|---|---|---|---|---|---|---|---|
| R1 | Build | Host C++ compiler | GCC >= 11.3 (FATAL_ERROR otherwise) | GCC 10.5: **fails** | `CMakeLists.txt:22-28` | None (could use another GCC) | Yes if GCC >= 11.3 is installed (not a GPU limit) |
| R2 | Build | C++20 for CUDA sources | nvcc with `-std=c++20` (CUDA 12.x+) | nvcc 11.4 supports up to c++17: **fails** | `CMakeLists.txt:17-18` `set(CMAKE_CUDA_STANDARD 20)` / `CMAKE_CUDA_STANDARD_REQUIRED ON`; local `nvcc --help` lists only c++03/11/14/17 | None | No |
| R3 | Build | CUDA arch list for `_C_stable_libtorch` | >= 7.0 (toolkit < 12.8), >= 7.5 (toolkit >= 12.8) | 3.7 is filtered out. If 3.7 is the only arch, the build stops with FATAL_ERROR "No supported CUDA architectures" | `CMakeLists.txt:116-132` (`CUDA_SUPPORTED_ARCHS` "7.0;7.5;8.0;8.6;8.7;8.9;9.0" for toolkit < 12.8), `:231-239` (intersection plus FATAL_ERROR) | None | No |
| R4 | Build | Torch stable C ABI (`STABLE_TORCH_LIBRARY`, `torch/csrc/stable/*`, `torch/headeronly/*`) | torch >= 2.11 headers (`TORCH_TARGET_VERSION=0x020B...`), expected 2.13.0 | No torch >= 2.11 build targets sm_37 or CUDA 11.x (see R13) | `CMakeLists.txt:1232-1237`; `csrc/libtorch_stable/torch_bindings.cpp:5`; `csrc/libtorch_stable/torch_utils.h:4-7`; `CMakeLists.txt:71,192-195` (2.13.0 expected, warning only) | None (the legacy `_C` target is HIP-only: `CMakeLists.txt:367-370`) | No |
| R5 | Runtime import | `vllm._C_stable_libtorch` must import | Built extension | Blocked by R2/R3/R4 | `vllm/platforms/cuda.py:23` (unconditional `import vllm._C_stable_libtorch`) | `import_kernels` swallows ImportError (`cuda.py:237-245`), but the module-level import at `:23` does not | No |
| R6 | Core C kernels (FP16) | `__hmul2` / `__hadd` on `half` | sm_53 (CUDA 11.4 `cuda_fp16.hpp:1626-2397` `#if __CUDA_ARCH__ >= 530`; `__hmul2` at :1817, `__hadd` at :1857) | Not available on sm_37 | `csrc/libtorch_stable/cuda_vec_utils.cuh:350-358` `packed_mul` -> `__hmul2` (used by `activation_kernels.cu:390` `act_and_mul_kernel_with_param`, vec path); `csrc/custom_collective_common.cuh:105-108` `assign_add(half&)` -> `__hadd` with no arch guard | Scalar fallback in the activation kernel (`activation_kernels.cu:396-400`) is chosen at runtime, but the vec path is still instantiated, so compilation fails | No (would need source patches) |
| R7 | Core C kernels (BF16) | bf16 types/intrinsics | sm_80 | n/a | `csrc/libtorch_stable/type_convert.cuh:72-94` (guarded `__CUDA_ARCH__ >= 800`); `csrc/custom_collective_common.cuh:111-121` (guarded) | Guards exist | Guarded paths compile (not a blocker by itself) |
| R8 | Core C kernels (FP16 converter) | `_typeConvert<Half>` | CUDA_VERSION >= 12000 | With 11.4 the half specialization is disabled, so rms_norm uses the generic (non-vectorized) kernel | `csrc/libtorch_stable/type_convert.cuh:50-51`; `layernorm_kernels.cu:107,180` (`enable_if` on `_typeConvert<scalar_t>::exists`) | Generic kernel `layernorm_kernels.cu:179-180` | Would compile but is moot (R2-R4) |
| R9 | Core C kernels (cp.async) | cp.async | sm_80 | n/a | `csrc/libtorch_stable/async_util.cuh:26-31,53-68,87` | Guarded `#elif defined(__CUDA_ARCH__)` fallback | Yes (compiles) |
| R10 | Dtype policy | BF16 | CC >= 8.0 | Rejected | `vllm/platforms/cuda.py:656-675` `check_if_supports_dtype` (ValueError), called at `gpu_worker.py:486` | `--dtype half` / float32 | Yes (policy only) |
| R11 | Dtype policy | Auto dtype | CC >= 6.0 for fp16 in `supported_dtypes` | sm_37 gets `[torch.float32]` only ("Kepler and Maxwell ... only FP32 ... though vLLM doesn't support these GPUs") | `vllm/platforms/cuda.py:255-265`; `vllm/config/model.py:2400-2435` (auto dtype becomes fp32) | Explicit `--dtype half` is not blocked (`model.py:2467-2470`) | fp32 is the natural choice. fp16 math is emulated through float in torch, but vLLM's C kernels need sm_53 (R6) |
| R12 | Engine | V1 is the only engine | n/a | n/a | `vllm/engine/llm_engine.py:4-6` (`LLMEngine = V1LLMEngine`); `vllm/engine/async_llm_engine.py:4-6` (`AsyncLLMEngine = AsyncLLM`). No V0 code and no `VLLM_USE_V1` remain (grep finds nothing) | None (V0 removed) | n/a |
| R13 | PyTorch | torch==2.13.0 wheels | CUDA 12.6 / 13.0 (stable), 13.2 (experimental). cu130 archs: Turing 7.5 and newer. The cu126 line lists Maxwell 5.0 and newer for 2.14 | No CUDA 11 build. CUDA 12+ cannot target sm_35/37. Driver 470 is below CUDA 12 (>= 525) and 13 (>= 580) | `requirements/cuda.txt` `torch==2.13.0`; lock `requirements/test/cuda.txt:1275` `torch==2.13.0+cu130`; https://raw.githubusercontent.com/pytorch/pytorch/main/RELEASE.md (2.13 row; CUDA 13.0 archs "Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0+PTX)"); CUDA 12 drops SM35/SM37: https://docs.nvidia.com/cuda/archive/12.6.2/cufft/deprecated-functionality.html; driver minimums: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html | None in vLLM. Last CUDA 11 PyTorch wheels are 2.7 (cu118) per RELEASE.md | No: an older torch lacks the stable ABI (R4) and the `is_torch_equal_or_newer("2.10"/"2.11")` paths |
| R14 | Engine/compile | torch.compile + Inductor (default `CompilationMode.VLLM_COMPILE`, backend inductor) | Inductor GPU codegen = Triton. Inductor raises `GPUTooOldForTriton` for CC < 7.0 | Fails | `vllm/config/vllm.py:1827-1831` (mode defaults to VLLM_COMPILE when opt level > O0); `vllm/platforms/interface.py:170` `simple_compile_backend = "inductor"`; message in https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/exc.py ("Triton only supports devices of CUDA Capability >= 7.0") | `--enforce-eager` sets `mode=NONE` and `cudagraph_mode=NONE` (`vllm/config/vllm.py:1746-1748`, `:2025-2029`). custom_ops then become "all" (`:1840-1847`). IR priority becomes `["vllm_c","native"]` (`vllm/platforms/cuda.py:730-738`) | Yes for the model graph. Not for the stray `@torch.compile(backend=inductor)` helpers (R15) |
| R15 | Sampler helpers | Unconditional `@torch.compile(backend="inductor")` | Inductor -> Triton (CC >= 7.0) | Fails when hit | `vllm/v1/sample/ops/logprobs.py:10` `batched_count_greater_than` (used by `vllm/v1/sample/sampler.py:221,351` only when logprobs are requested); `vllm/v1/sample/ops/topk_topp_sampler.py:452` `compiled_random_sample` (no callers found); `vllm/model_executor/layers/vocab_parallel_embedding.py:175` `get_masked_input_and_mask` (TP > 1 and not `use_fused_embedding`, `:519-537`) | Avoid logprobs. TP > 1 uses fused C embedding when enabled | Partial |
| R16 | CUDA graphs | Graph capture (default FULL_AND_PIECEWISE) | CUDA 10+ | Not the limiting factor | `vllm/platforms/cuda.py:617-619` `CUDAGraphWrapper`; `vllm/config/vllm.py:2025-2029` | `--enforce-eager` | Yes |
| R17 | Attention | A valid backend for the platform | See the Attention backends table. On sm_37 only TRITON_ATTN and FLEX_ATTENTION pass `validate_configuration` (fp32, kv `auto`) | Both need Triton codegen for the GPU, which fails (R19) | `vllm/platforms/cuda.py:170-179` (priority FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT); `:510-514` (ValueError if none is valid) | None. TORCH_SDPA is ViT-only (`vllm/v1/attention/backends/registry.py:70` "this tag is only used for ViT"). xformers is gone (only stray mentions in `vllm/config/multimodal.py`, `vllm/model_executor/models/exaone4_5.py`). The CUDA paged-attention v1/v2 kernels are gone from `csrc/` (no paged_attention sources; `csrc/attention/` has only dtype headers) | No |
| R18 | Model runner | Model Runner V2 (default on CUDA when Triton is present) | Triton | Triton cannot compile for sm_37 | `vllm/config/vllm.py:719-778` (`if not HAS_TRITON: ... using the V1 model runner`); V2 Triton kernels such as `vllm/v1/worker/gpu/sample/gumbel.py:21`, `vllm/v1/worker/gpu/block_table.py` | `VLLM_USE_V2_MODEL_RUNNER=0` or no Triton -> V1 runner (`vllm/config/vllm.py:736-738,764-768`) | V1 runner still needs Triton (R19) |
| R19 | Triton (core, unconditional) | `@triton.jit` kernels on every step | Triton 3.7.1 (lock, via torch). Triton README: "NVIDIA GPUs (Compute Capability 8.0+)". Bundled ptxas 12.8.93 (CUDA 12 cannot target sm_37) | **Fails** | Pin: `requirements/test/cuda.txt:1351-1354` `triton==3.7.1 # via torch`. Unconditional use: `vllm/v1/worker/block_table.py:201-228,398-548` (slot mapping, V1 runner); `vllm/v1/worker/utils.py:59` (`_zero_kv_blocks_kernel`); attention `vllm/v1/attention/backends/triton_attn.py:37-42,699,834`. External: https://raw.githubusercontent.com/triton-lang/triton/release/3.7.x/README.md ; https://raw.githubusercontent.com/triton-lang/triton/release/3.7.x/cmake/nvidia-toolchain-version.json (`"ptxas": "12.8.93"`) | `HAS_TRITON` (`vllm/triton_utils/importing.py`) only checks that a Triton driver is active, not the arch. When false, `@triton.jit` becomes a placeholder decorator, but slot mapping and attention have no torch fallback | No |
| R20 | Sampling | Top-k/top-p | FlashInfer sampler: CC 8.0-12.1. Triton path: Triton | FlashInfer rejected. Triton fails | `vllm/v1/sample/ops/topk_topp_sampler.py:77-135` (`FlashInferBackend.supports_compute_capability`), `:216-226`, `:463-470` (`apply_top_k_top_p`: Triton if `HAS_TRITON`, else `apply_top_k_top_p_pytorch`); `:26-33` registers Triton warmups | `forward_native` plus `apply_top_k_top_p_pytorch` when `HAS_TRITON` is False. `VLLM_USE_FLASHINFER_SAMPLER=0` skips the FlashInfer import | Yes in isolation (pure torch), but only reachable with Triton absent, which breaks R19 |
| R21 | Sampling import | `flashinfer` Python import | flashinfer-python 0.7.0.post1 | Import should work (no GPU use) | `vllm/v1/sample/ops/topk_topp_sampler.py:101` imports `vllm.v1.attention.backends.flashinfer` (top-level `from flashinfer import ...`, `flashinfer.py:12-22`) | `VLLM_USE_FLASHINFER_SAMPLER=0` (`:95-99`) | Yes |
| R22 | Norm/RoPE/act | rms_norm, rotary, silu_and_mul | vllm_c kernels (R2-R6) or native torch | C kernels unavailable | `vllm/kernels/vllm_c.py:23-43`; `rotary_embedding/base.py:238-250`; `activation.py:138-146` | `forward_native` for each CustomOp (`custom_ops=["none"]` or `-rms_norm` etc.). Under inductor the native path is compiled (R14) | Native torch in eager mode would run, but CustomOp `forward_cuda` is chosen by default in eager mode |
| R23 | Distributed | NCCL via torch / PyNccl | NCCL from the torch cu13 wheel (`nvidia-nccl-cu13==2.29.7`) | Needs a CUDA 13 driver (>= 580). A CUDA 11 NCCL build would be needed | `requirements/test/cuda.txt:676`; `vllm/utils/nccl.py:20-32` (`VLLM_NCCL_SO_PATH` override); `vllm/platforms/cuda.py:621-650` | `VLLM_NCCL_SO_PATH` | UNVERIFIED (depends on a Kepler-capable NCCL plus matching torch) |
| R24 | Distributed | Custom all-reduce | World size in {2,4,6,8,16}. More than 2 GPUs require NVLink full connectivity. P2P test must pass | K80: PCIe only, so TP > 2 disables it. TP = 2 is allowed if P2P works | `vllm/distributed/device_communicators/custom_all_reduce.py:121,214-222` (sizes), `:262-276` ("not supported on more than two PCIe-only GPUs"), `:277-291` (P2P check); `vllm/platforms/cuda.py:828-849` `is_fully_connected` (NVML NVLink caps). Kernel has a pre-sm70 flag path (`csrc/custom_collective_common.cuh:163-192`) but unguarded `__hadd(half)` (`:105-108`) | Falls back to PyNccl automatically; `disable_custom_all_reduce` | Yes (fallback = NCCL, see R23) |
| R25 | Platform | NVML | `nvmlDeviceGetCudaComputeCapability`, NVLink P2P status | Works with driver 470 | `vllm/platforms/cuda.py:781-790`; `vllm/platforms/__init__.py:57-105` | n/a | Yes |
| R26 | Platform | FP8 | CC >= 8.9 | No | `vllm/platforms/cuda.py:605-606`; Triton attention rejects fp8 KV below SM89 (`triton_attn.py:519-532`) | Do not use fp8 | Yes |
| R27 | Build | vllm-flash-attn (FA2 `_vllm_fa2_C`) is built for CUDA | FA2 runtime gate CC >= 8.0. Python requirement skipped unless CUDA 12 | Not usable | `setup.py:1352` (always adds `_vllm_fa2_C`); `setup.py:1340-1343` ("vllm-flash-attn is built only for CUDA 12.x"); `vllm/v1/attention/backends/flash_attn.py:451-452` | Not selected below 8.0 | n/a |

### Attention backends

The CUDA, non-MLA decoder priority list is in `vllm/platforms/cuda.py:158-179`. The selector tries each entry in order and validates it against head size, dtype, kv dtype, block size and compute capability (`vllm/v1/attention/backend.py:300-337`). It keeps the valid backend with the lowest index and raises ValueError if none is valid (`cuda.py:510-514`). `--attention-backend` forces one and errors if it is invalid (`cuda.py:447-470`).

| Backend | File | Min CC | Dtypes | Needs Triton? | Evidence |
|---|---|---|---|---|---|
| FLASH_ATTN (vllm-flash-attn FA2/3/4) | `vllm/v1/attention/backends/flash_attn.py` | 8.0 | fp16, bf16 (no fp32) | No (CUDA/CuTe kernels) | `flash_attn.py:287` dtypes; `:451-452` `return capability >= DeviceCapability(8, 0)` |
| FLASHINFER | `vllm/v1/attention/backends/flashinfer.py` | 8.0 (FlashInfer supports SM75, but vLLM raised the floor because of a bug); max 12.1 | fp16, bf16 | Yes, some helpers (`flashinfer.py:45` imports triton) plus FlashInfer JIT | `flashinfer.py:417` dtypes; `:514-522` |
| TRITON_ATTN | `vllm/v1/attention/backends/triton_attn.py` | None declared (`return True`) | fp16, bf16, fp32 | **Yes** (unified_attention, reshape_and_cache are Triton) | `triton_attn.py:325-329` dtypes; `:413-414`; `:37-42` imports; bf16 KV requires SM80 (`:534-539`); fp8 KV requires SM89 (`:519-532`) |
| FLEX_ATTENTION | `vllm/v1/attention/backends/flex_attention.py` | None declared (base default `True`, `backend.py:259-260`) | fp16, bf16, fp32 | **Yes** (`torch.compile(flex_attention)` goes to Inductor, then Triton) | `flex_attention.py:55-58` (`torch.compile(create_block_mask...)`, `torch.compile(flex_attention, fullgraph=True)`); `:91-95` dtypes |
| TURBOQUANT | `vllm/v1/attention/backends/turboquant_attn.py` | (default) | fp16, bf16; only `turboquant_*` KV dtypes | Yes | `turboquant_attn.py:34,66,129-137` |
| TRITON_FLASH_ATTN (composite) | `vllm/v1/attention/backends/triton_flash_attn.py` | Needs FA v3/v4 (SM90+) | as FA | Yes | `triton_flash_attn.py:47-50`; only added for `major == 9` with mm-prefix (`cuda.py:172-175`) |
| TRITON_FLASHINFER (composite) | `vllm/v1/attention/backends/triton_flashinfer.py` | as FlashInfer | as FlashInfer | Yes | only added for `major == 10` with mm-prefix (`cuda.py:160`) |
| TORCH_SDPA | (enum tag, empty path) | n/a | n/a | No | `registry.py:70` "this tag is only used for ViT". Used only by `get_vit_attn_backend` (`cuda.py:539-593`). **Not selectable for decoder attention** |
| xformers | removed | n/a | n/a | n/a | No backend file. `grep -ril xformers vllm` finds only `vllm/config/multimodal.py` and `vllm/model_executor/models/exaone4_5.py` |
| Legacy CUDA paged attention (paged_attention_v1/v2) | removed | n/a | n/a | n/a | No `paged_attention*` sources in `csrc/`. `vllm/v1/attention/ops/paged_attn.py` only splits and writes the cache via `_custom_ops` |

**No decoder attention backend for CUDA both avoids Triton and accepts CC < 7.0.** The two backends that pass the capability gate on sm_37 (TRITON_ATTN and FLEX_ATTENTION) both generate Triton code at runtime.

### Hard blockers for K80

- **Triton cannot target sm_37, and vLLM uses Triton on every step with no fallback.** Pinned Triton 3.7.1 (`requirements/test/cuda.txt:1351`) supports "NVIDIA GPUs (Compute Capability 8.0+)" and bundles ptxas 12.8.93 (https://raw.githubusercontent.com/triton-lang/triton/release/3.7.x/README.md, https://raw.githubusercontent.com/triton-lang/triton/release/3.7.x/cmake/nvidia-toolchain-version.json). CUDA 12 cannot target SM35/SM37 (https://docs.nvidia.com/cuda/archive/12.6.2/cufft/deprecated-functionality.html). Unconditional Triton call sites: slot mapping `vllm/v1/worker/block_table.py:201-228` (kernel at `:398-548`), `vllm/v1/worker/utils.py:59`, and the attention backends (below). Model Runner V2 is Triton-based as well (`vllm/config/vllm.py:764-768`).
- **No attention backend for sm_37 without Triton.** FLASH_ATTN and FLASHINFER require CC >= 8.0 (`flash_attn.py:451-452`, `flashinfer.py:514-522`). TRITON_ATTN and FLEX_ATTENTION are Triton or Inductor (`triton_attn.py:37-42`, `flex_attention.py:55-58`). TORCH_SDPA is ViT-only (`registry.py:70`). xformers and the CUDA paged-attention kernels are removed. If none is valid, the selector raises ValueError (`vllm/platforms/cuda.py:510-514`).
- **The C++/CUDA extension cannot be built for sm_37.** The arch list floor is 7.0 (7.5 with CUDA >= 12.8), and the build stops with FATAL_ERROR when no supported arch remains (`CMakeLists.txt:116-132,231-239`). `CMAKE_CUDA_STANDARD 20` (`CMakeLists.txt:17-18`) needs an nvcc that CUDA 11.4 is not (its nvcc tops out at c++17). GCC >= 11.3 is required (`CMakeLists.txt:22-28`; local GCC 10.5). Core kernels call `__hmul2`/`__hadd` on `half` with no arch guard (`csrc/libtorch_stable/cuda_vec_utils.cuh:350-358`, `csrc/custom_collective_common.cuh:105-108`); CUDA 11.4 defines these only for `__CUDA_ARCH__ >= 530` (`/usr/local/cuda-11.4/include/cuda_fp16.hpp:1626-2397`).
- **The extension is mandatory at import.** `vllm/platforms/cuda.py:23` does an unconditional `import vllm._C_stable_libtorch`.
- **PyTorch 2.13 plus the stable ABI: no Kepler/CUDA-11 path.** `_C_stable_libtorch` targets the torch >= 2.11 stable C ABI (`CMakeLists.txt:1232-1237`). Torch 2.11-2.13 wheels ship only CUDA 12.6/12.8/13.x (https://raw.githubusercontent.com/pytorch/pytorch/main/RELEASE.md). CUDA 12+ cannot target SM37, and CUDA 12/13 need driver >= 525/580 against the K80's 470 (https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The lock uses `torch==2.13.0+cu130` (`requirements/test/cuda.txt:1275`), whose arch list starts at Turing 7.5.
- **The default graph compilation (Inductor) refuses CC < 7.0.** Mode defaults to VLLM_COMPILE with inductor (`vllm/config/vllm.py:1827-1831`; `vllm/platforms/interface.py:170`). The `GPUTooOldForTriton` message is in https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/exc.py. `--enforce-eager` turns this off for the model (`vllm/config/vllm.py:1746-1748`), but the stray `@torch.compile` helpers remain (`vllm/v1/sample/ops/logprobs.py:10`, `vocab_parallel_embedding.py:175`). This one has a partial fallback, so it is the least hard of the blockers.

### Open questions

- UNVERIFIED: which of the last CUDA-11 PyTorch wheels (<= 2.7, cu118) still include sm_37 in `torch.cuda.get_arch_list()`. A PyTorch forum post states 1.13 binaries were built for CC 3.7-8.6 (https://discuss.pytorch.org/t/gpu-compute-capability-support-for-each-pytorch-version/62434). The 2.x cu117/cu118 arch lists were not confirmed. Searched: pytorch RELEASE.md and web search. Either way this is moot for official vLLM, which needs the torch >= 2.11 stable ABI (R4).
- UNVERIFIED: whether driver 470 can run a cu118 build via CUDA 11.x minor-version compatibility (doc says >= 450 for 11.x, https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). This matters only for a forked or old-torch path.
- UNVERIFIED: whether an NCCL build that supports sm_37 (CUDA 11.x) is compatible with the torch/vLLM combination for TP (R23). Only `VLLM_NCCL_SO_PATH` override was found.
- UNVERIFIED: whether `flashinfer` 0.7.0.post1 imports cleanly on a host whose GPUs are all sm_37. The import is reached via the sampler (`topk_topp_sampler.py:101`) unless `VLLM_USE_FLASHINFER_SAMPLER=0`. Not tested; torch is not installed on this host.
- Not checked in depth: per-kernel shared memory use (48 KB/block on sm_37). Kernels that opt into larger dynamic shared memory were found only in non-core files (`cooperative_topk.cu`, `topk.cu`, `dsv3_fused_a_gemm.cu`, `hisparse_kernels.cu`). `topk.cu` is in the core extension source list (`CMakeLists.txt:449`), but no core-path caller was identified for a dense model.
- Not checked: whether `torch.accelerator.get_memory_info` (`vllm/v1/worker/gpu_worker.py:279`) and other newer torch APIs set a floor above 2.11. The stable-ABI floor (2.11) already dominates.

## 3. Qwen3 dense and Qwen3.5 / 3.6 / 3.8 on Tesla K80 (sm_37)


Sources:
- Official vLLM: `/home/jack/src/vllm` (main 3ca00a8261). All `file:line` references below point there unless prefixed `vllm37:`.
- HF listings: `hf models ls --author Qwen --search {Qwen3-,Qwen3.5,Qwen3.6,Qwen3.8} --expand createdAt,downloads,safetensors,config --json`. These were saved under a local scratch directory (not kept).
- The `config.json` files came from `hf download`, one directory per repo, under a local scratch directory (not kept).
- "Released" is the listing's `created_at` field. "Params" is the listing's `safetensors.total`.

Facts shared by both families:
- **Dtype on K80.** `CudaPlatform.supported_dtypes` returns only `[torch.float32]` below CC 6.0 (`vllm/platforms/cuda.py:254-264`). `_resolve_auto_dtype` picks from that list (`vllm/config/model.py:2392-2416`), so BF16 checkpoints load as FP32 on K80. Weight memory is therefore params × 4 bytes.
- **Build.** The official CMake supported-arch floor is 7.0 (`CMakeLists.txt:117-131`), and the build pins torch 2.13.0 (`CMakeLists.txt:71`, `requirements/cuda.txt:7`). The shared runtime is covered by another agent.
- **CustomOp dispatch.**
  - An enabled CustomOp runs `forward_cuda` on CUDA (`vllm/model_executor/custom_op.py:183-205`). A disabled one runs `forward_native` (`:189-192`).
  - The default is `custom_ops=["none"]` under inductor and `["all"]` otherwise, for example with enforce_eager (`vllm/config/vllm.py:1840-1847`).
  - `maybe_compile` skips torch.compile when mode is NONE or backend is eager (`custom_op.py:218-228`).
  - So on K80 the run must be eager (Inductor emits Triton) **and** `custom_ops=["none"]` to force the PyTorch-native paths.
- **IR rms_norm.** In eager mode the priority is `["vllm_c","native"]`; under inductor it is `["native"]` (`vllm/platforms/cuda.py:730-751`). The `native` implementation is the pure-torch body at `vllm/ir/ops/layernorm.py:10-21`.
- **Triton absence.** If Triton is missing, `vllm/triton_utils/importing.py:96-120` installs a `TritonPlaceholder`. `@triton.jit` then becomes a no-op, so imports succeed but any Triton kernel launch fails at call time.

---

### Qwen3 dense

Scope: architecture `Qwen3ForCausalLM` (`model_type` `qwen3`). Qwen3-30B-A3B and Qwen3-235B-A22B are `Qwen3MoeForCausalLM` (`qwen3_moe`), a different architecture, so they are excluded.

#### Official releases

| Repo | Released | Params | Format | Quant config | Context | Layers / KV heads / head_dim | Vision |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B (+ -Base 2025-04-28) | 2025-04-27 | 0.75B (Base 0.60B) | BF16 safetensors (`torch_dtype` bfloat16) | none | 40960 | 28 / 8 / 128; tie_word_embeddings=true | no |
| Qwen/Qwen3-0.6B-FP8 | 2025-04-28 | 0.75B | FP8 | `quant_method=fp8, fmt=e4m3, activation_scheme=dynamic, weight_block_size=[128,128]` | 40960 | 28 / 8 / 128; tie=true | no |
| Qwen/Qwen3-0.6B-GPTQ-Int8 | 2025-05-08 | 0.60B | GPTQ (`torch_dtype` float16) | `quant_method=gptq, bits=8, group_size=128, desc_act=false, sym=true, checkpoint_format=gptq` | 40960 | 28 / 8 / 128; tie=true | no |
| Qwen/Qwen3-1.7B (+ -Base) | 2025-04-27 | 2.03B (Base 1.72B) | BF16 | none | 40960 | 28 / 8 / 128; tie=true | no |
| Qwen/Qwen3-1.7B-FP8 | 2025-04-28 | 2.03B | FP8 | `quant_method=fp8` (from listing config; file not downloaded) | 40960 | same as 1.7B | no |
| Qwen/Qwen3-1.7B-GPTQ-Int8 | 2025-05-08 | 1.72B | GPTQ, float16 | `gptq, bits=8, group_size=128, desc_act=false, sym=true` | 40960 | 28 / 8 / 128; tie=true | no |
| Qwen/Qwen3-4B (+ -Base) | 2025-04-27 | 4.02B | BF16 | none | 40960 | 36 / 8 / 128; tie=true | no |
| Qwen/Qwen3-4B-FP8 | 2025-04-28 | 4.41B | FP8 | `fp8` (listing) | 40960 | 36 / 8 / 128 | no |
| Qwen/Qwen3-4B-AWQ | 2025-05-05 | 4.02B | AWQ, float16 | `quant_method=awq, bits=4, group_size=128, version=gemm, zero_point=true` | 40960 | 36 / 8 / 128; tie=true | no |
| Qwen/Qwen3-4B-Instruct-2507 (+ Thinking-2507) | 2025-08-05 | 4.02B | BF16 | none | **262144** (rope_theta 5e6) | 36 / 8 / 128; tie=true | no |
| Qwen/Qwen3-4B-Instruct-2507-FP8 (+ Thinking-2507-FP8) | 2025-08-06 | 4.41B | FP8 | `fp8, fmt=e4m3, dynamic, weight_block_size=[128,128]`, `modules_to_not_convert` lists lm_head and norms | 262144 | 36 / 8 / 128 | no |
| Qwen/Qwen3-4B-SafeRL | 2025-09-30 | 4.41B | BF16 (listing) | none | not downloaded | qwen3 arch (listing) | no |
| Qwen/Qwen3-8B (+ -Base) | 2025-04-27 | 8.19B | BF16 | none | 40960 | 36 / 8 / 128; tie=false | no |
| Qwen/Qwen3-8B-FP8 / -AWQ | 2025-04-28 / 2025-05-03 | 8.19B | FP8 / AWQ | `fp8` / `awq` (listing quant_method; files not downloaded) | 40960 | as 8B | no |
| Qwen/Qwen3-14B (+ -Base) | 2025-04-27 | 14.77B | BF16 | none | 40960 | 40 / 8 / 128; tie=false | no |
| Qwen/Qwen3-14B-FP8 / -AWQ | 2025-04-28 / 2025-05-01 | 14.77B | FP8 / AWQ | `fp8` / `awq` (listing) | 40960 | as 14B | no |
| Qwen/Qwen3-32B | 2025-04-27 | 32.76B | BF16 | none | 40960 | 64 / 8 / 128; tie=false | no |
| Qwen/Qwen3-32B-FP8 | 2025-04-28 | 32.76B | FP8 | `fp8, fmt=e4m3, dynamic, weight_block_size=[128,128]` | 40960 | 64 / 8 / 128 | no |
| Qwen/Qwen3-32B-AWQ | 2025-05-01 | 32.76B | AWQ, float16 | `awq, bits=4, group_size=128, version=gemm, zero_point=true` | 40960 | 64 / 8 / 128 | no |
| Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}-GGUF | 2025-05-01..05 | n/a | GGUF. Qwen3-4B-GGUF files are Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0 (`hf models info`) | n/a (no config.json) | n/a | n/a | no |
| Qwen/Qwen3-{0.6B..32B}-MLX-{4bit,6bit,8bit,bf16} | 2025-05-23..06-12 | as base | MLX | Qwen3-4B-MLX-4bit: `{"group_size":128,"bits":4}`, with **no quant_method**; maxpos 65536 | 65536 (4B-MLX-4bit) | 36 / 8 / 128 | no |

The Qwen3 dense configs set no `rope_scaling` and no `rope_parameters`, only `rope_theta` (1e6, or 5e6 for 2507). YaRN is opt-in only.

#### Code path (official vLLM)

1. **Registry.** `"Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM")` at `vllm/model_executor/models/registry.py:194`.
2. **Top model.** `Qwen3ForCausalLM` is at `models/qwen3.py:272`. `Qwen3Model` (`qwen3.py:265`) subclasses `Qwen2Model` (`models/qwen2.py:324`).
3. **Embedding.**
   - Class: `VocabParallelEmbedding` (`qwen2.py:371`).
   - TP=1: `F.embedding` (`layers/vocab_parallel_embedding.py:84-85, 518-520`).
   - TP>1: optional fused `ops.vocab_parallel_embedding` (`:522-531`), otherwise a masked `F.embedding` (`:536-545`).
4. **Per layer** (`Qwen3DecoderLayer`, `qwen3.py:174`; forward at `:227-247`):
   1. **input_layernorm.** `RMSNorm` (`qwen3.py:222`; `layers/layernorm.py:37`). Calls `forward_cuda` → `forward_native` → `ir.ops.rms_norm` / `fused_add_rms_norm` (`layernorm.py:74-115`). In eager mode the IR picks `vllm_c` = `torch.ops._C.rms_norm` (`vllm/kernels/vllm_c.py:23-44`), with `native` as fallback (`vllm/ir/ops/layernorm.py:10`).
   2. **QKV projection.** `QKVParallelLinear` (`qwen3.py:105`) → `UnquantizedLinearMethod` → `default_unquantized_gemm` (torch linear/cuBLAS) (`layers/linear.py:165,175`; `layers/utils.py:616-627`).
   3. **QK-norm.** Per-head `RMSNorm` on q and k (`qwen3.py:151-152, 161-167`). Same op as the layernorm.
   4. **Rotary.**
      - `get_rope` (`qwen3.py:122`) builds `RotaryEmbedding`, the default type with full rotary_dim (`layers/rotary_embedding/__init__.py:110,145-152`).
      - `forward_cuda` → `ops.rotary_embedding` (`_C`, `rotary_embedding/base.py:221-252`). Native fallback: `base.py:203-219`.
      - YaRN, if the user enables it: `YaRNScalingRotaryEmbedding` (`__init__.py:240,270`) inherits the same forward.
   5. **Attention.**
      - `Attention` (`layers/attention/attention.py:229`). The backend comes from `get_attn_backend` (`attention.py:350`), and the forward goes through `unified_attention_with_output` (`:551-571`).
      - CUDA priority list: FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT (`platforms/cuda.py:158-178`).
   6. **O projection.** `RowParallelLinear` (cuBLAS).
   7. **post_attention_layernorm.** Fused add + RMSNorm.
   8. **MLP.** `Qwen2MLP` (`qwen3.py:58`; `qwen2.py:81-113`). It runs `MergedColumnParallelLinear` gate_up, then `SiluAndMul`, then `RowParallelLinear` down. `SiluAndMul.forward_cuda` calls `torch.ops._C.silu_and_mul`; native fallback at `layers/activation.py:141-144`.
5. **Final norm.** `RMSNorm` (`qwen2.py:395`).
6. **LM head.** `ParallelLMHead`, tied to `embed_tokens` when `tie_word_embeddings` (`qwen3.py:299-306`). `LogitsProcessor` (`qwen3.py:335`) is a matmul.
7. **Compilation.** `@support_torch_compile` on `Qwen3Model` (`qwen3.py:255-264`). On K80 it must be disabled (eager), see the shared facts above.

#### Requirements

| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Embedding | `F.embedding` (TP=1); `_C vocab_parallel_embedding` (TP>1 fused) | torch; `_C` for the fused path | torch's | OK if torch runs on sm_37 | vocab_parallel_embedding.py:84-85, 518-531 | masked F.embedding, vocab_parallel_embedding.py:536-545 | yes |
| RMSNorm (layer norms, final norm, q_norm/k_norm) | `torch.ops._C.rms_norm` / `fused_add_rms_norm` (`csrc/libtorch_stable/layernorm_kernels.cu`) | vLLM `_C` built for sm_37 | official build floor 7.0 (CMakeLists.txt:117-131) | blocked in the official build; the kernel source has no half2 intrinsics (grep count 0) | kernels/vllm_c.py:23-44; platforms/cuda.py:738 | `ir.ops.rms_norm` native body, vllm/ir/ops/layernorm.py:10-21. Select it with ir_op_priority=native or inductor mode | yes (pure torch, fp32) |
| Linear (QKV, O, gate_up, down, lm_head) | `default_unquantized_gemm` (torch matmul → cuBLAS) | torch + cuBLAS | torch's | OK in FP32 (cuBLAS SGEMM) | layers/utils.py:616-627 | same op | yes |
| Rotary (NeoX, full dim) | `_C rotary_embedding` (`pos_encoding_kernels.cu`) | `_C` | build floor 7.0 | blocked in the official build | rotary_embedding/base.py:221-252 | `RotaryEmbedding.forward_native`, base.py:203-219 | yes |
| SiLU·Mul | `_C silu_and_mul` | `_C` | build floor 7.0 | blocked in the official build | activation.py:116-150 | activation.py:141-144 | yes |
| Paged attention (decoder, head_dim 128, GQA 8 KV heads) | FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION | FA: CC ≥ 8.0; FlashInfer: 8.0..; Triton; Flex uses torch.compile → Triton | FA 80 (flash_attn.py:451-452), FlashInfer 80 (flashinfer.py:514-520), Triton backend reports "True" (triton_attn.py:413-414), Flex "True" (flex_attention.py:259-260) | **BLOCKER**: every CUDA choice needs FA, FlashInfer, or Triton | platforms/cuda.py:158-178 | none for the CUDA platform. `cpu_attn.py` is CPU-only | no |
| torch.compile (default) | Inductor → Triton | Triton | n/a | must be disabled | qwen3.py:255; config/vllm.py:1840-1847 | enforce_eager / `-O0` | yes |

#### Quantized format support

| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| FP8 block 128x128 (`quant_method=fp8`) | `Fp8Config` | 75 | quantization/fp8.py:146-147; capability gate raises at config/vllm.py:943-949 |
| GPTQ int8 (`quant_method=gptq`) | `"gptq"` → `AutoGPTQConfig` (`quantization/__init__.py:167-169`) | 60, and act dtypes are only half or bf16 | auto_gptq.py:177-183. Its CC≥60 kernel is Exllama, which needs fp16 activations (`kernels/linear/mixed_precision/exllama.py:24-25, 44-45`). The `gptq_gemm` CUDA kernel uses half2 intrinsics (`csrc/libtorch_stable/quantization/gptq/q_gemm.cu`, 20 matches of `__hfma2/__hmul2/__hadd2`), which need sm_53 |
| AWQ int4 gemm (`quant_method=awq`) | `"awq"` → `AutoAWQConfig` (`__init__.py:157-159`) | 75 | auto_awq.py:233-234 |
| GGUF | none: no `gguf` entry in `QuantizationMethods` (`quantization/__init__.py:14-48`) and no GGUF loader in `model_executor`/`config` (grep) | n/a | same |
| MLX 4/6/8-bit | none. The config has `{"group_size","bits"}` with no quant_method | n/a | Qwen3-4B-MLX-4bit config.json |

#### In the K80 fork (vllm37)
- Model file: present, `vllm37:vllm/model_executor/models/qwen3.py` (`Qwen3Attention` at :56, `Qwen3ForCausalLM` at :264).
- Registry: present, `vllm37:vllm/model_executor/models/registry.py:129` `"Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM")`.

#### Verdict for K80
Qwen3 dense has no model-specific blocker. Every model-level op (RMSNorm with QK-norm, full-dim NeoX RoPE, SiLU·Mul, plain linear, embedding) has a pure-PyTorch fallback in official vLLM, or a `_C` kernel without sm_53+ intrinsics.

Blockers, all shared runtime:
1. **Attention backend.** Official CUDA backends need FA (CC 8.0), FlashInfer (CC 8.0), or Triton (`platforms/cuda.py:158-178`). Fix: the fork's own sm_37 attention backend, already part of the fork's work.
2. **`_C` built for ≥ 7.0 and torch 2.13** (`CMakeLists.txt:71,117-131`). Fix: the fork's toolchain (CUDA 11.4, torch 2.0.1).
3. **FP32-only dtype below CC 6.0** (`platforms/cuda.py:262-264`). This is memory, not correctness:

   | Model | FP32 weights | Needs |
   |---|---|---|
   | 0.6B | ≈3.0 GB | fits one 12 GB K80 die |
   | 1.7B | ≈8.1 GB | fits one die |
   | 4B / 4B-2507 | ≈16.1 GB | TP=2 (8 KV heads, divisible) |
   | 8B | ≈32.8 GB | TP=4 |
   | 14B / 32B | ≈59 / 131 GB | impractical |

   Fix: allow FP16 storage with FP32 compute in the fork, if the fork supports it.
4. **All official quantized repos fail the capability gate on CC 37** (`config/vllm.py:943`): FP8 needs 75, AWQ 75, GPTQ 60 plus fp16 activations plus half2. GGUF and MLX have no vLLM method. Fix: BF16 checkpoints converted to FP32/FP16, or a new sm_37 weight-only int8/int4 dequant kernel.

The fork already has the model file and registry entry, so no model porting is needed for Qwen3 dense. The 4B-2507 checkpoints only change `rope_theta` and `max_position_embeddings`, so the same code path applies.

---

### Qwen3.5 / Qwen3.6 / Qwen3.8 (hybrid Gated DeltaNet + gated full attention)

Scope:
- `model_type` `qwen3_5` / `qwen3_5_moe`, architectures `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`.
- Qwen3.8-2.4T-A95B is text-only `Qwen3_5MoeForCausalLM` (`qwen3_5_moe_text`) and is out of the size range.
- Qwen3.8-Flash-Next (180B) is `qwen4_exp` / `Qwen4ExpForConditionalGeneration`, a different architecture, and is excluded.
- 122B-A10B and 397B-A17B are listed for completeness but exceed 40B.

#### Official releases

All rows below come from downloaded configs. All of them share these values:
- `max_position_embeddings=262144`.
- `rope_parameters={rope_type: default, rope_theta: 1e7, partial_rotary_factor: 0.25, mrope_section: [11,11,10], mrope_interleaved: true}`.
- `attn_output_gate=true`, `full_attention_interval=4`.
- Linear attention: `linear_key_head_dim=linear_value_head_dim=128`, `linear_conv_kernel_dim=4`, `mamba_ssm_dtype=float32`.
- `mtp_num_hidden_layers=1`, `torch_dtype=bfloat16`, and a vision tower (`vision_config` present).

| Repo | Released | Params | Format | Quant config | Context | Layers / KV heads / head_dim | Vision |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B (+ -Base) | 2026-02-28 | 0.87B | BF16 | none | 262144 | 24 (18 linear + 6 full) / 2 / 256; GDN heads k16/v16; tie=true | yes, depth 12 |
| Qwen/Qwen3.5-2B (+ -Base) | 2026-02-28 | 2.27B | BF16 | none | 262144 | 24 (18+6) / 2 / 256; GDN k16/v16; tie=true | yes, depth 24 |
| Qwen/Qwen3.5-4B (+ -Base 2026-02-27) | 2026-02-27 | 4.66B | BF16 | none | 262144 | 32 (24+8) / 4 / 256; GDN k16/v32; tie=true | yes, depth 24 |
| Qwen/Qwen3.5-9B (+ -Base 2026-02-26) | 2026-02-27 | 9.65B | BF16 | none | 262144 | 32 (24+8) / 4 / 256; GDN k16/v32; tie=false | yes, depth 27 |
| Qwen/Qwen3.5-27B | 2026-02-24 | 27.78B | BF16 | none | 262144 | 64 (48+16) / 4 / 256; GDN k16/v48; tie=false | yes, depth 27 |
| Qwen/Qwen3.5-27B-FP8 | 2026-02-25 | 27.78B | FP8 | `quant_method=fp8, activation_scheme=dynamic, weight_block_size=[128,128]`, `modules_to_not_convert` includes lm_head, embed_tokens, `linear_attn.conv1d`… | 262144 | as 27B | yes |
| Qwen/Qwen3.5-27B-GPTQ-Int4 | 2026-03-03 | 27.78B | GPTQ | `gptq, bits=4, group_size=128, desc_act=false, sym=true`, `dynamic={lm_head, embed_tokens, "-:.*attn.*", "-:.*shared_expert.*", "-:.*mtp.*", "-:.*visual.*"}`, so all attention and GDN layers stay BF16 | 262144 | as 27B | yes |
| Qwen/Qwen3.5-35B-A3B (+ -Base) | 2026-02-24 | 35.95B | BF16 | none | 262144 | 40 (30+10) / 2 / 256; GDN k16/v32; MoE 256 experts top-8, moe_inter 512, shared 512; tie=false | yes, depth 27 |
| Qwen/Qwen3.5-35B-A3B-FP8 | 2026-02-25 | 35.95B | FP8 | `fp8, dynamic, weight_block_size=[128,128]` | 262144 | as 35B-A3B | yes |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 2026-03-03 | 35.95B | GPTQ | `gptq, bits=4, group_size=128, sym=true, desc_act=false`, same `dynamic` exclusions as 27B | 262144 | as 35B-A3B | yes |
| Qwen/Qwen3.5-122B-A10B (+FP8, GPTQ-Int4) / Qwen3.5-397B-A17B (+FP8, GPTQ-Int4) | 2026-02-16..03-03 | 125B / 403B | BF16 / FP8 / GPTQ | listing quant_method fp8 / gptq | n/d | > 40B, not downloaded | yes (listing arch) |
| Qwen/Qwen3.6-27B | 2026-04-21 | 27.78B | BF16 | none | 262144 | 64 (48+16) / 4 / 256; GDN k16/v48 | yes, depth 27 |
| Qwen/Qwen3.6-27B-FP8 | 2026-04-21 | 27.78B | FP8 | `fp8, fmt=e4m3, dynamic, weight_block_size=[128,128]`, visual blocks not converted | 262144 | as 27B | yes |
| Qwen/Qwen3.6-35B-A3B (+FP8) | 2026-04-15 | 35.95B | BF16 / FP8 | none / `fp8, e4m3, dynamic, [128,128]` | 262144 | 40 (30+10) / 2 / 256; MoE 256 top-8 | yes |
| Qwen/Qwen3.8-27B | 2026-08-05 | 27.78B | BF16 | none | 262144 | 64 (48+16) / 4 / 256; GDN k16/v48 | yes, depth 27 |
| Qwen/Qwen3.8-27B-FP8 | 2026-08-13 | 27.78B | FP8 | `fp8, fmt=e4m3, dynamic, weight_block_size=[128,128]` | 262144 | as 27B | yes |

Format gaps: the `--search Qwen3.5/3.6/3.8` listings show no Qwen-published GGUF, AWQ, MLX, or NVFP4 repos. Qwen3.6 and Qwen3.8 have no GPTQ repos.

#### Code path (official vLLM)

1. **Registry.**
   - `Qwen3_5ForConditionalGeneration` → `qwen3_5` (`registry.py:588`); `Qwen3_5MoeForConditionalGeneration` (`registry.py:589-591`).
   - Text-only `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` (`registry.py:196-197`).
   - Config classes are vendored at `vllm/transformers_utils/configs/qwen3_5.py` and `qwen3_5_moe.py`.
2. **Top model.**
   - `Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid)` at `models/qwen3_5.py:470`.
   - It builds `self.visual = Qwen3_VisionTransformer` inside `_mark_tower_model` (`qwen3_5.py:510-517`) and `self.language_model = Qwen3_5ForCausalLM` (`:519-522`).
   - Its forward calls only `language_model.model` (`:589-596`).
   - The MTP head weights are dropped by the mapper (`"mtp.": None`, `qwen3_5.py:473-476`).
3. **Vision skip.**
   - `_mark_tower_model` replaces the tower with `StageMissingLayer` when `get_limit_per_prompt(m)==0` for both image and video (`models/interfaces.py:338-380`).
   - `--language-model-only` makes every limit 0 (`config/multimodal.py:304-306, 699-700`).
   - If the tower is used instead, `Qwen3_VisionTransformer` is at `qwen3_vl.py:559` (Conv3dLayer patch embed at `:390`). Its ViT attention can fall back to TORCH_SDPA (`platforms/cuda.py:592`).
4. **Language model.**
   - `Qwen3_5Model(Qwen3NextModel)` (`qwen3_5.py:220`). Embedding is `VocabParallelEmbedding` (`:248`).
   - Layers are built from `config.layer_types` (`:253-262`), with final norm `GemmaRMSNorm` (`:273`).
   - The model is `@support_torch_compile` (`:210`).
5. **Decoder layer.**
   - `Qwen3_5DecoderLayer` (`qwen3_5.py:123`); forward inherited from `Qwen3NextDecoderLayer.forward` (`models/qwen3_next.py:549-622`).
   - **Norms**: `GemmaRMSNorm` computes x·(1+w) (`qwen3_5.py:43,185-190`; `layers/layernorm.py:140-175`). `forward_cuda` → `forward_native` → `ir.ops.rms_norm(x, w.float()+1)`.
   - **linear_attention layers**: `QwenGatedDeltaNetAttention` (`qwen3_5.py:147-154`), see step 6.
   - **full_attention layers**: `Qwen3NextAttention` (`qwen3_5.py:155-163`; `qwen3_next.py:273`). It works in this order:
     1. `QKVParallelLinear` with `num_heads*(1+attn_output_gate)` query heads, so it emits q and a gate (`qwen3_next.py:309-317`).
     2. `get_rope` builds `MRotaryEmbedding`: `rope_type` is default, `mrope_section` is present, and rotary_dim = 256·0.25 = 64 (`rotary_embedding/__init__.py:68-71,110-121`).
     3. `Attention(head_dim=256, kv_heads 2 or 4)` (`qwen3_next.py:344-359`) and GemmaRMSNorm q_norm/k_norm (`:361-362`).
     4. Either the fused Triton `fused_qk_rmsnorm_rope_gate` (`layers/fused_qk_norm_rope.py:16,138`) or the eager split → q/k norm → rotary (`qwen3_next.py:427-447`). The fused path is chosen only on CUDA/XPU **and** when the rotary dtype is fp16/bf16 (`qwen3_next.py:375-385`), so on K80 (FP32) it picks eager.
     5. Then `attn`, then `attn_output * sigmoid(gate)` (`:456-458`), then o_proj.
   - **Rotary on the eager path**: `MRotaryEmbedding.forward_cuda` takes 2-D positions → `triton_mrope` (`rotary_embedding/mrope.py:437-467`; kernel at `:16`).
   - **MLP (dense)**: `Qwen2MoeMLP` (gate_up, SiluAndMul, down) (`qwen3_5.py:174-181`).
   - **MLP (MoE)**: `Qwen3NextSparseMoeBlock` (`qwen3_5.py:169-173`; `qwen3_next.py:131`). It holds `GateLinear` router, a shared expert with `shared_expert_gate`, and `FusedMoEFactory` (`qwen3_next.py:186-235`). Unquantized CUDA backends: FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON (`fused_moe/oracle/unquantized.py:68-74`).
6. **Gated DeltaNet** (`layers/mamba/gdn/qwen_gdn_linear_attn.py`).
   1. **Dispatch.** `forward_cuda` on CUDA, `forward_cpu` on CPU, `forward_hip` on ROCm (`:405-416`).
   2. **Input projections.** `in_proj_qkvz` and `in_proj_ba` linears (`:432-447`, cuBLAS). The split happens at `:944-951`.
   3. **Core.** The custom op `torch.ops.vllm.qwen_gdn_attention_core` (`:964-970`, registered at `:1940-1944`) calls `_forward_core` (`:1250`).
   4. **Default decode.** `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` (`envs.py:132`) routes pure-decode batches to `_forward_core_decode_non_spec` (`:1278-1290, 1637-1687`). That function runs `causal_conv1d_update` (Triton, `ops/causal_conv1d.py:762,1096`) and then `fused_recurrent_gated_delta_rule_packed_decode` (Triton, `third_party/flash_linear_attention/ops/fused_recurrent.py:256,343`).
   5. **Prefill.**
      - `causal_conv1d_fn` (Triton, `causal_conv1d.py:16,481`; call at `qwen_gdn_linear_attn.py:1361`).
      - `fused_post_conv_prep`: l2norm and g/beta gating, Triton (`fused_gdn_prefill_post_conv.py:20,152`; call at `:1424`).
      - `ChunkGatedDeltaRule` (`:243-263`) dispatches the chunk kernel:
        - FlashInfer only for CC 9.0, the 10.x family, or the 12.x family (`:118-149`, `fi_chunk_gated_delta_rule` at `:190`).
        - CuteDSL opt-in on 10.x (`:327-370`).
        - Otherwise `forward_native`, which is **FLA Triton** `chunk_gated_delta_rule` (`:297-325` → `third_party/flash_linear_attention/ops/chunk.py:138`). That kernel runs cumsum, `chunk_scaled_dot_kkt`, `solve_tril`, `recompute_w_u`, `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o`, all Triton with `tl.dot`.
   6. **Mixed batches.** Decode tokens in a mixed batch use `fused_sigmoid_gating_delta_rule_update` (Triton, `fused_sigmoid_gating.py:24,181`; calls at `:1453, 1480, 1538`).
   7. **Fused CUDA decode kernel.** `fused_gdn_decode_post_conv_mtp` is gated to BF16, K=V=128, and CC ≥ 8.0 (`qwen_gdn_linear_attn.py:545-567`). It is built only for 8.0+ (`CMakeLists.txt:1129-1130`).
   8. **Output.** `RMSNormGated` (`:490-497`; `layernorm.py:289`). `forward_cuda` → FLA Triton `rmsnorm_fn` (`layernorm.py:404-420`; `layernorm_guard.py:74,562`). Then `out_proj` (`qwen_gdn_linear_attn.py:850-862`).
7. **State cache.**
   - Model hooks: `IsHybrid` / `HasInnerState` (`qwen3_5.py:295-303`), with `get_mamba_state_{dtype,shape,copy_func}_from_config` (`qwen3_5.py:384-421, 605-640`).
   - Shape comes from `MambaStateShapeCalculator.gated_delta_net_state_shape` (`mamba_utils.py:280`): a conv state plus a recurrent state per layer.
   - Dtype comes from `gated_delta_net_state_dtype` (`mamba_utils.py:122`). The recurrent dtype is taken from the HF `mamba_ssm_dtype="float32"` (`models/config.py:925-948`).
   - v1 objects: `MambaSpec` (`v1/kv_cache_interface.py:1050`), `MambaManager` (`v1/core/single_type_kv_cache_manager.py:1455`), and the `GDNAttentionBackend` / metadata builder (`v1/attention/backends/gdn_attn.py:31,89`). The builder is plain torch; `prepare_chunk_indices` (`ops/index.py`) has no Triton kernels.
   - Block size is aligned to the mamba page size by `HybridAttentionMambaModelConfig` (`models/config.py:579-593`).
8. **Text-only variant.** `Qwen3_5ForCausalLMConfig` strips `mrope_section` / `mrope_interleaved` (`models/config.py:952-965`), so text-only checkpoints use plain partial NeoX `RotaryEmbedding`. For text tokens, `Qwen3_5ForCausalLMBase.get_mrope_input_positions` returns three identical rows (`qwen3_5.py:439-445`), so MRoPE reduces to plain partial RoPE.

#### Requirements

| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Embedding / LM head / all dense linears (in_proj_qkvz, in_proj_ba, qkv, o, out_proj, MLP) | torch F.embedding / cuBLAS gemm | torch | torch's | OK (FP32) | vocab_parallel_embedding.py:84; layers/utils.py:616-627 | n/a | yes |
| GemmaRMSNorm (layer norms, final norm, q_norm/k_norm) | `ir.ops.rms_norm` → `_C rms_norm` (eager) | `_C` | build floor 7.0 | blocked in the official build | layernorm.py:159-175; kernels/vllm_c.py:23 | IR native, vllm/ir/ops/layernorm.py:10-21 | yes |
| Fused QK-norm + RoPE + gate split | Triton `_fused_qk_rmsnorm_rope_gate_kernel` | Triton; fp16/bf16 rotary | Triton | not selected in FP32 | fused_qk_norm_rope.py:16,138; selection qwen3_next.py:375-385 | eager split/norm/rope, qwen3_next.py:427-447 | yes |
| MRoPE (interleaved, partial 64/256) | Triton `triton_mrope` for 2-D positions | Triton | Triton | **blocked** | rotary_embedding/mrope.py:16,437-467 | `MRotaryEmbedding.forward_native`, mrope.py:374-436 (pure torch, handles interleaved). Text-only: plain RoPE via config.py:952-965 | yes (with `custom_ops=["none"]`) |
| Partial NeoX RoPE (text-only path) | `_C rotary_embedding` | `_C` | 7.0 build | blocked in the official build | base.py:221-252 | base.py:203-219 | yes |
| Full attention, head_dim **256**, GQA 2 or 4 KV heads | FA / FlashInfer / TRITON_ATTN / FLEX | FA 8.0, FlashInfer 8.0, Triton | see Qwen3 dense | **BLOCKER** (shared runtime). The fork's backend must also support head_dim 256 | platforms/cuda.py:158-178 | none on CUDA | no |
| Output gate `attn*sigmoid(gate)` | torch | torch | n/a | OK | qwen3_next.py:457-458 | n/a | yes |
| causal_conv1d (prefill `_fn`, decode `_update`, with cache-line indices) | Triton `_causal_conv1d_fwd_kernel`, `_causal_conv1d_update_kernel` | Triton | Triton | **BLOCKER** | ops/causal_conv1d.py:12,16,762 | `causal_conv1d_update_torch` (pure torch F.conv1d), ops/cpu/causal_conv1d.py:119-148, decode only. It takes a dense `conv_state`, not slot indices, and is not wired into the CUDA path. CPU path `_causal_conv1d_fwd_cpu` uses `ops.causal_conv1d_fwd_cpu` (CPU C++), ops/cpu/gdn_attention.py:48-125 | partial. The decode math is portable; prefill and the indexed state need new glue |
| GDN prefill post-conv prep (l2norm, g = -exp(A_log)·softplus(a+dt_bias), beta = sigmoid(b)) | Triton `_fused_post_conv_kernel` | Triton | Triton | **BLOCKER** | fused_gdn_prefill_post_conv.py:20,152 | no runtime fallback. Test refs `ref_l2norm` / `ref_gdn_gating` at tests/kernels/mamba/cpu/test_cpu_gdn_ops.py:92-110 | the refs are pure torch, so yes once ported |
| GDN chunked prefill (`chunk_gated_delta_rule`) | FLA Triton (default on CC < 9.0); FlashInfer (CC 9.0 / 10.x / 12.x); CuteDSL (10.x opt-in) | Triton or FlashInfer or CuteDSL | FlashInfer 90 (qwen_gdn_linear_attn.py:128-143) | **BLOCKER** | qwen_gdn_linear_attn.py:93-149, 243-325; FLA ops/chunk.py:138 (Triton kernels with `tl.dot`: chunk_delta_h 8, solve_tril 18, chunk_o 3, wy_fast 2, chunk_scaled_dot_kkt 3) | none on GPU. `forward_native` is the FLA Triton path despite its name (:297-325). CPU path `ops.chunk_gated_delta_rule_cpu` is CPU C++ (ops/cpu/gdn_attention.py:339). Pure-torch token loop exists only in tests: `ref_gated_delta_rule` tests/kernels/mamba/cpu/test_cpu_gdn_ops.py:113 | the test reference is pure torch and would run (slowly) |
| GDN decode recurrent update | Triton `fused_recurrent_gated_delta_rule_packed_decode` (default) / `fused_sigmoid_gating_delta_rule_update` | Triton | Triton | **BLOCKER** | fused_recurrent.py:256,343; fused_sigmoid_gating.py:24,181; envs.py:132 | CUDA `fused_gdn_decode_post_conv_mtp` needs BF16 and CC 8.0 (qwen_gdn_linear_attn.py:545-567; CMakeLists.txt:1129-1130); CPU C++ `fused_sigmoid_gating_delta_rule_update_cpu` (ops/cpu/gdn_attention.py:268) | no runtime fallback. The per-token reference loop is pure torch |
| RMSNormGated (norm_before_gate, SiLU gate) | FLA Triton `rmsnorm_fn` | Triton | Triton | blocked when enabled | layernorm.py:404-420 | `RMSNormGated.forward_native` / `forward_static`, layernorm.py:340-401 (pure torch) | yes (`custom_ops=["none"]` or `-rms_norm_gated`) |
| Dense MLP SiLU·Mul | `_C silu_and_mul` | `_C` | 7.0 build | blocked in the official build | activation.py:116-150 | activation.py:141-144 | yes |
| MoE experts (35B-A3B family) | FlashInfer TRTLLM / CUTLASS, TritonExperts, BatchedTritonExperts | FlashInfer or Triton | FlashInfer ≥ 8.0 | **BLOCKER**; FP32 weights ≈144 GB, so it does not fit anyway | fused_moe/oracle/unquantized.py:68-74,134-148 | CPU backend only on CPU platform (:104-105). `naive_batched_moe` lives only in tests (tests/kernels/moe/utils.py:175) | no |
| GDN state cache (conv + float32 recurrent) | v1 `MambaSpec` / `MambaManager` / `GDNAttentionMetadataBuilder` | v1 engine | n/a | the fork is v0 | kv_cache_interface.py:1050; single_type_kv_cache_manager.py:1455; gdn_attn.py:31-231 | the fork has a v0 `MambaCacheManager` (vllm37:vllm/model_executor/models/mamba_cache.py:25) | yes as a porting base |
| Vision tower (Conv3d + ViT) | MMEncoderAttention (FA or TORCH_SDPA) | torch SDPA fallback | n/a | skippable | qwen3_vl.py:559; platforms/cuda.py:592 | `--language-model-only` (multimodal.py:304,699-700; interfaces.py:375-379) | yes (skip) |
| torch.compile | Inductor → Triton | Triton | n/a | must be disabled | qwen3_5.py:210; config/vllm.py:1840-1847 | enforce_eager / `-O0` | yes |

Derived sizes on K80 (FP32 only, `platforms/cuda.py:262-264`):

| Model | FP32 weights (incl. vision tower) | Needs |
|---|---|---|
| Qwen3.5-0.8B | ≈3.5 GB | one die |
| Qwen3.5-2B | ≈9.1 GB | one die |
| Qwen3.5-4B | ≈18.6 GB | TP=2 |
| Qwen3.5-9B | ≈38.6 GB | TP=4 |

Per-sequence GDN recurrent state for Qwen3.5-4B: num_v_heads 32 × 128 × 128 × 4 B ≈ 2 MiB per layer, × 24 linear layers ≈ 48 MiB per sequence (from the config fields).

#### Quantized format support

| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| FP8 block 128x128, dynamic act (Qwen3.5/3.6/3.8 -FP8) | `Fp8Config` | 75 | quantization/fp8.py:146-147; enforced at config/vllm.py:943-949 |
| GPTQ int4 g128 sym, attention and GDN excluded via `dynamic` (Qwen3.5 -GPTQ-Int4) | `AutoGPTQConfig` (`"gptq"` alias, `quantization/__init__.py:167-169`). Dense uses an MPLinear kernel (Exllama, CC 60, fp16 act only); MoE uses `make_wna16_moe_kernel` (auto_gptq.py:349, 691-698) | 60, act dtype half/bf16 only | auto_gptq.py:177-183; exllama.py:24-25,44-45 |
| NVFP4 / AWQ / GGUF / MLX | no Qwen-published repos for this family | n/a | HF listings (search Qwen3.5 / 3.6 / 3.8) |

#### In the K80 fork (vllm37)
- Model file: **absent**. `vllm37:vllm/model_executor/models/` has `qwen3.py` and `qwen3_moe.py` but no `qwen3_5.py` or `qwen3_next.py`. No GDN/FLA code exists: a grep for `gated_delta|GatedDeltaNet|qwen3_5|qwen3_next` over `vllm37:vllm/` returns nothing.
- Registry: **absent**. `vllm37:vllm/model_executor/models/registry.py` lists only Qwen3 (:129) and Qwen3Moe (:130).
- Related infrastructure that is present: v0 `MambaCacheManager` (`vllm37:vllm/model_executor/models/mamba_cache.py:25`), and `layers/mamba/ops/causal_conv1d.py` (Triton) plus `layernorm_gated.py` (Triton).

#### Verdict for K80

Not runnable on K80 today. The model-specific blockers, each with the porting work that removes it:

1. **Gated DeltaNet core.** Prefill chunk, post-conv prep, and decode recurrence are Triton (FLA) or FlashInfer/CuteDSL (CC ≥ 9.0/10.x) or a BF16 CC ≥ 8.0 CUDA kernel. There is no GPU fallback (`qwen_gdn_linear_attn.py:93-149, 243-325, 545-567, 1250-1568`).
   - Option 1: port `ref_gated_delta_rule` (`tests/kernels/mamba/cpu/test_cpu_gdn_ops.py:113`) as a pure-torch FP32 per-token recurrence: state ← state·exp(g) + β·k⊗(v − state·k); o = state·q·scale. It is correct but O(T) serial per layer for prefill.
   - Option 2 (better): write sm_37 CUDA kernels in FP32 with no tensor cores, sized for 48 KB smem. One kernel is a recurrent step (decode and short prefill). The other is a chunked prefill built on cuBLAS batched GEMMs plus a small triangular solve.
   - The K=V=128 head dims are fixed by every config above.
2. **causal_conv1d with per-slot cache indices** is Triton only (`ops/causal_conv1d.py:16,762`). Port: a simple FP32 CUDA depthwise-conv kernel (width 4) with state gather and scatter, or torch `F.conv1d` plus `index_select`/`index_copy_` built on `causal_conv1d_update_torch` (`ops/cpu/causal_conv1d.py:119`).
3. **MRoPE Triton** (`mrope.py:437-467`). Use `forward_native` (`mrope.py:374`). Better for text-only: strip the mrope fields as `Qwen3_5ForCausalLMConfig` does (`models/config.py:952-965`), then use the fork's existing NeoX rotary with rotary_dim 64 (partial 0.25). For text tokens the three position rows are identical (`qwen3_5.py:439-445`).
4. **RMSNormGated and the fused QK-norm/RoPE/gate** are Triton. Use `RMSNormGated.forward_native` (`layernorm.py:340-401`) and the eager QK path (`qwen3_next.py:427-447`). The fused path is already skipped in FP32 (`:375-385`).
5. **Hybrid cache.** The official code depends on the v1 `MambaSpec` / `MambaManager` / `GDNAttentionMetadata`. The fork is v0. Port by adapting the fork's `MambaCacheManager` (`vllm37:.../mamba_cache.py:25`) to hold two tensors per GDN layer:
   - conv state `[slots, conv_dim/TP, 3]` in model dtype;
   - recurrent state `[slots, nv/TP, 128, 128]` in float32.
   Then pass the slot indices the way the fork's Mamba2 models do.
6. **Model glue.** Port `qwen3_5.py` together with the needed parts of `qwen3_next.py`:
   - `Qwen3NextAttention` with the output gate;
   - the GemmaRMSNorm x·(1+w) variant;
   - weight-name mapping: `in_proj_qkv`/`in_proj_z` → `in_proj_qkvz`, `in_proj_b`/`in_proj_a` → `in_proj_ba` (`qwen3_5.py:223-230`), and drop `mtp.*`;
   - vendor `transformers_utils/configs/qwen3_5.py` for the fork's older transformers (fork transformers version UNVERIFIED);
   - skip or omit the vision tower (`--language-model-only` equivalent);
   - register `Qwen3_5ForConditionalGeneration` (or the text-only class) in the fork registry.
7. **Shared runtime.** This family adds head_dim 256 for the full-attention layers on top of the dense Qwen3 blockers (attention backend, `_C` build, FP32-only memory). The fork's attention kernel must support head_dim 256.
8. **MoE variants** (35B-A3B, Qwen3.6-35B-A3B) need fused MoE (Triton/FlashInfer only, `oracle/unquantized.py:68-74`). At ≈144 GB in FP32 they do not fit K80 memory. Out of scope.
9. **Quantized repos.** FP8 needs CC 75 and GPTQ needs 60 plus fp16 activations. In the GPTQ-Int4 repos the GDN and attention layers are unquantized BF16 anyway. Only the BF16 checkpoints are usable (cast to FP32).

Smallest practical targets: Qwen3.5-0.8B and 2B on one die, Qwen3.5-4B with TP=2. TP=2 is valid for 4B because num_kv_heads 4, linear k-heads 16, and v-heads 32 are all divisible by 2.

## 4. Gemma 4 and Gemma 3 on Tesla K80 (sm_37): release formats, vLLM code path, kernel requirements


Sources:
- Unprefixed paths are in `/home/jack/src/vllm` (main, 3ca00a8261). `vllm37/...` means `/home/jack/src/vllm37`.
- Hub metadata came from `hf models ls --author google|nvidia|RedHatAI --search gemma-N --expand createdAt,safetensors,gated,config --limit 300 --json`.
- configs were downloaded to a local scratch directory (not kept). Gemma 3 Google repos are gated with `gated=manual` and returned "Access denied … requires approval" (gated, license not accepted). For Gemma 3 the architecture fields come from the hub listing (`config.architectures`, `model_type`). The numeric fields come from the ungated mirrors `unsloth/gemma-3-{270m,1b,4b,12b,27b}-it` (a local scratch directory (not kept)). The mirror configs carry `"unsloth_fixed": true` and are labelled "mirror" wherever they are used.
- "Released" is the HF repo `created_at`. It is not necessarily the public announcement date.
- fp32/fp16 weight sizes are arithmetic: safetensors `total` param count × 4 or × 2 bytes.

---

### Gemma 4

#### Official releases

| Repo | Released (HF created_at) | Params (safetensors total) | Format | Quant config | Context | Layers / KV heads / head_dim | Sliding window | Vision/Audio |
|---|---|---|---|---|---|---|---|---|
| google/gemma-4-E2B-it (+ `-E2B` base) | 2026-03-02 | 5,123,178,051 (BF16) | BF16 safetensors, arch `Gemma4ForConditionalGeneration`, model_type gemma4 / text gemma4_text | none | 131072 | 35 (28 sliding / 7 full) / kv 1 (global kv null → 1) / 256 sliding, 512 full (`global_head_dim`) | 512 | vision `gemma4_vision` 16 layers h=768; audio `gemma4_audio` 12 layers. PLE: `hidden_size_per_layer_input`=256, `vocab_size_per_layer_input`=262144. `num_kv_shared_layers`=20, `use_double_wide_mlp`=true |
| google/gemma-4-E4B-it (+ base) | 2026-03-02 | 7,996,156,490 (BF16) | BF16 | none | 131072 | 42 (35 sliding / 7 full) / kv 2 / 256, 512 | 512 | vision gemma4_vision 16L h=768; audio gemma4_audio 12L. PLE 256. `num_kv_shared_layers`=18 |
| google/gemma-4-12B-it (+ base) | 2026-05-23 | 11,959,730,224 (BF16) | BF16, arch `Gemma4UnifiedForConditionalGeneration`, model_type gemma4_unified / text gemma4_unified_text | none | 262144 | 48 (40 sliding / 8 full) / kv 8 sliding, `num_global_key_value_heads` 1 / 256, 512; `attention_k_eq_v`=true | 1024 | encoder-free: `gemma4_unified_vision` (patch_size 16, model_patch_size 48, mm_embed_dim 3840) and `gemma4_unified_audio` (raw frames, hidden 640). PLE off (`hidden_size_per_layer_input`=0) |
| google/gemma-4-26B-A4B-it (+ base) | 2026-03-11 | 25,805,936,206 (BF16) | BF16 | none | 262144 | 30 (25 sliding / 5 full) / kv 8, global 2 / 256, 512; k_eq_v=true | 1024 | vision gemma4_vision 27L h=1152; audio_config null. MoE: `num_experts` 128, `top_k_experts` 8, `moe_intermediate_size` 704, dense `intermediate_size` 2112 |
| google/gemma-4-31B-it (+ base) | 2026-03-11 | 31,273,088,876 (BF16) | BF16 | none | 262144 | 60 (50 sliding / 10 full) / kv 16, global 4 / 256, 512; k_eq_v=true | 1024 | vision gemma4_vision 27L h=1152; no audio |
| google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it-qat-q4_0-unquantized | 2026-04-28 … 2026-06-04 | same as BF16 parent (e.g. E2B 5,104,297,539) | BF16, QAT-trained weights intended for q4_0 export | none (26B config read: no `quantization_config`) | as parent | as parent | as parent | as parent |
| google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it-qat-q4_0-gguf | 2026-05-01 / 2026-06-05 (12B) | n/a (GGUF) | GGUF Q4_0 plus mmproj GGUF. Files: `gemma-4-31B_q4_0-it.gguf` 17.65 GB, `gemma-4-26B_q4_0-it.gguf` 14.44 GB, `gemma-4-E4B_q4_0-it.gguf` 5.15 GB, `gemma-4-12b-it-qat-q4_0.gguf` 6.98 GB | GGUF Q4_0 | — | — | — | mmproj file per repo |
| google/gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct (no 26B-A4B variant listed) | 2026-06-04/05 | 31B: 33,597,586,080 (I64/I32/BF16) | compressed-tensors `pack-quantized` | `quant_method: compressed-tensors`; group_0 targets `Linear`, weights int4 sym, `strategy: group`, `group_size: 32`, `input_activations: null`. Vision/audio embedders and `lm_head` are in `ignore` | as parent | as parent | as parent | towers kept BF16 (ignore list) |
| google/gemma-4-{E2B,E4B}-it-qat-mobile-ct | 2026-06-01 | E4B 8,680,352,300 (I64/F32/I32/BF16/I8) | compressed-tensors `pack-quantized`, mixed | E4B: g0 `embed_tokens`+`lm_head` int2 per-channel W-only. g1 audio tower int2 + static int8 in/out acts. g2 `embed_tokens_per_layer` int2 group 256. g3 294 LM attn/MLP linears int4 per-channel + static per-tensor int8 input and output acts. g4 PLE gate/proj + vision int8 W + int8 acts | as parent | as parent | as parent | as parent |
| google/gemma-4-{E2B,E4B}-it-qat-mobile-transformers | 2026-06-02 | E4B 3,376,527,178 (F32/BF16/I8/U8) | custom transformers format, `architectures` absent | `quant_method: "gemma"`, `num_bits` 4, per-module 2/4/8-bit regexes, `quantize_embeddings: true` | as parent | as parent | as parent | as parent |
| google/gemma-4-*-it-assistant (+ `-qat-q4_0-unquantized-assistant`) | 2026-04-23 … 2026-05-29 | 31B-assistant 469,518,596 | BF16 MTP drafter, arch `Gemma4AssistantForCausalLM` / `Gemma4UnifiedAssistantForCausalLM` | none | 262144 | 31B-assistant: 4 layers (3 sliding / 1 full), `num_kv_shared_layers` 4 | 1024 | — |
| nvidia/Gemma-4-31B-IT-NVFP4 | 2026-04-02 | (BF16/U8) | ModelOpt NVFP4 | `quant_method: modelopt`, `quant_algo: NVFP4`, W and A fp4 `group_size` 16, `kv_cache_scheme` fp8 (8-bit float). Ignore: lm_head, embed_vision, all `self_attn` | — | — | — | — |
| nvidia/Gemma-4-26B-A4B-NVFP4 | 2026-05-01 | (BF16/F8_E4M3/U8) | ModelOpt NVFP4 | same `quant_algo: NVFP4`, group 16, fp8 KV. Ignore also covers some `mlp*`/`router*` | — | — | — | — |
| (3rd-party, not Google) RedHatAI/gemma-4-{12B,26B-A4B,31B}-it-{FP8-dynamic,FP8-block,NVFP4} | 2026-04-03 … 06-08 | — | compressed-tensors | `quant_method: compressed-tensors` (from hub listing; schemes not read) | — | — | — | — |

Shared text config (all Gemma 4 configs read):
- `hidden_activation: gelu_pytorch_tanh`
- `final_logit_softcapping: 30.0`. The 31B-assistant has null. No `attn_logit_softcapping` key appears.
- `rope_parameters`: `sliding_attention` uses default rope with `rope_theta` 10000. `full_attention` uses `rope_type: proportional`, `rope_theta` 1e6, `partial_rotary_factor` 0.25.
- `vocab_size` 262144.
- `use_bidirectional_attention`: "vision" for 12B/26B/31B, null for E2B/E4B.
- `dtype: bfloat16`, `transformers_version` 5.5.0.dev0, or 5.10.0.dev0 for 12B.

#### Code path (official vLLM)

1. **Registry and arch dispatch.** Three registry entries:
   - `Gemma4ForCausalLM` → `gemma4` at `vllm/model_executor/models/registry.py:109`.
   - `Gemma4ForConditionalGeneration` → `gemma4_mm` at `registry.py:410`.
   - `Gemma4UnifiedForConditionalGeneration` → `gemma4_unified` at `registry.py:411-414`.

   The draft models `Gemma4MTPModel` (`registry.py:676`) and `Gemma4DSparkModel` (`registry.py:650`) are separate. Assistant repos are rewritten to `gemma4_mtp` (`vllm/config/speculative.py:1020-1029`). The per-arch config hook `Gemma4Config` is mapped at `vllm/model_executor/models/config.py:1137-1139`.
2. **Attention backend forced by the model.** `Gemma4Config.verify_and_update_config` (`models/config.py:221-283`) sees heterogeneous head dims (256 and 512). If FA4 exists it forces FA4 (`:255-275`). Otherwise it sets `attention_config.backend = TRITON_ATTN` (`:276-283`). FA4 requires CC 9.x, 10.x or 11.x (`vllm/vllm_flash_attn/flash_attn_interface.py:72-86`).
3. **Multimodal wrapper.** `Gemma4ForConditionalGeneration` (`gemma4_mm.py:1015`) builds three parts:
   - The vision tower via HF `AutoModel.from_config` inside `_mark_tower_model(..., {"image","video"})` (`gemma4_mm.py:1097-1108`).
   - The audio tower, also via `AutoModel`, when `audio_config` is present (`:1111-1133`).
   - The LM via `init_vllm_registered_model` inside `_mark_language_model` (`:1137-1138`).

   The module docstring says the towers run eagerly as HF modules (`gemma4_mm.py:5-8`). **Text-only serving can skip the towers.** With `--language-model-only` or `--limit-mm-per-prompt` 0, `get_limit_per_prompt` returns 0 (`vllm/config/multimodal.py:699-700`). The tower children are then replaced by `StageMissingLayer` (`vllm/model_executor/models/interfaces.py:372-380`). `is_mm_prefix_lm` is forced off when MM is disabled (`vllm/transformers_utils/model_arch_config_convertor.py:369-370`).

   The 12B Unified model has no towers (`gemma4_unified.py:259-262`). Its small `vision_embedder` (ColumnParallelLinear patch_dense), `embed_vision` and `embed_audio` are always built and are not tower-marked (`gemma4_unified.py:264-285`). Its LM is again `Gemma4ForCausalLM` (`:288-294`). A text-only `Gemma4ForCausalLM` drops `audio_tower.`, `vision_tower.`, `embed_audio.` and `embed_vision.` weights (`gemma4.py:1408-1422`).
4. **Token embedding.** `VocabParallelEmbedding` → `F.embedding` (`vllm/model_executor/layers/vocab_parallel_embedding.py:84-85`). The result is multiplied by `normalizer = sqrt(hidden)`, held in model dtype (`gemma4.py:1092-1100`).
5. **Per-layer embeddings (E2B/E4B only).**
   - Table: `embed_tokens_per_layer = VocabParallelEmbedding(vocab_size_per_layer_input, hidden_size_per_layer_input × num_layers)` (`gemma4.py:1018-1031`). That is 262144 × 35·256 = 2.35 B params for E2B (9.4 GB in fp32) and 2.82 B params for E4B (11.3 GB in fp32).
   - Lookup and scale: lookup with ×sqrt(256) (`gemma4.py:869-892`).
   - Projection: `per_layer_model_projection` (ColumnParallelLinear), then RMSNorm, then (proj + emb) × 1/√2 (`gemma4.py:894-921`).
   - Inside each layer: gate ReplicatedLinear → `torch.nn.functional.gelu(approximate="tanh")` → multiply → projection → RMSNorm → residual add (`gemma4.py:763-773`).
   - **No AltUp or LAuReL in Gemma 4.** `grep -i "altup\|laurel" gemma4*.py` returns 0 hits. Those blocks belong to `gemma3n.py`, with 69 `altup` hits.
6. **Norms.** Gemma 4 uses plain `RMSNorm` (output = norm(x)·w), not `GemmaRMSNorm`. The layer norms are at `gemma4.py:619-628`. q_norm and k_norm carry weights and v_norm does not (`gemma4.py:433-447`). The router norm also has no weight (`:311`). `RMSNorm.forward_cuda` → `forward_native` → IR op `ir.ops.rms_norm` (`vllm/model_executor/layers/layernorm.py:74-115`). The IR op picks `vllm_c` (`torch.ops._C.rms_norm`, `vllm/kernels/vllm_c.py:23-44`) or the `native` pure-torch implementation (`vllm/ir/ops/layernorm.py:9-21, 43-62`). The CUDA default priority is `["native"]` under inductor and `["vllm_c","native"]` without compilation (`vllm/platforms/cuda.py:734-751`).
7. **Decoder layer.** `Gemma4DecoderLayer` (`gemma4.py:561-779`) runs this sequence: input_norm → attn → post_attn_norm → +res → pre_ff_norm → MLP → [MoE branch] → post_ff_norm → +res → PLE → × `layer_scalar` (`gemma4.py:714-779`).
8. **Attention.** `Gemma4Attention` (`gemma4.py:366-558`).
   - Head dim and KV heads come per layer from `gemma4_layer_config`. Full layers get `global_head_dim`=512 and, with k_eq_v, `num_global_key_value_heads` (`vllm/transformers_utils/configs/gemma4.py:10-34`; `gemma4.py:578-581`).
   - `scaling=1.0` (`gemma4.py:396-399`).
   - The sliding window comes from `layer_types` (`:449-452`).
   - **KV sharing:** the last `num_kv_shared_layers` layers have only `q_proj` and reuse the KV of the last earlier layer of the same type through `kv_sharing_target_layer_name` (`gemma4.py:404-418, 472-493, 528-535`). This is E2B: 20 layers, E4B: 18 layers.
   - `Attention(..., logits_soft_cap=attn_logit_softcapping (absent → None), per_layer_sliding_window, kv_sharing_target_layer_name, mm_prefix_clamp_sliding_window)` (`gemma4.py:502-520`).
   - Optional YOCO fast-prefill split (`gemma4.py:1103-1141`, enabled by `cache_config.kv_sharing_fast_prefill`).
9. **Rotary.** `get_rope(..., is_neox_style=True)` (`gemma4.py:495-500`):
   - Sliding layers use `RotaryEmbedding` with theta 1e4.
   - Full layers use `scaling_type == "proportional"` → `Gemma4RotaryEmbedding` (`vllm/model_executor/layers/rotary_embedding/__init__.py:153-162`; `gemma4_rope.py:16-77`) with theta 1e6 and partial factor 0.25. The non-rotated dims get zero inv_freq, so the base kernel is applied to all dims.
   - Kernel: `ops.rotary_embedding` in `forward_cuda` (`rotary_embedding/base.py:221-244`). Fallback: `forward_native` (`base.py:203`).
10. **MLP.** `Gemma4MLP` (`gemma4.py:257-287`) runs MergedColumnParallelLinear → `GeluAndMul(approximate="tanh")`. `get_act_and_mul_fn("gelu_pytorch_tanh")` resolves through `activation.py:848` to `torch.ops._C.gelu_tanh_and_mul` (`activation.py:427-439, 455-460`). The fallback is `forward_native` with `F.gelu(…, approximate="tanh")` (`activation.py:446-453`). KV-shared layers in E2B use a double-width MLP (`gemma4.py:598-608`).
11. **MoE (26B-A4B).** The MoE block runs in parallel with the dense MLP, and the two outputs are summed (`gemma4.py:744-757`).
    - `Gemma4Router` applies RMSNorm (no weight) × root_size × scale, then `GateLinear(out_dtype=fp32)` (`gemma4.py:290-363`). `GateLinear`'s ultimate fallback is `F.linear` (`fused_moe/router/gate_linear.py:31-32`).
    - **Routing** uses the Triton kernel `_gemma4_routing_kernel` whenever `is_cuda_alike()` (`gemma4.py:131-219, 350-353`). The torch routing `gemma4_routing_function_torch` (`gemma4.py:222-242`) runs only on non-CUDA platforms.
    - Experts: `FusedMoEFactory(num_experts=128, top_k=8, activation="gelu_tanh", custom_routing_function=…)` (`gemma4.py:649-659`; factory at `fused_moe/layer.py:88`). The unquantized oracle's CUDA candidates are FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON and BATCHED_TRITON (`fused_moe/oracle/unquantized.py:68-74`).
    - Candidate guards: TRTLLM bf16 requires SM100 family (`experts/trtllm_bf16_moe.py:98-100`). FlashInfer CUTLASS requires SM90/SM100 (`experts/flashinfer_cutlass_moe.py:130-134`). TritonExperts accepts any `is_cuda_alike()` (`experts/triton_moe.py:118-119`).
12. **Final norm and logits.** Final norm is `RMSNorm` (`gemma4.py:1090`). `LogitsProcessor(soft_cap=final_logit_softcapping=30)` (`gemma4.py:1459-1462`) applies `torch.tanh(logits/cap)*cap` (`vllm/model_executor/layers/logits_processor.py:113-116, 221-222`). The LM head is tied to the embedding (`gemma4.py:1456-1457`).
13. **Vision bidirectional attention.** It applies to sliding layers only. mm_prefix is cleared on full layers (`gemma4_mm.py:1164-1172, 2121-2197`) and clamped on sliding layers (`gemma4.py:512-518`). It is not used in text-only serving (step 3).

#### Requirements

| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Decoder paged attention, sliding 512/1024 + full, head_dim 256 **and 512**, KV sharing | Gemma4Config forces FA4, or else TRITON_ATTN | FA4: CC 9.x/10.x/11.x. Triton: Triton compiler | FA4 90. TRITON_ATTN has no CC guard (`supports_compute_capability` → True) but needs Triton | **Blocked.** Triton does not target sm_37, and FA4 is impossible | `models/config.py:240-283`; `flash_attn_interface.py:72-86`; `triton_attn.py:413-414` (no CC check), `:37-42` (Triton ops), `:367-368` (sliding OK), `:387-388` (head ≥32, so 512 OK), `:325-328` (fp32 OK) | Other CUDA backends in priority order (`platforms/cuda.py:168-178`): FLASH_ATTN needs CC ≥ 8.0 (`flash_attn.py:451-452`) and head ≤256 without FA4 (`:421-428`). FLASHINFER needs CC 8.0-12.1 (`flashinfer.py:514-522`). FLEX_ATTENTION uses `torch.compile(flex_attention)` (`flex_attention.py:55-58`) and raises on KV sharing (`:1227-1228`) and on softcap (`:1219-1222`). TURBOQUANT uses Triton (`turboquant_attn.py:61-66`). No pure-PyTorch paged decoder attention exists for CUDA | **No** |
| Attention logit softcap | not used: no `attn_logit_softcapping` key in any Gemma 4 config, so `getattr(..., None)` | — | — | N/A | `gemma4.py:592` | FA (`flash_attn.py:128-140`), FlashInfer (`flashinfer.py:316-352`) and Triton (`triton_unified_attention.py:550`) support it. Flex does not (`flex_attention.py:1219-1222`) | N/A |
| Final logit softcap 30 | `torch.tanh` | PyTorch | any | OK if the PyTorch build has sm_37 | `logits_processor.py:113-116` | itself pure torch | Yes |
| RMSNorm (all norms, q/k/v norm, router norm) | IR op: `vllm_c` → `torch.ops._C.rms_norm`, or `native` | `_C` CUDA ext (cub BlockReduce) or torch | — | `_C` build for sm_37 is the shared-runtime question. Native works | `layernorm.py:96-115`; `vllm/kernels/vllm_c.py:23-44`; `csrc/libtorch_stable/layernorm_kernels.cu:68` | `vllm/ir/ops/layernorm.py:9-21, 43-62` (native). Selected when `["native"]` priority or under inductor (`platforms/cuda.py:738`) | Yes (eager; inductor would emit Triton) |
| GELU-tanh-and-mul | `torch.ops._C.gelu_tanh_and_mul` | `_C` CUDA ext | — | shared-runtime question | `activation.py:427-439, 455-460`; `csrc/libtorch_stable/activation_kernels.cu` | `GeluAndMul.forward_native` (`activation.py:446-453`) | Yes |
| PLE gate GELU | `torch.nn.functional.gelu(approximate="tanh")` | torch | any | OK | `gemma4.py:767` | — | Yes |
| RoPE: default for sliding, proportional partial 0.25 for full | `ops.rotary_embedding` (`_C`) | `_C` CUDA ext | — | shared-runtime question | `rotary_embedding/base.py:221-244`; `gemma4_rope.py:16-77`; `csrc/libtorch_stable/pos_encoding_kernels.cu` | `RotaryEmbedding.forward_native` (`base.py:203`) | Yes |
| Embedding and PLE embedding | `F.embedding` | torch | any | OK (memory permitting) | `vocab_parallel_embedding.py:84-85` | — | Yes |
| Dense GEMMs (QKV, O, MLP, PLE projections) | `UnquantizedLinearMethod` (cuBLAS through torch) | torch/cuBLAS | any CUDA | OK in fp32 | — (UNVERIFIED: the unquantized linear dispatch was not opened for this report) | — | Yes |
| MoE routing (26B-A4B) | Triton `_gemma4_routing_kernel` | Triton | n/a | **Blocked** | `gemma4.py:131-219, 350-353` | `gemma4_routing_function_torch` (`gemma4.py:222-242`), but it is reached only when not `is_cuda_alike()` | Code exists. It needs a 1-line platform guard change |
| MoE router GEMM | `GateLinear` → specialized or cuBLAS kernels, or `F.linear` | SM90+ for the specialized paths | — | Falls back to F.linear | `gate_linear.py:31-32, 63-64` | `F.linear` via ReplicatedLinear | Yes |
| MoE experts, 128 × top-8, gelu_tanh (26B-A4B) | TritonExperts (fused_moe) | Triton | none declared (`is_cuda_alike`) | **Blocked** | `oracle/unquantized.py:68-74`; `experts/triton_moe.py:118-119` | No CUDA pure-torch experts. CPU experts require `is_cpu()` (`experts/cpu_moe.py:144-145`). `int4_emulation_moe.py:29` also subclasses TritonExperts | **No** (would need a new torch loop-over-experts impl) |
| Vision tower (E2B/E4B/26B/31B) | HF `AutoModel` (transformers ≥ 5.5.0.dev0 per config) eager | transformers 5.x | — | Skippable for text-only | `gemma4_mm.py:5-8, 1097-1108` | skip with `--language-model-only` (`config/multimodal.py:699-700`; `interfaces.py:372-380`) | Yes (skip) |
| Audio tower (E2B/E4B) | HF `AutoModel` | transformers 5.x | — | Skippable | `gemma4_mm.py:1111-1133` | same | Yes (skip) |
| Unified 12B patch embedder | ColumnParallelLinear + LayerNorm | torch | any | always built. Small | `gemma4_unified.py:74-130, 264-285` | — | Yes |
| dtype | auto: platform list minus Gemma-blocked dtypes | — | — | Kepler `supported_dtypes` = `[torch.float32]` (`platforms/cuda.py:262-264`), so auto resolves to fp32. Gemma 4 is **not** in `_FLOAT16_NOT_SUPPORTED_MODELS` (`config/model.py:2369-2374`), so `--dtype half` is allowed. fp16 numerics for Gemma 4 are UNVERIFIED. bf16 is rejected on <SM80 (`platforms/cuda.py:656-658`) | `config/model.py:2395-2433` | — | fp32 OK (4 B/param) |
| torch.compile / inductor | `@support_torch_compile` on Gemma4Model and the YOCO halves | Inductor → Triton | — | Must run with `--enforce-eager` / compilation mode NONE | `gemma4.py:807-810, 954-957, 989-992` | eager mode | Yes |

#### Quantized format support

| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| BF16 safetensors (`*-it`, `*-qat-q4_0-unquantized`) | unquantized; loaded as fp32 on Kepler | none (dtype → fp32) | `platforms/cuda.py:262-264` |
| QAT w4a16 CT (`*-qat-w4a16-ct`: int4 sym group 32, act none) | `CompressedTensorsConfig` → `CompressedTensorsWNA16` → `choose_mp_linear_kernel` | Config 70 (`compressed_tensors.py:117-118`). Scheme 75 (`compressed_tensors_wNa16.py:102-104`). Enforced at `vllm/config/vllm.py:943-949`. CUDA kernels: Machete 90, Marlin 75, Conch 80, Exllama 60 (fp16 activations only, `exllama.py:43-44`), TritonW4A16 0 but Triton (`triton_w4a16.py:313-330`). List at `kernels/linear/__init__.py:507-514` | Blocked on sm_37: 37 < 70. No pure-torch CUDA MP kernel exists. `CPUWNA16LinearKernel` is CPU-only (`__init__.py:528-530`) |
| QAT mobile CT (`*-qat-mobile-ct`: int4 per-channel W + static int8 in/out acts; int2 embeddings/lm_head; int8 PLE) | `CompressedTensorsWNA8O8Int` (fake-quant of activations around an MP weight-only matmul) plus `CompressedTensorsEmbeddingWNA16Int` | WNA8O8 70 (`compressed_tensors_wNa8o8.py:83-84`). The embedding dequant-gather is a Triton kernel (`compressed_tensors_embedding.py:26-90, 159-165`). The matmul uses the same MP kernel list as above. The kernel serving the int2 `uint2b2` lm_head is UNVERIFIED (only Exllama mentions uint2b2, as untested, `exllama.py:20`) | Blocked (CC 70 gate + Triton embedding) |
| QAT mobile transformers (`quant_method: "gemma"`) | none: "gemma" is not in `QuantizationMethods` | — | `vllm/model_executor/layers/quantization/__init__.py:15-49, 156-182` |
| GGUF Q4_0 (`*-qat-q4_0-gguf`) | not in-tree. Moved to the out-of-tree `vllm-gguf-plugin` | n/a in tree (plugin not inspected: UNVERIFIED) | `docs/features/quantization/gguf.md:7-12`. "gguf" is absent from `quantization/__init__.py:15-49` |
| NVFP4 (nvidia/Gemma-4-*-NVFP4, `quant_method: modelopt`, `quant_algo: NVFP4`) | overridden to `modelopt_fp4` → `ModelOptNvFp4Config` | 75 (`modelopt.py:754-755`). act dtypes bf16/half/fp8 only (`modelopt.py:750-751`), so fp32 is rejected by `vllm/config/vllm.py:950-955` | `modelopt.py:757-763` |
| FP8-dynamic / FP8-block (RedHatAI, CT) | `CompressedTensorsW8A8Fp8` | 89 (`compressed_tensors_w8a8_fp8.py:83-85`) | — |
| NVFP4 (RedHatAI, CT) | `CompressedTensorsW4A4Fp4` | 75 (`compressed_tensors_w4a4_nvfp4.py:35-36`) | — |

#### In the K80 fork (vllm37)

- Model file: **absent**. There is no `gemma4*.py` in `vllm37/vllm/model_executor/models/` (`ls | grep -i gemma` lists only gemma, gemma2, gemma3, gemma3_mm, gemma3n, gemma3n_mm and paligemma). `grep -rli gemma4 vllm37/vllm` returns 0 files.
- Registry entry: **absent**. `vllm37/vllm/model_executor/models/registry.py` has no `Gemma4*` key (Gemma entries only at lines 69-72, 150, 206-207 and 232).

#### Verdict for K80

Gemma 4 is **not runnable on sm_37 with official vLLM as-is, and the K80 fork has no Gemma 4 implementation.**

Blockers, with what would remove each:

1. **Decoder attention has no usable CUDA backend.** Gemma4Config forces TRITON_ATTN when FA4 is missing (`models/config.py:276-283`). FA and FlashInfer are gated to CC ≥ 8.0 (`flash_attn.py:451-452`, `flashinfer.py:514-522`). Flex needs torch.compile/Triton and rejects KV sharing (`flex_attention.py:55-58, 1227-1228`).
   - Fix: a new non-Triton backend for sm_37 (CUDA C paged attention or pure torch SDPA over gathered blocks). It must support per-layer sliding windows, head_dim 512 on full layers and 256 on sliding layers, KV-sharing targets, and fp32 KV.
   - The vllm37 v0 paged-attention kernels may cover the head sizes. Head-size support for 512 in vllm37 is UNVERIFIED and was not inspected per instructions.
2. **MoE (26B-A4B):** both routing and experts are Triton-only on CUDA (`gemma4.py:350-353`; `oracle/unquantized.py:68-74`).
   - Fix: route `gemma4_routing_function_torch` (`gemma4.py:222-242`) on CUDA, and add a torch or CUDA-C expert loop.
   - Moot for memory anyway: 26B-A4B is 103 GB in fp32 and 52 GB in fp16 (computed).
3. **Model port into the v0 fork:** the files `gemma4.py`, `gemma4_mm.py`, `gemma4_unified.py`, `gemma4_rope.py`, `transformers_utils/configs/gemma4.py` and the Gemma4Config hook would all need backporting. They depend on transformers 5.x config classes (config `transformers_version` 5.5.0.dev0 / 5.10.0.dev0) and on v1 KV-sharing (`kv_sharing_target_layer_name`).
   - Fix: backport the text-only `Gemma4ForCausalLM`.
   - Implement KV sharing in the v0 cache engine, or materialise per-layer K/V (losing the memory saving).
   - Implement per-layer heterogeneous head_dim KV cache. vLLM main handles this with different page sizes per group (`v1/core/single_type_kv_cache_manager.py:1064`).
4. **dtype and memory.** Kepler forces fp32 (`platforms/cuda.py:262-264`), so weights cost 4 B/param.
   - E2B: 20.5 GB fp32 (PLE table alone 9.4 GB), which needs both 12 GB K80 dies with TP=2 or a fp16 override.
   - E4B: 32 GB fp32.
   - 12B: 48 GB fp32. 31B: 125 GB fp32.
   - Fix: allow `--dtype half` (vLLM does not block it for gemma4, `config/model.py:2369-2374`). fp16 still needs fp32 accumulation, since sm_37 has no native fp16 arithmetic. Gemma 4 fp16 overflow behaviour is UNVERIFIED.
   - Or offload or quantize the PLE table (it is an embedding lookup, so CPU-resident is feasible).
5. **All published quantized formats are blocked.**
   - CT w4a16 needs CC ≥ 70/75 (`compressed_tensors.py:117-118`, `compressed_tensors_wNa16.py:102-104`).
   - Mobile CT needs CC 70 plus a Triton embedding gather.
   - NVFP4 needs 75 and bf16/fp16.
   - "gemma" quant_method is unsupported.
   - GGUF is out-of-tree.
   - Fix: a sm_37 int4 group-32 dequant GEMM, or dequantize-to-fp16/fp32 at load. Dequant at load works for `*-qat-q4_0-unquantized` BF16 weights trivially, but brings no memory saving.
6. **Torch compile** must be disabled (`--enforce-eager`). Native and `_C` fallbacks exist for RMSNorm, GELU-tanh and RoPE (rows above).

The multimodal towers are not a blocker: they are skippable for text-only serving (`config/multimodal.py:699-700`; `interfaces.py:372-380`).

---

### Gemma 3

#### Official releases

| Repo | Released (HF created_at) | Params (safetensors total) | Format | Quant config | Context | Layers / KV heads / head_dim | Sliding window | Vision/Audio |
|---|---|---|---|---|---|---|---|---|
| google/gemma-3-270m-it (+ `-270m` base) | 2025-07-30 (base 2025-08-05) | 268,098,176 (BF16) | BF16, `Gemma3ForCausalLM`, model_type gemma3_text (hub listing) | none (gated: config from mirror) | 32768 (mirror) | 18 (15 sliding / 3 full, explicit `layer_types`) / 1 / 256 (mirror) | 512 | none (text-only). `use_bidirectional_attention` false |
| google/gemma-3-1b-it (+ `-1b-pt`) | 2025-03-10 (pt 2025-02-20) | 999,885,952 (BF16) | BF16, `Gemma3ForCausalLM`, gemma3_text | none | 32768 (mirror) | 26 / 1 / 256 (mirror). `sliding_window_pattern` 6 → 22 sliding / 4 full (derived; transformers not installed locally, UNVERIFIED) | 512 | none |
| google/gemma-3-4b-it (+ pt) | 2025-02-20 | 4,300,079,472 (BF16) | BF16, `Gemma3ForConditionalGeneration`, gemma3 | none | 131072 (rope linear ×8) (mirror) | 34 / 4 / 256. Pattern 6 → 29/5 (derived) | 1024 | vision `siglip_vision_model` 27L h=1152, 256 tokens/image (mirror). No audio |
| google/gemma-3-12b-it (+ pt) | 2025-03-01 | 12,187,325,040 (BF16) | BF16, Gemma3ForConditionalGeneration | none | 131072 (mirror) | 48 / 8 / 256. Pattern 6 → 40/8 (derived) | 1024 | SigLIP (as 4b) |
| google/gemma-3-27b-it (+ pt) | 2025-03-01 | 27,432,406,640 (BF16) | BF16, Gemma3ForConditionalGeneration | none | 131072 (mirror) | 62 / 16 / 128 (`query_pre_attn_scalar` 168). Pattern 6 → 52/10 (derived) | 1024 | SigLIP 27L h=1152 (mirror) |
| google/gemma-3-{1b,4b,12b,27b}-{it,pt}-qat-q4_0-gguf | 2025-03-10 … 2025-03-20 | n/a | GGUF Q4_0 (`gemma-3-27b-it-q4_0.gguf` 17.23 GB + `mmproj-model-f16-27B.gguf` 0.86 GB; `gemma-3-1b-it-q4_0.gguf` 1.00 GB, no mmproj) | GGUF Q4_0 | — | — | — | mmproj (f16) for 4b/12b/27b |
| google/gemma-3-{270m,270m-it,1b-it,4b-it,12b-it,27b-it}-qat-q4_0-unquantized | 2025-04-08 … 2025-08-07 | same as parent | BF16 QAT weights | none expected. Gated: config not readable (UNVERIFIED) | as parent | as parent | as parent | as parent |
| google/gemma-3-{1b,4b,12b}-it-qat-int4-unquantized | 2025-04-09 | same as parent | BF16 QAT weights (int4 target) | gated, UNVERIFIED | as parent | as parent | as parent | as parent |
| (3rd-party) RedHatAI/gemma-3-{1b,4b,12b,27b}-it-{FP8-dynamic, quantized.w4a16, quantized.w8a8} | 2025-04-28 … 2025-06-05 | — | compressed-tensors | `quant_method: compressed-tensors` (hub listing; schemes not read) | — | — | — | — |

Shared fields (mirror configs):
- `hidden_activation: gelu_pytorch_tanh`
- `attn_logit_softcapping: null` and `final_logit_softcapping: null`
- `rope_theta` 1e6 (global) and `rope_local_base_freq` 1e4 (local)
- `rope_scaling` `{linear, factor 8}` for 4b/12b/27b; null for 270m/1b
- `vocab_size` 262144
- `torch_dtype: bfloat16`

No per-layer embeddings, AltUp, KV sharing or MoE. Those are Gemma 3n features (`gemma3n.py`), and Gemma 3n is out of scope.

#### Code path (official vLLM)

1. **Registry.** `Gemma3ForCausalLM` → `gemma3` (`registry.py:106`). `Gemma3TextModel` → `gemma3.Gemma3Model`, a pooling arch (`registry.py:220`). `Gemma3ForConditionalGeneration` → `gemma3_mm` (`registry.py:401`). `Gemma3TextModelConfig` sets `is_causal = not use_bidirectional_attention` (`models/config.py:73-77`, mapped at `:1136`).
2. **dtype rule.** `gemma3` and `gemma3_text` are in `_FLOAT16_NOT_SUPPORTED_MODELS` ("Numerical instability. Please use bfloat16 or float32 instead.") at `config/model.py:2369-2374`. Auto resolution drops fp16 (`config/model.py:2377-2381, 2395-2405`). An explicit `--dtype half` raises (`:2384-2389, 2476`). On Kepler the candidate list is `[torch.float32]` (`platforms/cuda.py:262-264`), so Gemma 3 resolves to **fp32**.
3. **Multimodal wrapper (4b/12b/27b).** `Gemma3ForConditionalGeneration` (`gemma3_mm.py:500`) builds:
   - `SiglipVisionModel` + `Gemma3MultiModalProjector` inside `_mark_tower_model(vllm_config, "image")` (`gemma3_mm.py:552-558`).
   - The LM as `Gemma3ForCausalLM` (`:560-566`).

   **Text-only serving skips SigLIP** via `--language-model-only` / limit 0 (`config/multimodal.py:699-700`; `interfaces.py:372-380`). Gemma 3 is in the mm-prefix (bidirectional image attention) list (`model_arch_config_convertor.py:374-383`), but that is forced off when MM is disabled (`:369-370`).
4. **Embedding.** `VocabParallelEmbedding` → `F.embedding` (`vocab_parallel_embedding.py:84-85`), multiplied by the `normalizer` buffer sqrt(hidden) (`gemma3.py:313-341`).
5. **Decoder layer.** `Gemma3DecoderLayer` (`gemma3.py:224-289`) has 4 × `GemmaRMSNorm` (input, post_attn, pre_ff, post_ff) (`:254-263`). The forward uses fused residual-add norms (`:265-289`).
6. **Norm.** `GemmaRMSNorm` computes x·(1+w) (`layernorm.py:138-168`). `forward_cuda` → `forward_native` → `ir.ops.rms_norm` / `fused_add_rms_norm` with `weight.float()+1` (`layernorm.py:159-175`). The IR implementation is `vllm_c` (`vllm_c.py:23-44`, which needs weight dtype == x dtype, true in fp32) or native (`ir/ops/layernorm.py:9-62`).
7. **Attention.** `Gemma3Attention` (`gemma3.py:96-221`):
   - QKVParallelLinear (`:132-140`).
   - q_norm and k_norm are GemmaRMSNorm (`:149-150, 211-216`).
   - `scaling = query_pre_attn_scalar**-0.5` (`:130`).
   - Sliding vs full comes from `config.layer_types` (`:152-155`).
   - `Attention(..., logits_soft_cap=None, per_layer_sliding_window=sliding_window)` (`:189-200`). The decoder passes `attn_logits_soft_cap=None` explicitly (`:243`).
   - `EncoderOnlyAttention` is used when `is_causal` is false (`:178-187`).
   - vLLM's hybrid KV-cache manager handles the 5:1 sliding/full ratio (`v1/core/kv_cache_utils.py:1570`).
8. **Rotary.**
   - transformers v5 per-layer-type `rope_parameters` or the v4 fallback. Local layers get `rope_type default`, `rope_theta=rope_local_base_freq` (1e4) (`gemma3.py:157-169`).
   - The global rope uses linear ×8 scaling for 4b+ (`LinearScalingRotaryEmbedding`, `rotary_embedding/__init__.py:20, 185`).
   - Kernel: `ops.rotary_embedding` (`base.py:221-244`). Fallback: `forward_native` (`base.py:203`).
   - Context: vLLM does not re-apply the rope factor to max_model_len for gemma3 (`config/model.py:2572-2574`).
9. **MLP.** `Gemma3MLP` (`gemma3.py:63-93`) → `GeluAndMul(approximate="tanh")` → `_C.gelu_tanh_and_mul`, with fallback `forward_native` (`activation.py:427-453`).
10. **Final.** Final `GemmaRMSNorm` (`gemma3.py:326, 372`). `LogitsProcessor(soft_cap=final_logit_softcapping)` gets null here, so no softcap is applied (`gemma3.py:414-416`). The LM head is tied (`:411-412`).
11. **Vision (if enabled).** SigLIP uses `MMEncoderAttention`. On CC < 8.0 the ViT backend loop returns `TORCH_SDPA` (`platforms/cuda.py:539-574`), which goes to `_forward_sdpa` (`layers/attention/mm_encoder_attention.py:529, 802-803`).

#### Requirements

| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Decoder paged attention: sliding 512/1024 + full, head_dim 256 (128 for 27B), no softcap | backend chosen from `FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT` | FA: CC ≥ 8.0. FlashInfer: 8.0-12.1. Triton/Flex/TurboQuant: Triton | 80 for FA/FI. Triton: no declared CC guard | **Blocked**: FA and FI are rejected by the CC check, and Triton does not target Kepler | `platforms/cuda.py:168-178`; `flash_attn.py:451-452`; `flash_attn_interface.py:52-59` (FA2 CC ≥ 8); `flashinfer.py:514-522`; `triton_attn.py:37-42, 413-414`; `flex_attention.py:55-58` | No pure-PyTorch CUDA decoder backend (`cpu_attn.py` is the CPU platform's backend) | **No** |
| Sliding window support | FA (`flash_attn.py:386-387`), FlashInfer (`flashinfer.py:476-477`), Triton (`triton_attn.py:367-368`), Flex (`flex_attention.py:113-114`) | — | — | none usable on sm_37 | — | — | No |
| Logit softcap | not used (null in config, `None` passed) | — | — | N/A | `gemma3.py:243, 414-416` | — | N/A |
| GemmaRMSNorm | IR `vllm_c` (`_C.rms_norm` / `fused_add_rms_norm`) or native | `_C` ext or torch | — | shared runtime. Native works | `layernorm.py:159-175`; `vllm_c.py:23-80` | `ir/ops/layernorm.py:9-62` | Yes (eager) |
| GELU-tanh-and-mul | `_C.gelu_tanh_and_mul` | `_C` ext | — | shared runtime | `activation.py:427-439` | `activation.py:446-453` | Yes |
| RoPE (default local; linear ×8 global) | `_C.rotary_embedding` | `_C` ext | — | shared runtime | `rotary_embedding/base.py:221-244` | `base.py:203` | Yes |
| Embedding × normalizer | `F.embedding` + mul | torch | any | OK | `gemma3.py:338-341` | — | Yes |
| SigLIP vision encoder (4b+) | `MMEncoderAttention` → TORCH_SDPA on CC < 8.0 | torch SDPA | any | OK if torch supports sm_37. Skippable | `platforms/cuda.py:539-574`; `mm_encoder_attention.py:529, 802-803`; `gemma3_mm.py:552-558` | skip tower via `--language-model-only` | Yes |
| dtype | fp16 blocked for gemma3/gemma3_text, bf16 needs SM80, so the Kepler list is fp32 only | — | — | **fp32 forced** (4 B/param) | `config/model.py:2369-2389`; `platforms/cuda.py:262-264, 656-658` | — | fp32 OK |
| torch.compile | `@support_torch_compile` on Gemma3Model | inductor → Triton | — | must use eager | `gemma3.py:292` | eager | Yes |

#### Quantized format support

| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| BF16 safetensors (`-it`, `-pt`, `-qat-q4_0-unquantized`, `-qat-int4-unquantized`) | unquantized → fp32 on Kepler | none | `platforms/cuda.py:262-264`; `config/model.py:2369-2374` |
| GGUF Q4_0 (`*-qat-q4_0-gguf`) | not in-tree in vLLM main (`vllm-gguf-plugin`) | n/a in tree. Plugin UNVERIFIED | `docs/features/quantization/gguf.md:7-12`; absent from `quantization/__init__.py:15-49` |
| W4A16 CT (RedHatAI `quantized.w4a16`) | `CompressedTensorsWNA16` | 75 (scheme), 70 (config) | `compressed_tensors_wNa16.py:102-104`; `compressed_tensors.py:117-118` |
| W8A8 INT8 CT (RedHatAI `quantized.w8a8`) | `CompressedTensorsW8A8Int8` | 75 | `compressed_tensors_w8a8_int8.py:35-37` |
| FP8-dynamic CT (RedHatAI) | `CompressedTensorsW8A8Fp8` | 89 | `compressed_tensors_w8a8_fp8.py:83-85` |
| GGUF in the K80 fork | `GGUFConfig` | 60 (still > 37) | `vllm37/vllm/model_executor/layers/quantization/gguf.py:43-44`; registered at `vllm37/.../quantization/__init__.py:21, 133` |

#### In the K80 fork (vllm37)

- Model file present: `vllm37/vllm/model_executor/models/gemma3.py`. Multimodal: `vllm37/vllm/model_executor/models/gemma3_mm.py`.
- Registry entries present:
  - `vllm37/vllm/model_executor/models/registry.py:71`: `"Gemma3ForCausalLM": ("gemma3", "Gemma3ForCausalLM")`.
  - `registry.py:206`: `"Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration")`.
- The fork has the same fp16 block (`vllm37/vllm/config/__init__.py:3111-3112`) and the same Kepler fp32-only dtype list (`vllm37/vllm/platforms/cuda.py:64-74`).

#### Verdict for K80

Gemma 3 is the more tractable family: plain dense GQA, head_dim 256/128, no KV sharing, no MoE, no PLE. The model file and registry entries already exist in vllm37.

Blockers in official vLLM, with what would remove each:

1. **Decoder attention.** No CUDA backend passes on sm_37. FA and FlashInfer are CC-gated (`flash_attn.py:451-452`, `flashinfer.py:514-522`). TRITON_ATTN and FLEX need Triton (`triton_attn.py:37-42`, `flex_attention.py:55-58`).
   - Fix (official v1): add a CUDA-C or torch-SDPA paged backend with per-layer sliding window and fp32 KV.
   - Fix (fork): the v0 attention backend in vllm37 is the relevant path. Whether it honours `per_layer_sliding_window` with Gemma 3's interleaved 5:1 pattern on sm_37 is UNVERIFIED (not inspected per instructions).
2. **fp32 forced.**
   - vLLM blocks fp16 for gemma3/gemma3_text (`config/model.py:2369-2374`). Kepler only lists fp32 (`platforms/cuda.py:262-264`).
   - Memory in fp32: 270m 1.1 GB and 1b 4.0 GB fit one 12 GB die. 4b is 17.2 GB (~16 GB without SigLIP; split UNVERIFIED), which needs TP=2 across both dies. 12b (48.7 GB) and 27b (110 GB) do not fit.
   - Fix: lift the fp16 block with fp32 residual/norm accumulation. This is not recommended without validation, given Google's documented instability that vLLM cites. Or keep fp32 and target only 270m/1b (and 4b with TP=2).
3. **Quantized formats** all need CC ≥ 60–89 (table above).
   - Fix: run the BF16 `qat-q4_0-unquantized` / `qat-int4-unquantized` checkpoints upcast to fp32. These are just BF16 tensors, but they give no memory win.
   - Or port a sm_37 dequant GEMM for GGUF Q4_0. The fork's GGUFConfig gate (60) would need lowering and its kernels need a sm_37 build. Both UNVERIFIED.
4. **torch.compile** must be disabled (`gemma3.py:292`). The RMSNorm, GELU-tanh and RoPE ops all have native fallbacks (rows above) if the `_C` extension is not built for sm_37.

The SigLIP vision tower is not a blocker: it is skippable, and on CC < 8.0 it uses TORCH_SDPA anyway.

## 5. IBM Granite 4.x (dense and hybrid) and NVIDIA Nemotron 3 / Nemotron-H on Tesla K80 (sm_37)


Sources:
- Official vLLM main at `/home/jack/src/vllm`, commit 3ca00a8261. Unless a path says otherwise, every `path:line` is relative to that checkout.
- The K80 fork at `/home/jack/src/vllm37`. Paths into it are prefixed `vllm37:`.
- Hugging Face data came from `hf models ls --author ... --expand created_at,downloads,safetensors,config --json`. Each repo's `config.json` and `hf_quant_config.json` were downloaded to a local scratch directory (not kept). No weights were downloaded.
- "Released" is the HF repo `created_at` date, used here as the release date.
- "Params" is the safetensors `total` value from the listing.

Shared fact for all three families:
- On sm_37, official vLLM supports only FP32 as a dtype. `vllm/platforms/cuda.py:255-264` returns `[torch.float32]` below cc 6.0, with the comment "Kepler and Maxwell ... only FP32 is supported, though vLLM doesn't support these GPUs".
- BF16 needs cc ≥ 8.0 (`vllm/platforms/cuda.py:656-672`).
- Every checkpoint below ships as BF16. On K80 each one must be up-cast to FP32, or FP16 support has to be re-enabled in the fork.
- One K80 die has 12 GB. In FP32 a model needs 4 B/param, so anything above about 2.8B params does not fit on one die without TP=2. This is arithmetic on the safetensors counts.

---

### IBM Granite 4.x dense (GraniteForCausalLM, model_type `granite`)

#### Official releases
Shape facts for all repos in this table:
- All repos use `architectures=GraniteForCausalLM` and `model_type=granite`.
- Every layer is full attention with a dense SwiGLU MLP. There are no Mamba or MoE layers, and no `layer_types` or `sliding_window` field.
- RoPE has no scaling (`rope_scaling: null`).
- Context is `max_position_embeddings=131072` for every repo in the table.
- Base and instruct repos share the same shapes.

| Repo | Released | Params | Format | Quant config | Context | Layers (attn/mamba/moe) / KV heads / head_dim |
|---|---|---|---|---|---|---|
| ibm-granite/granite-4.1-3b (+ -base) | 2026-04-06 | 3.40B | BF16 | none | 131072 | 40/0/0 / 8 / 64 (2560/40) |
| ibm-granite/granite-4.1-8b (-base is 8.38B) | 2026-04-06 | 8.79B | BF16 | none | 131072 | 40/0/0 / 8 / 128 (4096/32) |
| ibm-granite/granite-4.1-30b (+ -base) | 2026-04-06 | 28.87B | BF16 | none | 131072 | 64/0/0 / 8 / 128 |
| granite-4.1-{3b,8b,30b}-fp8 | 2026-04-20 | 3.66/8.79/29.28B | FP8 E4M3 | `compressed-tensors`, `float-quantized`, W8 float per-channel static, A8 float per-token dynamic, targets `Linear`, ignore `lm_head` (config.json of all 3 repos read) | 131072 | as base |
| granite-4.1-{3b,8b,30b}-GGUF | 2026-04-16/17 | - | GGUF Q2_K…Q8_0, bf16 (file list of 4.1-8b-GGUF) | n/a | - | - |
| ibm-granite/granite-4.2-3b | 2026-08-07 | 3.66B | BF16 | none | 131072 | 40/0/0 / 8 / 64 |
| ibm-granite/granite-4.2-8b | 2026-08-07 | 8.79B | BF16 | none | 131072 | 40/0/0 / 8 / 128 |
| ibm-granite/granite-4.2-30b | 2026-08-07 | 29.28B | BF16 | none | 131072 | 64/0/0 / 8 / 128 |
| granite-4.2-{3b,8b,30b}-fp8 | 2026-08-13 | 3.66/8.79/29.28B | FP8 E4M3 | same compressed-tensors FP8 W8A8 scheme as 4.1-fp8 (config.json of all 3 read) | 131072 | as base |
| granite-4.2-{3b,8b,30b}-mxfp4 | 2026-08-13 | 3.66/5.06/15.94B | MXFP4 (U8 packed) | `compressed-tensors`, `mxfp4-pack-quantized`, W4 float group 32 (uint8 scales), A4 float group 32 dynamic | 131072 | as base |
| granite-4.2-{3b,8b,30b}-nvfp4 | 2026-08-13 | 3.66/5.31/16.83B | NVFP4 (U8 packed) | `compressed-tensors`, `nvfp4-pack-quantized`, W4 float tensor_group 16 (fp8_e4m3 scales), A4 tensor_group 16 dynamic=local | 131072 | as base |
| granite-4.2-{3b,8b,30b}-GGUF | 2026-08-12 | - | GGUF Q2_K…Q8_0, bf16 (file list of 4.2-8b-GGUF) | n/a | - | - |
| granite-4.2-*-{bf16,q4,q8}-mlx | 2026-09-01 | - | Apple MLX | not vLLM-relevant | - | - |

muP multipliers and RoPE, from each repo's config.json:

| Repo | embedding_multiplier | residual_multiplier | attention_multiplier | logits_scaling | rope_theta | tie_word_embeddings |
|---|---|---|---|---|---|---|
| granite-4.1-3b | 12.0 | 0.22 | 0.015625 (=1/64) | 10.0 | 1e7 | true |
| granite-4.1-8b | 12.0 | 0.22 | 0.0078125 (=1/128) | 16.0 | 1e7 | true |
| granite-4.1-30b | 12.0 | 0.175 | 0.0078125 | 16.0 | 5e7 | true |
| granite-4.2-3b | 1.0 | 1.0 | 0.015625 | 1.0 | 1e7 | **false** |
| granite-4.2-8b | 1.0 | 1.0 | 0.0078125 | 1.0 | 1e7 | false |
| granite-4.2-30b | 1.0 | 1.0 | 0.0078125 | 1.0 | 5e7 | false |

Granite 4.2 drops muP. All its multipliers are 1.0, except `attention_multiplier`, which equals 1/head_dim (not 1/sqrt).

#### Code path (official vLLM)
1. Registry: `"GraniteForCausalLM": ("granite", "GraniteForCausalLM")` at `vllm/model_executor/models/registry.py:123`. `GraniteSWAForCausalLM` (line 128) maps to the same class.
2. Embedding: `GraniteModel.embed_tokens = VocabParallelEmbedding` at `vllm/model_executor/models/granite.py:335`. The embedding is multiplied by `embedding_multiplier` at `granite.py:373`. The op is a plain `F.embedding` plus a mul.
3. Per layer (`GraniteDecoderLayer`, `granite.py:245`):
   1. `input_layernorm = RMSNorm` at `granite.py:283`.
   2. `GraniteAttention` (`granite.py:145`):
      - QKV projection: `QKVParallelLinear` at `granite.py:183`. Unquantized, this is a GEMM.
      - Softmax scale is `config.attention_multiplier`, not 1/sqrt(d) (`granite.py:180`).
      - Per-layer window, rope_theta and sinks come from `granite_layer_attn_params` (`granite.py:74-104`). Granite 4.x configs have no `layer_types`, so they get full attention and no sinks.
      - RoPE: `get_rope(...)` at `granite.py:206`, applied at `granite.py:239`. This is `RotaryEmbedding` (`vllm/model_executor/layers/rotary_embedding/base.py:139`). `forward_cuda` calls `ops.rotary_embedding` (`base.py:221-244`).
      - Attention: `Attention(...)` at `granite.py:219`. This is the generic v1 attention layer, and the backend is chosen by the platform.
      - Output projection: `o_proj = RowParallelLinear` at `granite.py:192`.
   3. Residual: `residual + h * residual_multiplier` at `granite.py:300`.
   4. `post_attention_layernorm` (RMSNorm) at `granite.py:284`.
   5. `GraniteMLP` (`granite.py:107`): `gate_up_proj` MergedColumnParallelLinear (`:118`), then `SiluAndMul` (`:136`, CUDA op `torch.ops._C.silu_and_mul`, `vllm/model_executor/layers/activation.py:138`), then `down_proj` (`:125`).
   6. Second residual with `residual_multiplier` at `granite.py:305`.
4. Final `norm` (RMSNorm) at `granite.py:353`. RMSNorm `forward_cuda` delegates to `forward_native` (`vllm/model_executor/layers/layernorm.py:96-116`), which dispatches the vLLM IR op `ir.ops.rms_norm` (`layernorm.py:81`). That op has two implementations:
   - native PyTorch at `vllm/ir/ops/layernorm.py:10-21`;
   - `vllm_c` → `torch.ops._C.rms_norm` at `vllm/kernels/vllm_c.py:24-45`.
5. LM head and logits: `ParallelLMHead` at `granite.py:421`. When `tie_word_embeddings` is set, its weights are tied to `embed_tokens`. `LogitsProcessor(scale = 1/logits_scaling)` is built at `granite.py:430-436`.

#### Requirements
Abbreviations used in this and later tables:
- **SR (shared runtime)**: the vLLM-wide runtime that a separate agent is verifying for sm_37.
- **`_C`**: vLLM's compiled C++/CUDA extension. Whether it builds for sm_37 is an SR question.

| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Embedding × multiplier | torch `F.embedding` | none | any | OK | `granite.py:335,373` | - | - |
| RMSNorm | `torch.ops._C.rms_norm` (`vllm_c`) or native IR op | `_C` built for sm_37 (SR) | build-dependent | depends on SR build | `vllm/kernels/vllm_c.py:24-45` | `vllm/ir/ops/layernorm.py:10-21` (FP32 math in torch) | Yes, pure torch |
| Linear (BF16/FP16/FP32) | `UnquantizedLinearMethod` → `F.linear` / cuBLAS | dtype support | any | FP32 only per `cuda.py:263` | `vllm/platforms/cuda.py:255-264` | - | FP32 GEMM works |
| RoPE (neox) | `ops.rotary_embedding` (`_C`) | `_C` (SR) | build-dependent | depends on SR | `rotary_embedding/base.py:221-244` | `RotaryEmbedding.forward_native` at `base.py:203` | Yes, pure torch |
| Attention (full, GQA, head_dim 64/128, no sinks) | v1 backends: FLASH_ATTN (cc ≥ 8.0, `vllm/v1/attention/backends/flash_attn.py:451-452`), FlashInfer (≥ SM80, `flashinfer.py:514+`), TRITON_ATTN (says any CC, `triton_attn.py:413-414`, but it is Triton), FlexAttention (Triton via inductor) | FA2 or Triton | ≥ 8.0 or Triton | **No working backend on sm_37** in official vLLM (SR) | as cited | none in official vLLM for CUDA. The fork has a K80 SDPA backend: `vllm37: vllm/attention/backends/torch_sdpa.py:5,217` | fork-only |
| SiluAndMul | `torch.ops._C.silu_and_mul` | `_C` (SR) | build-dependent | depends on SR | `activation.py:138,146-151` | `SiluAndMul.forward_native` at `activation.py:141-144` | Yes |
| Logits scale | torch mul in LogitsProcessor | none | any | OK | `granite.py:430-436` | - | - |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| compressed-tensors FP8 W8A8 (channel/token-dynamic): 4.1-*-fp8, 4.2-*-fp8 | `CompressedTensorsConfig` (overall min 70). Scheme `CompressedTensorsW8A8Fp8` (min 89). Below 89 it falls back to `CompressedTensorsW8A16Fp8`, Marlin (min 75). | 89 (W8A8) / 75 (W8A16 fallback) | `quantization/compressed_tensors/compressed_tensors.py:117-118`. Selection at `:834-875`. Scheme check `:950`. `schemes/compressed_tensors_w8a8_fp8.py:83-85`, `schemes/compressed_tensors_w8a16_fp8.py:53-55`. Enforced at `vllm/config/vllm.py:943`. |
| compressed-tensors MXFP4 (4.2-*-mxfp4) | `CompressedTensorsW4A4Mxfp4` | 80 | `compressed_tensors.py:767-768`, `schemes/compressed_tensors_w4a4_mxfp4.py:45-46` |
| compressed-tensors NVFP4 (4.2-*-nvfp4) | `CompressedTensorsW4A4Fp4` | 75 | `compressed_tensors.py:756-765`, `schemes/compressed_tensors_w4a4_nvfp4.py:35-36` |
| GGUF (4.1/4.2-*-GGUF) | **No GGUF method in official vLLM main**. `QuantizationMethods` (`quantization/__init__.py:15-50`) lists no `gguf`. `vllm/model_executor/model_loader/` has no gguf loader. | n/a | as cited. A grep of `vllm/` for "gguf" finds only `models/exaone_moe.py:225`. |
| MLX | not supported by vLLM | n/a | - |

Conclusion: every vendor-quantized Granite dense format needs cc ≥ 75 in official vLLM, so none is usable on sm_37. Only the BF16 checkpoints can be used, up-cast to FP32 or FP16.

#### In the K80 fork (vllm37)
- Model file present: `vllm37: vllm/model_executor/models/granite.py`. It reads `rope_theta` via `getattr(config,"rope_theta",10000)` at `granite.py:192`. All 4.1/4.2 configs carry `rope_theta`.
- Registry entry present: `vllm37: vllm/model_executor/models/registry.py:81` (`"GraniteForCausalLM": ("granite","GraniteForCausalLM")`).
- Attention backend for K80 present: `vllm37: vllm/attention/backends/torch_sdpa.py` (its docstring names Tesla K80, lines 5 and 217).

#### Verdict for K80
Granite dense is the **most portable** family here. Its ops are attention, RoPE, RMSNorm, SwiGLU and GEMM, with no Mamba and no MoE.

Blockers in official vLLM:
1. **dtype**: official vLLM allows only FP32 on cc < 6.0 (`cuda.py:263`). 4.1-3b and 4.2-3b need about 13.6/14.6 GB in FP32, which is more than one 12 GB K80 die. FP16 would need about 6.8/7.3 GB.
   - Fix: allow FP16 storage in the fork, or use TP=2 across both dies.
   - **UNVERIFIED** whether Granite 4.1's muP scaling (×12 embedding) overflows in FP16. I did not test it.
2. **Attention**: official vLLM has no sm_37 backend (FA ≥ 8.0, FlashInfer ≥ 8.0, the rest are Triton). Fix: use the fork's `torch_sdpa` backend.
3. **`_C` ops** (rms_norm, silu_and_mul, rotary_embedding): build-dependent. Fix: either build `_C` for sm_37 (SR), or use the pure-torch `forward_native` paths that already exist (`vllm/ir/ops/layernorm.py:10`, `activation.py:141`, `rotary_embedding/base.py:203`).
4. **Quantized repos**: unusable on K80 (min cc 75/80/89, see table). GGUF is not supported in official main at all.

The fork already has the model file, the registry entry and an SDPA backend. Granite 4.x dense should therefore load in the fork with `dtype=float16/float32`, provided the fork's `granite.py` matches the 4.x configs. Running it on the K80 is **UNVERIFIED**.

---

### IBM Granite 4.0 hybrid (GraniteMoeHybridForCausalLM, model_type `granitemoehybrid`)

#### Official releases
All rows below come from config.json (`layer_types`, `num_local_experts`, `position_embedding_type`). "a" = attention, "m" = mamba2. Every layer also has a shared MLP (`shared_intermediate_size>0`). Every layer is MoE when `num_local_experts>0`.

| Repo | Released | Params | Format | Quant config | Context | Layers (attn/mamba/moe) / KV heads / head_dim |
|---|---|---|---|---|---|---|
| granite-4.0-h-small (+ -base) | 2025-09-16 | 32.21B | BF16 | none | 131072 | 4/36/40 (72 experts, top-10, expert inter 768, shared 1536) / 8 / 128; NoPE |
| granite-4.0-h-small-FP8 | 2025-10-01 | 32.64B | FP8 | `compressed-tensors` float-quantized W8 channel, A8 token-dynamic; targets `Linear` and `GraniteMoeHybridParallelExpertsLinear`; router `...block_sparse_moe.router.layer` ignored | 131072 | as above |
| granite-4.0-h-tiny (+ -base) | 2025-09-16 | 6.94B | BF16 | none | 131072 | 4/36/40 (64 experts, top-6, inter 512, shared 1024) / 4 / 128; NoPE |
| granite-4.0-h-micro (+ -base) | 2025-09-16 | 3.19B | BF16 | none | 131072 | 4/36/0 (dense shared MLP 8192) / 8 / 64; NoPE |
| granite-4.0-h-1b (+ -base) | 2025-10-07 | 1.46B | BF16 | none | 131072 | 4/36/0 (dense 4096) / 4 / 128; NoPE |
| granite-4.0-h-350m (+ -base) | 2025-10-07 | 0.34B | BF16 | none | 32768 | 4/28/0 (dense 2048) / 4 / 64; NoPE |
| granite-4.0-micro (+ -base) | 2025-09-16 | 3.40B | BF16 | none | 131072 | **40/0/0** (attention-only, dense 8192) / 8 / 64; RoPE θ=1e7 |
| granite-4.0-1b (+ -base) | 2025-10-07 | 1.63B | BF16 | none | 131072 | **40/0/0** / 4 / 128; RoPE θ=1e7 |
| granite-4.0-350m (+ -base) | 2025-10-07 | 0.35B | BF16 | none | 32768 | **28/0/0** / 4 / 64; RoPE θ=1e7 |
| granite-4.0-tiny-preview (+ -base-preview) | 2025-04-30 | 6.67B | BF16 (base: F32) | none | not read | not read |
| *-GGUF for every model above | 2025-09-24 … 10-23 | - | GGUF Q2_K…Q8_0, bf16 (file list of h-tiny-GGUF) | n/a | - | - |

Mamba and muP parameters:
- **h-tiny, h-small, h-micro**: `mamba_d_state=128`, `mamba_d_conv=4`, `mamba_expand=2`, `mamba_n_groups=1`, `mamba_chunk_size=256`, `mamba_conv_bias=true`.
- **Mamba heads × head_dim**: h-small 128×64, h-tiny 48×64, h-micro 64×64, h-1b 48×64, h-350m 48×32.
- **Attention layer positions** in h-* (from `layer_types`): indices 5, 15, 25, 35 for the 40-layer models. h-350m: 10, 13, 17, 27.
- **muP**: `embedding_multiplier=12`, `residual_multiplier=0.22` (0.246 for h-350m, 0.263 for 350m), `attention_multiplier` 0.0078125 (h-small, h-tiny, h-1b) or 0.015625 (h-micro, h-350m, micro, 350m).
- **logits_scaling**: h-small 16, h-tiny 6, h-micro 8, h-1b 6, h-350m 3, micro 10, 1b 8, 350m 4.
- **Non-h models**: `granite-4.0-{micro,1b,350m}` are labelled hybrid, but their `layer_types` are all `"attention"`. They contain no Mamba layers.

#### Code path (official vLLM)
1. Registry: `"GraniteMoeHybridForCausalLM": ("granitemoehybrid", ...)` at `vllm/model_executor/models/registry.py:125`. The class declares `HasInnerState` and `IsHybrid` (`granitemoehybrid.py:587-592`).
2. Embedding: `VocabParallelEmbedding` at `granitemoehybrid.py:347`, multiplied by `embedding_multiplier` at `:389`.
3. Layer dispatch: `ALL_DECODER_LAYER_TYPES[config.layer_types[i]]` at `granitemoehybrid.py:355`. The map at `:322-329` accepts both "attention"/"mamba" and "full_attention"/"linear_attention".
4. **Mamba layer** (`GraniteMoeHybridMambaDecoderLayer`, `granitemoehybrid.py:58`): RMSNorm (`:110`), then `MambaMixer2` (`:73-89`), then residual × `residual_multiplier` (`:124`). `MambaMixer2` is in `vllm/model_executor/layers/mamba/mamba_mixer2.py:264`.
   1. `in_proj` (MergedColumnParallelLinear or ColumnParallelLinear, `mamba_mixer2.py:357/382`) is a GEMM.
   2. `torch.ops.vllm.mamba_mixer2` is a custom op (`:596`, registered `:1228-1232`) that runs `conv_ssm_forward` (`:712`).
   3. **Prefill**:
      - Conv: `causal_conv1d_fn` (`:821`), a **Triton** kernel `_causal_conv1d_fwd_kernel` at `ops/causal_conv1d.py:16-17,481`.
      - SSD scan: `mamba_chunk_scan_combined_varlen` (`:850`, `ops/ssd_combined.py:157`). It runs five **Triton** kernels: `_chunk_cumsum_fwd` and `_chunk_state_fwd` (`ops/ssd_chunk_state.py:303,353`), `_state_passing_fwd` (`ops/ssd_state_passing.py:102`), `_bmm_chunk_fwd` (`ops/ssd_bmm.py:148`), and `_chunk_scan_fwd` (`ops/ssd_chunk_scan.py:418`). Call order is at `ssd_combined.py:66-110`. The kernels use `tl.dot` (`ssd_chunk_scan.py:301,328,370`; `ssd_bmm.py:131`; `ssd_chunk_state.py:284`).
      - Before the KV cache is allocated, `_warmup_ssd_kernels` (`:614-680`) runs Triton autotune.
   4. **Decode**:
      - Conv: `causal_conv1d_update` (`:937`), **Triton** `_causal_conv1d_update_kernel` (`ops/causal_conv1d.py:762-763,1096`).
      - State update: `selective_state_update` (`:1078`). It goes through `ops/ssu_dispatch.py:660` to the backend chosen in `initialize_mamba_ssu_backend` (`ssu_dispatch.py:593-647`):
        - the default `MambaConfig.backend = TRITON` (`vllm/config/mamba.py:40`) uses `TritonSSUBackend` (`ssu_dispatch.py:275`) and the Triton `_selective_scan_update_kernel` (`ops/mamba_ssm.py:239-240,496`);
        - the alternative is `FlashInferSSUBackend` (`ssu_dispatch.py:333-339`, `flashinfer.mamba.selective_state_update`);
        - the CPU backend is forced only when `current_platform.is_cpu()` (`ssu_dispatch.py:614-620`).
      - With `use_replayssm` (off by default, `mamba_mixer2.py:531-533`), the Triton or FlashInfer replay kernels are used instead.
   5. Gated norm: `Mixer2RMSNormGated` (`mamba_mixer2.py:88`).
      - With `n_groups==1` and TP=1 (true for every Granite h-* model, `mamba_n_groups=1`), it uses the **Triton** `rms_norm_gated` (`mamba_mixer2.py:180-190` → `ops/layernorm_gated.py:13-14,145`).
      - Otherwise it uses `forward_native` (pure torch, `:119-168`).
   6. `out_proj` (RowParallelLinear, `:496`).
   7. After the mixer: RMSNorm (`:111`). Then one of: MoE (`GraniteMoeMoE`, `granitemoe.py:74`) plus shared MLP (`GraniteMoeSharedMLP`, `granitemoeshared.py:42`, SiluAndMul at `:72`); or shared MLP only when there are 0 experts (`granitemoehybrid.py:128-139`). Then residual × `residual_multiplier`.
5. **Attention layer** (`GraniteMoeHybridAttentionDecoderLayer`, `granitemoehybrid.py:146`; `GraniteMoeHybridAttention`, `:226`):
   - `head_dim = hidden/heads` (`:241`).
   - RoPE only when `position_embedding_type == "rope"` (`:277-285`). All h-* models are `"nope"`, so `rotary_emb=None`. The non-h models use rope.
   - `Attention(..., scale=attention_multiplier)` at `:287-290`. MoE and shared MLP follow, as in the Mamba layer.
6. **MoE** (`GraniteMoeMoE`):
   - Router: `GateLinear` (`granitemoe.py:99`). On non-Hopper/Blackwell GPUs it falls back to `F.linear` (`fused_moe/router/gate_linear.py:287`).
   - Experts: `FusedMoEFactory(..., renormalize=True)` (`granitemoe.py:106-117`; factory at `fused_moe/layer.py:88`). Softmax top-k uses `ops.topk_softmax` (`_C`) at `fused_moe/router/fused_topk_router.py:26-33`.
   - Unquantized backend order on CUDA (`fused_moe/oracle/unquantized.py:69-74`): FlashInfer TRT-LLM, then FlashInfer CUTLASS, then **TRITON** (`fused_moe/experts/triton_moe.py:65`, kernel `fused_moe_kernel` at `fused_moe/fused_moe.py:297-298`), then BATCHED_TRITON.
   - `moe_align_block_size` is a `_C` op (`fused_moe/moe_align_block_size.py:153`; `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`).
7. Final norm (`granitemoehybrid.py:372`). `LogitsProcessor(scale=1/logits_scaling)` at `:680-684`.
8. **State cache manager**:
   - Spec and shape: `MambaMixer2.get_kv_cache_spec/get_state_shape/get_state_dtype` (`mamba_mixer2.py:1096-1145`) produce a `MambaSpec` (`vllm/v1/kv_cache_interface.py:1050`).
   - Shapes come from `MambaStateShapeCalculator.mamba2_state_shape` (`layers/mamba/mamba_utils.py:195-221`): conv state `(conv_dim, d_conv-1)` and SSM state `(heads, head_dim, d_state)`. Dtypes come from `mamba_utils.py:75`.
   - Allocation and paging: v1 `MambaManager` (`vllm/v1/core/single_type_kv_cache_manager.py:1455`). Metadata: `Mamba2AttentionBackend` and its builder (`vllm/v1/attention/backends/mamba2_attn.py:94,122`).
   - Model hooks: `get_mamba_state_shape_from_config` and friends at `granitemoehybrid.py:612-655`.
   - Prefix-caching "align" mode adds more Triton kernels (`vllm/v1/worker/mamba_utils.py:38-634`, called at `gpu_model_runner.py:4288-4297,1588-1593`). That mode is off by default (`mamba_cache_mode="none"`, `vllm/config/cache.py:189`).

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| causal_conv1d prefill | Triton `_causal_conv1d_fwd_kernel` | Triton | Triton target (not Kepler) | **Blocked** | `ops/causal_conv1d.py:16-17,481` | `causal_conv1d_fn_cpu` at `ops/cpu/causal_conv1d.py:13-84`: a per-sequence loop over `F.conv1d` with `.item()` calls. It is swapped in only when `is_cpu()` (`causal_conv1d.py:1281-1288`). | Plausible: device-agnostic torch code, but slow and needs re-wiring. Not tested (**UNVERIFIED**). |
| causal_conv1d decode | Triton `_causal_conv1d_update_kernel` | Triton | Triton | **Blocked** | `causal_conv1d.py:762-763,1096` | `causal_conv1d_update_torch` (pure torch) at `ops/cpu/causal_conv1d.py:119`. It has no `conv_state_indices` support, so a gather/scatter wrapper is needed. `causal_conv1d_update_cpu` is a C++ CPU op (`:87-116`), not usable on GPU. | Partially, with a wrapper |
| SSD chunked scan (prefill) | 5 Triton kernels with `tl.dot` | Triton | Triton | **Blocked** | `ops/ssd_combined.py:27-155`; `ssd_chunk_scan.py:148,418` | CPU path `_mamba_chunk_scan_combined_fwd_cpu` (`ops/cpu/mamba_ssm.py:10-80`) calls the **C++ CPU** op `ops.mamba_chunk_scan_fwd_cpu` (`csrc/cpu/mamba_cpu.cpp`). Pure torch exists only in tests: `ssd_minimal_discrete` at `tests/kernels/mamba/test_mamba_ssm_ssd.py:43`. | **None in vllm/**. Needs a new torch or CUDA implementation. |
| selective_state_update (decode) | Triton `_selective_scan_update_kernel` (default) or FlashInfer `selective_state_update` | Triton / FlashInfer | Triton / FI | **Blocked** | `ops/mamba_ssm.py:239,496`; `ssu_dispatch.py:275-339`; `vllm/config/mamba.py:40` | `CPUSSUBackend` (`ssu_dispatch.py:408-424`) → C++ `ops.selective_state_update_cpu`. Pure torch only in tests: `selective_state_update_ref` at `tests/kernels/mamba/utils.py:9`. | None in vllm/ (simple to write: per-token einsum) |
| Stochastic-rounding fp16 state | Triton inline PTX `cvt.rs.f16x2.f32` | PTX `cvt.rs` (Blackwell-class; **UNVERIFIED** exact SM) | n/a | off by default | `mamba_ssm.py:208-220`; `enable_stochastic_rounding=False` at `vllm/config/mamba.py:43` | disable it | n/a |
| Gated RMSNorm (n_groups=1) | Triton `_layer_norm_fwd_1pass_kernel` | Triton | Triton | **Blocked** | `mamba_mixer2.py:180-190`; `ops/layernorm_gated.py:13-14,145` | `Mixer2RMSNormGated.forward_native` at `mamba_mixer2.py:119-168` | Yes, pure torch |
| MoE experts (h-tiny, h-small) | Triton `fused_moe_kernel` (FlashInfer TRT-LLM/CUTLASS preferred if available) | Triton / FlashInfer | Triton / FI | **Blocked** | `fused_moe/oracle/unquantized.py:69-74`; `experts/triton_moe.py:118-119`; `fused_moe.py:297-298` | No CUDA pure-torch expert path in `vllm/`. CPU experts use C++ `cpu_fused_moe` (`experts/cpu_moe.py:303`). A reference exists only in tests: `torch_moe` at `tests/kernels/utils.py:295`. The fork has `vllm37: vllm/model_executor/layers/fused_moe/moe_torch_iterative.py`, which is not referenced from fork `layer.py` (grep). | Must be written or wired in |
| MoE top-k softmax | `_C` `topk_softmax` (cub, bf16 headers) | `_C` (SR) | build-dependent | SR | `router/fused_topk_router.py:26-33`; `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` | `_softmax_topk` (pure torch) at `router/cpu_router.py:138` | Yes |
| moe_align_block_size | `_C` op | `_C` (SR) | build-dependent | SR (only needed by Triton MoE) | `moe_align_block_size.py:153` | not needed by a torch MoE | n/a |
| Router GEMM | `F.linear` tier-6 fallback | none | any | OK | `router/gate_linear.py:287` | - | - |
| Attention (4 layers, NoPE) | as in the dense section | FA / Triton | ≥ 8.0 / Triton | **No official backend** | `flash_attn.py:451`, `triton_attn.py:413` | fork `torch_sdpa` | fork-only |
| RMSNorm, SiluAndMul, GEMMs | as in the dense section | `_C` / cuBLAS | - | as in the dense section | - | native | Yes |
| Mamba state copy/align (prefix caching) | Triton | Triton | Triton | avoid: off by default | `vllm/v1/worker/mamba_utils.py:38+`; `cache.py:189` | keep `mamba_cache_mode="none"` | Yes |
| SSD warmup / autotune | Triton | Triton | Triton | Blocked (runs on profile) | `mamba_mixer2.py:614-680,777` | remove when a non-Triton path is used | n/a |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| compressed-tensors FP8 W8A8 (h-small-FP8), linear layers | `CompressedTensorsW8A8Fp8` (89), else `CompressedTensorsW8A16Fp8` Marlin (75) | 75 | as in the dense section |
| compressed-tensors FP8 W8A8 (h-small-FP8), MoE experts | `CompressedTensorsW8A8Fp8MoEMethod`, which selects an FP8 MoE backend: Triton FP8 needs `supports_fp8()`; Marlin needs cc ≥ 7.5 | 75 | `compressed_tensors_moe/compressed_tensors_moe.py:147-158`; `experts/triton_moe.py:156-163`; `experts/marlin_moe.py:587-589`; `utils/marlin_utils_fp8.py:30-31` |
| GGUF | not supported in official main | n/a | `quantization/__init__.py:15-50` (no `gguf`) |

#### In the K80 fork (vllm37)
- Model file present: `vllm37: vllm/model_executor/models/granitemoehybrid.py`.
  - The layer map accepts only `"attention"` and `"mamba"` (`granitemoehybrid.py:306-309`). That matches the downloaded configs, which use those strings.
  - Supports `position_embedding_type == "rope"`/NoPE (`:261`).
- Registry entry present: `vllm37: vllm/model_executor/models/registry.py:83`.
- Mamba-2 stack present: `vllm37: vllm/model_executor/layers/mamba/mamba_mixer2.py` with ops `causal_conv1d.py`, `ssd_*.py`, `mamba_ssm.py`, `layernorm_gated.py`.
  - All are **Triton** (`@triton.jit` counts: causal_conv1d 2, ssd_chunk_state 3, mamba_ssm 3, …).
  - The v0 cache manager is `vllm37: vllm/model_executor/models/mamba_cache.py`.
  - `MambaMixer2.forward_native` is an empty stub (`vllm37: .../mamba_mixer2.py:422-430`, body `pass`).
- No CUDA Mamba-2 kernel. `vllm37: csrc/mamba/mamba_ssm/selective_scan_fwd.cu` exists, but it is the Mamba-1 selective scan (`CMakeLists.txt:248`).

#### Verdict for K80
- **granite-4.0-{micro,1b,350m}** (attention-only `layer_types`) never instantiate `MambaMixer2`. They need the same ops as Granite dense, plus `GraniteMoeSharedMLP` (SwiGLU). They are about as portable as Granite dense.
- **h-1b and h-350m** fit on one die in FP32 (1.46B/0.34B params). h-micro (3.19B) needs FP16 or TP=2.

Blockers for the **h-*** models, with evidence:
1. **Mamba-2 prefill SSD scan** is Triton-only on CUDA (`ssd_combined.py`). There is no GPU or torch fallback in `vllm/`.
   - Fix: write a torch SSD that handles varlen/`cu_seqlens` and initial states (start from `tests/kernels/mamba/test_mamba_ssm_ssd.py:43`), or a sequential CUDA scan for sm_37.
   - head_dim 32 or 64 and d_state 128 make the SSM state 48–128 × 64 × 128 FP32 per layer. A naive sequential scan in torch is O(seq) kernel launches.
   - A CUDA C kernel that keeps state in registers or global memory, not in 48 KB of smem, is the realistic option.
2. **Decode `selective_state_update`** is Triton (default) or FlashInfer. Fix: a short torch implementation (`tests/kernels/mamba/utils.py:9` pattern plus state-index gather/scatter), or port `csrc/cpu/mamba_cpu.cpp` logic to CUDA.
3. **causal_conv1d** prefill and decode are Triton. Fix: wire `ops/cpu/causal_conv1d.py:13` (`causal_conv1d_fn_cpu`, pure torch) for CUDA, and wrap `causal_conv1d_update_torch` (`:119`) with index gather/scatter.
4. **Gated RMSNorm** uses Triton for n_groups=1. Fix: force `Mixer2RMSNormGated.forward_native` (`mamba_mixer2.py:119`).
5. **MoE** (h-tiny, h-small) needs Triton or FlashInfer experts. Fix: a torch expert loop (pattern: `tests/kernels/utils.py:295`, or the fork's `moe_torch_iterative.py`) plus the torch top-k from `router/cpu_router.py:138`.
6. Attention, dtype and `_C` blockers are the same as for Granite dense. The fork already handles attention (SDPA) and its v0 Mamba cache. Remove the Triton SSD warmup (`mamba_mixer2.py:614`).
7. h-small (32B, 72 experts) needs about 64 GB in FP16. It does not fit one K80 card (2×12 GB), so it is out of scope for K80. h-tiny (6.94B) needs about 13.9 GB in FP16, which means TP=2 across both dies.

---

### NVIDIA Nemotron 3 / Nemotron-H (NemotronHForCausalLM, model_type `nemotron_h`)

#### Official releases
Pattern legend (`hybrid_override_pattern`, or `layers_block_type` for 3.5): M = Mamba-2, `-` = dense MLP (relu²), `*` = attention, E = MoE.

Facts common to all rows:
- Every Mamba config has `ssm_state_size=128`, `n_groups=8`, `conv_kernel=4`, `mamba_hidden_act=silu`, and `mlp_hidden_act=relu2`.
- No model has RoPE: `NemotronHAttention` (see below) builds no rotary embedding.
- Attention scale is `head_dim**-0.5`.
- No muP multipliers.

| Repo | Released | Params | Format | Quant config | Context | Layers (attn/mamba/moe) / KV heads / head_dim |
|---|---|---|---|---|---|---|
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 2026-03-07 | 3.97B | BF16 | none | 262144 | 42 total: 4 `*` / 21 M / 0 E (17 `-`) / 8 / 128; Mamba 96 heads × 80, chunk 256 |
| NVIDIA-Nemotron-3-Nano-4B-FP8 | 2026-03-12 | 3.97B | FP8 | `hf_quant_config.json`: modelopt 0.29.0, `quant_algo=FP8`, `kv_cache_quant_algo=FP8`, 46 excluded modules (lm_head, some mixer in/out_proj, attention q/k/v/o) | 262144 | as above |
| NVIDIA-Nemotron-3-Nano-4B-GGUF | 2026-03-07 | - | GGUF Q4_K_M only (file list) | n/a | - | - |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 (+ -Base, -Japanese) | 2025-08-12 | 8.89B | BF16 | none | 131072 | 56: 4 `*` / 27 M / 0 E (25 `-`) / 8 / 128; Mamba 128 × 80, chunk 128 |
| NVIDIA-Nemotron-Nano-9B-v2-FP8 | 2025-09-22 | 8.89B | FP8 | modelopt 0.34.1, `FP8`, kv none, 44 excluded (lm_head, all `mixer.conv1d`, …) | 131072 | as above |
| NVIDIA-Nemotron-Nano-9B-v2-NVFP4 | 2025-10-07 | 5.86B (packed) | NVFP4 | modelopt 0.34.1, `NVFP4`, group 16, kv none, 52 excluded | 131072 | as above |
| nvidia/NVIDIA-Nemotron-Nano-12B-v2 (+ -Base) | 2025-08-21 | 12.31B | BF16 | none | 131072 | 62: 6 `*` / 28 M / 0 E (28 `-`) / 8 / 128; Mamba 128 × 80 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (+ -Base) | 2025-12-04 | 31.58B (about 3B active per name) | BF16 | none; `mamba_ssm_cache_dtype=float32` | 262144 | 52: 6 `*` / 23 M / 23 E / 2 / 128; 128 routed experts top-6 (inter 1856) + 1 shared (3712); sigmoid router with `e_score_correction_bias`, `routed_scaling_factor=2.5`, `n_group=1`, `topk_group=1`; Mamba 64 × 64, chunk 128 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 | 2025-12-06 | 31.58B | FP8 | modelopt 0.29.0, `FP8`, `kv_cache_quant_algo=FP8`, 60 excluded | 262144 | as above |
| NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | 2025-12-20 | 18.24B (packed) | NVFP4 | modelopt 0.29.0, `NVFP4`, group 16, kv FP8, 60 excluded | 262144 | as above |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (+ -Base) | 2026-08-01 | 31.58B | BF16 | none; `mamba_ssm_cache_dtype=float32`; `num_nextn_predict_layers=1` (MTP) | 262144 | `layers_block_type`: 6 attention / 23 mamba / 23 moe, same shapes as Nano-30B / 2 / 128 |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 2026-08-04 | 17.82B (packed) | NVFP4 + FP8 mixed | modelopt 0.44.0rc5, `quant_algo=MIXED_PRECISION`: 5935 layers `W4A16_NVFP4`, 46 `FP8`; `kv_cache_quant_algo=FP8`; 72 excluded (embeddings, conv1d, gates, some attention) | 1048576 (config.json) | as above |
| …-NVFP4-DFlash / -DSpark | 2026-08-05 | 0.66B / 0.76B | NVFP4 draft models (`DFlashDraftModel`/`Qwen3DSparkModel`, model_type qwen3) | modelopt | - | speculative drafters, not Nemotron-H |

Excluded from this study: Super 120B and Ultra 550B (too large), Omni and VL variants, and Nemotron-H-4B/8B/47B/56B (2025 "Nemotron-H" family, `architectures` empty in the listing).

#### Code path (official vLLM)
1. Registry: `"NemotronHForCausalLM": ("nemotron_h", "NemotronHForCausalLM")` at `vllm/model_executor/models/registry.py:178`. `NemotronHPuzzleForCausalLM` is at `:179`, and the MTP head `NemotronHMTPModel` at `:680`. The class is `IsHybrid` (`nemotron_h.py:718-725`).
   - `NemotronHForCausalLMConfig` sets `mamba_ssm_cache_dtype` from the HF config when it is "auto" (`vllm/model_executor/models/config.py:784-804`). Nano-30B and Lightning set float32.
   - Config class: vLLM's own `NemotronHConfig` (`vllm/transformers_utils/configs/nemotron_h.py`) requires `hybrid_override_pattern` (`:209-230`). Lightning ships only `layers_block_type` and no `auto_map`. It then depends on Transformers' native `nemotron_h` config. **UNVERIFIED**: Transformers is not installed locally, so I could not check how it maps `layers_block_type` to the pattern.
2. Embedding: `VocabParallelEmbedding` at `nemotron_h.py:585`. There is no embedding multiplier.
3. Layer dispatch: `ALL_DECODER_LAYER_TYPES` `{"M","-","*","E"}` at `nemotron_h.py:549-554`, indexed by `hybrid_override_pattern[i]` at `:596`. Every layer is pre-norm with a fused residual: `RMSNorm(x, residual)`, which dispatches `ir.ops.fused_add_rms_norm` (`layernorm.py:88`).
4. **M layer**: `NemotronHMambaDecoderLayer` (`nemotron_h.py:378`) runs `MambaMixer2` (`:391-407`, `intermediate_size = mamba_num_heads*mamba_head_dim`). The kernel path is identical to the Granite hybrid section: Triton causal_conv1d, Triton SSD, Triton or FlashInfer SSU.
   - Difference: `n_groups=8`, so `Mixer2RMSNormGated.forward_cuda` takes **`forward_native`** (pure torch). The condition `self.n_groups != 1` is at `mamba_mixer2.py:180-181`.
5. **`-` layer**: `NemotronHMLPDecoderLayer` (`:279`) runs `NemotronHMLP` (`:93`):
   - `up_proj` ColumnParallelLinear (`:110`);
   - `get_act_fn("relu2")`, which is `ReLUSquaredActivation` (`activation.py:653-675`, CUDA op `torch.ops._C.relu_squared` at `:662`), called through `maybe_fused_act_quant` (`:131`);
   - `down_proj` (`:118`).
   - This is not a gated MLP: there is no `SiluAndMul`.
6. **`*` layer**: `NemotronHAttention` (`:427`):
   - `qkv_proj` (`:461`);
   - `self.scaling = head_dim**-0.5` (`:459`);
   - **no rotary**: `forward` goes straight from qkv to `attn` (`:492-501`);
   - `Attention(..., per_layer_sliding_window=config.sliding_window)` (`:481-490`). `sliding_window` is null in every downloaded config.
7. **E layer**: `NemotronHMoEDecoderLayer` (`:335`) runs `NemotronHMoE` (`:136`).
   - Router: `GateLinear(out_dtype=float32, force_fp32_compute=True)` (`:160-166`), which falls back to `F.linear` (`gate_linear.py:86,287`). `e_score_correction_bias` is FP32 (`:168-170`).
   - Shared expert: `NemotronHMLP` (relu²) at `:179-192`.
   - Experts: `FusedMoEFactory(use_grouped_topk=True, num_expert_group=1, topk_group=1, scoring_func="sigmoid", e_score_correction_bias, activation=relu2 no-mul, routed_scaling_factor=2.5)` (`:230-254`).
   - Router choice: `create_fused_moe_router` treats `n_group<=1 && topk_group<=1` as degenerate and falls through to `FusedTopKBiasRouter` (`fused_moe/router/router_factory.py:176-240`). That router calls the `_C` `topk_sigmoid` (`router/fused_topk_bias_router.py:56-65,228-236`). Expert GEMMs use the same unquantized backend order as Granite (Triton on generic CUDA).
   - No latent MoE: `moe_latent_size` is None in the 30B and Lightning configs (code at `:153-228`).
8. Final `norm_f` (`:619`, applied at `:659`), then `lm_head` (`:842`) and the logits processor. `tie_word_embeddings` is false in every downloaded config.
9. State cache: `get_mamba_state_shape_from_config` and `get_mamba_state_dtype_from_config` (`nemotron_h.py:762-824`) feed the same `MambaSpec`, `MambaManager` and `Mamba2AttentionBackend` path as the Granite hybrid. The SSM state is FP32 for Nano-30B and Lightning.

#### Requirements
Rows identical to the Granite hybrid table also apply here: causal_conv1d (both), SSD scan, `selective_state_update`, MoE Triton experts, attention, GEMM, and the Triton warmup. The rows below are the ones specific to Nemotron or that differ.

| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| SSD scan, conv, SSU | Triton (as in the Granite hybrid section) | Triton / FlashInfer | Triton | **Blocked** | `mamba_mixer2.py:821,850,937,1078` | none in `vllm/` for SSD or SSU. Torch conv at `ops/cpu/causal_conv1d.py:13,119`. | SSD and SSU must be written |
| Gated RMSNorm (n_groups=8) | `forward_native` (torch) | none | any | **OK as is** | `mamba_mixer2.py:180-181,119-168` | - | Yes |
| relu² activation | `torch.ops._C.relu_squared` | `_C` (SR) | build-dependent | SR | `activation.py:660-675` | `forward_native` = `torch.square(F.relu(x))` at `activation.py:666-668` | Yes |
| Fused add + RMSNorm | `ir.ops.fused_add_rms_norm` → `_C` or native | `_C` (SR) | - | SR | `layernorm.py:88-94` | native IR implementation at `vllm/ir/ops/layernorm.py:43+` | Yes |
| Router GEMM (FP32) | `F.linear` (FP32 weights) | none | any | OK | `nemotron_h.py:160-166`; `gate_linear.py:86,287` | - | - |
| Sigmoid + bias top-k (`routed_scaling` 2.5) | `_C` `topk_sigmoid` | `_C` (SR) | build-dependent | SR | `router_factory.py:176-240`; `fused_topk_bias_router.py:56-65,228-236` | Pure torch `grouped_topk` at `router/grouped_topk_router.py:81-130`. It is wrapped in `@torch.compile(backend=simple_compile_backend="inductor")` (`:76-80`, `vllm/platforms/interface.py:170`), and inductor needs Triton. Pure torch also exists at `cpu_router.py:26` (`_grouped_topk`) and in the `fused_topk_bias` tail at `:311+` (reached only on the ROCm-aiter path). | Yes, if called eagerly |
| MoE experts, relu² no-mul | Triton `fused_moe_kernel` | Triton / FlashInfer | Triton | **Blocked** | as in the Granite hybrid section | `tests/kernels/utils.py:295` (reference only) | Must be written |
| Attention (NoPE, GQA 2 or 8 KV heads, head_dim 128) | FA / FlashInfer / Triton | as in the dense section | ≥ 8.0 / Triton | **No official backend** | `flash_attn.py:451` | fork `torch_sdpa` | fork-only |
| FP8 KV cache (FP8 and NVFP4 repos specify `kv_cache_quant_algo=FP8`) | FP8 cache | FP8 hardware conversion | **UNVERIFIED** threshold | Not applicable on K80. Use `kv_cache_dtype=auto`. | `hf_quant_config.json` of Nano-4B-FP8, Nano-30B-FP8/NVFP4, Lightning-NVFP4 | - | - |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| ModelOpt FP8 (Nano-4B-FP8, 9B-v2-FP8, Nano-30B-FP8) | `ModelOptFp8Config` ("modelopt") | **80** | `quantization/modelopt.py:400,430-438` |
| ModelOpt NVFP4 (9B-v2-NVFP4, Nano-30B-NVFP4) | `ModelOptNvFp4Config` ("modelopt_fp4") | 75 | `modelopt.py:717,747-755` |
| ModelOpt MIXED_PRECISION, W4A16_NVFP4 + FP8 (Lightning-NVFP4) | `ModelOptMixedPrecisionConfig` ("modelopt_mixed"): Marlin W4A16 and Marlin FP8 | 75 | `modelopt.py:1501,1554-1569` (comment: "Turing and up (SM75+) … Marlin") |
| GGUF (Nano-4B-GGUF Q4_K_M) | not supported in official main | n/a | `quantization/__init__.py:15-50` |

Global check: `vllm/config/vllm.py:943-947` raises an error if `capability < quant_config.get_min_capability()`. So every Nemotron quantized repo is rejected on sm_37.

#### In the K80 fork (vllm37)
- Model file present: `vllm37: vllm/model_executor/models/nemotron_h.py`. Its `ALL_DECODER_LAYER_TYPES` holds only `"M"`, `"-"` and `"*"` (`nemotron_h.py:290-294`), with **no `"E"` (MoE)**. So **Nano-30B-A3B and Lightning-30B-A3B cannot load in the fork**. Nano-4B and Nano-9B/12B-v2 (M/-/* only) are structurally supported.
- Registry entry present: `vllm37: vllm/model_executor/models/registry.py:114`.
- The Mamba-2 path is the same Triton-only stack as in the Granite hybrid section (`vllm37: vllm/model_executor/layers/mamba/ops/*`). `MambaMixer2.forward_native` is a stub (`vllm37: .../mamba_mixer2.py:422-430`).

#### Verdict for K80
Ranking by porting effort: Nano-4B < Nano-9B-v2 < 12B-v2 << Nano-30B-A3B ≈ Lightning-30B-A3B.

Blockers, with evidence:
1. **Mamba-2 SSD prefill and SSU decode**: Triton-only on CUDA (`ssd_combined.py`, `mamba_ssm.py:239`). There is no in-tree GPU or torch fallback, and this blocks every Nemotron model. The fix is the same as for the Granite hybrid: a torch or CUDA SSD (varlen, initial states, FP32 state) and a torch SSU.
   - Nemotron state per Mamba layer is 96 × 80 × 128 FP32 (about 3.9 MB) for Nano-4B and 128 × 80 × 128 (about 5.2 MB) for 9B. This is arithmetic from the config.
   - `n_groups=8` means B and C are shared across 12–16 heads each.
2. **causal_conv1d**: Triton. Fix: wire the pure-torch `ops/cpu/causal_conv1d.py:13/119` for CUDA.
3. **MoE** (Nano-30B, Lightning only): Triton experts plus `_C` `topk_sigmoid`. Fix: a torch expert loop with relu² (no gate), a sigmoid+bias top-k in eager torch (`grouped_topk_router.py:81` logic without `torch.compile`), and `routed_scaling_factor=2.5`.
   - The fork also lacks the `"E"` layer type, so the newer `nemotron_h.py` MoE code (`nemotron_h.py:136-277,335-376`) must be backported.
   - Size: 31.6B params is about 63 GB in FP16, which does not fit on a 24 GB K80 card. These models are effectively out of scope.
4. **Attention, dtype, `_C`**: same as the Granite sections.
   - Nemotron has no RoPE, so the rotary op is not needed.
   - Nano-4B needs about 15.9 GB in FP32 or about 7.9 GB in FP16 (from 3.97B params). That means FP16 on one die or FP32 with TP=2.
   - **UNVERIFIED**: FP16 numerical safety of relu² activations and Mamba states. NVIDIA ships BF16 only, and the 30B and Lightning configs ask for an FP32 SSM cache.
5. **Quantized repos**: ModelOpt FP8 needs cc ≥ 80, NVFP4 and mixed precision need 75, and GGUF is unsupported in official vLLM. None is usable on K80.
6. **Lightning config**: it uses `layers_block_type`, and the fork's `nemotron_h.py` reads `hybrid_override_pattern` (`vllm37: nemotron_h.py:322`). **UNVERIFIED** whether the fork's Transformers provides it, but this is moot because Lightning is MoE.

## 6. Mistral (Ministral 3 / Magistral / Devstral Small 2) and AllenAI Olmo 3


Sources: official vLLM `/home/jack/src/vllm` (main 3ca00a8261, 2026-10-07; all `file:line` below are in this tree unless prefixed `vllm37:`), K80 fork `/home/jack/src/vllm37`.
HF data: `hf models ls --author mistralai|allenai --expand created_at,safetensors` (saved locally, not kept). Configs: `hf download <repo> config.json params.json` into a local scratch directory (not kept). The "Released" column is the HF repo `created_at`. The public announcement may come later.

---

### Mistral (Ministral 3 / Magistral Small 2509 / Devstral Small 2)

#### Official releases
All repos below ship `config.json` (HF), `params.json` + `consolidated*.safetensors` (Mistral native format) and `tekken.json` (the mistral_common tokenizer), per `hf models info --expand siblings`. The -GGUF repos ship only `.gguf` files (+ `-mmproj.gguf` for vision).

| Repo | Released (repo created_at) | Params (safetensors total) | Format | Quant config | Context (yarn original) | Layers / KV heads / head_dim | Sliding window | Vision |
|---|---|---|---|---|---|---|---|---|
| mistralai/Ministral-3-3B-Instruct-2512 | 2025-10-31 | 3.85B (BF16 0.82B + F8_E4M3 3.03B) | FP8 HF + consolidated | config.json `quant_method: fp8`, `activation_scheme: static`, `weight_block_size: null` (per-tensor), `modules_to_not_convert: [vision_tower, multi_modal_projector, lm_head]`; params.json `quantization: {qformat_weight: fp8_e4m3, qscheme_act: TENSOR}` | 262144 (yarn factor 16, orig 16384); `llama_4_scaling_beta 0.1`; rope_theta 1e6 | 26 / 8 / 128 (hidden 3072, 32 q-heads, ff 9216, tied emb) | `sliding_window: null` | Pixtral 24L, 1024 hidden, image 1540, patch 14 |
| mistralai/Ministral-3-3B-Instruct-2512-BF16 | 2025-10-31 | 4.25B total (3.85B BF16 reported) | BF16 HF + consolidated | none | same | 26 / 8 / 128 | null | Pixtral (same) |
| mistralai/Ministral-3-3B-Base-2512, -3B-Reasoning-2512 | 2025-10-31 | 4.25B | BF16 only | none | (not downloaded; Reasoning listing shows BF16 only) | (same family) | – | Pixtral |
| mistralai/Ministral-3-8B-Instruct-2512 | 2025-10-31 | 8.92B (BF16 1.50B + F8 7.42B) | FP8 HF + consolidated | same FP8 per-tensor static scheme | 262144 (factor 16, orig 16384), theta 1e6, l4 beta 0.1 | 34 / 8 / 128 (hidden 4096, ff 14336) | null | Pixtral 24L |
| mistralai/Ministral-3-8B-Instruct-2512-BF16, -8B-Base, -8B-Reasoning | 2025-10-31 | 8.92B | BF16 | none | (BF16 config not downloaded; listing dtype BF16) | 34 / 8 / 128 | – | Pixtral |
| mistralai/Ministral-3-14B-Instruct-2512 | 2025-10-31 | 13.95B (BF16 1.78B + F8 12.16B) | FP8 HF + consolidated | same FP8 per-tensor static scheme | 262144 (factor 16, orig 16384), theta 1e9, l4 beta 0.1 | 40 / 8 / 128 (hidden 5120, ff 16384) | null | Pixtral 24L |
| mistralai/Ministral-3-14B-Reasoning-2512 (also -14B-Instruct-BF16, -14B-Base) | 2025-10-31 | 13.95B | BF16 | none (config has no quantization_config) | 262144 (factor 16, orig 16384), theta 1e9 | 40 / 8 / 128 | null | Pixtral |
| mistralai/Ministral-3-{3B,8B,14B}-{Instruct,Reasoning}-2512-GGUF | 2025-10-31 | – | GGUF: BF16, Q8_0, Q5_K_M, Q4_K_M + BF16 mmproj (file list for 3B and 14B Instruct) | GGUF | – | – | – | mmproj file |
| mistralai/Magistral-Small-2509 | 2025-09-12 | 24.01B BF16 | BF16 HF + consolidated | none | 131072, **no yarn**, rope_theta 1e9; text `model_type: mistral` (not ministral3) | 40 / 8 / 128 (hidden 5120, ff 32768) | null | Pixtral 24L |
| mistralai/Magistral-Small-2509-GGUF | 2025-09-12 | – | GGUF BF16/Q8_0/Q5_K_M/Q4_K_M (no mmproj listed) | – | – | – | – | – |
| mistralai/Devstral-Small-2-24B-Instruct-2512 | 2025-11-28 | 24.01B (BF16 1.78B + F8 22.23B) | **FP8 only** HF + consolidated (2 shards); no -BF16 or -GGUF sibling in the mistralai listing | FP8 per-tensor static, same as Ministral 3 | 393216 (yarn factor 48, orig 8192), theta 1e8, l4 beta 0.1 (orig 8192) | 40 / 8 / 128 (hidden 5120, ff 32768) | null | Pixtral 24L |

#### Code path (official vLLM)
0. Format auto-detection. `config_format=auto` selects **mistral** when the repo has `consolidated*.safetensors` and `params.json` (`vllm/transformers_utils/config.py:868-877`, `repo_utils.py:230-242`). `load_format=auto` selects **mistral** when `consolidated*.safetensors` exists (`model_executor/model_loader/default_loader.py:161-173`). `tokenizer_mode=auto` selects mistral the same way (`vllm/tokenizers/registry.py:142-155`). So all mistralai repos above default to the native `params.json`/consolidated/tekken path.
   - Native path: `adapt_config_dict` (`transformers_utils/configs/mistral.py:12-93`) sets `architectures=["MistralForCausalLM"]` (:53). It maps `yarn` to `rope_parameters{rope_type: yarn, beta_fast, beta_slow, factor, original_max_position_embeddings}`, and `apply_scale: false` becomes `attention_factor=1.0` (:114-143). It keeps the `llama_4_scaling` dict (:58-68), maps `quantization` fp8_e4m3/TENSOR to `{quant_method: fp8, activation_scheme: static}` (:199-211), and wraps a vision model as `PixtralForConditionalGeneration` (:96-111).
   - HF path (`--config-format hf --load-format hf`): `Mistral3ForConditionalGeneration` (`registry.py:516-519`, `models/mistral3.py:405`). The text config `model_type: ministral3` resolves through transformers' `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` (`config/vllm.py:985-994`) to `Ministral3ForCausalLM`, which maps to `mistral.MistralForCausalLM` (`registry.py:170`). Magistral's text `model_type: mistral` maps to `MistralForCausalLM` (`registry.py:171`).
1. Embedding → `VocabParallelEmbedding` (`models/llama.py:382`; class `layers/vocab_parallel_embedding.py:205`, forward :518) → plain gather (torch `F.embedding`).
2. Per layer (`MistralDecoderLayer`, `models/mistral.py:144-204`, extends `LlamaDecoderLayer` `llama.py:251`): `input_layernorm` RMSNorm (`llama.py:308`, `layers/layernorm.py:37`). Fused add+norm `forward_cuda` :96 → CUDA `csrc/libtorch_stable/layernorm_kernels.cu`.
3. QKV → `QKVParallelLinear` (`llama.py:165`) → unquantized: `F.linear`/cuBLAS. FP8 checkpoints: `Fp8LinearMethod` (see the Quantized format table).
4. RoPE → `get_rope(...)` (`llama.py:243`) → `rope_type == "yarn"` branch (`layers/rotary_embedding/__init__.py:240-276`) → **`YaRNScalingRotaryEmbedding`** (`layers/rotary_embedding/yarn_scaling_rope.py:10-85`). It precomputes a cos/sin cache of `original_max_position * factor` rows (:76-85): 262144 rows for Ministral 3 and 393216 for Devstral 2, ×128×fp32, so 128 MB and 192 MB. mscale = 1 (native: `attention_factor=1.0`; HF: `mscale/mscale_all_dim = 1/1`, :38-46). Applied by `RotaryEmbedding.forward_cuda` → `ops.rotary_embedding` (`rotary_embedding/base.py:221-252`, CUDA `csrc/libtorch_stable/pos_encoding_kernels.cu`). FlashInfer rope is disabled (base.py:38-48).
5. **Llama-4 attention scaling** (`MistralAttention`, `mistral.py:106-139`): `q *= 1 + beta*log(1+floor(pos/orig_max))` in plain torch (:117-126, applied :136-138). It is enabled only when `config.llama_4_scaling` exists, which only the native params.json path sets (`configs/mistral.py:58`). Grep for `llama_4_scaling_beta` in `vllm/` returns no hits. So with `--config-format hf`, vLLM ignores `rope_parameters.llama_4_scaling_beta`. Transformers applies it (`get_llama_4_attn_scale`, https://github.com/huggingface/transformers/blob/main/src/transformers/models/ministral3/modeling_ministral3.py). The factor is exactly 1 for pos < 16384 (Ministral 3) or < 8192 (Devstral 2), so this only matters for long contexts.
6. Attention → `Attention(..., per_layer_sliding_window=None)` (`llama.py:211-221`). Ministral 3 has no sliding window: `sliding_window: null` in every config.json above.
7. O-proj `RowParallelLinear` → `post_attention_layernorm` RMSNorm (fused add) → MLP `LlamaMLP` (`llama.py:300`): `MergedColumnParallelLinear` gate_up → `SiluAndMul` (`llama.py:116`, `layers/activation.py:116`, CUDA `csrc/libtorch_stable/activation_kernels.cu`) → `down_proj`.
8. Final `RMSNorm` (`llama.py:395`) → `ParallelLMHead`, tied for 3B (`tie_word_embeddings: true`; `llama.py:503-511`) → `LogitsProcessor`.
9. Weight remap for the native format: `mistral_mapping` (`mistral.py:242-264`; `qscale_weight → weight_scale`, `qscale_act → input_scale`). wq/wk are permuted for the rotary layout (`mistral.py:292-333`). HF FP8 suffixes are remapped `activation_scale → input_scale` and `weight_scale_inv → weight_scale` (`mistral3.py:420-437`).
10. Vision (Pixtral). Native path: `PixtralForConditionalGeneration.vision_encoder = VisionTransformer` (`pixtral.py:346`, :950). HF path: `PixtralHFVisionModel` (`pixtral.py:1501`) + `Mistral3MultiModalProjector` (RMSNorm + `Mistral3PatchMerger` using `F.unfold` + `nn.Linear` + GELU MLP, `mistral3.py:79-175`). Ops:
    - `Conv2dLayer` patch conv (`pixtral.py:960`/`:1517`) and RMSNorm.
    - 2D RoPE. Native: complex64 `view_as_complex` (`pixtral.py:712-723`). HF: transformers `apply_rotary_pos_emb` (`pixtral.py:22`, :1330).
    - `MMEncoderAttention` (`pixtral.py:799`, :1306). On CUDA the backend list is FLASH_ATTN, TORCH_SDPA, ... (`platforms/cuda.py:539-553`). FLASH_ATTN requires CC ≥ 8.0 (`v1/attention/backends/flash_attn.py:451-452`), so CC < 8.0 returns TORCH_SDPA (`cuda.py:570-573`) → `F.scaled_dot_product_attention(..., scale=, enable_gqa=)` (`v1/attention/ops/vit_attn_wrappers.py:263-264`).
    - **Text-only serving skips the tower.** With `--limit-mm-per-prompt '{"image":0}'` or `--language-model-only` (`config/multimodal.py:304`, :699-700), `_mark_tower_model` replaces the tower and projector with `StageMissingLayer` and skips their init (`models/interfaces.py:337-380`). Both wrappers use it (`mistral3.py:465`, `pixtral.py:~399`).

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Embedding / LM head | torch gather + cuBLAS GEMM | cuBLAS | any | OK (FP16/FP32) | `vocab_parallel_embedding.py:518`, :637 | – | yes |
| RMSNorm (+fused add) | CUDA `layernorm_kernels.cu` (CustomOp `forward_cuda`) | vLLM `_C` built for sm_37 | no arch guard found by grep in `layernorm_kernels.cu` | depends on shared-runtime build (other agent) | `layers/layernorm.py:96` | `RMSNorm.forward_native` `layernorm.py:74` | yes (pure torch) |
| QKV / O / MLP linear (BF16 ckpt) | `UnquantizedLinearMethod` → cuBLAS | dtype FP16 or FP32; BF16 rejected below CC 8.0 | – | OK only as `--dtype half`/`float32`. Auto dtype on Kepler resolves to FP32 (`platforms/cuda.py:262-264`); BF16 raises (`cuda.py:656-674`) | – | – | – |
| Linear (FP8 ckpt) | `Fp8LinearMethod` | FP8 HW (CUTLASS/torch `_scaled_mm`) or Marlin | 75 | **Blocked** | `quantization/fp8.py:146-147`; check enforced `config/vllm.py:943-949` | Marlin FP8 weight-only needs CC ≥ 7.5 (`quantization/utils/marlin_utils_fp8.py:30-31`, `kernels/linear/scaled_mm/marlin.py:35-45`); no pure-torch FP8 dequant path found in `kernels/linear/scaled_mm/` (only `pytorch.py`, which needs `platform.supports_fp8()`, :37-49) | **no** |
| YaRN rope | `ops.rotary_embedding` CUDA (`pos_encoding_kernels.cu`); cache built in torch | `_C` | no arch guard found (grep) | shared runtime | `rotary_embedding/base.py:221-252` | `RotaryEmbedding.forward_native` `base.py:203-219` (`forward_static` :161) | yes |
| Llama-4 q scaling | torch elementwise | none | any | OK | `mistral.py:117-138` | – | yes |
| SiluAndMul | CUDA `activation_kernels.cu` (a cp.async path is guarded `__CUDA_ARCH__ >= 800` at :522) | `_C` | – | shared runtime | `activation.py:146` | `SiluAndMul.forward_native` `activation.py:141` | yes |
| Decoder attention (full causal, GQA 32/8, hd 128) | v1 backends: FLASH_ATTN (CC ≥ 8.0, `flash_attn.py:451`), FLASHINFER (CC ≥ 8.0, `flashinfer.py:514-519`), TRITON_ATTN (CC check `True`, `triton_attn.py:413`, but Triton does not target sm_37) | FA2/FlashInfer/Triton | 80 / Triton | **Blocked in official tree** (shared runtime) | as cited | none model-specific; K80 fork uses patched xformers/SDPA on V0 (`vllm37: vllm/platforms/cuda.py:363-385`) | via fork only |
| Pixtral ViT attention | MMEncoderAttention → TORCH_SDPA on CC < 8 | torch SDPA with `scale=` (torch ≥ 2.1) and `enable_gqa=` (torch ≥ 2.5) kwargs | any | OK in official runtime. **torch 2.0.1 lacks `enable_gqa`/`scale` kwargs** (UNVERIFIED against torch 2.0 docs; `enable_gqa` is documented from 2.5: https://pytorch.org/docs/2.5/generated/torch.nn.functional.scaled_dot_product_attention.html) | `vit_attn_wrappers.py:263-264`, `platforms/cuda.py:570-573` | vllm37 Pixtral uses xformers or `nn.functional.scaled_dot_product_attention` (`vllm37: pixtral.py:675-681`, :1102-1112) | yes in fork |
| Pixtral 2D rope | torch complex64 | none | any | OK | `pixtral.py:712-723` | – | yes |
| Patch conv | `Conv2dLayer` (cuDNN/unfold+GEMM) | – | any | OK (UNVERIFIED cuDNN on sm_37) | `pixtral.py:960`, :1517 | – | – |
| torch.compile of decoder | `@support_torch_compile` on `MistralModel` (`mistral.py:206`) → Inductor/Triton | Triton | – | must run `--enforce-eager` / compilation off (shared runtime) | `mistral.py:206` | eager CustomOp `forward_native` | yes |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| FP8 E4M3 per-tensor W8A8 static (default Ministral-3-*-Instruct-2512, Devstral-Small-2) | `Fp8Config` / `Fp8LinearMethod` (`quant_method: fp8`) | **75** | `quantization/fp8.py:92`, :146-147, :225; raise at `config/vllm.py:943-949` |
| FP8 weight-only fallback for GPUs without FP8 HW | `MarlinFP8ScaledMMLinearKernel` | 75 (`has_device_capability(75)`) | `kernels/linear/scaled_mm/marlin.py:29-45`; `marlin_utils_fp8.py:30-31`; comment `fp8.py:247-248` |
| FP8 via torch `_scaled_mm` | `PerTensorTorchFP8ScaledMMLinearKernel` etc. | needs `current_platform.supports_fp8()` | `kernels/linear/scaled_mm/pytorch.py:37-65` |
| Same FP8 in the native format (`params.json quantization`) | remapped to `Fp8Config` | 75 | `configs/mistral.py:199-211` |
| GGUF (official Ministral-3/Magistral -GGUF repos) | **none.** `gguf` is absent from `QuantizationMethods` (`quantization/__init__.py:15-48`) and from `LoadFormats` (`model_loader/__init__.py:32-47`). grep -i gguf in `vllm/` finds only comments | n/a | as cited |
| GGUF in K80 fork | `GGUFConfig` min capability 60 (above sm_37) | 60 | `vllm37: vllm/model_executor/layers/quantization/gguf.py:43-44` |
| FP8 in K80 fork | `Fp8Config` 80; Marlin FP8 80 | 80 | `vllm37: quantization/fp8.py:108-109`, `vllm37: quantization/utils/marlin_utils_fp8.py:19-20` |

#### In the K80 fork (vllm37)
- `vllm37/vllm/model_executor/models/mistral3.py`: present. Registry `"Mistral3ForConditionalGeneration": ("mistral3", ...)` at `vllm37/vllm/model_executor/models/registry.py:228`.
- `vllm37/vllm/model_executor/models/pixtral.py`: present. Registry `PixtralForConditionalGeneration` at `registry.py:236`.
- `vllm37/vllm/model_executor/models/mistral.py`: **absent**. `"MistralForCausalLM": ("llama", "LlamaForCausalLM")` at `registry.py:106`. The native weight mapping lives in `vllm37: models/llama.py:484-489`.
- `Ministral3ForCausalLM` (text model_type `ministral3`): **no registry entry** (grep in `vllm37: registry.py`).
- Native-format yarn remap: present (`vllm37: vllm/transformers_utils/configs/mistral.py:25-26`, :68-80). `llama_4_scaling`: **absent** (grep in `configs/mistral.py`, `models/llama.py`).
- Format auto-detection prefers HF when config.json exists (`vllm37: vllm/transformers_utils/config.py:354-362`), and `load_format=auto` loads `*.safetensors` (`vllm37: default_loader.py:107-108`). Mistral repos therefore need explicit `--tokenizer-mode mistral --config-format mistral --load-format mistral` in the fork.

#### Verdict for K80
Blockers:
1. **Default Ministral-3 Instruct and Devstral-Small-2 checkpoints are FP8.** `Fp8Config.get_min_capability()=75` raises at `config/vllm.py:943`. Marlin FP8 needs CC ≥ 7.5 (fork: 80). Devstral-Small-2-24B-Instruct-2512 has no official BF16 or GGUF variant (mistralai listing).
   → Use the `-BF16` repos (3B/8B/14B Instruct), or the BF16-only Base/Reasoning repos, with `--dtype half`. For Devstral 2, add a load-time dequant: per-tensor scale (`weight_block_size: null`, `qscheme_act: TENSOR`), so `w_fp16 = w_fp8.float() * weight_scale`. A tiny `process_weights_after_loading` in torch, or an offline conversion script. Note: torch 2.0.1 has no `float8_e4m3fn` dtype (UNVERIFIED in this session; float8 dtypes were added in PyTorch 2.1), so the conversion must happen offline on a newer torch.
2. **BF16 → FP16/FP32.** CC < 8.0 rejects BF16 (`platforms/cuda.py:656-674`) and auto picks FP32 on Kepler (`cuda.py:262-264`). FP32 doubles memory: 3B ≈ 15 GB > 12 GB per K80 die. FP16 overflow risk for these checkpoints is UNVERIFIED.
   → Run `--dtype half`. 3B (≈ 8.5 GB FP16 incl. vision; ≈ 7.7 GB text-only) fits one die. 8B needs TP=2, 14B TP=4, 24B (Magistral) TP=4–8.
3. **Decoder attention backend**: no v1 backend runs on sm_37 (FA/FlashInfer CC ≥ 8.0, Triton not Kepler). Shared runtime, other agent.
   → Use the fork's V0 xformers/SDPA path (`vllm37: platforms/cuda.py:363-385`). Ministral 3 has no sliding window, so plain causal paged attention is enough.
4. **Missing model plumbing in the fork**: no `Ministral3ForCausalLM` registry entry and no `llama_4_scaling`.
   → Use native format (`--config-format mistral`, which yields `MistralForCausalLM` + yarn already remapped in the fork). Port the 10-line `_get_llama_4_attn_scale` from `mistral.py:106-138` into the fork's Llama attention (needed for positions ≥ 16384, or ≥ 8192 for Devstral 2). Keep context below those limits otherwise. The HF-format path also needs transformers ≥ 5 (`config.json "transformers_version": "5.0.0.dev0"`), while the fork pins `transformers >= 4.55.0` (`vllm37: requirements/common.txt:10`).
5. **Vision tower**: works through SDPA/xformers in the fork. Official code passes `enable_gqa=` (torch ≥ 2.5).
   → Serve text-only (`--limit-mm-per-prompt image=0`) or keep the fork's Pixtral implementation.
6. **GGUF repos**: unsupported in official vLLM (removed). Fork GGUF min CC is 60.
   → Not a viable path without porting the GGUF dequant kernels to sm_37 (out of scope here).
7. **YaRN cos/sin cache** costs 128–192 MB FP32 (yarn_scaling_rope.py:76-85).
   → Cap `max_model_len`. The cache is still built for `orig*factor` rows; either patch it to size by `max_model_len` or accept the cost.

---

### AllenAI Olmo 3

#### Official releases
| Repo | Released (repo created_at) | Params | Format | Quant config | Context (yarn original) | Layers / KV heads / head_dim | Sliding window | Vision |
|---|---|---|---|---|---|---|---|---|
| allenai/Olmo-3-1025-7B (base) | 2025-09-12 | 7.30B | BF16 safetensors | none | 65536 (yarn factor 8, orig 8192, attention_factor 1.2079); rope_theta 5e5 | 32 / **32 (MHA)** / 128 (4096/32) | 4096; layer_types 24 sliding + 8 full (pattern S,S,S,F) | none |
| allenai/Olmo-3-7B-Instruct | 2025-11-19 | 7.30B | BF16 | none | 65536 (same yarn) | 32 / 32 / 128, ff 11008, vocab 100278, untied | 4096; 24 S + 8 F | none |
| allenai/Olmo-3-7B-Think | 2025-11-18 | 7.30B | BF16 | none | 65536 (same) | 32 / 32 / 128 | 4096; 24 S + 8 F | none |
| allenai/Olmo-3-32B-Think | 2025-11-19 | 32.23B | BF16 | none | 65536 (same) | 64 / **8** / 128 (5120/40), ff 27648 | 4096; 48 S + 16 F | none |
| allenai/Olmo-3.1-32B-Instruct (and -Think) | 2025-12-10 | 32.23B | BF16 | none | 65536 (same) | 64 / 8 / 128 | 4096; 48 S + 16 F | none |
| Other variants: -SFT/-DPO/-RL-Zero-*, Olmo-3-1125-32B (base) | 2025-10-14 … 2025-12-12 | 7.30B / 32.23B | BF16 (Olmo-3.1-32B-Instruct-SFT/DPO are F32) | none | – | – | – | – |
| GGUF / FP8 / AWQ from allenai | none in `hf models ls --author allenai --search Olmo-3` | – | – | – | – | – | – | – |

QK-norm: Olmo3 normalizes q and k over the full projection width, `q_norm = RMSNorm(num_heads*head_dim)` and `k_norm = RMSNorm(num_kv_heads*head_dim)`, applied before the reshape. The block is **post-norm**: `post_attention_layernorm` and `post_feedforward_layernorm` sit on the branch output. Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo3/modeling_olmo3.py (fetched). Transformers maps legacy `rope_scaling` (yarn) **only to full_attention layers**; sliding layers keep default rope (`configuration_olmo3.py`, fetched: `self.rope_parameters["full_attention"].update(rope_scaling)`).

#### Code path (official vLLM)
1. Registry: `"Olmo3ForCausalLM": ("transformers", "TransformersForCausalLM")` (`registry.py:717`, in `_TRANSFORMERS_SUPPORTED_MODELS`). **No native olmo3/olmo2 model file exists** in `vllm/model_executor/models/` (only `olmoe.py`, `olmo_hybrid.py`). The model is the HF `Olmo3ForCausalLM`, instantiated by the Transformers modeling backend (`models/transformers/base.py`, `causal.py:38`), which needs `transformers >= 5.16.1` (`requirements/common.txt:10`).
2. Module substitution (`base.py:470-578`, `recursive_replace`):
   - `nn.Embedding` → `VocabParallelEmbedding` (`transformers/utils.py:263-298`).
   - q/k/v fused into `QKVParallelLinear` (`fusers/qkv.py:37`, :176).
   - gate/up GLU → `MergedColumnParallelLinear` + `SiluAndMul` (`fusers/glu.py:50`, :115-118).
   - Other `nn.Linear` → vLLM linear (`utils.py:109-151`).
   - Olmo3RMSNorm (including **q_norm/k_norm**) → `TPAwareRMSNorm` (vLLM `RMSNorm` CustomOp). Under TP it all-gathers the sharded q/k before norming (`fusers/rms_norm.py:107-136`, :278-297).
3. Rotary: no rope fuser exists (`ls transformers/fusers/`). HF `Olmo3RotaryEmbedding` computes cos/sin per layer type in **pure torch** each forward. Yarn inv_freq comes from transformers `ROPE_INIT_FUNCTIONS["yarn"]` for full layers, and sliding layers use default rope (fetched modeling file).
4. Attention: `config._attn_implementation = "vllm"` (`base.py:209`) → `vllm_attention_forward` (`transformers/__init__.py:67-97`, registered :128) → vLLM `Attention`. `_create_attention_instances` passes `per_layer_sliding_window=text_config.sliding_window` for layers whose `layer_types[i] == "sliding_attention"` (`base.py:673-676`).
5. Sliding-window-capable v1 backends: FLASH_ATTN (`flash_attn.py:386`, CC ≥ 8.0 :451), FLASHINFER (:476, CC ≥ 8.0 :514), TRITON_ATTN (:367, Triton), FLEX_ATTENTION (:113, torch.compile/Triton). The base default is `False` (`v1/attention/backend.py:189-190`). **None runs on sm_37.**
6. Final norm → lm_head (`ParallelLMHead`, untied) → logits. Decoder is wrapped with `support_torch_compile` (`transformers/base.py:39`, :257) → needs eager mode on K80.

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Embedding / lm_head | torch gather / cuBLAS | – | any | OK | `transformers/utils.py:263-298` | – | yes |
| RMSNorm incl. QK-norm (full-width 4096 / 5120) | vLLM `RMSNorm` CustomOp (CUDA `layernorm_kernels.cu`) | `_C` | no arch guard found | shared runtime | `fusers/rms_norm.py:291-297`, `layernorm.py:96` | `RMSNorm.forward_native` `layernorm.py:74` | yes |
| QKV / MLP linear (BF16 ckpt) | cuBLAS | FP16/FP32 (BF16 rejected below CC 8.0) | – | OK with `--dtype half` | `platforms/cuda.py:656-674` | – | – |
| Rope (yarn on full layers, default on sliding) | HF transformers pure torch | none | any | OK (math is pure torch) | fetched `modeling_olmo3.py` | – | yes |
| SiluAndMul | `activation_kernels.cu` | `_C` | – | shared runtime | `fusers/glu.py:115-118` | `activation.py:141` | yes |
| Interleaved sliding-window attention (24/32 layers, window 4096) + MHA (7B: 32 KV heads) | FA2 / FlashInfer / Triton / Flex | CC ≥ 8.0 or Triton | 80 / Triton | **Blocked in official tree** | `flash_attn.py:386,451`; `flashinfer.py:476,514`; `triton_attn.py:367,413`; `flex_attention.py:113` | none model-specific | no |
| torch.compile | Inductor/Triton | Triton | – | must use `--enforce-eager` | `transformers/base.py:39,257` | eager | yes |
| Transformers backend itself | transformers ≥ 5.16.1 | torch version required by transformers 5.x (UNVERIFIED; not checked) | – | not usable on the fork's torch 2.0.1 (UNVERIFIED) | `requirements/common.txt:10` | – | – |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| BF16 (all official Olmo 3 repos) | unquantized; must cast to FP16/FP32 on K80 | BF16 needs 80 | `platforms/cuda.py:255-264`, :656-674 |
| Official FP8/GGUF/AWQ from allenai | none published | – | `hf models ls --author allenai --search Olmo-3` (no such repos) |

#### In the K80 fork (vllm37)
- Olmo3 model file: **absent.** `vllm37/vllm/model_executor/models/` has `olmo.py`, `olmo2.py`, `olmoe.py` only. grep `olmo3|Olmo3` in `vllm37/vllm` returns nothing.
- Registry: **no `Olmo3ForCausalLM`.** `"Olmo2ForCausalLM": ("olmo2", ...)` at `vllm37/vllm/model_executor/models/registry.py:116`.
- `vllm37/vllm/model_executor/models/olmo2.py` already has full-width QK-norm (`olmo2.py:105-109`, :139-145, applied :160). It has **no** `layer_types`/sliding-window handling (grep).
- Transformers fallback backend: `vllm37/vllm/model_executor/models/transformers.py` (handles `layer_types` sliding at :540-555). The fork pins `transformers >= 4.55.0`, but Olmo3 appeared in 4.57 (configs report `transformers_version` 4.57.0/4.57.1).
- Fork V0 interleaved-SWA handling: with XFORMERS/FLASHINFER set via env, it **disables sliding window and caps max_len to 4096** (`vllm37/vllm/config/__init__.py:718-731`). The V0 paged decode `forward_decode` has no sliding-window argument (`vllm37/vllm/attention/ops/paged_attn.py:90-107`). Prefill xformers applies `make_local_attention` (`vllm37/vllm/attention/backends/xformers.py:721-723`).

#### Verdict for K80
Blockers:
1. **No native implementation.** The official tree reaches Olmo3 only through the Transformers backend, which needs transformers ≥ 5.16.1 (and the torch that comes with it), plus Triton-free attention.
   → Port Olmo3 into the fork as `olmo3.py`, cloned from `vllm37: olmo2.py`. It already has full-width QK-norm and the post-norm layout. Add three things:
   - (a) read `layer_types` and pass `per_layer_sliding_window=4096` to `Attention` on sliding layers;
   - (b) build **two** rotary embeddings: `get_rope(... rope_scaling=yarn{factor 8, orig 8192, attention_factor 1.2079})` for full layers and default rope (theta 5e5) for sliding layers, to match transformers' `rope_parameters["full_attention"].update(rope_scaling)`;
   - (c) register `"Olmo3ForCausalLM"` and add an `Olmo3Config` shim if transformers 4.55 lacks it.
2. **Interleaved sliding-window decode** is unsupported in the fork's V0 paged attention (`paged_attn.py:90` has no window argument; `config/__init__.py:718-731` caps at 4096).
   → Either accept `max_model_len ≤ 4096`, where SWA equals full attention and the output is exact, or add a per-layer window to the V0 paged-attention decode kernel (mask positions `< seq_len - window`) and the SDPA/xformers prefill paths.
3. **Memory.** 7B is MHA (32 KV heads × 128 × 2 × 2 B × 32 layers = 512 KB/token at FP16). Weights ≈ 14.6 GB FP16 → TP=2 on one K80 board (2 × 12 GB). 32B ≈ 64 GB FP16 → ≥ 6–8 dies (40 heads/8 KV heads → TP ∈ {2,4,8}). FP32 doubles all of this.
   → `--dtype half` and TP=2 for 7B. Sliding-window KV savings need the hybrid allocator, which V0 lacks.
4. **BF16 checkpoints** → FP16 cast (`--dtype half`). FP16 activation overflow for Olmo 3 is UNVERIFIED.
5. **Attention backend and torch.compile**: shared-runtime blockers (other agent). Use the fork's xformers/SDPA V0 path with eager mode.

## 7. LiquidAI LFM2 / LFM2.5 and OpenAI gpt-oss on Tesla K80 (sm_37)


Sources: official vLLM checkout `/home/jack/src/vllm` (main 3ca00a8261, 2026-10-07); K80 fork `/home/jack/src/vllm37`; HF Hub metadata
from `hf models ls --author LiquidAI|openai --expand createdAt,downloads,safetensors,config,gguf --json` and `config.json` files
downloaded to a local scratch directory (not kept) (no weights downloaded). All paths below are relative to `/home/jack/src/vllm` unless prefixed `vllm37/`.

### LiquidAI LFM2 / LFM2.5

#### Official releases
Text-only causal-LM repos from `LiquidAI` (VL / Audio / Encoder / ColBERT / DSpark-draft repos omitted). Released = HF `createdAt`.
Params = HF safetensors total. head_dim = hidden_size / num_attention_heads (no explicit `head_dim` field). Every config has
`conv_L_cache: 3` (short-conv kernel size 3, state length 2) and `conv_bias: false`.

| Repo | Released | Params (total/active) | Format | Quant config | Context | Layer pattern / KV heads / head_dim |
|---|---|---|---|---|---|---|
| LiquidAI/LFM2-350M | 2025-07-10 | 354.5M dense | safetensors BF16 | none | UNVERIFIED (config not downloaded) | lfm2 (not downloaded) |
| LiquidAI/LFM2-700M | 2025-07-10 | 742.5M dense | safetensors BF16 | none | UNVERIFIED | lfm2 |
| LiquidAI/LFM2-1.2B | 2025-07-10 | 1.170B dense | safetensors BF16 | none | UNVERIFIED | lfm2 |
| LiquidAI/LFM2-2.6B | 2025-09-22 | 2.569B dense | safetensors BF16 | none | UNVERIFIED | lfm2 |
| LiquidAI/LFM2-8B-A1B | 2025-10-07 | 8.34B MoE | safetensors BF16 (+704 F32 = expert bias) | none | UNVERIFIED | lfm2_moe |
| LiquidAI/LFM2-24B-A2B | 2026-02-24 | 23.84B MoE / active ~2B (name) | safetensors BF16 | none | 128000 | 40 layers = 30 conv + 10 full_attention; 32 q / 8 kv heads; head_dim 64 (2048/32); 64 experts, top-4, moe_intermediate 1536, 2 dense layers, sigmoid router w/ expert bias |
| LiquidAI/LFM2.5-1.2B-Base / -Instruct / -Thinking / -JP | 2026-01-05 / 2026-01-06 / 2026-01-20 / 2026-01-04 | 1.170B dense | safetensors BF16 | none | 128000 (`max_position_embeddings`) | -Instruct config: 16 layers = 10 conv + 6 full_attention (idx 2,5,8,10,12,14); 32 q / 8 kv; head_dim 64; hidden 2048; ff 12288 (auto-adjusted); vocab 65536; rope_theta 1e6 |
| LiquidAI/LFM2.5-350M (+ -Base) | 2026-03-31 | 354.5M dense | safetensors BF16 | none | 128000 | 16 layers = 10 conv + 6 attn; 16 q / 8 kv; head_dim 64 (1024/16); ff 6656 |
| LiquidAI/LFM2.5-230M (+ -Base) | 2026-06-24 / 2026-06-16 | 229.7M dense | safetensors BF16 | none | 128000 | 14 layers = 8 conv + 6 attn; 16 q / 8 kv; head_dim 64; ff 2560 |
| LiquidAI/LFM2.5-8B-A1B (+ -Base) | 2026-05-28 | 8.468B MoE / active ~1B (name) | safetensors BF16 (+704 F32) | none | 128000 | 24 layers = 18 conv + 6 attn; 32 q / 8 kv; head_dim 64; 32 experts, top-4 (`num_experts_per_tok`), moe_intermediate 1792, `num_dense_layers: 2`, `use_expert_bias: true`, `norm_topk_prob: true`; vocab 128000 |
| LiquidAI/LFM2.5-2.6B (+ -Base) | 2026-07-28 / 2026-08-01 | 2.697B dense | safetensors BF16 | none | 131072 | 30 layers = 22 conv + 8 attn; 32 q / 8 kv; head_dim 64; ff 10752; vocab 128000; rope_theta 1e7 |
| LiquidAI/LFM2.5-*-GGUF (1.2B-Instruct/Base/Thinking/JP, 350M, 230M, 2.6B, 8B-A1B) | 2026-01-04 .. 2026-08-01 | same as base | GGUF: BF16, F16, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0 (+ QAD-Q4_0 for 1.2B-Instruct) — file listing of LFM2.5-1.2B-Instruct-GGUF and LFM2.5-8B-A1B-GGUF | GGUF (no HF quant config) | GGUF `context_length` 128000 (2.6B: 131072) | gguf `architecture`: `lfm2` / `lfm2moe` |
| LiquidAI/LFM2.5-*-ONNX (1.2B-Base/Instruct/Thinking, 350M, 230M, 2.6B, 8B-A1B; LFM2-8B-A1B/24B-A2B) | 2026-01-04 .. 2026-08-01 | — | ONNX (`onnx/` dir) | none in config | — | — |
| LiquidAI/LFM2.5-*-MLX-{bf16,4bit,5bit,6bit,8bit} (+ 2.6B: mxfp4, mxfp8, nvfp4) | 2026-01-06 .. 2026-08-06 | — | MLX (U32-packed) | not a vLLM `quantization_config` (`qc=None` in listing) | — | — |

Notes: no official GPTQ/AWQ/FP8/compressed-tensors LFM repos exist (search of all 178 LiquidAI repos for `quantization_config`
returned none). LFM2.5-2.6B/8B-A1B configs use transformers 5.x `rope_parameters` (both handled: `lfm2.py:147-152` passes
`config.rope_parameters`).

#### Code path (official vLLM)
1. Registry → `vllm/model_executor/models/registry.py:144` `"Lfm2ForCausalLM": ("lfm2", ...)`, `:145` `"Lfm2MoeForCausalLM": ("lfm2_moe", ...)`.
2. Model class → `Lfm2ForCausalLM` (`lfm2.py:395-396`, interfaces `HasInnerState, IsHybrid`); hybrid state shape/dtype via
   `get_mamba_state_shape_from_config` (`lfm2.py:431-452` → `MambaStateShapeCalculator.short_conv_state_shape`, `mamba_utils.py:257-266`,
   shape `(conv_dim/TP, conv_L_cache-1 [+num_spec])`) and dtype = model dtype (`mamba_utils.py:113-119`).
3. Embedding → `VocabParallelEmbedding` (`lfm2.py:325`) — plain torch embedding.
4. Layer selection → `layer_types[i] == "full_attention"` → `Lfm2AttentionDecoderLayer` else `Lfm2ShortConvDecoderLayer` (`lfm2.py:329-334`; MoE: `lfm2_moe.py:430-437`).
5. Norms → `RMSNorm` (`lfm2.py:222-223, 273-274`) → `layernorm.py:37` CustomOp; `forward_cuda` (`layernorm.py:96`) = CUDA `ops.rms_norm`/`fused_add_rms_norm`; `forward_native` (`layernorm.py:74`).
6. Short-conv layer → `ShortConv` (`vllm/model_executor/layers/mamba/short_conv.py:36`, PluggableLayer). `forward` → custom op
   `torch.ops.vllm.short_conv` (`short_conv.py:204-213`, registered `:376-380`). Dispatcher `short_conv()` (`:363-373`): **on any non-CPU
   platform calls `forward_cuda`**, only CPU calls `forward_native`.
   - `in_proj` = `MergedColumnParallelLinear` dim→3·dim (`:67-73`), split B, C, x; `Bx = B*x`.
   - prefill: `causal_conv1d_fn` (`short_conv.py:289-298`) → `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:481`, **Triton** kernel `_causal_conv1d_fwd_kernel` (`@triton.jit`, `causal_conv1d.py:16-17`).
   - decode: `causal_conv1d_update` (`short_conv.py:312-334`) → `causal_conv1d.py:1096`, **Triton** kernel `_causal_conv1d_update_kernel` (`causal_conv1d.py:762-763`; launch uses `launch_pdl=current_platform.is_arch_support_pdl()`, `:1274`).
   - There is no CUDA C++ causal_conv1d: `csrc` matches only `csrc/cpu/*` and a ROCm KDA file (grep `causal_conv1d` in `csrc`); `_custom_ops.py:1999-2009, 3663-3706` are CPU ops.
   - `y = C * conv(Bx)`, `out_proj` RowParallelLinear (`short_conv.py:74-80, 339`).
7. Short-conv metadata/state → `MambaAttentionBackendEnum.SHORT_CONV` (`short_conv.py:358-360`; `vllm/v1/attention/backends/registry.py:219`)
   → `ShortConvAttentionBackend`/`ShortConvAttentionMetadataBuilder` (`vllm/v1/attention/backends/short_conv_attn.py:26-50`, base
   `mamba_attn.py`); KV spec `MambaSpec` (`vllm/v1/kv_cache_interface.py:1050`); manager `MambaManager`
   (`vllm/v1/core/single_type_kv_cache_manager.py:1455`). With `mamba_cache_mode == "align"` (prefix caching) the runner calls
   `preprocess_mamba`/`postprocess_mamba_align_gpu` (`vllm/v1/worker/gpu_model_runner.py:4289-4297, 1588-1593`) which use **Triton**
   kernels (`vllm/v1/worker/mamba_utils.py:38, 102, 190, 368, 502, 548, 633 batch_memcpy_kernel`).
8. Attention layer → `Lfm2Attention` (`lfm2.py:93`): `QKVParallelLinear` (`:131`), per-head **q/k RMSNorm** (`:161-162, 173-174`), NeoX RoPE
   `get_rope` (`:147-152`), `Attention` (`:153-160`, no sliding window, no sinks), `out_proj` (`:140`). Backend chosen by CUDA platform
   priority list FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION → TURBOQUANT (`vllm/platforms/cuda.py:174-178`).
9. Dense MLP → `Lfm2MLP` (`lfm2.py:49-90`): `w13` MergedColumnParallelLinear (`:70`), `SiluAndMul` (`:84`; `activation.py:116`, `forward_cuda` `:146`, `forward_native` `:141`), `w2` RowParallelLinear (`:77`).
10. MoE (lfm2_moe only, layers ≥ `num_dense_layers`) → `Lfm2MoeSparseMoeBlock` (`lfm2_moe.py:96`): `GateLinear` router (`:128-133`), F32
    `e_score_correction_bias` (`:134-139`), `FusedMoEFactory(..., use_grouped_topk=True, num_expert_group=1, topk_group=1,
    scoring_func="sigmoid", renormalize=norm_topk_prob)` (`:141-158`).
    - Routing: `create_fused_moe_router` treats `num_expert_group<=1 and topk_group<=1` as degenerate (`router/router_factory.py:175-199`) and,
      because a bias is present, returns `FusedTopKBiasRouter` (`router_factory.py:226-230`) → `fused_topk_bias` (`router/fused_topk_bias_router.py:146`)
      → `vllm_topk_sigmoid` → CUDA `ops.topk_sigmoid` (`fused_topk_bias_router.py:56-75`; kernel `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`, built in `_moe_C`, `CMakeLists.txt:1297-1301`).
    - Experts: unquantized BF16/FP16 → `UnquantizedFusedMoEMethod` + `fused_moe/oracle/unquantized.py` (see gpt-oss section for backend list).
11. Final norm + `ParallelLMHead`/tied embedding (`lfm2.py:352, 468`) + `LogitsProcessor`.

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Short-conv prefill (causal_conv1d_fn) | Triton `_causal_conv1d_fwd_kernel` | Triton | Triton's floor (sm_75+ in current Triton; Kepler never supported) | BLOCKED | `mamba/ops/causal_conv1d.py:16, 481`; `short_conv.py:289, 370-371` | `causal_conv1d_fn_cpu` = per-sequence `F.conv1d(groups=dim)` in pure torch (`mamba/ops/cpu/causal_conv1d.py:13-84`), reached via `ShortConv.forward_native` (`short_conv.py:95-171`) only when `current_platform.is_cpu()` (`short_conv.py:370-373`) | YES in principle: device-agnostic torch (F.conv1d, cat, copy_); uses `.item()` per sequence (`cpu/causal_conv1d.py:39-41, 50, 56`) → host syncs, not CUDA-graph safe. Needs dispatcher change at `short_conv.py:370`. |
| Short-conv decode (causal_conv1d_update) | Triton `_causal_conv1d_update_kernel` | Triton | as above | BLOCKED | `causal_conv1d.py:762, 1096`; `short_conv.py:312-334` | (a) `causal_conv1d_update_cpu` → `torch.ops._C.causal_conv1d_update_cpu_vec` (`cpu/causal_conv1d.py:87-116`, `_custom_ops.py:1999-2009`) = CPU C++ op, used by `forward_native` on x86 (`short_conv.py:190-199`); (b) `causal_conv1d_update_torch` pure torch (`cpu/causal_conv1d.py:119-148`) used on ARM with gather/scatter by state index (`short_conv.py:181-189`) | (a) NO (CPU-only op); (b) YES (torch cat/F.conv1d/index); a port must route CUDA to branch (b). Not CUDA-graph-hostile (no `.item()`). |
| Mamba state copy for prefix caching (`mamba_cache_mode="align"`) | Triton `batch_memcpy_kernel`, fused pre/post-process kernels | Triton | Triton floor | BLOCKED only if prefix caching on hybrid model | `v1/worker/mamba_utils.py:633-657, 368, 502`; gated at `gpu_model_runner.py:4289, 1588` | none in code; avoid by keeping `mamba_cache_mode` ≠ "align" (no prefix caching) | YES by configuration (disable prefix caching) |
| q/k RMSNorm, block RMSNorm | CUDA `rms_norm` / `fused_add_rms_norm` | `_C` ext built for sm_37 | shared runtime | depends on shared `_C` build | `layernorm.py:96` | `RMSNorm.forward_native` `layernorm.py:74` | YES (pure torch, FP32 accumulation) |
| SiluAndMul | CUDA `silu_and_mul` | `_C` | shared runtime | depends on `_C` | `activation.py:146` | `activation.py:141` | YES |
| RoPE (default, NeoX) | CUDA `rotary_embedding` | `_C` | shared runtime | depends on `_C` | `lfm2.py:147-152` | `RotaryEmbedding.forward_native` (shared runtime; not re-verified here) | UNVERIFIED here (other agent) |
| Full attention (GQA 32/8 or 16/8, head_dim 64, no window, no sinks) | FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION | FA: CC ≥ 8.0 (`v1/attention/backends/flash_attn.py:451-452`); FlashInfer: 8.0 ≤ CC ≤ 12.1 (`flashinfer.py:514-522`); Triton attn: Triton (`triton_attn.py:413-414` returns True for any CC); Flex: `torch.compile(flex_attention)` → Inductor/Triton (`flex_attention.py:55-58`) | 8.0 for FA/FI; Triton floor for others | BLOCKED (shared runtime issue; no LFM2-specific need) | `platforms/cuda.py:174-178` | none for decoder attention on CUDA in official vLLM | NO (needs fork's attention backend) |
| Model dtype | — | CUDA platform `supported_dtypes`: CC ≥ 6.0 for FP16, ≥ 8.0 for BF16, else FP32 only | 6.0 (FP16) | FP32 only on sm_37 | `platforms/cuda.py:255-264` | — | FP32 works (memory cost ×2) |
| Dense linears (qkv, in_proj, out_proj, w13/w2) | `UnquantizedLinearMethod` → torch `F.linear` / cuBLAS | cuBLAS | any (FP16/FP32 GEMM on sm_37 via cuBLAS) | OK in FP16/FP32 (BF16 checkpoint must be cast) | shared | — | YES |
| MoE routing (lfm2_moe) | CUDA `topk_sigmoid` (`_moe_C`) | `_moe_C` built for sm_37; CUB; uses `__nv_bfloat16` type conversions | not arch-guarded in source (no `__CUDA_ARCH__` in `topk_softmax_kernels.cu` grep) | UNVERIFIED (compiles only if `_moe_C` and libtorch_stable ABI build for sm_37) | `fused_topk_bias_router.py:56-75, 229`; `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:285-287, 360-371` | pure-torch `grouped_topk` (`router/grouped_topk_router.py:81-164`, sigmoid + bias + topk + renorm) reached via `GroupedTopKRouter` only when grouping is non-degenerate (`router_factory.py:196-212`) | YES (pure torch) but needs router-factory change for num_expert_group=1 |
| MoE experts (lfm2_moe, BF16) | Unquantized oracle: Triton `fused_moe` / FlashInfer / … | see gpt-oss section | — | BLOCKED (see gpt-oss "unquantized" row) | `lfm2_moe.py:141-158` | see gpt-oss section | see gpt-oss section |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| BF16 safetensors (all official text repos) | unquantized; K80 has no BF16 → must load as `--dtype float16` or `float32` | n/a | HF listing `safetensors: {'BF16': ...}`; no `quantization_config` in any LiquidAI config |
| GGUF (official `-GGUF` repos; BF16/F16/Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0) | **none in official main**: no `gguf*.py` in `vllm/model_executor/layers/quantization/` and no `gguf_loader.py` in `vllm/model_executor/model_loader/` (only a stale comment `models/qwen2_moe.py:502`). The fork has `vllm37/vllm/model_executor/layers/quantization/gguf.py` (`get_min_capability()` = 60, line 43-44) + `vllm37/.../model_loader/gguf_loader.py`, but no `lfm2` mapping (grep `lfm` in it: none) and no LFM2 model | n/a (main); 60 (fork GGUF, blocks sm_37 as-is) | ls/grep above; GGUF file listing |
| ONNX, MLX (incl. MLX mxfp4/mxfp8/nvfp4) | not loadable by vLLM (no ONNX/MLX loader) | n/a | repo listings (`onnx/` dir; MLX U32 packed tensors, `qc=None`) |

#### In the K80 fork (vllm37)
- Model file: ABSENT — `ls vllm37/vllm/model_executor/models | grep -i lfm` returns nothing.
- Registry entry: ABSENT — `grep -i lfm vllm37/vllm/model_executor/models/registry.py` returns nothing.
- ShortConv layer: ABSENT — `grep -rn "short_conv\|ShortConv" vllm37/vllm` returns nothing; fork has `vllm37/vllm/model_executor/layers/mamba/ops/causal_conv1d.py` (Triton, `@triton.jit` at line 16) and v0 `mamba_mixer`/`mamba_mixer2` only.

#### Verdict for K80
Blockers (LFM2-specific):
1. **Short-conv conv1d is Triton-only on CUDA** (`short_conv.py:370-371` → `causal_conv1d.py:16, 762`). Fix: make `short_conv()` dispatch to
   a CUDA-safe native path: prefill via `causal_conv1d_fn_cpu` (pure torch, `cpu/causal_conv1d.py:13`) and decode via
   `causal_conv1d_update_torch` (`cpu/causal_conv1d.py:119`) with the ARM-style gather/scatter (`short_conv.py:181-189`), never the
   CPU C++ `causal_conv1d_update_cpu_vec`. Kernel size is only 3, so a small hand-written CUDA kernel (depthwise conv + state shift,
   FP16 storage/FP32 math, no sm_53+ intrinsics) is also easy if the torch path is too slow.
2. **Hybrid (attention + conv-state) cache plumbing is v1-only** (`MambaSpec`, `MambaManager`, `ShortConvAttentionMetadataBuilder`);
   the v0 fork has no ShortConv. Fix: port `lfm2.py` + `short_conv.py` into vllm37 and wire the conv state into the fork's v0
   `MambaCacheManager`-style state (the fork already carries `mamba_mixer2` v0 plumbing, `vllm37/vllm/model_executor/layers/mamba/`),
   state shape `(conv_dim, 2)` per layer per sequence. Disable prefix caching (Triton state copy, `v1/worker/mamba_utils.py:633`).
3. **BF16 weights / dtype** — official vLLM allows only FP32 below CC 6.0 (`vllm/platforms/cuda.py:255-264`: "Kepler and Maxwell ... only FP32 is supported"). FP32 is affordable for LFM2 (1.2B ≈ 4.7 GB, 2.6B ≈ 10.8 GB per 12 GB die); FP16 needs that list relaxed in the port (FP16 overflow risk in conv/MLP UNVERIFIED).
4. **Attention backend** — generic (no sinks/sliding window); whatever backend the fork uses for Llama-class models suffices
   (head_dim 64, GQA). q/k-norm uses RMSNorm (native fallback exists).
5. **lfm2_moe only**: sigmoid+bias top-k needs `_moe_C topk_sigmoid` or the torch `grouped_topk` (`grouped_topk_router.py:81-164`);
   expert GEMMs need a non-Triton MoE path (see gpt-oss verdict). LFM2.5-8B-A1B BF16 = ~17 GB → FP16 fits only across both K80 dies (TP=2) or not at all with KV; dense 230M–2.6B are the realistic targets.

### OpenAI gpt-oss

#### Official releases
`hf models ls --search gpt-oss --author openai` returns exactly 4 repos; only the two 20B ones are ≤ 40B. openai publishes no
GGUF/ONNX/BF16 repo (GGUF is `ggml-org/gpt-oss-20b-GGUF`, 2025-08-02, not openai). Configs: a local scratch directory (not kept).

| Repo | Released | Params (total/active) | Format | Quant config | Context | Layer pattern / KV heads / head_dim |
|---|---|---|---|---|---|---|
| openai/gpt-oss-20b | 2025-08-04 (HF createdAt) | 20.91B total (safetensors: BF16 1.804B + U8 19.11B packed); active: 4 of 32 experts/token (`num_experts_per_tok: 4`, `num_local_experts: 32`); "3.6B active" per model card https://huggingface.co/openai/gpt-oss-20b — not fetched, UNVERIFIED | safetensors: MXFP4 experts (`*_blocks` U8 + `*_scales`), everything else BF16 (index: `mlp.experts.gate_up_proj_blocks/_scales/_bias`, `down_proj_blocks/_scales/_bias`, `self_attn.sinks`) | `{'quant_method': 'mxfp4', 'modules_to_not_convert': ['model.layers.*.self_attn', 'model.layers.*.mlp.router', 'model.embed_tokens', 'lm_head']}` | 131072 (`max_position_embeddings`; YaRN factor 32 over `original_max_position_embeddings` 4096, rope_theta 150000) | 24 layers alternating `sliding_attention` / `full_attention` (12/12, layer 0 sliding), `sliding_window: 128`; 64 q / 8 kv heads; `head_dim: 64`; hidden 2880; expert intermediate 2880; `attention_bias: true`; `swiglu_limit: 7.0`; per-head attention sinks (`self_attn.sinks`); vocab 201088 |
| openai/gpt-oss-safeguard-20b | 2025-09-18 | 21.51B total (BF16 1.804B + U8 19.71B) — same architecture | same MXFP4 layout | identical `quantization_config` | 131072 | identical to gpt-oss-20b (config diff: only `transformers_version`) |
| openai/gpt-oss-120b, gpt-oss-safeguard-120b | 2025-08-04 / 2025-09-18 | 116.8B / 120.4B | MXFP4 | mxfp4 | — | out of scope (> 40B) |
| Community BF16 (not official): unsloth/gpt-oss-20b-BF16 (2025-08-05), unsloth/gpt-oss-safeguard-20b-BF16 (2025-10-29), FriendliAI/gpt-oss-20b-BF16, huihui-ai/… | — | 20.91B, all BF16 | safetensors BF16; names `mlp.experts.gate_up_proj` (fused, no `_blocks`), `gate_up_proj_bias`, `down_proj`, `down_proj_bias`, `self_attn.sinks` (from `model.safetensors.index.json` of unsloth/gpt-oss-20b-BF16) | none (`quantization_config` absent, `torch_dtype: bfloat16`) | same | same |
| Community other: unsloth/gpt-oss-20b-bnb-4bit (`qc=bitsandbytes`), amd/gpt-oss-20b-w-mxfp4-a-bf16 (`qc=quark`), ggml-org/unsloth GGUF | — | — | — | — | — | — |

#### Code path (official vLLM)
1. Registry → `registry.py:120` `"GptOssForCausalLM": ("gpt_oss", "GptOssForCausalLM")`.
2. Config hook → `GptOssForCausalLMConfig.verify_and_update_model_config` rewrites `quant_method` "mxfp4" → "gpt_oss_mxfp4" (`models/config.py:519-534`); quant registry `"mxfp4": Mxfp4Config`, `"gpt_oss_mxfp4": GptOssMxfp4Config` (`layers/quantization/__init__.py:176-177`); `GptOssMxfp4Config.override_quantization_method` (`quantization/mxfp4.py:126-143`).
3. Model → `GptOssForCausalLM` (`gpt_oss.py:1411`) → `GptOssModel` (`:473`) → `TransformerBlock` (`:425`) ×24; `RMSNorm` (`:449-450, 500`).
4. Attention → `OAIAttention` (`gpt_oss.py:91`): `QKVParallelLinear(bias=True)` (`:137-145`); YaRN RoPE in fp32, NeoX (`:110-126`);
   `sinks` parameter `[num_heads/TP]` (`:130-132`); `Attention(..., per_layer_sliding_window = sliding_window if layer_idx % 2 == 0 else None, sinks=self.sinks)` (`:176-188`); `o_proj(bias=True)` (`:147-153`).
   With MXFP4, `Mxfp4Config.get_quant_method` gives `UnquantizedLinearMethod` for linears and `None` for Attention (`mxfp4.py:88-110`).
   Backend selection (CUDA priority `platforms/cuda.py:174-178`) filtered by sink support:
   - FLASH_ATTN: `supports_sink` needs FA3/FA4 (`fa_utils.py:319-322`, `flash_attn.py:445-448`); `"sink not supported on compute capability < 9.0"` (`flash_attn.py:467-468`); CC ≥ 8.0 overall (`:451-452`).
   - FLASHINFER: sinks only via TRT-LLM attention on SM90 / SM100 / SM120 (`flashinfer.py:525-544`); CC 8.0–12.1 (`:514-522`).
   - TRITON_ATTN: `supports_sink` True (`triton_attn.py:395-396`), any CC (`:413-414`); sliding window `(w-1, 0)` (`:511-516`); kernel `unified_attention(..., sinks=...)` Triton (`v1/attention/ops/triton_unified_attention.py:807, 833, 871-886`).
   - FLEX_ATTENTION: no `supports_sink` override → base `False` (`v1/attention/backend.py:169-170`); also `torch.compile(flex_attention)` (`flex_attention.py:55-58`).
5. MoE block → `MLPBlock` (`gpt_oss.py:360`): router `GateLinear(bias=True)` (`:383-389`); `FusedMoEFactory(num_experts=32, top_k=4, renormalize=True, has_bias=True, activation="swigluoai", routed_experts_cls=GptOssRoutedExperts)` (`:391-404`); output sliced to hidden (`:418`).
   - Routing: softmax top-k → `fused_topk` → `vllm_topk_softmax` → CUDA `ops.topk_softmax` (`router/fused_topk_router.py:26-43, 80-116`; `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`).
   - MXFP4 experts: `GptOssMxfp4MoEMethod` (`mxfp4.py:145-155`) → `select_mxfp4_moe_backend` → oracle `fused_moe/oracle/mxfp4.py`, gpt-oss priority (`:368-400`): FLASHINFER_TRTLLM_MXFP4_BF16 → …_MXFP8 → AITER ×4 (ROCm) → TRITON (OAI `triton_kernels` matmul_ogs) → FLASHINFER_CUTLASS ×2 → MARLIN → BATCHED_MARLIN → HUMMING → XPU → CPU → EMULATION; none → `NotImplementedError` (`:685-691`).
   - BF16 experts: `UnquantizedFusedMoEMethod` (`fused_moe/unquantized_fused_moe_method.py:47`, backend select `:54`, bias params `:99-106`) → `oracle/unquantized.py` CUDA list FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON (`:68-74`), FlashInfer demoted for SWIGLUOAI (`:88-94`), none → `NotImplementedError` (`:369-371`).
   - SwiGLU-OAI (interleaved gate/up, clamp 7.0, alpha 1.702): inside MoE calls `torch.ops._C.swigluoai_and_mul` directly (`fused_moe/activation.py:380-381`); CUDA kernel `csrc/libtorch_stable/activation_kernels.cu:409-422`.
6. Weights → `GptOssModel.load_weights` (`gpt_oss.py:1340-1408`): mxfp4 → `_load_weights_mxfp4` (`:555`); quark → `_load_weights_quark` (`:751`); else (BF16) → `_load_weights_other` (`:1167`; w13 `[E,H,2I]` sliced `2*tp` on last dim then permuted to `[E,2I,H]`, w2 permuted, biases, sinks narrowed per head ≈ `:1209-1265`).

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| MXFP4 quant config gate | `Mxfp4Config.get_min_capability()` = 80; act dtype BF16 only | — | 8.0 | BLOCKED at config time ("Minimum capability: 80. Current capability: 37") | `quantization/mxfp4.py:67-69, 75-77`; enforced `config/vllm.py:940-955` | none | NO |
| MXFP4 MoE — FlashInfer TRT-LLM (BF16/MXFP8 act) | FlashInfer trtllm-gen | FlashInfer | 10.x only | BLOCKED | `fused_moe/experts/trtllm_mxfp4_moe.py:121-123` | — | — |
| MXFP4 MoE — TRITON (OAI `triton_kernels` matmul_ogs) | Triton + `triton_kernels` | Triton | 9.0 ≤ CC < 13.0 | BLOCKED | `experts/gpt_oss_triton_kernels_moe.py:38-50, 1046-1047`; `utils/import_utils.py:548-555` | — | — |
| MXFP4 MoE — FlashInfer CUTLASS | FlashInfer CUTLASS fused MoE | FlashInfer, CUTLASS | 9.0 / 10.x / 12.x | BLOCKED | `experts/flashinfer_cutlass_moe.py:128-139` | — | — |
| MXFP4 MoE — Marlin (W4A16, FP4 e2m1 + e8m0 scales) | vLLM CUDA Marlin MoE (`csrc/libtorch_stable/moe/marlin_moe_wna16/`) | mma.sync/ldmatrix/cp.async; MXFP4 kernels generated only with `a_type kBFloat16` (`generate_kernels.py:102-110`), SM75 kernels only for FP16/INT8 act (`:193`); `MARLIN_MOE_ARCHS "8.0+PTX;…"` (`CMakeLists.txt:1332-1336`) | Python gate 7.5 (`experts/marlin_moe.py:587-589`, `utils/marlin_utils_fp4.py:34`); effectively 8.0 for MXFP4 | BLOCKED | as cited | — | — |
| MXFP4 MoE — HUMMING | `humming` pkg | external | 7.5 | BLOCKED | `experts/fused_humming_moe.py:402-408` | — | — |
| MXFP4 MoE — CPU | x86 AMX CPU kernel | CPU platform | n/a | N/A on CUDA | `experts/cpu_moe.py:624-634` | — | — |
| MXFP4 MoE — EMULATION | `OCP_MXQuantizationEmulationTritonExperts` subclass of `TritonExperts`: per-forward dequant via `quark.torch.kernel.mx.dq_mxfp4`, then Triton `fused_moe_kernel` | amd-quark + Triton | none checked in Python | BLOCKED (GEMM is Triton) | `experts/ocp_mx_emulation_moe.py:84, 123-139, 158-210`; `quantization/utils/mxfp4_utils.py:146-158, 199-206`; `fused_moe.py:298, 864` | none (no pure-torch MXFP4 GEMM wired into any oracle) | NO |
| BF16/FP16 MoE experts (unquantized) | TritonExperts (`fused_moe_kernel`, `HAS_BIAS`), FlashInfer TRTLLM BF16 (CC 10.x, no swigluoai), FlashInfer CUTLASS (CC 9.0/10/12) | Triton / FlashInfer | Triton: no Python CC check (`triton_moe.py:118-119`) | BLOCKED (would be selected, fails at Triton launch) | `oracle/unquantized.py:68-74`; `experts/triton_moe.py:436, 557`; `fused_moe.py:350, 505-508`; `experts/trtllm_bf16_moe.py:95-102, 118-120` | `CPUUnquantizedExperts` supports SWIGLUOAI+bias (`experts/cpu_moe.py:170-176, 241-309`) but CPU-platform only | NO (not reachable on CUDA) |
| Router top-k (softmax, renorm) | CUDA `topk_softmax` (`_moe_C`) | `_moe_C` built for sm_37 | not arch-guarded in source | UNVERIFIED (depends on `_moe_C` build; CMake min arch 7.0, `CMakeLists.txt:116-132`) | `router/fused_topk_router.py:26-43` | torch `grouped_topk` softmax branch (`router/grouped_topk_router.py:112-164`) is correct with 1 group, but not selected for gpt-oss | YES if rewired |
| SwiGLU-OAI activation in MoE | CUDA `_C.swigluoai_and_mul` | `_C` build | not arch-guarded (dispatch on float types, `activation_kernels.cu:1019`) | UNVERIFIED (build) | `fused_moe/activation.py:380-381`; `csrc/libtorch_stable/activation_kernels.cu:409-422` | `SwigluOAIAndMul.forward_native` (`layers/activation.py:486-493`) — not used by the MoE path | YES if MoE path is changed to call it |
| Attention with sinks + sliding window 128 | TRITON_ATTN `unified_attention` (only sink-capable backend below CC 9.0) | Triton | Triton floor | BLOCKED | `triton_attn.py:395-396, 413-414`; `flash_attn.py:467-468`; `flashinfer.py:525-544`; `backend.py:169-170` | none in runtime (test reference `tests/kernels/attention/test_triton_unified_attention.py:125-171` has sliding window but no sinks) | NO |
| YaRN RoPE (fp32 cache), RMSNorm, qkv/o bias linears | `_C` rotary/rms_norm; cuBLAS | `_C` | shared runtime | shared runtime | `gpt_oss.py:110-126, 137-153`; `layernorm.py:74, 96` | `forward_native` (RMSNorm `layernorm.py:74`) | YES |
| Dtype | CUDA platform: CC < 6.0 → `[torch.float32]` only | — | FP16 needs 6.0 | Model must run FP32 (≈ 84 GB for BF16 checkpoint, 20.9B × 4 B) | `platforms/cuda.py:255-264` | — | FP32 does not fit 2×12 GB |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| MXFP4 (official openai/gpt-oss-20b, -safeguard-20b) | `GptOssMxfp4Config` (subclass of `Mxfp4Config`) → `GptOssMxfp4MoEMethod`; linears unquantized | 80; act dtype BF16 only | `quantization/__init__.py:176-177`; `mxfp4.py:67-69, 75-77, 126-155` |
| BF16 (unsloth/gpt-oss-20b-BF16, unsloth/gpt-oss-safeguard-20b-BF16, FriendliAI/…) | no quant method → `_load_weights_other` + `UnquantizedFusedMoEMethod` — vLLM **would load it** (names `gate_up_proj`/`down_proj`/`_bias`/`sinks` handled, `gpt_oss.py:1167-1265, 1424-1433` per sub-trace); on K80 dtype forced to FP32 (`platforms/cuda.py:264`) and MoE GEMM is Triton | n/a (no quant gate) | unsloth index file; `oracle/unquantized.py:68-74` |
| Quark MXFP4 (amd/gpt-oss-20b-w-mxfp4-a-bf16) | `QuarkConfig` → `_load_weights_quark` | 70 (`quark/quark.py:127-128`) | `__init__.py:172`; `gpt_oss.py:751` — MoE still needs an MXFP4 backend from the oracle above |
| bitsandbytes (unsloth/gpt-oss-20b-bnb-4bit) | none: no `bitsandbytes` key in main's quantization registry | n/a | grep `bitsandbytes` in `quantization/__init__.py`: none |
| GGUF (ggml-org/unsloth) | none: no GGUF quant method/loader in main | n/a | see LFM2 GGUF row |

#### In the K80 fork (vllm37)
- Model file: PRESENT — `vllm37/vllm/model_executor/models/gpt_oss.py` (sinks param line 75, `layer_idx % 2` sliding window line 107, `Attention(..., sinks=...)` line 119; MoE `FusedMoE(..., has_bias=True, activation="swiglu_oai")` lines 164-165; loaders `_load_weights_mxfp4` line 267 and `_load_weights_other` line 475).
- Registry entry: PRESENT — `vllm37/vllm/model_executor/models/registry.py:76`; config hook `vllm37/vllm/model_executor/models/config.py:250, 376`.
- MXFP4: `vllm37/vllm/model_executor/layers/quantization/mxfp4.py` `get_min_capability()` = 90 (lines 44-45).
- Sinks in fork: only `vllm37/vllm/v1/attention/backends/{flash_attn,flashinfer,triton_attn}.py` and Triton ops `vllm37/vllm/attention/ops/{triton_unified_attention,prefix_prefill,chunked_prefill_paged_decode}.py`; none of the v0 backends in `vllm37/vllm/attention/backends/` (xformers, torch_sdpa, flash_attn, …) mention `sinks`.
- BF16 MoE in fork: `swiglu_oai` implemented in torch inside Triton `fused_experts` path (`vllm37/vllm/model_executor/layers/fused_moe/fused_moe.py:1625-1648`) — still Triton GEMMs.

#### Verdict for K80
Blockers:
1. **MXFP4 checkpoint is rejected at config time** (min CC 80, BF16 activations only; `mxfp4.py:67-77`, `config/vllm.py:940-955`), and every MXFP4 MoE backend needs CC ≥ 7.5–10.0 or Triton; the "emulation" backend still runs Triton GEMMs (`ocp_mx_emulation_moe.py:84, 210`). Fix: offline-dequantize MXFP4 → FP16 (or use a community BF16 repo and cast to FP16) — i.e. never run MXFP4 on device.
2. **No dtype that fits**: CC < 6.0 → FP32 only (`platforms/cuda.py:255-264`); 20.9B params × 4 B ≈ 84 GB ≫ 24 GB (2 × 12 GB K80 dies). Even FP16 (≈ 42 GB) does not fit; only an INT4/INT8 weight-only format with a K80-capable kernel (e.g. the fork's GGUF path, min CC 60 as-is in `vllm37/.../gguf.py:43-44`, would need lowering and gpt-oss arch mapping) or CPU expert offload could fit. Fix: enable FP16 storage on sm_37 (FP16 storage + FP32 math, as the fork presumably does) **and** a ≤ 4-bit expert format with a non-Triton dequant-GEMM kernel.
3. **Attention sinks + sliding window**: on CC < 9.0 only TRITON_ATTN supports sinks (`triton_attn.py:395-396`; FA `flash_attn.py:467-468`; FlashInfer `flashinfer.py:525-544`); Triton does not target Kepler; the fork's v0 backends have no sinks. Fix: add sinks (extra per-head logit in the softmax denominator) + window 128 to the fork's K80 paged-attention kernel / a torch reference path.
4. **MoE expert GEMM (any dtype) is Triton/FlashInfer/Marlin only** (`oracle/unquantized.py:68-74`); fix: a torch per-expert loop (index_select + matmul + bias + `SwigluOAIAndMul.forward_native` `activation.py:486-493`) or a CUDA grouped GEMM without tensor cores; router via torch top-k (`grouped_topk_router.py:112-164`) if `_moe_C` is not built for sm_37.
Net: gpt-oss-20b is impractical on a K80 (memory, not just kernels); LFM2 dense models (230M–2.6B) are realistic after the ShortConv port.

## 8. Hunyuan dense, ERNIE 4.5 and SmolLM3 on Tesla K80 (sm_37)


Evidence conventions
- `V:` = `/home/jack/src/vllm` (official, main @ 3ca00a8261). Paths below are relative to `V:/vllm/` unless they start with `csrc/`, `requirements/` or `CMakeLists.txt` (relative to `V:`).
- `F:` = `/home/jack/src/vllm37` (K80 fork).
- `HF-T:` = Hugging Face Transformers **v5.18.0** source, fetched from `https://raw.githubusercontent.com/huggingface/transformers/v5.18.0/src/transformers/...` into a local scratch directory (not kept). Official vLLM pins `transformers >= 5.16.1, < 5.19.0` (`V:requirements/common.txt:10`), so v5.18.0 is inside the supported range.
- `cfg:<repo>` = field of that repo's `config.json`, downloaded with `hf download <repo> config.json` into a local scratch directory (not kept). "Released" = HF `createdAt` from `hf models ls --expand createdAt,safetensors`. "Params" = HF `safetensors.total`.

Shared facts that apply to all three families (cited once here, referenced below as S1–S6)
- **S1 — no vLLM CUDA kernels for sm_37.** `CMakeLists.txt:120-131`: `CUDA_SUPPORTED_ARCHS` lowest entry is 7.0 (CUDA < 12.8) or 7.5; `CMakeLists.txt:231-232` intersects the requested archs with this list, so every `torch.ops._C.*` / `torch.ops._moe_C.*` op is absent for sm_37 in an official build. Torch pin: `torch==2.13.0` (`V:requirements/cuda.txt:7`).
- **S2 — dtype.** `platforms/cuda.py:255-264`: below CC 6.0, `supported_dtypes` = `[torch.float32]` ("Kepler and Maxwell ... only FP32 is supported, though vLLM doesn't support these GPUs"), so `--dtype auto` on bf16 checkpoints resolves to **fp32**; `platforms/cuda.py:656-674` raises for bf16 below CC 8.0. All three families ship bf16 (`torch_dtype: "bfloat16"` in every `cfg:` below, except Hunyuan AWQ = float16).
- **S3 — attention backends on CUDA.** Priority list `platforms/cuda.py:166-178` = FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT. FLASH_ATTN needs CC ≥ 8.0 (`v1/attention/backends/flash_attn.py:451-452`); FLASHINFER needs CC ≥ 8.0 (`v1/attention/backends/flashinfer.py:514-520`); TRITON_ATTN accepts any CC (`triton_attn.py:413-414`) but is Triton (`triton_attn.py:37-42` imports `triton_prefill_attention`, `triton_reshape_and_cache_flash`, `triton_unified_attention`); FLEX_ATTENTION wraps `torch.compile(flex_attention)` (`flex_attention.py:55-58`), i.e. Inductor→Triton. No non-Triton CUDA attention backend works below CC 8.0.
- **S4 — CustomOp fallback switch.** `model_executor/custom_op.py:172-205`: if an op is disabled via `compilation_config.custom_ops` it dispatches to `forward_native` (line 189-192); otherwise to `forward_cuda`. `forward_native` is compiled only when compilation mode ≠ NONE (`custom_op.py:207-227`).
- **S5 — RMSNorm IR op.** `RMSNorm.forward_cuda` → `forward_native` → `ir.ops.rms_norm` (`model_executor/layers/layernorm.py:74-114`). CUDA default priority is `["vllm_c","native"]` when not using Inductor (`platforms/cuda.py:730-752`); `vllm_c` impl = `torch.ops._C.rms_norm` (`kernels/vllm_c.py:23-43`); `native` impl = pure torch (`ir/ops/layernorm.py:9-21`). Priority is user-settable (`config/kernel.py:31-35`).
- **S6 — Unquantized linear.** `dispatch_unquantized_gemm` returns `default_unquantized_gemm` = `torch.nn.functional.linear` on CUDA unless a FlashInfer BF16 backend is requested (`model_executor/layers/utils.py:83-89, 616-636`). Embedding = `F.embedding` (`model_executor/layers/vocab_parallel_embedding.py:84-85`).
- **S7 — quant min-capability gate.** `config/vllm.py:935-949` raises `ValueError` if `capability < quant_config.get_min_capability()`. sm_37 → 37.
- **S8 — K80 memory.** 12 GB per GK210 die, 2 dies per board (UNVERIFIED in this session — no datasheet fetched; standard NVIDIA spec).

---

### Tencent Hunyuan dense (HunYuanDenseV1ForCausalLM)

#### Official releases
| Repo | Released | Params | Format | Quant config | Context | Layers / KV heads / head_dim | Rope |
|---|---|---|---|---|---|---|---|
| tencent/Hunyuan-0.5B-Instruct (+ -Pretrain 2025-07-28) | 2025-07-30 | 0.539 B | BF16 safetensors | none | 262144 (`max_position_embeddings`) | 24 / 8 (16 q) / 128 | `rope_scaling={"type":"dynamic","alpha":1000.0,"factor":1.0,...}`, `rope_theta=10000` |
| tencent/Hunyuan-0.5B-Instruct-FP8 | 2025-07-30 | 0.539 B (415 M F8_E4M3) | FP8 E4M3 | `quant_method: compressed-tensors`, `format: naive-quantized`, weights+input `num_bits 8, type float, strategy tensor, dynamic false`, `ignored_layers [lm_head, model.embed_tokens]` | 262144 | 24 / 8 / 128 | same |
| tencent/Hunyuan-0.5B-Instruct-GPTQ-Int4 | 2025-07-30 | 0.539 B (I32 packed) | GPTQ int4 | `bits 4, group_size 128, desc_act true, sym true, static_groups true, checkpoint_format gptq` | 262144 | 24 / 8 / 128 | same |
| tencent/Hunyuan-0.5B-Instruct-AWQ-Int4 | 2025-07-31 | 0.539 B | AWQ int4 (`torch_dtype float16`) | `quant_method awq, bits 4, group_size 128, version gemm, zero_point true` | 262144 | 24 / 8 / 128 | same |
| tencent/Hunyuan-1.8B-Instruct (+ -FP8/-GPTQ-Int4 2025-07-30, -AWQ-Int4 2025-07-31, -Pretrain) | 2025-07-30 | 1.791 B | BF16 (+FP8/GPTQ/AWQ siblings, same configs as 0.5B — safetensors dtypes F8_E4M3 / I32 confirm) | none | 262144 | 32 / 4 (16 q) / 128 | same |
| tencent/Hunyuan-4B-Instruct (+ FP8/GPTQ/AWQ/Pretrain) | 2025-07-30 | 4.222 B | BF16 (+siblings) | none | 262144 | 36 / 8 (32 q) / 128 | same |
| tencent/Hunyuan-7B-Instruct (+ -Pretrain) | 2025-07-30 | 7.505 B | BF16 | none | **32768** (config) | 32 / 8 (32 q) / 128 | same |
| tencent/Hunyuan-7B-Instruct-FP8 | 2025-07-30 | 7.505 B | FP8 E4M3 | same compressed-tensors block as 0.5B-FP8 | 32768 | 32 / 8 / 128 | same |
| tencent/Hunyuan-7B-Instruct-GPTQ-Int4 | 2025-07-30 | 7.505 B | GPTQ int4 | same as 0.5B GPTQ (`desc_act true`) | 32768 | 32 / 8 / 128 | same |
| tencent/Hunyuan-7B-Instruct-AWQ-Int4 | 2025-07-31 | 7.505 B | AWQ int4 (fp16) | same as 0.5B AWQ | 32768 | 32 / 8 / 128 | same |
| tencent/Hunyuan-7B-Instruct-0124 | 2025-01-24 | n/a (no safetensors metadata) | `pytorch_model-00001-of-00001.bin` (15.0 GB) + remote code `modeling_hunyuan.py` (file listing) | none | 32768 | 32 / 8 / 128 | same |
| tencent/Hunyuan-MT-7B / -MT-Chimera-7B (+ -fp8) | 2025-08-28/29 | 8.030 B | BF16 / FP8 compressed-tensors | fp8: same block as above | 32768 | 32 / 8 / 128 | `alpha 100000.0` |
| GGUF | — | — | none from `tencent` for dense models (only `tencent/Hunyuan-A13B-Instruct-GGUF`, MoE 80 B) | — | — | — | — |

Common to all dense configs: `use_qk_norm: true`, `tie_word_embeddings: true`, `use_cla: false`, `attention_bias/mlp_bias false`, `hidden_act silu`, `rms_norm_eps 1e-5`, `model_type hunyuan_v1_dense`; vocab 120818 (0.5/1.8/4B), 128167 (7B), 128256 (MT). No NoPE layers.

#### Code path (official vLLM)
Official vLLM has **no native Hunyuan dense model file**; it is served by the generic Transformers modeling backend.
1. Registry → `models/registry.py:710` `"HunYuanDenseV1ForCausalLM": ("transformers", "TransformersForCausalLM")` (in `_TRANSFORMERS_SUPPORTED_MODELS`, `registry.py:702`) → class `TransformersForCausalLM(CausalMixin)` (`models/transformers/__init__.py:129`).
2. Model construction → `Base.__init__` (`models/transformers/base.py:121-199`): sets `_attn_implementation="vllm"` (`base.py:209`), builds HF `HunYuanDenseV1Model` on `meta` via `AutoModel.from_config` (`base.py:174-179`; HF class `HF-T:models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py:383`), then `recursive_replace` (`base.py:467-587`) and `_create_attention_instances` (`base.py:590-691`).
3. torch.compile → `base.py:287-300` decorates the decoder with `enable_if=can_enable_torch_compile`; `models/transformers/utils.py:359-369` returns **False** when any `rope_type == "dynamic"` ("Dynamic rope scaling is not compatible with torch.compile") → Hunyuan runs eager (no Inductor/Triton codegen for the model body). `type`→`rope_type` normalisation: `transformers_utils/config.py:628-630`.
4. Embedding → vocab `nn.Embedding` replaced by `VocabParallelEmbedding` (`base.py:565-568`, `transformers/utils.py:263`) → `F.embedding` (S6).
5. RMSNorm (input, post-attn, final, **and QK-norm**) → HF `HunYuanDenseV1RMSNorm` (`HF-T:…dense.py:47-64`) structurally matched by `RMSNormFuser` (`transformers/fusers/rms_norm.py:144-171, 278-297`) → `TPAwareRMSNorm` (subclass of vLLM `RMSNorm`, `rms_norm.py:135`, resolved in `transformers/layers.py:38`) → S5 (`_C.rms_norm` or native).
6. QKV → `q_proj/k_proj/v_proj` matched by `QKVFuser` (`transformers/fusers/qkv.py:83, 165-200`) → `QKVParallelLinear`; `o_proj` → `replace_linear_class` (`transformers/utils.py:109-135`, called at `base.py:553-562`) → `F.linear` (S6).
7. Rotary (dynamic NTK-alpha) → **HF `HunYuanDenseV1RotaryEmbedding`**, not a vLLM rope class (no file in `models/transformers/` references rotary — grep `rotary|rope` hits only `utils.py:365-369` and MLA dims). `HF-T:…dense.py:323-331`: if `rope_type=="dynamic"` and `alpha`: `base = rope_theta * alpha ** (head_dim/(head_dim-2))`, `inv_freq` computed once, `attention_scaling=1.0`. `forward` (`:364-380`) is decorated with `@dynamic_rope_update`, which on every call does `seq_len = torch.max(position_ids)+1` and, only if `seq_len > max_position_embeddings`, recomputes `inv_freq` with the generic `ROPE_INIT_FUNCTIONS["dynamic"]` (ignores `alpha`) (`HF-T:modeling_rope_utils.py:82-128`). `apply_rotary_pos_emb` (`:92`) is plain torch (cos/sin via fp32 matmul+cat, `:366-380`). vLLM's own `DynamicNTKAlphaRotaryEmbedding` (`model_executor/layers/rotary_embedding/dynamic_ntk_alpha_rope.py:9-43`, chosen by `get_rope` at `rotary_embedding/__init__.py:209-219`) is **not used** on this path.
8. QK-norm → applied **after** RoPE, per head (`HF-T:…dense.py:180-181, 199-201`), replaced with vLLM RMSNorm as in step 5.
9. Attention → HF calls `ALL_ATTENTION_FUNCTIONS["vllm"]` = `vllm_attention_forward` (`transformers/__init__.py:67-94, 124`) → vLLM `Attention` (`model_executor/layers/attention/attention.py:229`) instantiated at `base.py:681` → backend per S3.
10. MLP → `gate_proj/up_proj/act_fn/down_proj` matched by `GLUFuser` (`transformers/fusers/glu.py:50, 122, 160-191`) → `MergedColumnParallelLinear` + `get_act_and_mul_fn("silu")` (`transformers/layers.py:42-64`) = `SiluAndMul` (`model_executor/layers/activation.py:116-150`) + `RowParallelLinear`.
11. LM head → tied `ParallelLMHead.tie_weights(embed)` (`transformers/causal.py:48-57`), `LogitsProcessor` (`causal.py:60`).

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| Build of `_C` ops | CUDA C++ | arch in CUDA_SUPPORTED_ARCHS | 7.0 | not built | S1 | per-op native paths below | yes where listed |
| dtype | — | bf16 needs CC ≥ 8.0 | — | auto → fp32 | S2 | `--dtype half` (fp16) | fp16 storage OK; fp16 GEMM speed/support on sm_37 cuBLAS UNVERIFIED |
| Embedding | `F.embedding` | torch | any | OK | S6 | — | — |
| Linear (QKV/O/gate_up/down) | `F.linear` → cuBLAS | torch | any | OK (fp32) | S6; `transformers/fusers/qkv.py:176`, `glu.py:181-191` | — | — |
| RMSNorm + QK-norm | `_C.rms_norm` (vllm_c) | `_C` | 7.0 build | missing | S5, `kernels/vllm_c.py:23-43` | `ir/ops/layernorm.py:9-21` (set `ir_op_priority.rms_norm=["native"]`, `config/kernel.py:31-35`) | yes (pure torch) |
| SiluAndMul | `torch.ops._C.silu_and_mul` | `_C` | 7.0 build | missing | `activation.py:136-150` | `activation.py:141-144` via `custom_ops=["none"]` (S4) | yes |
| RoPE dynamic-alpha | HF torch code | torch | any | OK | `HF-T:…dense.py:312-380`, `modeling_rope_utils.py:82-128` | — | — (note: `torch.max(position_ids)` forces a host sync each step) |
| torch.compile | Inductor → Triton | Triton | Triton: not Kepler | auto-disabled for this model | `transformers/utils.py:359-369` | eager | yes |
| Attention | FA / FlashInfer / Triton / Flex | FA ≥ 8.0, FI ≥ 8.0, Triton | 8.0 or Triton | **blocker** | S3 | none on CUDA | no |
| Fusers (FX tracing) | `torch.fx` on meta device | torch ≥ version used by `fx_utils.py` | any | depends on torch build (shared runtime) | `transformers/fuser.py:53-61` | — | UNVERIFIED with a K80-compatible torch |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| FP8 E4M3 static per-tensor W8A8 (compressed-tensors) | `CompressedTensorsConfig` → `CompressedTensorsW8A8Fp8` if CC ≥ 89, else `CompressedTensorsW8A16Fp8` (Marlin FP8) | config 70; W8A8Fp8 89; W8A16Fp8 75 | `quantization/compressed_tensors/compressed_tensors.py:117-118, 834-868`; `schemes/compressed_tensors_w8a8_fp8.py:83-85`; `schemes/compressed_tensors_w8a16_fp8.py:53-55`; rejected at S7 (37 < 70) |
| GPTQ int4 g128 **desc_act=true** | `AutoGPTQConfig` ("gptq"/"gptq_marlin" → `auto_gptq`) | 60 | `quantization/__init__.py:168-169`; `auto_gptq.py:182-183`; **additionally rejected**: `auto_gptq.py:116` → `utils/gptq_utils.py:32-37` raises "GPTQ group activation ordering (desc_act=True) is no longer supported" — so the Tencent GPTQ repos fail on any GPU in official vLLM. Kernel candidates `kernels/linear/__init__.py:506-514`: Cutlass W4A8 90, Machete 90, Marlin 75, Conch 80, Exllama 60 (fp16-only, `exllama.py:24-25, 43-44`; `csrc/libtorch_stable/quantization/gptq/q_gemm.cu:55-60` uses `half2`/`__hfma2` = sm_53+), TritonW4A16 0 (Triton), Humming 75 |
| AWQ int4 g128 GEMM, zero_point | `AutoAWQConfig` ("awq"/"awq_marlin" → `auto_awq`) | 75 | `quantization/__init__.py:157-158`; `auto_awq.py:233-234`; non-Marlin CUDA kernel `csrc/libtorch_stable/quantization/awq/gemm_kernels.cu:30-31` asserts for `__CUDA_ARCH__ < 750` (uses `ldmatrix`/`mma.sync`, `:189-275`); Triton alt `_custom_ops.py:587-598` |

#### In the K80 fork (vllm37)
- Model file: present — `F:/vllm/model_executor/models/hunyuan_v1.py` (dense class `HunYuanDenseV1ForCausalLM` at line 932; QK-norm after RoPE at lines 219-227).
- Registry: present — `F:/vllm/model_executor/models/registry.py:88` `"HunYuanDenseV1ForCausalLM": ("hunyuan_v1", "HunYuanDenseV1ForCausalLM")` (MoE at :87).
- Rope: fork's `get_rope` builds `DynamicNTKAlphaRotaryEmbedding` for `type: dynamic` + `alpha` — `F:/vllm/model_executor/layers/rotary_embedding/__init__.py:130-135`.

#### Verdict for K80
Blockers (official vLLM):
1. Attention: no CUDA backend below CC 8.0 without Triton (S3). → Port work: use the fork's existing sm_37 attention path (fork v0 engine) — this is shared-runtime work, not model-specific.
2. No `_C` kernels for sm_37 (S1) → RMSNorm/SiluAndMul must run native (S4/S5) or `_C` must be rebuilt for sm_37 (fork already does this per its own build; not verified here).
3. Official path depends on Transformers v5 + torch.fx fusers on torch 2.13 (`requirements/cuda.txt:7`, `requirements/common.txt:10`) — not runnable on torch 2.0.1/CUDA 11.4. → Use the fork's **native** `hunyuan_v1.py` instead (registered, includes NTK-alpha rope); no backend port needed.
4. Quantized releases: FP8 (min 70/75/89), AWQ (75), GPTQ (60 + desc_act=true rejected) all fail S7 at 37. → Only BF16 releases are usable; convert to fp16 offline, or dequantize GPTQ/AWQ to fp16 on load. An sm_37 int4 GEMM would need a new non-`half2` kernel (fp32 accumulate, scalar `__half2float` conversions).
5. Memory (S8): fp32 auto-dtype weights = 0.5B 2.2 GB, 1.8B 7.2 GB, 4B 16.9 GB, 7B 30.0 GB; fp16 halves this. → 0.5B/1.8B fit one die; 4B needs fp16 (8.4 GB) on one die or TP=2; 7B needs fp16 + TP=2 (15.0 GB across 2×12 GB) with little KV headroom.
Model-specific ops (RMSNorm, QK-norm, SiLU-mul, NTK-alpha RoPE) all have pure-torch or fork-native implementations; **Hunyuan dense is not blocked by model-specific kernels on K80**.

---

### Baidu ERNIE 4.5 (Ernie4_5ForCausalLM, Ernie4_5_MoeForCausalLM)

#### Official releases
| Repo | Released | Params | Format | Quant config | Context | Layers / KV heads / head_dim | Rope |
|---|---|---|---|---|---|---|---|
| baidu/ERNIE-4.5-0.3B-PT (+ -Base-PT) | 2025-06-28 | 0.361 B | BF16 safetensors | none | 131072 | 18 / 2 (16 q) / 128 | default RoPE, `rope_theta 500000`, `rope_scaling null`; non-NeoX (interleaved) per vLLM `ernie45.py:44-53` |
| baidu/ERNIE-4.5-21B-A3B-PT | 2025-06-28 | 21.949 B (BF16 + 4.4 M F32) | BF16 | none | 131072 | 28 / 4 (20 q) / 128 (=2560/20, no `head_dim` field) | same; MoE: 64 experts, top-6, 2 shared, `moe_intermediate_size 1536`, `moe_layer_start_index 1`, `moe_layer_end_index 27`, `num_nextn_predict_layers` present |
| baidu/ERNIE-4.5-21B-A3B-Base-PT | 2025-06-28 | 21.825 B | BF16 | none | 131072 | 28 / 4 / 128 | same (config not downloaded; params from HF metadata) |
| baidu/ERNIE-4.5-21B-A3B-Thinking | 2025-09-08 | 21.825 B | BF16 | none | 131072 | 28 / 4 / 128 | same; no `moe_layer_end_index` (HF default −1 → num_layers−1, `HF-T:ernie4_5_moe/configuration_ernie4_5_moe.py:114-122`) |
| baidu/ERNIE-4.5-*-Paddle (0.3B, 21B-A3B, and 300B FP8/W4A8/2-bit variants) | 2025-06-28/07-08 | — | PaddlePaddle checkpoints (safetensors dtype `U16`/`I8`) | Paddle-specific | — | — | — | 
| GGUF / AWQ / GPTQ / FP8 (HF format) | — | — | **none** from `baidu` ≤ 40 B (`hf models ls --author baidu --search ERNIE-4.5`, 25 repos) | — | — | — | — |

Tied embeddings: `tie_word_embeddings: true` (both cfgs). QK-norm: none (no field; vLLM model has no q/k norm). NoPE: none. Paddle repos: no Paddle loader in `model_executor/model_loader` (grep -i paddle: no hits) → not loadable by vLLM.

#### Code path (official vLLM)
Dense (0.3B):
1. Registry `models/registry.py:96` → `models/ernie45.py:41` `Ernie4_5ForCausalLM(LlamaForCausalLM)` with `@support_torch_compile` (`ernie45.py:32-40`); post-init sets `rotary_emb.is_neox_style=False`, removes `o_proj` bias (`ernie45.py:44-53`).
2. Embedding `VocabParallelEmbedding` (`models/llama.py:382`) → `F.embedding` (S6).
3. RMSNorm input/post-attn with fused residual add (`llama.py:308-311, 322-328`), final norm (`llama.py:395`) → S5 (`ir.ops.rms_norm` / `fused_add_rms_norm`).
4. QKV `QKVParallelLinear` (`llama.py:165`), `o_proj` `RowParallelLinear` (`llama.py:175`) → `F.linear` (S6).
5. RoPE `get_rope(...)` default (`llama.py:236-248`) → `RotaryEmbedding` (`rotary_embedding/base.py:139`): `forward_cuda` → `ops.rotary_embedding` (`base.py:221-252`); `forward_native` (`base.py:203-219`) → `forward_static` (`base.py:161-201`) → `ApplyRotaryEmb.forward_static` (`rotary_embedding/common.py:134`), pure torch.
6. Attention `Attention(...)` (`llama.py:212-222`) → backend per S3.
7. MLP `MergedColumnParallelLinear` + `SiluAndMul` + `RowParallelLinear` (`llama.py:82-122`).
8. LM head tied (`llama.py:503-510`), `LogitsProcessor` (`llama.py:513`).

MoE (21B-A3B, -Thinking):
1. Registry `registry.py:97` → `models/ernie45_moe.py:510` `Ernie4_5_MoeForCausalLM`.
2. Attention `Ernie4_5_MoeAttention` (`ernie45_moe.py:213-303`): `QKVParallelLinear`, `RowParallelLinear`, `get_rope(..., is_neox_style=False)` (`:272-277`), `Attention` (`:278`) — same ops as dense steps 4-6.
3. Layer MLP choice (`ernie45_moe.py:337-365`): layer 0 dense `Ernie4_5_MoeMLP` (`:79-117`, Merged/Row linear + `SiluAndMul` at `:110`); layers 1-27 `Ernie4_5_MoeMoE`.
4. Router gate `GateLinear(..., out_dtype=fp32, params_dtype=fp32)` (`ernie45_moe.py:156-161`) → on non-SM90/100 falls to tier 6 `F.linear` (`fused_moe/router/gate_linear.py:20-33, 287-302`).
5. Routing: `e_score_correction_bias` (`ernie45_moe.py:164-166, 193`) → `create_fused_moe_router` (`fused_moe/layer.py:291-327`) → `FusedTopKBiasRouter` (`fused_moe/router/router_factory.py:228-237`) → `fused_topk_bias` → `vllm_topk_softmax` → `ops.topk_softmax` (`router/fused_topk_bias_router.py:35-53, 146-227`) → `torch.ops._moe_C.topk_softmax` (`_custom_ops.py:2300-2312`) → `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`.
6. Shared experts (2 × 1536 → one `Ernie4_5_MoeMLP` of 3072, `ernie45_moe.py:167-181`) passed into `FusedMoEFactory` (`:184-197`).
7. Routed experts: `FusedMoEFactory` (`fused_moe/layer.py:88`) → `MoERunner` → `UnquantizedFusedMoEMethod.forward_cuda` → `forward_native` → `self.moe_kernel.apply` (`fused_moe/unquantized_fused_moe_method.py:301-342`; "native" here is still the selected kernel). Backend oracle on CUDA (`fused_moe/oracle/unquantized.py:63-73`): FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, **TRITON**, BATCHED_TRITON. For pre-Ampere this resolves to `TritonExperts` (`experts/triton_moe.py:65, 118-119, 243, 416`) → `moe_align_block_size` CUDA op (`fused_moe/moe_align_block_size.py:153`, `_custom_ops.py:2176`) + Triton `fused_moe_kernel` (`fused_moe/fused_moe.py:35, 297-298, 761`).
8. **Non-Triton fused-MoE path?** On CUDA: none. CPU backend (`oracle/unquantized.py:104-105`) is CPU-only (`experts/cpu_moe.py:144-145` `is_cpu()`) and is a C++ kernel (`cpu_fused_moe`, `cpu_moe.py:303-315`). The only pure-torch MoE in the tree is test reference code `V:tests/kernels/utils.py:295-334` (`torch_moe`, `torch_moe_single`).
9. LM head tied; MTP (`num_nextn_predict_layers`) weights skipped unless speculative (`ErnieMTPModel`, `registry.py:677`).

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| dtype | — | bf16 ≥ 8.0 | — | auto → fp32 | S2 | `--dtype half` | fp16 storage OK (UNVERIFIED perf) |
| Embedding / Linear | `F.embedding` / `F.linear` | torch | any | OK | S6 | — | — |
| RMSNorm (+fused add) | `_C.rms_norm`, `_C.fused_add_rms_norm` | `_C` | 7.0 build | missing | S5, `kernels/vllm_c.py:23-86` | `ir/ops/layernorm.py:9-21, 43-62` | yes |
| RoPE (non-NeoX) | `_C.rotary_embedding` | `_C` | 7.0 build | missing | `rotary_embedding/base.py:221-252` | `base.py:203-219` + `common.py:134` via `custom_ops=["none"]` | yes |
| SiluAndMul | `_C.silu_and_mul` | `_C` | 7.0 build | missing | `activation.py:136-150` | `activation.py:141-144` | yes |
| torch.compile | Inductor → Triton | Triton | not Kepler | enabled by default (`ernie45.py:32`) | — | `-O0`/`--enforce-eager` | yes |
| Attention | FA / FI / Triton / Flex | see S3 | 8.0 / Triton | **blocker** | S3 | none on CUDA | no |
| MoE gate | `F.linear` fp32 | torch | any | OK | `gate_linear.py:287-302` | — | — |
| MoE top-k (+bias) | `_moe_C.topk_softmax` (cub, `__bfloat162float`/`__half2float` conversions only; fp32 input here) | `_moe_C` | 7.0 build | missing; source looks arch-portable (no `__CUDA_ARCH__` guard found by grep) — compile for sm_37 UNVERIFIED | `topk_softmax_kernels.cu:63-66, 83, 181-186, 285-287, 849` | no CUDA fallback (`GroupedTopKRouter.forward_native` exists at `router/grouped_topk_router.py:192` but ERNIE does not use grouped top-k) | no (needs port or a torch top-k) |
| MoE align | `_moe_C.moe_align_block_size` | `_moe_C` | 7.0 build | missing | `moe_align_block_size.py:153` | none | no |
| MoE experts | Triton `fused_moe_kernel` / FlashInfer | Triton / FI (SM90/100) | Triton | **blocker** | `oracle/unquantized.py:63-73, 134-146`; `fused_moe.py:297` | none on CUDA (CPU C++ only; torch ref in tests only) | no |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| (none in HF/PT format ≤ 40 B from baidu) | — | — | HF listing `baidu` / `ERNIE-4.5`: only BF16 `-PT` and Paddle repos |
| Paddle FP8 / W4A8 / 2-bit (300B only, >40 B anyway) | not supported (Paddle format) | — | no "paddle" in `model_executor/model_loader` |

#### In the K80 fork (vllm37)
- Model files: present — `F:/vllm/model_executor/models/ernie45.py`, `F:/vllm/model_executor/models/ernie45_moe.py` (uses `FusedMoE` at line 129, `get_rope(..., is_neox_style=False)` at 235-240).
- Registry: present — `F:/vllm/model_executor/models/registry.py:63` (`Ernie4_5ForCausalLM`), `:64` (`Ernie4_5_MoeForCausalLM`).

#### Verdict for K80
- **ERNIE-4.5-0.3B (dense)**: no model-specific blocker. It is Llama with non-NeoX RoPE; all ops have pure-torch fallbacks (RMSNorm, RoPE, SiLU-mul) or are cuBLAS. Blockers are shared runtime only: attention (S3) and `_C` build (S1). Fork already registers it. fp32 weights 1.44 GB — fits easily.
- **ERNIE-4.5-21B-A3B (-PT/-Thinking)**: blocked.
  1. Fused MoE experts are Triton/FlashInfer only on CUDA (`oracle/unquantized.py:63-73`) → port work: write a non-Triton expert path for sm_37 (e.g. torch loop over experts with `index_select`+`F.linear`, or a CUDA grouped GEMM in fp32 accumulate); check whether the fork's `FusedMoE` already has one (outside this task's scope).
  2. `topk_softmax` and `moe_align_block_size` are `_moe_C` CUDA ops (not built for sm_37, S1) → compile them for sm_37 (source has no arch guard in `topk_softmax_kernels.cu`, UNVERIFIED) or replace with `torch.softmax`+`torch.topk` with bias-corrected selection.
  3. Memory (S8): 21.95 B params = 87.8 GB fp32 / 43.9 GB fp16 vs 24 GB per K80 board; no official int4/GGUF release in HF format → does not fit even with TP=2 unless the port adds an sm_37 int4/int8 weight-only MoE path plus third-party quantization, or CPU offload of experts. Practically: **not viable on one K80**.
  4. Attention (S3) as for dense.

---

### HuggingFace SmolLM3 (SmolLM3ForCausalLM)

#### Official releases
| Repo | Released | Params | Format | Quant config | Context | Layers / KV heads / head_dim | Rope |
|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM3-3B | 2025-07-08 | 3.075 B | BF16 safetensors | none | 65536 | 36 / 4 (16 q) / 128 (2048/16, no `head_dim` field) | default RoPE `rope_theta 5000000`, `rope_scaling null`; **NoPE** on layers 3,7,11,15,19,23,27,31,35 (`no_rope_layers` has 0 at those indices; `no_rope_layer_interval 4`) |
| HuggingFaceTB/SmolLM3-3B-Base | 2025-06-19 | 3.075 B | BF16 | none | 65536 | 36 / 4 / 128 | same; no explicit `no_rope_layers` → HF derives every 4th layer NoPE (`HF-T:smollm3/configuration_smollm3.py:101-104`) |
| HuggingFaceTB/SmolLM3-3B-ONNX | 2025-07-08 | — | ONNX (not vLLM-loadable) | — | — | — | — |
| HuggingFaceTB/SmolLM3-3B-checkpoints | 2025-07-20 | — | training checkpoints (no safetensors metadata) | — | — | — | — |
| GGUF / AWQ / GPTQ / FP8 | — | — | **none** under `HuggingFaceTB` (listing shows only `smollm-360M-instruct-v0.2-Q8_0-GGUF`, a SmolLM v1 model) | — | — | — | — |

Other cfg fields (SmolLM3-3B): `tie_word_embeddings: true`, `use_sliding_window: false`, `sliding_window: null`, `layer_types` all `full_attention` (36), `attention_bias/mlp_bias false`, `rms_norm_eps 1e-6`, vocab 128256. QK-norm: none.

#### Code path (official vLLM)
Served by the generic Transformers backend (no native `smollm3.py`; `ls models | grep -i smol` → only `smolvlm.py`).
1. Registry `models/registry.py:720` `"SmolLM3ForCausalLM": ("transformers", "TransformersForCausalLM")` → `transformers/__init__.py:129`.
2. Construction as Hunyuan step 2 (`base.py:121-199`); HF classes `HF-T:smollm3/modeling_smollm3.py:350` (model), `:174` (attention), `:287` (decoder layer).
3. torch.compile → **enabled**: `rope_type` is `default`, so `can_enable_torch_compile` returns True (`transformers/utils.py:359-369`) → Inductor (Triton) by default.
4. Embedding → `VocabParallelEmbedding` (`base.py:565-568`) → `F.embedding`.
5. RMSNorm `SmolLM3RMSNorm` (`HF-T:…smollm3.py:251`) → `RMSNormFuser` → vLLM `RMSNorm` (S5).
6. QKV → `QKVFuser` → `QKVParallelLinear`; `o_proj` → `RowParallelLinear` (as Hunyuan step 6).
7. RoPE → HF `SmolLM3RotaryEmbedding` (`HF-T:…smollm3.py:49-100`, default init, `@dynamic_rope_update` is a no-op for `default`, `modeling_rope_utils.py:120-127`); **NoPE**: `self.use_rope = config.no_rope_layers[layer_idx]` (`:200`) and RoPE applied only `if self.use_rope` (`:222-224`). Entirely HF torch code; vLLM is not involved, so NoPE needs no vLLM support.
8. Attention → `vllm_attention_forward` → vLLM `Attention` (as Hunyuan step 9). Sliding window would be wired from `layer_types` (`base.py:674-676`), but all layers are `full_attention`.
9. MLP → `GLUFuser` → `MergedColumnParallelLinear` + `SiluAndMul` + `RowParallelLinear` (`HF-T:…smollm3.py:271-284`; `transformers/fusers/glu.py:160-191`).
10. LM head tied (`transformers/causal.py:48-57`).

#### Requirements
| Op | Kernel impl | Requires | Min CC | K80 status | Evidence | Fallback (file:line) | Fallback on K80? |
|---|---|---|---|---|---|---|---|
| dtype | — | bf16 ≥ 8.0 | — | auto → fp32 | S2 | `--dtype half` | fp16 storage OK (UNVERIFIED perf) |
| Embedding / Linear | `F.embedding` / `F.linear` | torch | any | OK | S6 | — | — |
| RMSNorm | `_C.rms_norm` | `_C` | 7.0 build | missing | S5 | `ir/ops/layernorm.py:9-21` | yes |
| SiluAndMul | `_C.silu_and_mul` | `_C` | 7.0 build | missing | `activation.py:136-150` | `activation.py:141-144` | yes |
| RoPE / NoPE | HF torch code | torch | any | OK | `HF-T:…smollm3.py:49-100, 200, 222-224` | — | — |
| torch.compile | Inductor → Triton | Triton | not Kepler | enabled by default | `transformers/utils.py:359-369` | `-O0` / `--enforce-eager` | yes |
| Attention | FA / FI / Triton / Flex | see S3 | 8.0 / Triton | **blocker** | S3 | none on CUDA | no |
| Transformers backend itself | HF Transformers v5 + torch.fx fusers | transformers ≥ 5.16.1, torch 2.13 | — | not runnable on torch 2.0.1 stack | `requirements/common.txt:10`, `requirements/cuda.txt:7` | native model file (none exists) | — |

#### Quantized format support
| Format | vLLM method | Min capability | Evidence |
|---|---|---|---|
| (none officially released) | — | — | `hf models ls --author HuggingFaceTB --search SmolLM3`: only BF16, ONNX, checkpoints repos |

#### In the K80 fork (vllm37)
- Model file: **absent** — no `smollm3*.py` in `F:/vllm/model_executor/models/` (only `smolvlm.py`); generic backend file `F:/vllm/model_executor/models/transformers.py` exists.
- Registry: **no explicit entry** (grep `SmolLM3` in `F:/vllm/model_executor/models/registry.py`: none). Fallback resolution via `_try_resolve_transformers` (`F:/vllm/model_executor/models/registry.py:506-567`, uses `getattr(transformers, architecture)` and `is_backend_compatible()`), which needs an installed Transformers that has `SmolLM3ForCausalLM` (fork requires `transformers >= 4.55.0`, `F:/requirements/common.txt:10`; SmolLM3 configs were written by 4.53/4.54.dev). Whether such a Transformers version runs on torch 2.0.1: UNVERIFIED (no Transformers with `smollm3`/`hunyuan_v1_dense` is installed anywhere on this machine — `find / -type d -name smollm3` returned nothing).

#### Verdict for K80
Blockers:
1. Attention (S3) and `_C` build (S1) — shared runtime.
2. No native model in either tree; the official path needs Transformers v5 + torch 2.13; the fork's Transformers fallback needs Transformers ≥ 4.53 on torch 2.0.1 (UNVERIFIED compatibility). → Porting work: add a native `smollm3.py` to the fork — it is Llama with a per-layer "skip RoPE" flag (`no_rope_layers[i]==0` → don't call `rotary_emb`), tied embeddings, no QK-norm, no sliding window; roughly a 20-line subclass of the fork's `llama.py`. No new kernels.
3. Memory (S8): 3.075 B → 12.3 GB fp32 (does not fit one 12 GB die) / 6.15 GB fp16 (fits, ~5 GB left for KV on one die) → use `--dtype half` or TP=2 in fp32.
NoPE itself is not a blocker (it is a control-flow flag, no kernel).

## 9. Why quantized kernels fail on K80, and the software workaround

### 9.1 The exact reason: missing arithmetic units, not missing data support

A quantized checkpoint is bytes plus a rule for turning those bytes back into numbers. The K80 can store and read any of these formats. What fails is the **kernel**. Every quantized kernel in vLLM is built on an instruction that only newer GPUs execute, and each format's `get_min_capability()` is the oldest GPU that has that instruction.

Instructions read from official vLLM (`csrc/libtorch_stable/…`) and the fork:

| Format | What the kernel does | Instruction it is built on | Unit the K80 lacks |
|---|---|---|---|
| GPTQ (Exllama) | Unpacks int4 into half2 pairs and multiply-adds pairs | `__hfma2` (`quantization/gptq/qdq_4.cuh:66`) | Packed FP16 arithmetic (K80 can store FP16, not compute in it; `cuda_fp16.hpp:1626` guards these at `__CUDA_ARCH__ >= 530`) |
| AWQ | Three-input logic op to unpack; tensor-core tile load; tensor-core multiply | `lop3.b32` (`quantization/awq/dequantize.cuh:43`), `ldmatrix` (`quantization/awq/gemm_kernels.cu:189`), `mma.sync…f16` (`:224`) | LOP3 (Maxwell+), `ldmatrix`, tensor cores |
| Marlin (GPTQ / AWQ / compressed-tensors w4a16 / FP8 and FP4 weight-only / MoE) | Asynchronous global→shared copy overlapped with tensor-core multiply | `cp.async` (`quantization/marlin/marlin.cuh:98`), `ldmatrix` (`marlin_template.h:85`), `mma.sync` (`marlin_mma.h:25`; FP8 `:79`), `lop3` (`dequant.h:75`) | Tensor cores, async copy |
| Machete | Warpgroup-wide matrix multiply | CUTLASS `arch::Sm90` (`quantization/machete/machete_collective_builder.cuh:16`) | Hopper warpgroup tensor cores |
| W8A8 INT8 / FP8 (CUTLASS scaled_mm) | Multiplies weights **and** activations in 8-bit | CUTLASS kernels tagged `arch::Sm75` / `Sm89` / `Sm90` / `Sm100` (`quantization/w8a8/cutlass/scaled_mm_c2x_sm75_dispatch.cuh:26`) | INT8 and FP8 tensor cores; K80 has no 8-bit multiply at all |
| FP8 values (e4m3) | Hardware FP8 conversion and FP8 multiply | `__nv_fp8_e4m3`, `mma…e4m3` (`quantization/marlin/marlin_mma.h:76-79`) | FP8 conversion / math units |
| NVFP4 / MXFP4 | Hardware FP4 conversion and block-scaled FP4 multiply | `cvt.rn.satfinite.e2m1x2.f32` (`quantization/fp4/nvfp4_utils.cuh:79`); MoE GEMM "on SM100" (`quantization/fp4/mxfp4_blockwise_moe_kernel.cu:5`) | FP4 conversion, block-scaled FP4 tensor cores |
| GGUF (fork's MMVQ/MMQ) | Four int8 multiply-adds in one instruction | `__dp4a` (`vllm37:csrc/quantization/gguf/vecdotq.cuh:57`) | Integer dot-product unit (sm_61) |

The K80 has: FP32 fused multiply-add, 32-bit integer shift / AND / OR, byte permute, warp shuffle, 48 KB shared memory per block (112 KB and 128 K registers per SM, `docs/port/kepler-vs-maxwell.md:82-83`).

### 9.2 Two kinds of quantization

- **Weight-only** (GPTQ, AWQ, compressed-tensors w4a16, GGUF, FP8 weight-only). The math is `w = (q − zero) × scale` per group, followed by an ordinary multiply. Shift, AND, subtract and FMA all exist on the K80, so these need **only a new kernel**. FP8 weights can be decoded the same way with bit operations.
- **Activation-quantized** (W8A8 INT8/FP8, NVFP4 W4A4, MXFP4). The format exists to multiply in 8 or 4 bits. The K80 has no such multiplier, so the only option is to decode to FP32. That keeps the storage saving and drops the compute speedup, which reduces them to weight-only formats.

### 9.3 Design rule: convert to what the K80 is good at

When the hardware lacks a unit, rewrite the work as FP32 FMA and 32-bit integer operations:

- **Storage:** pack 8 × int4 into one `uint32` along K. One coalesced 32-bit load then delivers 8 weights. Per-group (e.g. 128) FP32 scale and zero.
- **Unpack in registers, never in memory:** `(q >> 4j) & 0xF`, then either `(float)nib` or the mantissa trick `__int_as_float(0x4B000000 | nib) − 2^23` (an integer OR plus an FP32 subtract), then FMA with the activation in FP32.
- **Group the scale:** accumulate `Σ (q − z) · x` per 8-weight word, then multiply by the group scale once.
- **Repack at load time:** convert GPTQ / AWQ / compressed-tensors / GGUF Q4 checkpoints into this layout once, the way Marlin repacks for its kernel. One kernel then serves every weight-only int4 format.

### 9.4 Measured on this host

Prototype: `docs/port/k80-int4-gemv-proto.cu` (one warp per output row, `x` in shared memory, group 128). Built with `nvcc -O3 -arch=sm_37`, run on GPU0, 200 iterations each, compared against `cublasSgemv` on the same weights dequantized to FP32. **[measured]**

| Shape (N × K) | cuBLAS FP32 GEMV | int4, mantissa-trick unpack | int4, int→float unpack | Max rel. error vs FP32 |
|---|---|---|---|---|
| 4096 × 4096 | 597.5 µs (112 GB/s) | 209.1 µs (**2.86×**) | 211.0 µs (2.83×) | 5.0e-4 |
| 11008 × 4096 | 1242.9 µs (145 GB/s) | 438.2 µs (2.84×) | 407.3 µs (**3.05×**) | 2.4e-3 |
| 4096 × 11008 | 1510.7 µs (119 GB/s) | 757.9 µs (**1.99×**) | 784.8 µs (1.92×) | 5.2e-4 |

What the numbers say:
- A first, unoptimized kernel is already **2–3× faster than FP32 cuBLAS** for single-token decode, using **8× less weight memory**. The error comes from FP32 summation order.
- It reads packed data at only 33–62 GB/s, against the die's ~240 GB/s, so it is limited by instruction issue, not memory. The ceiling for a 4-bit layout is up to ~8× over FP32. Tiling, more weights per load, and keeping `x` in registers are the next steps.
- The two unpack methods are within noise of each other. The unpack instruction is not the bottleneck yet.
- The K = 11008 case uses 44 KB of shared memory per block for `x`, which limits occupancy. That explains why it is the slowest case.

### 9.5 Consequence for the plan

- #73: load weight-only int4 checkpoints (GPTQ / AWQ / compressed-tensors / GGUF Q4) and repack them to the K80 layout. A dequantize-to-FP32 path is the correctness reference.
- #74: the fused K80 kernel (GEMV for decode, a tiled GEMM for prefill) built on 9.3, replacing the narrower "software dp4a" plan.
- The same rule extends to int8 (4 per word) and to FP8 / FP4 weights decoded by bit operations, which unlocks those checkpoints as weight-only formats.

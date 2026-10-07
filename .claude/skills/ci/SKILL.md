---
name: ci
description: Run K80-specific CI workflows for vLLM on the self-hosted runner. Use when triggering a remote build or smoke test, or deciding between local vs CI.
---

# CI

GitHub Actions workflows run on the self-hosted runner `c246-pro-vllm37` (labels `self-hosted`, `vllm`, `k80`), which has the 2× K80. All are manual (`workflow_dispatch`); nothing runs on push or pull requests.

## Workflows

| File | Purpose | Timeout |
|---|---|---|
| `k80-pipeline.yml` | `k80-build` then `k80-runtime` | — |
| `k80-build.yml` | Build `vllm37-local` with `make build-local`; `rebuild_builder` rebuilds the builder | 60 / 180 min |
| `k80-runtime.yml` | xformers kernel test on GPU0, start the container, assert the attention backend, `/v1/completions` against `docker/k80/ci/golden.json` | 45 min |
| `k80-context-stress.yml` | Sweep `--max-model-len` (2K–16K) on TP=1 | 120 min |
| `k80-cutlass-repro.yml` | CUTLASS Sm37 SGEMM vs cuBLAS (manual check of the patch xformers uses) | 20 min |
| `k80-host-info.yml` | Read-only `nvidia-smi` / `nvcc` / OS info | — |

The hardware workflows share the `k80-hardware` concurrency group. Shared shell lives in `docker/k80/ci/`: `ensure-builder.sh` (pulls `dogkeeper886/vllm37-builder:latest` when `vllm37-builder:latest` is missing), `ensure-runtime.sh` (rebuilds `vllm37-local` unless its revision label is the branch SHA), `wait-ready.sh`.

## Trigger

```bash
gh workflow run k80-pipeline.yml --ref <branch>
gh workflow run k80-pipeline.yml --ref <branch> -f attention_backend=TORCH_SDPA
gh workflow run k80-pipeline.yml --ref <branch> -f builder_image=vllm37-builder:next   # test a new builder
gh workflow run k80-context-stress.yml --ref <branch> -f context_lengths=2048,8192
```

Defaults: model `Qwen/Qwen3-0.6B`, `tp_size` 1, backend `XFORMERS`, builder `vllm37-builder:latest`.

## Safety
- Keep `tp_size` at 1 or 2. TP=4 across both K80 boards can overload the shared CPU power cable and has halted the machine.
- Never change GPU power limits (`nvidia-smi -pl`); it has caused system halts.
- `NCCL_P2P_DISABLE=1` is set in the runtime image; K80 is PCIe-only.
- The ollama37 runner on this machine shares the GPUs and is outside the `k80-hardware` group.

## Golden output
`docker/k80/ci/golden.json` applies when `model` and `tp_size` match. After an intended output change, run the pipeline with defaults and copy `text`, `completion_tokens` and `finish_reason` from the runtime log.

## Local vs CI
- On the K80 machine: `tools/k80-host/serve.sh` (host build) or `cd docker/k80 && make build-local && make run`.
- Validating a branch in the clean container path: trigger `k80-pipeline.yml`.

## Related
- `/k80-build` — build locally
- `docker/k80/README.md` — runtime configuration

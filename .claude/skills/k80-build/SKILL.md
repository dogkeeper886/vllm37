---
name: k80-build
description: Build vLLM for Tesla K80 (sm_37) using the Docker builder/runtime pattern. Use when you need to produce a `vllm37-local` image, or are considering a host build (don't).
argument-hint: [local|github|builder]
---

# K80 Build

Build vLLM for Tesla K80 via Docker. **Never build on the host** — CUDA 11.4 +
GCC 10 + PyTorch 2.0.1 + sm_37 combination is pinned inside the builder image.

## When to use
- Producing a runtime image to test code changes
- Rebuilding after touching `csrc/`, `setup.py`, `docker/k80/requirements.txt` or `docker/k80/constraints.txt`
- Building on a clean machine (pull `dogkeeper886/vllm37-builder` first)

## Environment
- CUDA 11.4.4, GCC 10.5, CMake 4, Python 3.10, PyTorch 2.0.1
- Target: Tesla K80 (compute capability 3.7)
- Host toolchain: ignored — everything runs in Docker

## Build types
- **builder** — `make build-builder` (CUDA + GCC + PyTorch 2.0.1 base, ~120 min first time)
- **local** — `make build-local` (compile current checkout against the builder, ~10 min)

## Commands
```bash
cd docker/k80

# One-time: get the builder (prefer pull over rebuild)
docker pull dogkeeper886/vllm37-builder:latest
docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest
# OR: make build-builder  (~120 min)

# Iterate:
make build-local          # ~10 min incremental
make build-local JOBS=8   # override parallelism via .env
```

## Key build flags (already set in Dockerfiles)
- `TORCH_CUDA_ARCH_LIST="3.7"` — K80 compute capability
- `VLLM_BUILD_LEGACY_CUDA=1` — skip CUTLASS kernels (require sm_70+)
- `--no-deps` on `pip install .` — prevents PyTorch 2.7 from overwriting 2.0.1

## Outputs
- `vllm37-builder:latest` — base image with toolchain
- `vllm37-local:latest` — runtime image (from local source)

## Related
- `/ci` — trigger the build remotely on the self-hosted runner
- `docker/k80/README.md` — architecture and limitations
- `docker/k80/Makefile` — all build targets

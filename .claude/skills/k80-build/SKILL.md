---
name: k80-build
description: Build vLLM for Tesla K80 (sm_37) on the host or as the Docker builder/runtime images. Use when you need a host build to iterate on, or a `vllm37-local` image.
argument-hint: [host|local|builder]
---

# K80 Build

One staged script, `tools/k80-host/setup.sh`, builds the same stack everywhere:

| Stage | Builds | Run by |
|---|---|---|
| `toolchain` | Python 3.10.16, venv, cuDNN 8.7 for CUDA 11, PyTorch v2.4.1 sm_37 wheel | host, builder image |
| `deps` | `docker/k80/requirements.txt` with `docker/k80/constraints.txt`; removes Triton | host, runtime image |
| `xformers` | patched xformers v0.0.23 (`0.0.23+k80`) | host, runtime image |
| `vllm` | this checkout (`VLLM_BUILD_LEGACY_CUDA=1`) | host (editable), runtime image |

## Build types

- **host** — `tools/k80-host/setup.sh` (first run ~1.5–2 h; then `setup.sh vllm` after source changes). Needs driver 470, CUDA 11.4, GCC 10 and the Python build headers; see `tools/k80-host/README.md`. Serve with `tools/k80-host/serve.sh`.
- **local** — `cd docker/k80 && make build-local` builds `vllm37-local` on the builder. A source-only change rebuilds just the `vllm` stage (~4 min with `JOBS=8`).
- **builder** — `make build-builder` (~120 min). Or pull: `docker pull dogkeeper886/vllm37-builder:latest && docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest`.

`make build-local JOBS=8` sets parallelism on the command line; `JOBS` in `docker/k80/.env` sets the default.

## When to rebuild what
- `csrc/`, `setup.py`, `vllm/` → `setup.sh vllm` (host) or `make build-local`
- `docker/k80/requirements.txt` or `constraints.txt` → `setup.sh deps vllm` or `make build-local`
- `tools/k80-host/setup.sh` toolchain stage, `docker/k80/builder/Dockerfile` → builder

## Outputs
- `vllm37-builder:latest` — builder image (venv at `/opt/venv`)
- `vllm37-local:latest` — runtime image, labelled with the source commit
- Host: `~/opt/k80` (Python, cuDNN, wheels), `~/.venvs/vllm37`

## Related
- `/ci` — build and smoke-test on the self-hosted runner
- `docker/k80/Makefile` — all build targets

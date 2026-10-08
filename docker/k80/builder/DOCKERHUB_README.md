# vllm37-builder

Build environment for [vllm37](https://github.com/dogkeeper886/vllm37), a vLLM fork for the **Tesla K80** (compute capability 3.7). CUDA 12 and current PyTorch wheels dropped Kepler, so this image carries a toolchain that still targets `sm_37`.

## What's inside

| Component | Version | Note |
|---|---|---|
| Base OS | Rocky Linux 8 | |
| CUDA Toolkit | 11.4.4 | Highest CUDA that driver 470 (the last Kepler driver) runs |
| GCC | 10.5.0 | Host compiler for CUDA 11.4 and PyTorch |
| CMake | 4.0.1 | |
| Python | 3.10.16 | virtualenv at `/opt/venv`, first on `PATH` |
| cuDNN | 8.7.0 for CUDA 11 | Its support matrix lists SM 3.5 and later; in `/opt/k80/cudnn` |
| PyTorch | 2.4.1, built from source | `TORCH_CUDA_ARCH_LIST="3.7"`; reports `2.4.1` |
| NumPy | 1.26.4 | |

Python, cuDNN and PyTorch come from the `toolchain` stage of [`tools/k80-host/setup.sh`](https://github.com/dogkeeper886/vllm37/blob/main/tools/k80-host/setup.sh), the same script the host build uses.

## Use

The runtime image builds on this one. From a vllm37 checkout:

```bash
docker pull dogkeeper886/vllm37-builder:latest
docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest
cd docker/k80
make build-local      # vllm37 runtime image on top of this builder
make run              # serves on port 8000
```

Install packages into this image only with the fork's pins (`pip install -c docker/k80/constraints.txt ...`). A plain `pip install` of upstream vLLM replaces the `sm_37` PyTorch with a CUDA 12 build that cannot run on a K80.

## Rebuild

```bash
make -C docker/k80 build-builder JOBS=8    # about 2.5 h without cache
```

`JOBS` sets parallel compile jobs (about 2 GB of RAM each). Versions are set in `docker/k80/builder/Dockerfile` (GCC, CMake) and `tools/k80-host/setup.sh` (Python, cuDNN, PyTorch).

## Tags

- `latest` — current builder
- `YYYY-MM-DD` — builders kept by date

## Related

- [dogkeeper886/ollama37](https://hub.docker.com/r/dogkeeper886/ollama37) — Ollama built for the K80

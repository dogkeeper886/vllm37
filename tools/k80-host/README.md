# K80 host build

Builds and serves vllm37 on the host filesystem, for fast iteration on a machine that has the K80 driver and toolchain installed. CI keeps using the `docker/k80` images.

| | Host build | Docker build |
|---|---|---|
| Use for | Editing and testing on the K80 machine | CI, clean and portable builds |
| Python edits | Live (editable install) | Rebuild the image |
| Build steps | `setup.sh` | `setup.sh` stages, run in the builder and runtime images |

## Requirements

- NVIDIA driver 470, CUDA 11.4 at `/usr/local/cuda-11.4`, GCC 10 at `/usr/local/bin`, CMake ≥ 3.26, git
- Python build headers (Rocky 8): `openssl-devel libffi-devel bzip2-devel xz-devel zlib-devel readline-devel sqlite-devel tk-devel gdbm-devel ncurses-devel lz4-devel`

## Build

```bash
tools/k80-host/setup.sh          # all stages; first run about 1.5-2 h (PyTorch)
tools/k80-host/setup.sh vllm     # after a source change: rebuild only vLLM
```

Stages: `toolchain` (Python 3.10.16, venv, cuDNN 8.7, PyTorch 2.4.1), `deps`, `xformers`, `vllm`. The Docker builder runs `toolchain`; the runtime image runs the other three. Puts Python, cuDNN, sources and wheels in `~/opt/k80` (`K80_PREFIX`) and the virtualenv in `~/.venvs/vllm37` (`K80_VENV`). Every pip install uses `docker/k80/constraints.txt`.

## Serve

```bash
CUDA_VISIBLE_DEVICES=0 tools/k80-host/serve.sh                     # TinyLlama, TP=1, port 8000
MODEL=Qwen/Qwen3-0.6B MAX_MODEL_LEN=8192 tools/k80-host/serve.sh
```

Same engine settings as `docker/k80/docker-compose.yml`. Keep `TP_SIZE` at 1 or 2: TP=4 across both K80 boards can overload the shared CPU power cable.

## Work in the shell

```bash
source tools/k80-host/env.sh     # CUDA, cuDNN and venv on PATH
```

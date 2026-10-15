# K80 host build

Builds and serves vllm37 on the host filesystem, for fast iteration on a machine that has the K80 driver and toolchain installed. CI tests the `docker/k80` image by default, and this build when the repository variable `TEST_SERVER` is `local`.

| | Host build | Docker build |
|---|---|---|
| Use for | Editing and testing on the K80 machine | CI, clean and portable builds |
| Python edits | Live (editable install) | Rebuild the image |
| Build steps | `setup.sh` | `setup.sh` stages, run in the builder and runtime images |

## Requirements

- NVIDIA driver 470, CUDA 11.4 at `/usr/local/cuda-11.4`, GCC 10 at `/usr/local/bin`, CMake ≥ 3.26, git, wget
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

Same engine settings as `docker/k80/docker-compose.yml`.

`VLLM_K80_CUDA_GRAPHS=1` captures decode in CUDA graphs instead of passing `--enforce-eager` (#93). The outputs are the same, and decode is faster where kernel launches dominate: Qwen3-0.6B on two dies +32–42 %, Qwen3-8B int4 on four dies +11–20 %, one die of the board on the chipset link (GPU2) about +60 % for Qwen3-0.6B, one die of the CPU-attached board +1–6 %. The graphs take about 1–2 GiB per die and 15–140 s at startup, so leave room with a lower `GPU_MEM_UTIL` (0.78 for 8B int4).

## Clocks

```bash
tools/k80-host/clock_sample.py --dies 0,1 --csv trace.csv -- <command>
```

Records each die's SM clock, utilization, power, temperature and throttle reasons while the command runs, and the driver's own log of clock changes (NVML, read only; it never sets a clock or power limit). From Python, `with Sampler(dies=[0]) as s: ...` then `s.summary()`. A timing on these cards depends on the clock: under load a die sits at its 149 W cap at 600–800 MHz, not 875, and a die that reaches 88 °C is slowed by the hardware.

## Run as a service

```bash
tools/k80-host/install-service.sh        # writes ~/.config/systemd/user/vllm37.service
systemctl --user start vllm37
tail -f ~/.local/state/vllm37/serve.log
systemctl --user stop vllm37
```

The service runs `serve.sh` with the settings in `~/.config/vllm37/serve.env` (template: `serve.env.example`). It is installed disabled, so it runs only when started. Each start moves the previous log to `serve.log.1`. CI stops it for a test and starts it again afterwards.

## Work in the shell

```bash
source tools/k80-host/env.sh     # CUDA, cuDNN and venv on PATH
```

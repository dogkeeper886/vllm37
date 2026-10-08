# vllm37 — vLLM for Tesla K80

vLLM serves large language models fast on NVIDIA GPUs. Current vLLM needs a GPU of compute capability 7.0 or newer, CUDA 12 and Triton. The Tesla K80 has compute capability 3.7, driver 470 and CUDA 11.4 at most, so every current vLLM release fails on it. vllm37 rebuilds a mid-2025 vLLM on a CUDA 11.4 stack compiled for `sm_37` and replaces the kernels the K80 cannot run, so a K80 serves models through vLLM's OpenAI-compatible API.

> **Status:** experimental, tested only on 2× Tesla K80. For modern GPUs use [vllm-project/vllm](https://github.com/vllm-project/vllm).

## How vllm37 runs on a K80

vllm37 swaps each part of vLLM that needs a newer GPU for one the K80 can execute:

| Upstream vLLM needs | vllm37 uses |
|---|---|
| CUDA 12, compute capability ≥ 7.0 | CUDA 11.4, kernels compiled for `sm_37` |
| PyTorch 2.x wheels (no Kepler build) | PyTorch 2.4.1 built from source for `sm_37`, cuDNN 8.7 for CUDA 11 |
| FlashAttention, FlashInfer or Triton attention | xformers v0.0.23 `cutlassF` kernels, patched for `sm_37` |
| V1 engine, `torch.compile`, CUDA graphs | V0 engine, eager mode (`--enforce-eager`) |
| BF16 / FP16 math | FP32 (`--dtype float32`) |

The fork builds on vLLM ~v0.10 (August 2025), Python 3.10 and GCC 10. [`docs/port/model-family-requirements.md`](docs/port/model-family-requirements.md) explains, with file and line evidence, why each upstream path fails on `sm_37`.

## Quick start

On a machine with the K80, driver 470 and Docker with the NVIDIA runtime:

```bash
cd docker/k80
docker pull dogkeeper886/vllm37-builder:latest
docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest
make build-local          # runtime image from this checkout
make run && make logs     # serves on port 8000, TP=1
```

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello", "max_tokens": 32}'
```

`docker/k80/.env` sets the model, context length and tensor parallelism (template: `.env.example`). Keep tensor parallelism at 1 or 2; see [Hardware safety](#hardware-safety).

## What runs today

Measured on 2× Tesla K80 (4 GPU dies, 11.4 GiB each):

| Model | Setup | Result |
|---|---|---|
| `Qwen/Qwen3-0.6B` | TP=1, XFormers | CI golden output; context up to 16,384 tokens (11,252-token prompt: 22 s prefill) |
| `Qwen/Qwen3-0.6B` | TP=1, Torch SDPA | Same output as XFormers |
| `Qwen/Qwen3-0.6B` | TP=1, `--model-impl transformers` | Same output, through transformers' own model code |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | TP=1 | Previous CI golden output |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | TP=2 | Worked in earlier testing; not re-run since the 2026-10 build changes |

FP32 weights take 4 bytes per parameter, so one die holds models up to about 2B parameters and one board (TP=2) up to about 4B. [`idea.md`](idea.md) ranks candidate models by size and fit.

### Open problems

Tracked in [#78](https://github.com/dogkeeper886/vllm37/issues/78):

- Quantized checkpoints (GPTQ, AWQ, FP8, GGUF and others): their kernels use instructions the K80 lacks. A K80 int4 kernel is planned in #73 and #74.
- MoE models: `vllm._moe_C` fails to load (D11).
- Guided decoding returns unconstrained text (D12).
- Models that mix sliding-window and full attention layers are capped at the window size (Gemma 3 270m/1b: 512 tokens), and attention supports head dimensions up to 256.

## Build

One script, [`tools/k80-host/setup.sh`](tools/k80-host/setup.sh), builds the stack in four stages: `toolchain` (Python, cuDNN, PyTorch), `deps`, `xformers`, `vllm`. Three builds run it:

| Build | Command | Use for |
|---|---|---|
| Host | `tools/k80-host/setup.sh`, then `tools/k80-host/serve.sh` | Editing on the K80 machine; Python edits apply without a rebuild |
| Builder image | `make -C docker/k80 build-builder` (~2 h) | The `toolchain` stage, once per toolchain change |
| Runtime image | `make -C docker/k80 build-local` | CI and clean builds; a source-only change rebuilds in ~4 min |

The host build needs driver 470, CUDA 11.4 at `/usr/local/cuda-11.4`, GCC 10 and the Python build headers listed in [`tools/k80-host/README.md`](tools/k80-host/README.md). [`docker/k80/constraints.txt`](docker/k80/constraints.txt) pins every install. [`docker/k80/README.md`](docker/k80/README.md) covers image details and settings.

## CI

GitHub Actions workflows run on a self-hosted runner attached to the K80s. You start each one by hand:

```bash
gh workflow run k80-pipeline.yml --ref <branch>                          # build, then smoke test
gh workflow run k80-pipeline.yml --ref <branch> -f attention_backend=TORCH_SDPA
gh workflow run k80-context-stress.yml --ref <branch>                    # context sweep, 2K-16K
```

The smoke test checks the selected attention backend and compares `/v1/completions` with [`docker/k80/ci/golden.json`](docker/k80/ci/golden.json).

## Hardware safety

- **Keep tensor parallelism at 1 or 2.** TP=4 runs both K80 boards from one CPU power cable and has halted this machine.
- **Leave GPU power limits at their defaults.** Changing them (`nvidia-smi -pl`) has halted this machine.
- **Keep `NCCL_P2P_DISABLE=1`** for TP > 1. The K80 is PCIe-only; the runtime image and `serve.sh` set it.

## Project docs

| Document | Contents |
|---|---|
| [`docs/port/model-family-requirements.md`](docs/port/model-family-requirements.md) | 14 model families: release formats, code paths, requirements on sm_37; why quantized kernels fail and the int4 workaround |
| [`idea.md`](idea.md) | Model targets ranked by fit on K80 |
| [`docs/port/`](docs/port/) | Port studies: CUTLASS, FlashAttention, CUDA 11.4 pins, Kepler vs Maxwell, prior art |
| [#78](https://github.com/dogkeeper886/vllm37/issues/78) | Cleanup tracking and open runtime problems |

## Contributing

This fork takes changes that keep the K80 working. General vLLM changes go to [vllm-project/vllm](https://github.com/vllm-project/vllm).

## Upstream and license

vllm37 is an independent fork of [vllm-project/vllm](https://github.com/vllm-project/vllm), unaffiliated with the vLLM project and the PyTorch Foundation. If you use vLLM in research, cite the upstream paper: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).

Apache 2.0, inherited from upstream vLLM. See [`LICENSE`](LICENSE).

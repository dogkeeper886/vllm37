# K80 Docker images

The K80 stack builds as two images. The **builder** (`builder/Dockerfile`) holds Rocky Linux 8, CUDA 11.4, GCC 10.5 and CMake 4, plus the `toolchain` stage of [`tools/k80-host/setup.sh`](../../tools/k80-host/setup.sh): Python 3.10.16 in `/opt/venv`, cuDNN 8.7 for CUDA 11 and PyTorch 2.0.1 for `sm_37`. The **runtime** image (`runtime/Dockerfile.local`) adds the `deps`, `xformers` and `vllm` stages on top and serves vLLM's OpenAI-compatible API on port 8000. The host build runs the same script, so both get the same stack.

## Build and run

```bash
cd docker/k80

# Builder: pull the published one, or build it (~2 h)
docker pull dogkeeper886/vllm37-builder:latest
docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest
# make build-builder

make build-local     # runtime image from this checkout
make run             # docker compose up -d
make logs
make stop
```

`make build-local` orders the layers from least to most often changed (`deps` → `xformers` → `vllm`), so a source-only change rebuilds just the `vllm` stage (~4 min with `JOBS=8`). The image is labelled with the source commit (`org.opencontainers.image.revision`); CI rebuilds when the label differs from the branch.

## Settings

`docker-compose.yml` reads these from `.env` (copy `.env.example`) or the shell:

| Variable | Default | Meaning |
|---|---|---|
| `JOBS` | 4 | Parallel compile jobs for `make build-*` |
| `MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Model to serve |
| `TP_SIZE` | 1 | Tensor parallel size: 1 or 2 (see Safety) |
| `DTYPE` | `float32` | Weight and compute dtype |
| `MAX_MODEL_LEN` | 2048 | Context length |
| `GPU_MEM_UTIL` | 0.85 | Fraction of each die's memory vLLM may use |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | `XFORMERS` or `TORCH_SDPA` |
| `VLLM_K80_TRACE` | 0 | 1 = TP-init trace logging (below) |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache mounted into the container |
| `VLLM_LOG_DIR` | `/tmp/vllm-logs` | Host directory for vLLM and NCCL logs |

Engine settings that do not change per run (`VLLM_USE_V1=0`, `TORCHDYNAMO_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`) live in `runtime/Dockerfile.local`. Python packages come from `requirements.txt`, pinned by `constraints.txt`; both are shared with the host build.

## Safety

- **Keep `TP_SIZE` at 1 or 2.** TP=4 runs both K80 boards from one CPU power cable and has halted this machine; TP>1 has also hung the system before.
- **Leave GPU power limits at their defaults.** Changing them (`nvidia-smi -pl`) has halted this machine.
- **Keep `NCCL_P2P_DISABLE=1`.** The K80 is PCIe-only; P2P causes kernel timeouts with TP>1.

## Diagnosing a hang

Phase-1 instrumentation (issue #10) adds opt-in trace logging at the TP-init
stages most likely to hang, plus crash-safe stdio so final log lines actually
reach disk when the kernel wedges.

### Enabling trace logs

Set `VLLM_K80_TRACE=1` before `make run` (or via the `k80-runtime` workflow
input; default is on in CI):

```bash
VLLM_K80_TRACE=1 make run
```

Each vLLM log record is tagged `rank=N local_rank=N` while tracing is on. Trace
lines are prefixed `[k80-trace]` — grep the log bundle for those to reconstruct
the startup timeline.

### What gets logged, and where the evidence lands

| Log | Content | Where |
|---|---|---|
| `vllm.log` | Everything vLLM wrote to stdout (app logger, trace lines, Python exceptions) | `docker compose logs` → CI artifact |
| `nccl-*.log` | NCCL library's own debug output (one file per process via `%h-%p`) | Bind-mounted `/var/log/vllm/` → `$VLLM_LOG_DIR` → CI artifact |
| host `dmesg`, `journalctl` | Kernel / driver wedges (Xid errors, PCIe errors) | Host-side capture — **not yet wired into CI**, follow-up work |

The vLLM and NCCL logs are separate on purpose: if the app process hangs or
dies, the NCCL file is still intact because NCCL writes it directly via its C
runtime and never touches Python's stdio layer.

### What to look for

1. **Find the last trace line before the hang.** Each line tells you which
   rank reached which stage. Stages (in order): `spawning workers` →
   `worker process entered` → `init_device cuda ready` →
   `init_worker_distributed_environment begin` → `init_process_group begin` →
   `new_group begin` (per TP/PP group) → `load_model` →
   `determine_num_available_blocks` → `initialize_cache` → `warmup` →
   `post-warmup sync`.

2. **Compare `elapsed_ms` / `elapsed_s` fields** against expected. The
   numbers below are **initial estimates** derived from the flow study, not
   measured baselines — replace them with observed CI medians once a TP=1
   run establishes ground truth:
   - `init_process_group` should complete in < 2 s. > 5 s = NCCL probe stall.
   - `new_group` with NCCL backend: < 500 ms. Larger = PCIe / P2P issue.
   - `load_model`: depends on model size and disk; typically < 30 s for 3 B FP32.
   - `warmup`: < 10 s.

3. **Watch `free_mib` on each rank.** Drop below a few hundred MB before the
   warmup forward is a red flag — K80 has ~11.5 GB per die, a 3 B FP32 TP=2
   model leaves ~2-3 GB margin at peak.

4. **Cross-reference NCCL log timestamps** with vLLM trace timestamps. NCCL
   buffers some messages; the last NCCL line often tells you which collective
   was in flight when the app wedged.

### Phase-1 validation policy

TP=1 is the CI default. Before running TP=2, make sure host-side `dmesg` /
`nvidia-smi dmon` capture is running so a hang can be diagnosed afterwards; CI
does not capture these yet. TP=4 stays off (see Safety below).

## Files

```
docker/k80/
├── builder/Dockerfile        # OS, CUDA 11.4, GCC 10.5, CMake 4 + setup.sh toolchain
├── runtime/Dockerfile.local  # setup.sh deps, xformers, vllm; engine env; entrypoint
├── requirements.txt          # Python packages (common.txt + Ray)
├── constraints.txt           # torch==2.0.1, numpy<2, opencv<4.12
├── docker-compose.yml        # GPUs, ports, cache and log mounts, per-run settings
├── Makefile                  # build-builder, build-local, run, stop, logs, verify-*-patches
├── .env.example              # settings template
├── ci/                       # golden.json and scripts used by the k80-* workflows
├── xformers-build/           # xformers build script and GPU kernel test
├── xformers-patches/         # sm_37 patches for xformers v0.0.23
├── cutlass-patches/          # Sm37 arch trait for CUTLASS
└── cutlass-repro/            # CUTLASS Sm37 SGEMM vs cuBLAS check
```

## CI

The `k80-*` workflows in `.github/workflows/` build and test these images on the self-hosted runner; see the CI section of the [root README](../../README.md#ci). `k80-pipeline.yml` calls `k80-build.yml` and `k80-runtime.yml`, so their runs appear under "K80 Full Pipeline".

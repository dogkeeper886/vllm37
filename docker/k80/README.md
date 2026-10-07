# vLLM for Tesla K80 (sm_37) + CUDA 11.4

Two-stage Docker build system for running vLLM on Tesla K80 GPUs
(compute capability 3.7). Follows the `dogkeeper886/ollama37` build pattern:
Rocky Linux 8, CUDA 11.4, GCC 10, CMake 4.

## Hardware Target

- 2x Tesla K80 cards (4 GPU dies, 12GB each, 48GB total)
- Tensor parallelism across all 4 dies
- Models up to ~4B parameters in FP32

## Quick Start

```bash
cd docker/k80/

# 1. Build builder image (~120 min first time)
make build-builder

# 2. Build the runtime image from this checkout
make build-local

# 3. Run
make run
make logs

# 4. Test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello", "max_tokens": 50}'
```

## Configuration

Optional — the Makefile and compose file both have sensible built-in defaults.
To override, copy the template and edit:

```bash
cp .env.example .env
```

`.env` values apply to both `make build-*` and `make run`. Available settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `JOBS` | 4 | Parallel build jobs |
| `MODEL` | TinyLlama-1.1B | Model to serve |
| `TP_SIZE` | 1 | Tensor parallel size (see safety note below before raising) |
| `DTYPE` | float32 | Only float32 on K80 |

### TP safety

`TP_SIZE` defaults to `1` because TP>1 on 2x K80 has triggered full system hangs
on this hardware — even inside a KVM guest with vLLM inside a container. Before
raising it, enhance host-side logging (`dmesg`, `journalctl`, `nvidia-smi dmon`)
so a hang is diagnosable post-mortem. TP=4 is additionally power-risky on a
shared CPU EPS connector and is currently unvalidated. See the project
`CLAUDE.md` for the authoritative safety rules.

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

TP=1 is the current CI default and the only configuration validated to work on
this hardware. **Do not attempt TP >= 2** until (a) Phase-1 has demonstrated
that trace lines and NCCL logs actually land in the artifact on TP=1, and (b)
host-side `dmesg` / `nvidia-smi dmon` capture is wired into the workflow. See
issue #10 for status.

## Architecture

```
docker/k80/
├── builder/
│   └── Dockerfile          # CUDA 11.4 + GCC 10 + CMake 4 + Python 3.10 + PyTorch 2.0.1
├── runtime/
│   ├── Dockerfile          # Clone from GitHub, build vLLM
│   └── Dockerfile.local    # Copy local source, build vLLM
├── Makefile                # Build orchestration
├── docker-compose.yml      # Runtime with GPU access
├── .env.example            # Configuration template (copy to .env)
└── README.md               # This file
```

## CI

K80 workflows live under `.github/workflows/k80-*.yml`, run on the self-hosted
runner, and are manually triggered (`workflow_dispatch`).

`k80-pipeline.yml` composes `k80-build.yml` + `k80-runtime.yml` via
`workflow_call`. So `gh run list --workflow="K80 Docker Build"` showing zero
direct runs does NOT mean the workflow is unused — pipeline invocations surface
under the *caller* (`K80 Full Pipeline`).

Other workflows (`k80-host-info`, `k80-cutlass-repro`, `k80-xformers-build`,
`k80-context-stress`) are standalone and exercise specific stories — see each
file's header comment for purpose and related issue numbers.

## Key Build Flags

- `TORCH_CUDA_ARCH_LIST="3.7"` - Target K80 compute capability
- `VLLM_BUILD_LEGACY_CUDA=1` - Skip CUTLASS-dependent kernels (require sm_70+)
- `VLLM_USE_V1=0` - Use V0 engine (V1 requires PyTorch 2.5+)
- `VLLM_ATTENTION_BACKEND=TORCH_SDPA` - Use PyTorch SDPA (no xformers/flash-attn)

## Limitations

- FP32 only (K80 lacks native FP16 compute, no BF16)
- No CUDA graphs (requires sm_70+)
- No flash attention or xformers
- No CUTLASS quantization kernels
- Eager mode only (no torch.compile)

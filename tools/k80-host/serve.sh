#!/bin/bash
# Serve a model from the host build with the same settings as docker/k80/docker-compose.yml.
# Extra arguments go to the vLLM server, e.g. serve.sh --model-impl transformers
# Env: PROFILE, a model profile in docker/k80/models/ (a name or a path): its
#      flags go first, and MODEL, DTYPE, TP_SIZE and MAX_MODEL_LEN override its
#      own; CUDA_VISIBLE_DEVICES must name tensor x pipeline parallel dies.
#      MODEL, TP_SIZE (default 1), MAX_MODEL_LEN,
#      GPU_MEM_UTIL, DTYPE, PORT, CUDA_VISIBLE_DEVICES,
#      VLLM_ATTENTION_BACKEND (XFORMERS default, or TORCH_SDPA),
#      VLLM_K80_CUDA_GRAPHS (1 = CUDA graphs instead of --enforce-eager, #93;
#      leave room for the graphs with a lower GPU_MEM_UTIL, e.g. 0.78).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS} TORCHDYNAMO_DISABLE=1 \
       NCCL_P2P_LEVEL=PIX VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
profile=()
if [ -n "${PROFILE:-}" ]; then
  lines=$("$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/docker/k80/ci/profile.sh" "$PROFILE")
  while read -r kind value; do
    case "$kind" in
      MODEL) MODEL=${MODEL:-$value} ;;
      DTYPE) DTYPE=${DTYPE:-$value} ;;
      TP_SIZE) TP_SIZE=${TP_SIZE:-$value} ;;
      MAX_MODEL_LEN) MAX_MODEL_LEN=${MAX_MODEL_LEN:-$value} ;;
      PP_SIZE) profile+=(--pipeline-parallel-size "$value") ;;
      ARG) profile+=("$value") ;;
    esac
  done <<< "$lines"
fi
eager=(--enforce-eager)
[ "${VLLM_K80_CUDA_GRAPHS:-0}" = 1 ] && eager=()
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
  --dtype "${DTYPE:-float32}" \
  "${eager[@]}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-2048}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --swap-space 0 \
  --port "${PORT:-8000}" \
  "${profile[@]}" \
  "$@"

#!/bin/bash
# Usage: serve.sh [FLAG...]
#   FLAG  flags of vllm.entrypoints.openai.api_server, passed last
# Env:
#   PROFILE                 a profile name in docker/k80/models/ or a path
#                           (default none)
#   MODEL                   a Hugging Face model id or a model directory
#                           (default the profile's, else
#                           TinyLlama/TinyLlama-1.1B-Chat-v1.0)
#   DTYPE                   float32, half or auto (default the profile's, else
#                           float32)
#   TP_SIZE                 a die count that divides the model's attention
#                           heads (default the profile's, else 1)
#   MAX_MODEL_LEN           tokens, up to the model's own limit (default the
#                           profile's, else 2048)
#   CUDA_VISIBLE_DEVICES    comma-separated die indexes, TP_SIZE times the
#                           profile's pipeline parallel size of them
#                           (default all)
#   GPU_MEM_UTIL            a fraction above 0, at most 1 (default 0.85)
#   PORT                    a TCP port (default 8000)
#   VLLM_ATTENTION_BACKEND  XFORMERS or TORCH_SDPA (default XFORMERS)
#   VLLM_K80_CUDA_GRAPHS    1 for CUDA graphs, anything else for
#                           --enforce-eager (default 0)
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

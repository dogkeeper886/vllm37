#!/bin/bash
# Serve a model from the host build with the same settings as docker/k80/docker-compose.yml.
# Env: MODEL, TP_SIZE (default 1; TP=4 is power-risky on 2x K80), MAX_MODEL_LEN,
#      GPU_MEM_UTIL, DTYPE, PORT, CUDA_VISIBLE_DEVICES,
#      VLLM_ATTENTION_BACKEND (XFORMERS default, or TORCH_SDPA).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS} TORCHDYNAMO_DISABLE=1 \
       NCCL_P2P_DISABLE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
  --dtype "${DTYPE:-float32}" \
  --enforce-eager \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-2048}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --swap-space 0 \
  --port "${PORT:-8000}"

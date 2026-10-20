#!/bin/bash
# Usage: container-up.sh [TIMEOUT]
#   TIMEOUT       seconds, a whole number (default 600)
# Env:
#   COMPOSE_FILE  compose files in docker/k80, colon-separated
#                 (default docker-compose.yml)
#   VLLM_LOG_DIR  a directory (default VLLM_LOG_DIR in .env, else
#                 /tmp/vllm-logs)
#   HF_HOME       a directory (default HF_HOME in .env, else
#                 ~/.cache/huggingface)
#   HOST_UID      a user id (default this user's)
#   HOST_GID      a group id (default this user's group)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TIMEOUT=${1:-600}

LOG_DIR=${VLLM_LOG_DIR:-$(sed -n 's/^VLLM_LOG_DIR=//p' .env 2>/dev/null | tail -1)}
LOG_DIR=${LOG_DIR:-/tmp/vllm-logs}
mkdir -p "$LOG_DIR"
chmod 777 "$LOG_DIR"

export HOST_UID=${HOST_UID:-$(id -u)} HOST_GID=${HOST_GID:-$(id -g)}
HF_DIR=${HF_HOME:-$(sed -n 's/^HF_HOME=//p' .env 2>/dev/null | tail -1)}
HF_DIR=${HF_DIR:-$HOME/.cache/huggingface}
mkdir -p "${HF_DIR/#\~/$HOME}"

if ! docker compose up -d; then
  echo "Container did not start."
  exit 1
fi

start=$(date +%s)
while :; do
  elapsed=$(( $(date +%s) - start ))
  state=$(docker inspect --format='{{.State.Status}}' vllm37 2>/dev/null || echo missing)
  if [ "$state" != running ] && [ "$elapsed" -ge 10 ]; then
    docker compose logs --no-color --tail 200
    echo "Container state=$state after ${elapsed}s."
    exit 1
  fi
  if curl -fsS --max-time 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "Server ready after ${elapsed}s."
    exit 0
  fi
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    docker compose logs --no-color --tail 200
    echo "Server not ready after ${TIMEOUT}s."
    exit 1
  fi
  sleep 5
done

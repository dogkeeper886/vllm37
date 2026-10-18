#!/bin/bash
# Start the vllm37 container and wait until it answers /v1/models. When the
# container stops or the wait times out, print its last log lines and fail.
#
# Usage: container-up.sh [TIMEOUT_SECONDS]   (default 600)
# Exit 0 when ready, 1 otherwise; the last line printed states the outcome.
# Env:
#   MODEL, TP_SIZE, ...  the settings docker-compose.yml reads, from .env or
#                        the shell (see .env.example)
#   COMPOSE_FILE         compose files to use; CI sets
#                        docker-compose.yml:docker-compose.ci.yml
#   VLLM_LOG_DIR         host directory for vLLM and NCCL logs
#                        (default /tmp/vllm-logs)
#   HF_HOME              host Hugging Face cache, shared with the host build
#                        (default ~/.cache/huggingface)
#   HOST_UID, HOST_GID   user the container runs as (default: this user)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TIMEOUT=${1:-600}

# The log directory must exist and be writable before compose mounts it;
# otherwise Docker creates it root-owned and NCCL_DEBUG_FILE cannot write.
# Resolve it as compose does: shell, then .env, then the default.
LOG_DIR=${VLLM_LOG_DIR:-$(sed -n 's/^VLLM_LOG_DIR=//p' .env 2>/dev/null | tail -1)}
LOG_DIR=${LOG_DIR:-/tmp/vllm-logs}
mkdir -p "$LOG_DIR"
chmod 777 "$LOG_DIR"

# The container runs as this user (#87), so the Hugging Face cache it shares
# with the host build stays the user's. Create the cache first, or Docker
# creates it root-owned.
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
  # Give compose a few seconds to create the container before judging it.
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

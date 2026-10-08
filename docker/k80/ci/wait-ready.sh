#!/bin/bash
# Wait for the compose service `vllm` to answer /v1/models. Fails fast when
# the container stops, including when it has already exited (ps -a).
# Usage (from docker/k80): ci/wait-ready.sh [TIMEOUT_SECONDS]   (default 600)
# Exit 0 when ready, 1 otherwise; prints one line with the outcome.
set -uo pipefail
TIMEOUT=${1:-600}
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.ci.yml)
start=$(date +%s)

while :; do
  elapsed=$(( $(date +%s) - start ))
  cid=$("${COMPOSE[@]}" ps -a -q vllm 2>/dev/null)
  state=$(docker inspect --format='{{.State.Status}}' "$cid" 2>/dev/null || echo missing)
  # Give compose a few seconds to create the container before judging it.
  if [ "$state" != running ] && [ "$elapsed" -ge 10 ]; then
    echo "Container state=$state after ${elapsed}s."
    exit 1
  fi
  if curl -fsS --max-time 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "Server ready after ${elapsed}s."
    exit 0
  fi
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "Server not ready after ${TIMEOUT}s."
    exit 1
  fi
  sleep 5
done

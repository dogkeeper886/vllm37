#!/bin/bash
# Usage: service-stop.sh [--server]
#   --server  present: stop the server only and keep the K80s; absent: also give
#             them back
# Env:
#   VLLM37_RESULTS  a directory (default ci/tests/results)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
RESULTS=${VLLM37_RESULTS:-$HERE/tests/results}
UNIT=vllm37-ci
LOCAL_LOG=$XDG_RUNTIME_DIR/$UNIT.log

mkdir -p "$RESULTS"
if docker inspect vllm37 >/dev/null 2>&1; then
  docker logs vllm37 >> "$RESULTS/server.log" 2>&1
fi
if systemctl --user is-active --quiet "$UNIT"; then
  systemctl --user stop "$UNIT"
  echo "service-stop: stopped $UNIT"
fi
if [ -s "$LOCAL_LOG" ]; then
  cat "$LOCAL_LOG" >> "$RESULTS/server.log"
fi
rm -f "$LOCAL_LOG"

if [ "${1:-}" = --server ]; then
  exec "$HERE/../container-down.sh"
fi
exec "$HERE/gpu-restore.sh"

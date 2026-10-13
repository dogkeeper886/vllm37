#!/bin/bash
# The service step, stop: save the server's log beside the report, stop the
# server service-start.sh started, and give the K80s back. Safe to run when
# nothing was started, so a job runs it whatever happened before.
#
# Usage: service-stop.sh
# Env:
#   VLLM37_RESULTS  where server.log goes (default ci/tests/results)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
RESULTS=${VLLM37_RESULTS:-$HERE/tests/results}
UNIT=vllm37-ci
LOCAL_LOG=$XDG_RUNTIME_DIR/$UNIT.log

mkdir -p "$RESULTS"
if docker inspect vllm37 >/dev/null 2>&1; then
  docker logs vllm37 > "$RESULTS/server.log" 2>&1
fi
if systemctl --user is-active --quiet "$UNIT"; then
  systemctl --user stop "$UNIT"
  echo "service-stop: stopped $UNIT"
fi
if [ -s "$LOCAL_LOG" ]; then
  cp "$LOCAL_LOG" "$RESULTS/server.log"
fi
rm -f "$LOCAL_LOG"

# Removes the container and starts the services gpu-yield.sh stopped.
exec "$HERE/gpu-restore.sh"

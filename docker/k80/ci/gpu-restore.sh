#!/bin/bash
# Env:
#   VLLM37_YIELD_STATE  a file, the same as gpu-yield.sh's
#                       (default $XDG_RUNTIME_DIR/vllm37-ci-yielded)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
STATE=${VLLM37_YIELD_STATE:-$XDG_RUNTIME_DIR/vllm37-ci-yielded}
failed=0

if ! "$HERE/../container-down.sh"; then
  echo "gpu-restore: could not remove the vllm37 container" >&2
  failed=1
fi

[ -f "$STATE" ] || exit "$failed"
started=1
while read -r unit; do
  [ -n "$unit" ] || continue
  if systemctl --user start "$unit"; then
    echo "gpu-restore: started $unit"
  else
    echo "gpu-restore: could not start $unit" >&2
    started=0
    failed=1
  fi
done < "$STATE"
[ "$started" = 1 ] && rm -f "$STATE"
exit "$failed"

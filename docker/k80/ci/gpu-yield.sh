#!/bin/bash
# Free the K80s for the container under test: note which user systemd services
# are running, then stop them. gpu-restore.sh starts them again.
#
# The names go to a state file that outlives the job. A job killed before its
# restore leaves the file behind; the next yield adds to it instead of
# replacing it, so the next restore still starts what the killed job stopped.
#
# Usage: gpu-yield.sh
# Env:
#   VLLM37_YIELD_SERVICES  user units to stop when active (default "vllm37 ollama")
#   VLLM37_BUSY_SERVICES   user units that mean another CI holds the GPUs; the
#                          job fails when one is active (default "ollama37")
#   VLLM37_YIELD_STATE     state file (default $XDG_RUNTIME_DIR/vllm37-ci-yielded)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The runner is a system service, so its jobs may lack the user bus path.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
STATE=${VLLM37_YIELD_STATE:-$XDG_RUNTIME_DIR/vllm37-ci-yielded}

for unit in ${VLLM37_BUSY_SERVICES-ollama37}; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "gpu-yield: $unit is running, so another CI job holds the GPUs" >&2
    exit 1
  fi
done

# A container left by a killed job would be reused as it is.
"$HERE/../container-down.sh"

touch "$STATE"
for unit in ${VLLM37_YIELD_SERVICES-vllm37 ollama}; do
  if systemctl --user is-active --quiet "$unit"; then
    grep -qxF "$unit" "$STATE" || echo "$unit" >> "$STATE"
    systemctl --user stop "$unit"
    echo "gpu-yield: stopped $unit"
  else
    echo "gpu-yield: $unit is not running"
  fi
done
echo "gpu-yield: to restore: $(tr '\n' ' ' < "$STATE")"

#!/bin/bash
# Env:
#   VLLM37_YIELD_SERVICES  user unit names, space-separated, or empty for none
#                          (default "vllm37 ollama")
#   VLLM37_BUSY_SERVICES   user unit names, space-separated, or empty for none
#                          (default "ollama37")
#   VLLM37_YIELD_STATE     a file (default $XDG_RUNTIME_DIR/vllm37-ci-yielded)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
STATE=${VLLM37_YIELD_STATE:-$XDG_RUNTIME_DIR/vllm37-ci-yielded}

for unit in ${VLLM37_BUSY_SERVICES-ollama37}; do
  if systemctl --user is-active --quiet "$unit"; then
    echo "gpu-yield: $unit is running, so another CI job holds the GPUs" >&2
    exit 1
  fi
done

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

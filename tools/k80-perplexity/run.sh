#!/bin/bash
# Usage: run.sh <profile> [OPTION...]
#   profile  a profile name in docker/k80/models/ or a path
#   OPTION   perplexity.py measure's --text, --windows, --window-tokens,
#            --decode-prompts, --decode-prompt, --decode-tokens or --seed
# Env:
#   PORT                  a TCP port (default 8011)
#   CUDA_VISIBLE_DEVICES  comma-separated die indexes, the profile's tensor x
#                         pipeline parallel size of them, each with 100 MiB or
#                         less in use (default 0 up to that size less 1)
#   RESULTS               a directory (default ~/.local/state/k80-perplexity)
#   READY_TIMEOUT         whole seconds (default 3600)
#   GPU_MEM_UTIL          a fraction above 0, at most 1 (default 0.75)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PROFILE=${1:?usage: run.sh <profile> [perplexity.py measure options]}
shift
PORT=${PORT:-8011}
RESULTS=${RESULTS:-$HOME/.local/state/k80-perplexity}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}

lines=$("$ROOT/docker/k80/ci/profile.sh" "$PROFILE")
tp=1 pp=1
while read -r kind value; do
  case "$kind" in
    FILE) file=$value ;;
    TP_SIZE) tp=$value ;;
    PP_SIZE) pp=$value ;;
  esac
done <<< "$lines"
name=$(basename "$file" .yaml)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((tp * pp - 1)))}

for d in ${CUDA_VISIBLE_DEVICES//,/ }; do
  used=$(nvidia-smi -i "$d" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -gt 100 ]; then
    echo "run.sh: GPU$d holds $used MiB: another process is using it" >&2
    exit 1
  fi
done

source "$ROOT/tools/k80-host/env.sh"
src=$(cd / && python -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)
build=$(git -C "$src" rev-parse --short HEAD)
git -C "$src" diff --quiet HEAD -- vllm || build+="+dirty"

mkdir -p "$RESULTS"
stamp=$(date +%Y%m%d-%H%M%S)
log=$RESULTS/$name-$stamp.server.log
out=$RESULTS/$name-$stamp.json

PROFILE="$file" PORT="$PORT" setsid "$ROOT/tools/k80-host/serve.sh" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" > "$log" 2>&1 &
server=$!
trap 'kill -TERM -- -"$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
url=http://localhost:$PORT
echo "run.sh: serving $name ($build) on dies $CUDA_VISIBLE_DEVICES; log $log"
deadline=$((SECONDS + READY_TIMEOUT))
until curl -sf "$url/health" > /dev/null; do
  if ! kill -0 "$server" 2>/dev/null; then
    echo "run.sh: the server exited; the end of $log:" >&2
    tail -20 "$log" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "run.sh: $url gave no answer in $READY_TIMEOUT s" >&2
    exit 1
  fi
  sleep 10
done
python "$HERE/perplexity.py" measure --url "$url" --label "$name" --build "$build" \
  --out "$out" "$@" > /dev/null
echo "run.sh: $out"

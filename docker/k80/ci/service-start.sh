#!/bin/bash
# The service step, start: free the K80s, start the vllm37 server a test talks
# to, and report what it holds. service-stop.sh ends it and gives the K80s back.
#
# Usage: service-start.sh [docker|local]
#   docker   the vllm37 container from vllm37-local:latest (default)
#   local    tools/k80-host/serve.sh from the host build, as the temporary user
#            unit vllm37-ci. ~/.config/vllm37/serve.env and the vllm37 unit are
#            not touched.
# Env (the defaults are the first row of the use-case table):
#   MODEL           Hugging Face model id (default Qwen/Qwen3-0.6B)
#   DTYPE           float32 or half (default float32)
#   TP_SIZE         dies, from 1 up to what the host has, GPU0 first (default 1).
#                   vLLM starts only when the count divides the model's
#                   attention heads.
#   MAX_MODEL_LEN   context length in tokens (default 12288)
#   VLLM37_RESULTS  where server.json goes (default ci/tests/results)
# Output: a Server table on stdout and in $GITHUB_STEP_SUMMARY, and server.json.
# Exit 1 when the server does not start or holds memory on other dies than asked.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
SERVER=${1:-docker}
export MODEL=${MODEL:-Qwen/Qwen3-0.6B}
export DTYPE=${DTYPE:-float32}
export TP_SIZE=${TP_SIZE:-1}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
RESULTS=${VLLM37_RESULTS:-$HERE/tests/results}
UNIT=vllm37-ci
LOCAL_LOG=$XDG_RUNTIME_DIR/$UNIT.log
TIMEOUT=600

case "$SERVER" in docker|local) ;; *) echo "service-start: server must be docker or local, got '$SERVER'" >&2; exit 1 ;; esac
case "$DTYPE" in float32|half) ;; *) echo "service-start: DTYPE must be float32 or half, got '$DTYPE'" >&2; exit 1 ;; esac
case "$MAX_MODEL_LEN" in ''|*[!0-9]*) echo "service-start: MAX_MODEL_LEN must be a number, got '$MAX_MODEL_LEN'" >&2; exit 1 ;; esac
# No count is written here: the host says how many dies it has, and vLLM
# says whether the model can be split that many ways.
case "$TP_SIZE" in ''|*[!0-9]*|0*) echo "service-start: TP_SIZE must be a whole number from 1, got '$TP_SIZE'" >&2; exit 1 ;; esac
HOST_DIES=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$TP_SIZE" -gt "$HOST_DIES" ]; then
  echo "service-start: asked for $TP_SIZE dies, this host has $HOST_DIES" >&2
  exit 1
fi
# More than one die: keep the tensor-parallel start-up trace.
export VLLM_K80_TRACE=${VLLM_K80_TRACE:-$([ "$TP_SIZE" -gt 1 ] && echo 1 || echo 0)}

BUILD=$("$HERE/runtime.sh" "$SERVER" commit)
# A unit left by a killed local run would still hold the cards and port 8000,
# whichever kind of server starts now. gpu-yield.sh does the same for a container.
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
"$HERE/gpu-yield.sh"

if [ "$SERVER" = docker ]; then
  COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml:docker-compose.ci.yml} "$HERE/../container-up.sh" "$TIMEOUT"
else
  # The dies the container takes: the first TP_SIZE.
  DEVICES=$(seq -s, 0 $(( TP_SIZE - 1 )))
  : > "$LOCAL_LOG"
  systemd-run --user --unit="$UNIT" --collect --quiet \
    --setenv=MODEL="$MODEL" --setenv=DTYPE="$DTYPE" --setenv=TP_SIZE="$TP_SIZE" \
    --setenv=MAX_MODEL_LEN="$MAX_MODEL_LEN" --setenv=CUDA_VISIBLE_DEVICES="$DEVICES" \
    --setenv=VLLM_K80_TRACE="$VLLM_K80_TRACE" \
    /bin/sh -c 'exec "$0" >"$1" 2>&1' "$ROOT/tools/k80-host/serve.sh" "$LOCAL_LOG"
  start=$(date +%s)
  while :; do
    elapsed=$(( $(date +%s) - start ))
    if ! systemctl --user is-active --quiet "$UNIT"; then
      tail -n 200 "$LOCAL_LOG"
      echo "Unit $UNIT stopped after ${elapsed}s."
      exit 1
    fi
    if curl -fsS --max-time 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "Server ready after ${elapsed}s."
      break
    fi
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
      tail -n 200 "$LOCAL_LOG"
      echo "Server not ready after ${TIMEOUT}s."
      exit 1
    fi
    sleep 5
  done
fi

# What the server holds, from its own start-up log and from nvidia-smi.
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
if [ "$SERVER" = docker ]; then docker logs vllm37 > "$LOG" 2>&1; else cp "$LOCAL_LOG" "$LOG"; fi
WEIGHTS=$(grep -m1 -o 'model weights take [0-9.]*' "$LOG" | grep -o '[0-9.]*$' || true)
BLOCKS=$(grep -m1 -o '# cuda blocks: [0-9]*' "$LOG" | grep -o '[0-9]*$' || true)
# 16 tokens per KV cache block (vLLM's default --block-size).
KV=$(( ${BLOCKS:-0} * 16 ))

# Only the server's own processes: another tenant's memory is not the server's.
if [ "$SERVER" = docker ]; then
  PIDS=$(docker top vllm37 -eo pid | tail -n +2)
else
  CGROUP=$(systemctl --user show -p ControlGroup --value "$UNIT")
  PIDS=$(cat "/sys/fs/cgroup/systemd$CGROUP/cgroup.procs" 2>/dev/null || cat "/sys/fs/cgroup$CGROUP/cgroup.procs")
fi
MINE=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits \
  | awk -F', *' -v pids="$(echo $PIDS)" 'BEGIN { n = split(pids, a, " "); for (i = 1; i <= n; i++) p[a[i]] = 1 } ($2 in p)')
HELD=$(printf '%s\n' "$MINE" | awk -F', *' 'NF { u[$1] = 1 } END { n = 0; for (k in u) n++; print n }')
VRAM=$(printf '%s\n' "$MINE" | awk -F', *' 'NF { s += $3 } END { print s + 0 }')

mkdir -p "$RESULTS"
cat > "$RESULTS/server.json" <<EOF
{"server": "$SERVER", "build": "$BUILD", "model": "$MODEL", "dtype": "$DTYPE", "dies": $TP_SIZE, "maxModelLen": $MAX_MODEL_LEN, "weightsGib": ${WEIGHTS:-0}, "kvTokens": $KV, "vramMib": $VRAM, "diesHolding": $HELD}
EOF

emit() {
  printf '%s\n' "$1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf '%s\n' "$1" >> "$GITHUB_STEP_SUMMARY"; fi
}
emit ""
emit "### Server"
emit ""
emit "| | |"
emit "|---|---|"
emit "| Server | $SERVER, built from \`$BUILD\` |"
emit "| Model | \`$MODEL\` |"
emit "| Weight dtype | \`$DTYPE\` |"
emit "| Dies | asked for $TP_SIZE, memory held on $HELD |"
emit "| Context | $MAX_MODEL_LEN tokens |"
emit "| Weights | ${WEIGHTS:-unknown} GiB per die |"
emit "| KV cache | $KV tokens |"
emit "| VRAM | $VRAM MiB |"

if [ "$HELD" != "$TP_SIZE" ]; then
  echo "service-start: the server holds memory on $HELD die(s), asked for $TP_SIZE" >&2
  exit 1
fi

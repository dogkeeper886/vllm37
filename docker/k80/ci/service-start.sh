#!/bin/bash
# The service step, start: free the K80s, start the vllm37 server a test talks
# to, and report what it holds. service-stop.sh ends it and gives the K80s back.
#
# Usage: service-start.sh [docker|local]
#   docker   the vllm37 container from vllm37-local:latest (default)
#   local    tools/k80-host/serve.sh from the host build, as the temporary user
#            unit vllm37-ci. ~/.config/vllm37/serve.env and the vllm37 unit are
#            not touched.
# Env:
#   PROFILE         the model profile, a name in docker/k80/models/ or a path
#                   (default qwen3-0.6b, the function-test baseline). It holds
#                   the model's start-up flags; ci/profile.sh reads it.
#   MODEL, DTYPE, TP_SIZE, MAX_MODEL_LEN
#                   override the profile's model, dtype (weight and KV cache,
#                   such as float32 or half), tensor parallel size and context
#                   length; empty keeps the profile's
#   VLLM37_RESULTS  where server.json goes (default ci/tests/results)
# The server takes TP_SIZE x pipeline-parallel-size dies, GPU0 first, up to what
# the host has; vLLM says whether the model splits that many ways.
# Output: a Server table on stdout and in $GITHUB_STEP_SUMMARY, and server.json.
# Exit 1 when the server does not start or holds memory on other dies than asked.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
SERVER=${1:-docker}
RESULTS=${VLLM37_RESULTS:-$HERE/tests/results}
UNIT=vllm37-ci
LOCAL_LOG=$XDG_RUNTIME_DIR/$UNIT.log
TIMEOUT=600

case "$SERVER" in docker|local) ;; *) echo "service-start: server must be docker or local, got '$SERVER'" >&2; exit 1 ;; esac
# The Hugging Face cache as the server sees it: the container mounts it here.
[ "$SERVER" = docker ] && export HF_CACHE=/cache/huggingface
PROFILE=${PROFILE:-qwen3-0.6b}
PROFILE_LINES=$("$HERE/profile.sh" "$PROFILE")
P_DTYPE=float32 P_TP=1 P_PP=1 P_LEN=12288 EXTRA=()
while read -r kind value; do
  case "$kind" in
    FILE) PROFILE_FILE=$value ;;
    MODEL) P_MODEL=$value ;;
    DTYPE) P_DTYPE=$value ;;
    TP_SIZE) P_TP=$value ;;
    PP_SIZE) P_PP=$value ;;
    MAX_MODEL_LEN) P_LEN=$value ;;
    ARG) EXTRA+=("$value") ;;
  esac
done <<< "$PROFILE_LINES"
export MODEL=${MODEL:-$P_MODEL}
export DTYPE=${DTYPE:-$P_DTYPE}
export TP_SIZE=${TP_SIZE:-$P_TP}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-$P_LEN}
PP_SIZE=$P_PP

case "$MAX_MODEL_LEN" in ''|*[!0-9]*) echo "service-start: MAX_MODEL_LEN must be a number, got '$MAX_MODEL_LEN'" >&2; exit 1 ;; esac
# No count is written here: the host says how many dies it has, and vLLM
# says whether the model can be split that many ways.
case "$TP_SIZE" in ''|*[!0-9]*|0*) echo "service-start: TP_SIZE must be a whole number from 1, got '$TP_SIZE'" >&2; exit 1 ;; esac
case "$PP_SIZE" in ''|*[!0-9]*|0*) echo "service-start: pipeline-parallel-size must be a whole number from 1, got '$PP_SIZE'" >&2; exit 1 ;; esac
[ "$PP_SIZE" -gt 1 ] && EXTRA+=(--pipeline-parallel-size "$PP_SIZE")
DIES=$(( TP_SIZE * PP_SIZE ))
HOST_DIES=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$DIES" -gt "$HOST_DIES" ]; then
  echo "service-start: asked for $DIES dies, this host has $HOST_DIES" >&2
  exit 1
fi
# For the test step, which watches every die the server takes.
if [ -n "${GITHUB_ENV:-}" ]; then echo "VLLM37_DIES=$DIES" >> "$GITHUB_ENV"; fi
# More than one die: keep the tensor-parallel start-up trace.
export VLLM_K80_TRACE=${VLLM_K80_TRACE:-$([ "$DIES" -gt 1 ] && echo 1 || echo 0)}

BUILD=$("$HERE/runtime.sh" "$SERVER" commit)
# A unit left by a killed local run would still hold the cards and port 8000,
# whichever kind of server starts now. gpu-yield.sh does the same for a container.
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
"$HERE/gpu-yield.sh"

if [ "$SERVER" = docker ]; then
  export COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml:docker-compose.ci.yml}
  if [ "${#EXTRA[@]}" -gt 0 ]; then
    # docker-compose.yml's command is a fixed list, so the profile's other
    # flags go in an override that repeats it with them appended.
    OVERRIDE=$XDG_RUNTIME_DIR/vllm37-ci-profile.json
    (cd "$HERE/.." && docker compose config --format json) \
      | python3 -I -c 'import json, sys; c = json.load(sys.stdin)["services"]["vllm"]["command"]; json.dump({"services": {"vllm": {"command": c + sys.argv[1:]}}}, sys.stdout)' "${EXTRA[@]}" \
      > "$OVERRIDE"
    COMPOSE_FILE=$COMPOSE_FILE:$OVERRIDE
  fi
  "$HERE/../container-up.sh" "$TIMEOUT"
else
  # The dies the server takes: the first DIES.
  DEVICES=$(seq -s, 0 $(( DIES - 1 )))
  : > "$LOCAL_LOG"
  systemd-run --user --unit="$UNIT" --collect --quiet \
    --setenv=MODEL="$MODEL" --setenv=DTYPE="$DTYPE" --setenv=TP_SIZE="$TP_SIZE" \
    --setenv=MAX_MODEL_LEN="$MAX_MODEL_LEN" --setenv=CUDA_VISIBLE_DEVICES="$DEVICES" \
    --setenv=VLLM_K80_TRACE="$VLLM_K80_TRACE" --setenv=PROFILE= \
    /bin/sh -c 'log=$1; shift; exec "$0" "$@" >"$log" 2>&1' "$ROOT/tools/k80-host/serve.sh" "$LOCAL_LOG" "${EXTRA[@]}"
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
{"server": "$SERVER", "build": "$BUILD", "model": "$MODEL", "dtype": "$DTYPE", "dies": $DIES, "tensorParallel": $TP_SIZE, "pipelineParallel": $PP_SIZE, "profile": "$(basename "$PROFILE_FILE" .yaml)", "maxModelLen": $MAX_MODEL_LEN, "weightsGib": ${WEIGHTS:-0}, "kvTokens": $KV, "vramMib": $VRAM, "diesHolding": $HELD}
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
emit "| Profile | \`$(basename "$PROFILE_FILE")\` |"
emit "| Model | \`$MODEL\` |"
emit "| Weight dtype | \`$DTYPE\` |"
emit "| Dies | asked for $DIES (tensor $TP_SIZE x pipeline $PP_SIZE), memory held on $HELD |"
emit "| Context | $MAX_MODEL_LEN tokens |"
emit "| Weights | ${WEIGHTS:-unknown} GiB per die |"
emit "| KV cache | $KV tokens |"
emit "| VRAM | $VRAM MiB |"

if [ "$HELD" != "$DIES" ]; then
  echo "service-start: the server holds memory on $HELD die(s), asked for $DIES" >&2
  exit 1
fi

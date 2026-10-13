#!/bin/bash
# The runtime check: report the build a test is about to serve, and fail when
# it is not there. It never builds. The K80 Docker Build workflow makes the
# image; tools/k80-host/setup.sh makes the local build.
#
# Usage: runtime.sh [docker|local] [commit]
#   docker   the image vllm37-local:latest (default)
#   local    the host build in the venv, which installs a working tree in place
#   commit   print only the commit the build was made from
# Output: a table on stdout and in $GITHUB_STEP_SUMMARY.
# Env:
#   K80_VENV  the local build's venv (default ~/.venvs/vllm37)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER=${1:-docker}
ONLY=${2:-}
IMAGE=vllm37-local:latest
VENV=${K80_VENV:-$HOME/.venvs/vllm37}

emit() {
  printf '%s\n' "$1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf '%s\n' "$1" >> "$GITHUB_STEP_SUMMARY"; fi
}
missing() {
  if [ "$ONLY" = commit ]; then echo "$1" >&2; else emit ""; emit "### Runtime"; emit ""; emit "$1"; fi
  exit 1
}

case "$SERVER" in
  docker)
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
      || missing "No \`$IMAGE\` image on this runner. Run the K80 Docker Build workflow."
    REV=$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
    SIZE=$(docker image inspect "$IMAGE" --format '{{.Size}}')
    ;;
  local)
    [ -x "$VENV/bin/python" ] || missing "No local build at \`$VENV\`. Run tools/k80-host/setup.sh."
    PKG=$("$VENV/bin/python" -c "import importlib.util as u; s = u.find_spec('vllm'); print(s.submodule_search_locations[0] if s else '')")
    [ -n "$PKG" ] || missing "The venv \`$VENV\` has no vllm installed. Run tools/k80-host/setup.sh."
    TREE=$(git -C "$PKG" rev-parse --show-toplevel 2>/dev/null || true)
    if [ -n "$TREE" ]; then
      REV=$(git -C "$TREE" rev-parse HEAD)
      if [ -n "$(git -C "$TREE" status --porcelain --untracked-files=no)" ]; then REV="$REV (modified)"; fi
    else
      REV=unknown
    fi
    ;;
  *)
    echo "runtime.sh: server must be docker or local, got '$SERVER'" >&2
    exit 1
    ;;
esac

if [ "$ONLY" = commit ]; then
  echo "${REV:-unknown}"
  exit 0
fi

emit ""
emit "### Runtime"
emit ""
emit "| | |"
emit "|---|---|"
if [ "$SERVER" = docker ]; then
  emit "| Server | docker image \`$IMAGE\` |"
  emit "| Built from commit | \`${REV:-unknown}\` |"
  emit "| Size | $(awk -v b="$SIZE" 'BEGIN { printf "%.2f GiB", b / 1073741824 }') |"
else
  emit "| Server | local build in \`$VENV\` |"
  emit "| Working tree | \`${TREE:-not a git tree: $PKG}\` |"
  emit "| Commit | \`$REV\` |"
fi
emit "| This run's commit | \`${GITHUB_SHA:-$(git -C "$HERE" rev-parse HEAD)}\` |"

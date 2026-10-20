#!/bin/bash
# Usage: ensure-builder.sh [IMAGE]
#   IMAGE  an image on this machine, or vllm37-builder:latest, pulled from
#          Docker Hub when missing (default vllm37-builder:latest)
set -euo pipefail
IMAGE=${1:-vllm37-builder:latest}
HUB=dogkeeper886/vllm37-builder:latest

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Builder $IMAGE present."
  exit 0
fi
if [ "$IMAGE" = vllm37-builder:latest ]; then
  echo "Builder $IMAGE not found; pulling $HUB."
  docker pull "$HUB"
  docker tag "$HUB" "$IMAGE"
  exit 0
fi
echo "::error::Builder $IMAGE not found. Build it (k80-build.yml rebuild_builder=true) or tag it on the runner."
exit 1

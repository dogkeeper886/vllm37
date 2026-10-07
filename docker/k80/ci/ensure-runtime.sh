#!/bin/bash
# Build vllm37-local:latest unless it was already built from this commit
# (the image's org.opencontainers.image.revision label).
# Usage (from docker/k80): ci/ensure-runtime.sh [BUILDER_IMAGE]
set -euo pipefail
BUILDER=${1:-vllm37-builder:latest}
SHA=${GITHUB_SHA:-$(git rev-parse HEAD)}
IMAGE_SHA=$(docker inspect --format='{{index .Config.Labels "org.opencontainers.image.revision"}}' \
  vllm37-local:latest 2>/dev/null || true)

if [ "$IMAGE_SHA" = "$SHA" ]; then
  echo "vllm37-local:latest already built from $SHA; reusing it."
  exit 0
fi
echo "vllm37-local:latest is at '${IMAGE_SHA:-none}', branch is $SHA; building."
"$(dirname "$0")/ensure-builder.sh" "$BUILDER"
make build-local BUILDER_IMAGE="$BUILDER"

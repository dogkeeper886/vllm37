#!/bin/bash
# Stop and remove the vllm37 container.
#
# Usage: container-down.sh
# Env:
#   COMPOSE_FILE  compose files to use, as for container-up.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose down -v

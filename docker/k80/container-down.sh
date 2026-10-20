#!/bin/bash
# Env:
#   COMPOSE_FILE  compose files in docker/k80, colon-separated, the same as
#                 container-up.sh's (default docker-compose.yml)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose down -v

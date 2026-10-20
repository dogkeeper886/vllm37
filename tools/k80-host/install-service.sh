#!/bin/bash
# Env:
#   XDG_RUNTIME_DIR  the user's runtime directory (default /run/user/<uid>)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=$HOME/.config/systemd/user/vllm37.service
ENV_FILE=$HOME/.config/vllm37/serve.env
LOG=$HOME/.local/state/vllm37/serve.log
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

mkdir -p "$(dirname "$UNIT")" "$(dirname "$ENV_FILE")" "$(dirname "$LOG")"
if [ ! -f "$ENV_FILE" ]; then
  cp "$HERE/serve.env.example" "$ENV_FILE"
  echo "install-service: wrote $ENV_FILE"
fi

cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=vllm37 server (host build, Tesla K80)
After=network-online.target

[Service]
EnvironmentFile=-$ENV_FILE
ExecStartPre=-/bin/mv -f $LOG $LOG.1
ExecStart=/bin/sh -c 'exec "\$0" >"\$1" 2>&1' $HERE/serve.sh $LOG

[Install]
WantedBy=default.target
UNIT_EOF
echo "install-service: wrote $UNIT"

systemctl --user daemon-reload
echo "install-service: start with 'systemctl --user start vllm37'; log: $LOG"

#!/bin/bash
# Install the user systemd unit vllm37.service, which serves the host build of
# this checkout with tools/k80-host/serve.sh.
#
# Usage: install-service.sh
# Writes ~/.config/systemd/user/vllm37.service and, when it is missing,
# ~/.config/vllm37/serve.env (a copy of serve.env.example). The unit is left
# disabled: start it with `systemctl --user start vllm37`.
# The server logs to ~/.local/state/vllm37/serve.log; the previous run's log
# is kept as serve.log.1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=$HOME/.config/systemd/user/vllm37.service
ENV_FILE=$HOME/.config/vllm37/serve.env
LOG=$HOME/.local/state/vllm37/serve.log
# A shell without a login session (a system service, cron) has no user bus path.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

mkdir -p "$(dirname "$UNIT")" "$(dirname "$ENV_FILE")" "$(dirname "$LOG")"
if [ ! -f "$ENV_FILE" ]; then
  cp "$HERE/serve.env.example" "$ENV_FILE"
  echo "install-service: wrote $ENV_FILE"
fi

# No Restart=: a server that cannot get GPU memory stays down and visible
# instead of restarting in a loop on a power-sensitive machine.
# The log goes to a file because a user without journal access cannot read
# `journalctl --user`, and systemd 239 (Rocky 8) has no StandardOutput=append:.
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

#!/bin/bash
# Usage: profile.sh PROFILE
#   PROFILE   a name in docker/k80/models/ without .yaml, or a path with a / or
#             ending in .yaml or .yml
# Env:
#   HF_CACHE  a directory as the server sees it (default $HF_HOME)
#   HF_HOME   a directory (default ~/.cache/huggingface)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  '') echo "profile.sh: name a profile, one of: $(cd "$HERE/../models" && ls *.yaml | sed 's/\.yaml$//' | tr '\n' ' ')" >&2; exit 1 ;;
  */*|*.yaml|*.yml) FILE=$1 ;;
  *) FILE=$HERE/../models/$1.yaml ;;
esac
[ -f "$FILE" ] || { echo "profile.sh: no profile at $FILE" >&2; exit 1; }
export HF_CACHE=${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}
echo "FILE $(cd "$(dirname "$FILE")" && pwd)/$(basename "$FILE")"
python3 -I - "$FILE" <<'PY'
import os, sys, yaml
path = sys.argv[1]
data = yaml.safe_load(open(path)) or {}
if not isinstance(data, dict) or any(isinstance(v, (dict, list)) for v in data.values()):
    sys.exit(f"profile.sh: {path} must be a flat map of flag: value")
if not data.get("model"):
    sys.exit(f"profile.sh: {path} names no model")
known = {"model": "MODEL", "dtype": "DTYPE", "tensor-parallel-size": "TP_SIZE",
         "pipeline-parallel-size": "PP_SIZE", "max-model-len": "MAX_MODEL_LEN"}
for key, value in data.items():
    if isinstance(value, bool):
        if value:
            print(f"ARG --{key}")
        continue
    value = os.path.expanduser(os.path.expandvars(str(value)))
    if key in known:
        print(f"{known[key]} {value}")
    else:
        print(f"ARG --{key}")
        print(f"ARG {value}")
PY

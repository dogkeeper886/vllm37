#!/bin/bash
# Read a model profile, docker/k80/models/<name>.yaml: one flat YAML map of
# vLLM's own flags without the leading --. The profile is the only place a
# model's start-up flags live, so a new model is a new file, not a script edit.
#
# Usage: profile.sh <name or path>
# Env:   HF_CACHE  the Hugging Face cache as the server sees it, for ${HF_CACHE}
#                  in a value (default ${HF_HOME:-~/.cache/huggingface})
# Output, one line each:
#   FILE <path>                      the profile read
#   MODEL|DTYPE|TP_SIZE|PP_SIZE|MAX_MODEL_LEN <value>
#                                    the flags service-start.sh reads itself
#   ARG <token>                      every other flag, in order, for vLLM
# Exit 1 when the profile is missing, is not a flat map, or names no model.
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
        # As vLLM's own --config: true is the bare flag, false leaves it out.
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

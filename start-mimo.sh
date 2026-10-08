#!/usr/bin/env bash
# Starts the MiMo server: ./start-mimo.sh [config.json]
# The config is the first argument, else $STRATA_MIMO_CONFIG, else the strata-mimo-mxfp4.json setup writes in this
# folder, else strata-mimo.json there.  Its "port" is used (8095 if it has none); its "host" and "api_key" decide who can connect.  Ctrl+C stops it; ./stop-mimo.sh
# stops it from anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
default="$here/strata-mimo-mxfp4.json"
[ -f "$default" ] || default="$here/strata-mimo.json"
cfg="${1:-${STRATA_MIMO_CONFIG:-$default}}"
py="$here/.venv/bin/python"
[ -f "$cfg" ] || { echo "No MiMo config at $cfg: pass one, set STRATA_MIMO_CONFIG, or run ./setup.sh --family mimo."; exit 1; }
[ -x "$py" ] || { echo "No .venv in $here: run ./setup.sh --family mimo first."; exit 1; }
cd "$here"
exec "$py" serve/server.py --engine strata --config "$cfg"   # its port: "port" in the config (8095 when it has none)

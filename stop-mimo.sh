#!/usr/bin/env bash
# Stops the MiMo server and its engine started from this folder.  A hard stop: the engine's learned expert profile
# keeps its last periodic save.  Ctrl+C in the server's terminal is the gentle way.
here="$(cd "$(dirname "$0")" && pwd)"
pkill -f "$here/.venv/bin/python serve/server.py" && echo "stopped the server"
pkill -f "$here/build/strata-mimo" && echo "stopped the engine"
echo "Strata-MiMo stopped."

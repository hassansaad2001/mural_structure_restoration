#!/bin/bash
# Create the small Python environment for the GUIs on the HOST (Ubuntu or WSL): .venv-gui/
# Needs:  sudo apt install python3-venv python3-tk
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

if ! "$PY" -c "import tkinter" 2> /dev/null; then
    echo "tkinter is missing. Install it first:  sudo apt install python3-tk python3-venv"
    exit 1
fi
"$PY" -m venv "$REPO/.venv-gui"
"$REPO/.venv-gui/bin/pip" install --upgrade pip > /dev/null
"$REPO/.venv-gui/bin/pip" install -r "$REPO/requirements-gui.txt"
echo
echo "GUI environment ready: $REPO/.venv-gui"
echo "scripts/eval_gui.sh and scripts/inference_gui.sh use it automatically."

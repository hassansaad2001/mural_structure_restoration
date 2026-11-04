#!/bin/bash
# Run on the HOST: start inference/inference_server.py with a checkpoint in the container, open the
# inference GUI here, stop the server again when the GUI is closed.
#
#   bash scripts/inference_gui.sh <checkpoint.pth (host path)> [GUI options, e.g. --input-dir DIR --edge-dir DIR]
#
# The checkpoint must lie inside the folder mounted at /master (see docker/run.sh).
# Uses MSR_CONTAINER (default: msr), MSR_MOUNT and MSR_PYTHON like docker/run.sh.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_container.sh"
if [ -z "$1" ] || [ ! -f "$1" ]; then
    echo "Usage: bash scripts/inference_gui.sh <checkpoint.pth> [GUI options]"; exit 1
fi
CKPT="$(to_container_path "$1")"; shift
require_container

ADDR="$(server_addr 9999)"
if server_up 9999; then
    echo "An inference server is already running in '$CONTAINER' (port 9999). Stop it first."; exit 1
fi
echo "Starting inference server in container '$CONTAINER' with $CKPT ..."
docker exec -d -w "$REPO_IN_CONTAINER" "$CONTAINER" \
    bash -c "python3 inference/inference_server.py --model '$CKPT' --port 9999 > /tmp/inference_server.log 2>&1"
stop_server() {
    docker exec "$CONTAINER" pkill -f "inference/inference_server.py --model .* --port 9999" || true
    echo "Inference server stopped."
}
trap stop_server EXIT
if ! wait_server 9999 180; then     # loading the model can take a while
    echo "Inference server did not start; its log:"; docker exec "$CONTAINER" tail -20 /tmp/inference_server.log
    exit 1
fi

echo "Opening inference GUI ($ADDR)"
"$GUI_PY" "$REPO/inference/inference_gui.py" --server "$ADDR" "$@"

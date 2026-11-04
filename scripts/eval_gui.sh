#!/bin/bash
# Run on the HOST: start evaluation/eval_server.py in the container, open the evaluation GUI here,
# stop the server again when the GUI is closed (a server that was already running is left alone).
#
#   bash scripts/eval_gui.sh [GUI options, e.g. --settings my_eval.json]
#
# Uses MSR_CONTAINER (default: msr), MSR_MOUNT and MSR_PYTHON like docker/run.sh.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/_container.sh"
require_container

ADDR="$(server_addr 9998)"
STARTED=0
stop_server() {
    if [ "$STARTED" = 1 ]; then
        docker exec "$CONTAINER" pkill -f "evaluation/eval_server.py --port 9998" || true
        echo "Eval server stopped."
    fi
}
trap stop_server EXIT          # also stops the server if it fails to come up
if server_up 9998; then
    echo "Eval server already running at $ADDR"
else
    echo "Starting eval server in container '$CONTAINER'..."
    docker exec -d -w "$REPO_IN_CONTAINER" "$CONTAINER" \
        bash -c "python3 evaluation/eval_server.py --port 9998 > /tmp/eval_server.log 2>&1"
    STARTED=1
    if ! wait_server 9998 60; then
        echo "Eval server did not start; its log:"; docker exec "$CONTAINER" tail -20 /tmp/eval_server.log
        exit 1
    fi
fi

echo "Opening evaluation GUI ($ADDR, path map $MOUNT=/master)"
"$GUI_PY" "$REPO/evaluation/eval_checkpoints_gui.py" --server "$ADDR" --path-map "$MOUNT=/master" "$@"

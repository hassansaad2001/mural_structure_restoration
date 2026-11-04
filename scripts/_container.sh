#!/bin/bash
# Shared helpers for the host-side launchers (sourced, not run).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${MSR_CONTAINER:-msr}"
MOUNT="${MSR_MOUNT:-$(dirname "$REPO")}"                      # host folder mounted at /master
REPO_IN_CONTAINER="${MSR_CONTAINER_REPO:-/master/$(basename "$REPO")}"

# Python for the GUIs: $MSR_PYTHON, else .venv-gui (scripts/setup_gui_env.sh), else python3
if [ -n "$MSR_PYTHON" ]; then GUI_PY="$MSR_PYTHON"
elif [ -x "$REPO/.venv-gui/bin/python" ]; then GUI_PY="$REPO/.venv-gui/bin/python"
else GUI_PY=python3; fi

require_container() {
    if [ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER" 2> /dev/null)" != "true" ]; then
        echo "Container '$CONTAINER' is not running. Start it with:  bash docker/run.sh"
        echo "(or set MSR_CONTAINER=<name> to use another container)"
        exit 1
    fi
}

# host path -> path inside the container (only paths under $MOUNT can be seen there)
to_container_path() {
    local p; p="$(realpath -m "$1")"
    case "$p" in
        "$MOUNT"/*) echo "/master${p#"$MOUNT"}" ;;
        /master/*)  echo "$p" ;;
        *) echo "Path is not inside the mounted folder $MOUNT: $p" >&2; return 1 ;;
    esac
}

# host:port to reach container port $1: the published port if there is one, else the container IP
server_addr() {
    local pub ip
    pub="$(docker port "$CONTAINER" "$1/tcp" 2> /dev/null | head -1)"
    if [ -n "$pub" ]; then
        echo "localhost:${pub##*:}"
    else
        ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' "$CONTAINER" | awk '{print $1}')"
        echo "${ip:-localhost}:$1"
    fi
}

# Is a server listening on container port $1? Checked INSIDE the container: from the host, Docker's
# port forwarder accepts connections on a published port even when nothing listens behind it.
server_up() {
    docker exec "$CONTAINER" timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2> /dev/null
}

wait_server() {   # wait_server container_port seconds
    for _ in $(seq 1 "$2"); do server_up "$1" && return 0; sleep 1; done
    return 1
}

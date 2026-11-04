#!/bin/bash
# Start the project container (or restart it if it already exists). Works on native Ubuntu and WSL.
#
#   bash docker/run.sh                 build the image if needed, start container "msr" in the background
#   docker exec -it msr bash           open a shell in it (the repo is at /master/<repo folder>)
#
# Settings (environment variables):
#   MSR_CONTAINER  container name                        [msr]
#   MSR_IMAGE      image name                            [mural-structure:latest]
#   MSR_MOUNT      host folder mounted at /master        [parent folder of this repository]
#   MSR_DATA       optional extra host folder at /data   [none]
#   MSR_GPUS       value for --gpus, or "none"           [all]   e.g. MSR_GPUS='"device=0,1"'
#   MSR_SHM        shared memory for DataLoader workers  [32g]
#   MSR_EVAL_PORT / MSR_INFER_PORT / MSR_TB_PORT  host ports for the eval server, inference server
#                  and TensorBoard                       [9998 / 9999 / 6006]
#   MSR_AS_USER=1  run as your user instead of root (files written to the mount stay yours)
#   MSR_NETWORK=host  use the host network (no port publishing needed). Use it when downloads
#                  inside the container are very slow (Docker bridge/MTU problem, common on WSL)
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${MSR_CONTAINER:-msr}"
IMAGE="${MSR_IMAGE:-mural-structure:latest}"
MOUNT="${MSR_MOUNT:-$(dirname "$REPO")}"
GPUS="${MSR_GPUS:-all}"
SHM="${MSR_SHM:-32g}"

if docker container inspect "$NAME" > /dev/null 2>&1; then
    if [ "$(docker container inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
        docker start "$NAME" > /dev/null
        echo "Started existing container '$NAME'."
    else
        echo "Container '$NAME' is already running."
    fi
else
    if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
        echo "Building image $IMAGE (first time only, takes a few minutes)..."
        docker build --network=host -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO"
    fi
    args=(-d -it --name "$NAME" --shm-size "$SHM" --ulimit memlock=-1
          -v "$MOUNT:/master" -w "/master/$(basename "$REPO")")
    if [ "$MSR_NETWORK" = "host" ]; then
        args+=(--network host)       # servers listen directly on the host's ports 9998 / 9999 / 6006
    else
        args+=(-p "${MSR_EVAL_PORT:-9998}:9998" -p "${MSR_INFER_PORT:-9999}:9999" -p "${MSR_TB_PORT:-6006}:6006")
    fi
    [ "$GPUS" != "none" ] && args+=(--gpus "$GPUS")
    [ -n "$MSR_DATA" ] && args+=(-v "$MSR_DATA:/data")
    [ "$MSR_AS_USER" = "1" ] && args+=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
    docker run "${args[@]}" "$IMAGE" bash > /dev/null
    echo "Created container '$NAME' from $IMAGE."
fi

echo "  host folder : $MOUNT  ->  /master"
echo "  repository  : /master/$(basename "$REPO")"
echo "  open a shell: docker exec -it $NAME bash"

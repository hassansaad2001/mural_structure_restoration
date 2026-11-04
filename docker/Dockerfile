# Environment for training / testing / evaluation / inference servers.
# The code is NOT copied into the image: docker/run.sh mounts the folder that contains
# this repository at /master, so edits on the host are used immediately.
#
#   docker build --network=host -t mural-structure:latest -f docker/Dockerfile .   (from the repo root)
#   bash docker/run.sh                                                              (builds it if missing)
#
# --network=host: Docker's bridge network can be very slow on some machines (e.g. WSL, MTU
# mismatch); the host network avoids that and changes nothing else for the build.

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda

# libglib2.0-0: OpenCV; git/wget: convenience inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Versions known to work with torch 1.13.1 (numpy must stay < 2).
# The pip cache lives in a BuildKit cache mount: an interrupted build does not download again,
# and nothing of it ends up in the image.
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-docker.txt && rm /tmp/requirements-docker.txt

WORKDIR /master
CMD ["bash"]

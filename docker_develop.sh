#!/bin/bash

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <trained_model_location> <scene_location> <port> <ip> [additional host_render_server.py flags]"
  exit 1
fi

TRAINED_MODEL_LOCATION="$1"
SCENE_LOCATION="$2"
PORT="${3:-6009}"
IP="${4:-127.0.0.1}"
shift 4

# --user $(id -u):$(id -g) \

# Added user flag to allow VSCode development at the same time (prevents file permissions from getting overrun)
# (right now it treats the group user as 'nnamdiobi' instead of root which isn't a huge problem)
# Changed rm -r to -rf to allow for forcing deletion of EVER directory (since it's now locally owned)
docker run --rm --gpus all -it \
  -v /tmp/NVIDIA:/tmp/NVIDIA \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -v "$TRAINED_MODEL_LOCATION":/data/trained_model \
  -v "$SCENE_LOCATION":/data/scene \
  -p "$IP:$PORT:$PORT" \
  -v "$(pwd)":/ever_training2 \
  obinnamdi/ever:v1_full_build


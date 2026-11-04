#!/bin/bash
# Predict the missing structure for a folder of test samples with a trained checkpoint.
# Usage: scripts/test_example.sh <checkpoint.pth> <images_dir> <edges_dir> <masks_dir> <out_dir>
# Masks: white = hole. Edges: black lines on white background.
set -e
python generate_edges.py \
    --network="$1" \
    --dpath="$2" --epath="$3" --mpath="$4" \
    --outdir="$5" \
    --resolution=512 --save-raw

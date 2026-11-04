#!/bin/bash
# Baseline: ZITS TSR (edge only) + SSU, same data / masks / validation as scripts/train_example.sh.
# Run from the repository root. Recipe and deviations: docs/BASELINES.md.
set -e
DATA_ROOT=${DATA_ROOT:-/master/DhMurals-inpainting-dataset}

python train_tsr.py --preset=zits_tsr \
    --outdir=outputs/baselines \
    --gpus=4 --batch-gpu=8 --workers=6 \
    --kimg=1000 --snap=10 --seed=42 --allow-tf32=True \
    --data-edge="$DATA_ROOT/train/edges_512crops" \
    --data-image="$DATA_ROOT/train/images_512crops" \
    --mask-path="$DATA_ROOT/train/masks" --mask-aug=True --mirror=True \
    --face-boxes="$DATA_ROOT/face_boxes_train.json" --face-mask-prob=0.5 --face-oversample=6 \
    --data-val-edge="$DATA_ROOT/val/edges_512crops" \
    --data-val-image="$DATA_ROOT/val/images_512crops" \
    --val-mask-path="$DATA_ROOT/val/masks_heldout" \
    --val-face-boxes="$DATA_ROOT/face_boxes_val.json" \
    "$@"

#!/bin/bash
# Training configuration of the final model (run 00094: Swin stages after FFC blocks 1 and 3,
# LaMa discriminator, BCE + Dice + Hausdorff loss, face/hand holes).
# Run from the repository root. Set DATA_ROOT to your dataset folder (see docs/TRAINING.md).
set -e
DATA_ROOT=${DATA_ROOT:-/master/DhMurals-inpainting-dataset}

python train_tsr.py \
    --outdir=outputs/tsr \
    --gpus=4 --batch=16 --batch-gpu=2 --workers=6 \
    --kimg=1000 --snap=10 --seed=42 --allow-tf32=True \
    --data-edge="$DATA_ROOT/train/edges_512crops" \
    --data-image="$DATA_ROOT/train/images_512crops" \
    --mask-path="$DATA_ROOT/train/masks" --mask-aug=True --mirror=True \
    --face-boxes="$DATA_ROOT/face_boxes_train.json" --face-mask-prob=0.5 --face-oversample=6 \
    --data-val-edge="$DATA_ROOT/val/edges_512crops" \
    --data-val-image="$DATA_ROOT/val/images_512crops" \
    --val-mask-path="$DATA_ROOT/val/masks_heldout" \
    --val-face-boxes="$DATA_ROOT/face_boxes_val.json" \
    --swin-after=1,3 --swin-dim=256 --dec-norm=none \
    --discriminator=lama --d-norm=sn --d-layers=5 --d-mask-input=True \
    --d-blur=0.7 --d-fake-mode=legacy \
    --lr=0.0005 --lr-d=0.0005 --g-beta1=0.5 --d-beta1=0.5 \
    --lr-decay-start=200 --lr-min-ratio=0.05 --gamma=1 \
    --loss-mode=bce_dice_hd --bce-weight=1 --dice-weight=1 --hd-weight=0.5 \
    --adv-weight=1 --fm-weight=10 --bce-pos-weight=3 \
    "$@"

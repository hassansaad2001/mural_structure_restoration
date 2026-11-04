# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Write the validation FACE holes as mask PNGs, so the eval GUI can use them.

The holes are generated exactly like the training code's face validation bank
(datasets/dataset_tsr_512.py: TSRDataset.get_eval_sample(..., face=True)):
same crop order, same seeds, same shape generator. So GUI numbers on these
masks are comparable to the [Metrics FACE ...] lines in training logs.

Output: <out>/<crop_stem>_face0.png, _face1.png, ...  (white = hole, 512x512)
The GUI pairs them with the images by name (Mask pairing: auto / by_name).

Needs only numpy, opencv-python, Pillow (no torch), so it runs on the host. Example:
  python export_face_masks.py \
    --edges  <DATA_ROOT>/val/edges_512crops \
    --images <DATA_ROOT>/val/images_512crops \
    --boxes  <DATA_ROOT>/face_boxes_val.json \
    --out    <DATA_ROOT>/val/masks_face \
    --k 3
"""
import argparse
import json
import os
import random

import cv2
import numpy as np
import PIL.Image

PIL.Image.init()
EXTS = set(k.lower() for k in PIL.Image.EXTENSION.keys())


def find_files(path):
    return sorted(os.path.relpath(os.path.join(r, f), start=path)
                  for r, _d, fs in os.walk(path) for f in fs if os.path.splitext(f)[1].lower() in EXTS)


def face_hole(res, boxes, rng):
    # identical to TSRDataset._face_hole
    x0, y0, x1, y1 = boxes[rng.randrange(len(boxes))]
    bw, bh = x1 - x0, y1 - y0
    hole = np.zeros((res, res), np.uint8)
    if rng.random() < 0.5:
        s = rng.uniform(1.0, 1.25)
        cx, cy, ax, ay = (x0 + x1) / 2, (y0 + y1) / 2, s * bw / 2, s * bh / 2
    else:
        f = rng.uniform(0.25, 0.55)
        ax, ay = f * bw / 2, f * bh * rng.uniform(0.6, 1.2) / 2
        cx = rng.uniform(x0 + ax, max(x0 + ax, x1 - ax))
        cy = rng.uniform(y0 + ay, max(y0 + ay, y1 - ay))
    pts = []
    for i in range(16):
        t = 2 * np.pi * i / 16
        r = rng.uniform(0.8, 1.15)
        pts.append((cx + r * ax * np.cos(t), cy + r * ay * np.sin(t)))
    cv2.fillPoly(hole, [np.round(np.array(pts)).astype(np.int32)], 1)
    return hole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--boxes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=3, help='holes per image (= --val-masks-per-image)')
    ap.add_argument('--res', type=int, default=512)
    a = ap.parse_args()

    # same pairing / order as TSRDataset (edges are the source of truth)
    img_stems = {os.path.splitext(os.path.basename(f))[0] for f in find_files(a.images)}
    edges = [f for f in find_files(a.edges) if os.path.splitext(os.path.basename(f))[0] in img_stems]

    boxes = {}
    for k, v in json.load(open(a.boxes)).items():
        uniq = []
        for b in v:
            b = [float(t) for t in b]
            if b not in uniq:
                uniq.append(b)
        if uniq:
            boxes[os.path.splitext(os.path.basename(k))[0]] = uniq

    os.makedirs(a.out, exist_ok=True)
    n = 0
    for raw_idx, f in enumerate(edges):
        stem = os.path.splitext(os.path.basename(f))[0]
        bx = [b for b in boxes.get(stem, [])
              if min(b[2], a.res) - max(b[0], 0) >= 8 and min(b[3], a.res) - max(b[1], 0) >= 8]
        bx = [(max(0.0, b[0]), max(0.0, b[1]), min(float(a.res), b[2]), min(float(a.res), b[3])) for b in bx]
        if not bx:
            continue
        for k in range(a.k):
            mseed = 1_000_003 * (k + 1) + raw_idx
            hole = face_hole(a.res, bx, random.Random(mseed))
            PIL.Image.fromarray(hole * 255).save(os.path.join(a.out, f'{stem}_face{k}.png'))
            n += 1
    print(f'Wrote {n} face masks for {n // max(1, a.k)} images to {a.out}')


if __name__ == '__main__':
    main()

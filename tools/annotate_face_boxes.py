# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Quick face/hand box annotator for 512x512 mural crops (OpenCV, resumable).

Usage
-----
  # annotate (only crops whose name you list, or all crops in the folder)
  python annotate_face_boxes.py --images <DATA_ROOT>/train/images_512crops \
         --out face_boxes_train.json

  # check what the training holes will look like
  python annotate_face_boxes.py --images ... --out face_boxes_train.json --preview 20

Keys in the window (cv2.selectROIs):
  drag a box, SPACE/ENTER to accept it, draw more boxes, ESC when done with
  this image (ESC with no box = no face/hand on this crop).
  Before an image opens you can type 's' + ENTER in the terminal to skip it,
  'q' + ENTER to save and quit.

Tips
----
  * Box whole faces and hands (incl. small figures). Eyes/mouths are sampled
    automatically as 25-55 % sub-regions of each box during training.
  * 300-600 boxed TRAINING crops are enough for face_mask_prob=0.3.
  * Annotate the VALIDATION crops separately (--out face_boxes_val.json) and
    pass it with --val-face-boxes to get FACE metrics for the paper.
  * Tip for speed: first list crops that likely contain figures (e.g. sort by
    file name / source mural) and pass them with --list.
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np


def load_json(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_json(d, p):
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, p)


def face_hole(res, boxes, rng):
    # same generator as datasets/dataset_tsr_512.py :: TSRDataset._face_hole
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
    ap.add_argument('--images', required=True, help='folder of crops (gray or RGB)')
    ap.add_argument('--out', required=True, help='JSON to create / extend')
    ap.add_argument('--list', default=None, help='optional txt file with crop names to annotate')
    ap.add_argument('--preview', type=int, default=0, help='show N random generated face holes and exit')
    a = ap.parse_args()

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    names = sorted(f for f in os.listdir(a.images) if os.path.splitext(f)[1].lower() in exts)
    if a.list:
        wanted = {os.path.splitext(l.strip())[0] for l in open(a.list) if l.strip()}
        names = [n for n in names if os.path.splitext(n)[0] in wanted]
    boxes = load_json(a.out)

    if a.preview:
        keys = [k for k, v in boxes.items() if v]
        rng = random.Random(0)
        for k in rng.sample(keys, min(a.preview, len(keys))):
            fn = next((n for n in names if os.path.splitext(n)[0] == k), None)
            if fn is None:
                continue
            img = cv2.imread(os.path.join(a.images, fn))
            hole = face_hole(img.shape[0], boxes[k], rng)
            vis = img.copy()
            vis[hole > 0] = (0.4 * vis[hole > 0] + 0.6 * np.array([255, 120, 0])).astype(np.uint8)
            cv2.imshow('preview (any key = next, q = quit)', vis)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    todo = [n for n in names if os.path.splitext(n)[0] not in boxes]
    print(f'{len(names)} crops, {len(names) - len(todo)} already done, {len(todo)} to go.')
    for i, n in enumerate(todo):
        cmd = input(f'[{i + 1}/{len(todo)}] {n}  (ENTER=open, s=skip, q=quit) > ').strip().lower()
        if cmd == 'q':
            break
        key = os.path.splitext(n)[0]
        if cmd == 's':
            boxes[key] = []
            save_json(boxes, a.out)
            continue
        img = cv2.imread(os.path.join(a.images, n))
        rois = cv2.selectROIs('box faces/hands, ESC when done', img, showCrosshair=False)
        cv2.destroyAllWindows()
        boxes[key] = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h in np.array(rois).reshape(-1, 4)
                      if w > 4 and h > 4]
        save_json(boxes, a.out)
    n_pos = sum(1 for v in boxes.values() if v)
    print(f'Saved {a.out}: {n_pos} crops with boxes, {sum(len(v) for v in boxes.values())} boxes.')


if __name__ == '__main__':
    sys.exit(main())

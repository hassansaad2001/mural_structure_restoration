# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Shared fixtures: a tiny synthetic dataset and random checkpoints. CPU only.

Run inside the container (from the repository root):   python3 -m pytest tests -q
"""
import json
import os
import sys
import warnings

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')       # tests never use a GPU
warnings.filterwarnings('ignore')

RES = 512
MODELS = {   # name -> (generator class, extra G kwargs)
    'ours': ('networks.new_mat.GeneratorSketch', {}),
    'edgeconnect': ('networks.baselines.edgeconnect.EdgeConnectEdgeGenerator', {}),
    'sgtb_srn': ('networks.baselines.sgtb_srn.SGTBStructureGenerator', {}),
    'zits_tsr': ('networks.baselines.zits_tsr.ZITSTSREdge', {}),
}


def _sample(rng):
    """Gray mural-like image, its edge map (black strokes on white) and nothing else."""
    img = Image.fromarray((rng.random((RES // 8, RES // 8)) * 120 + 60).astype(np.uint8)).resize((RES, RES))
    edge = Image.new('L', (RES, RES), 255)
    di, de = ImageDraw.Draw(img), ImageDraw.Draw(edge)
    for _ in range(25):
        pts = [tuple(int(v) for v in rng.integers(0, RES, 2)) for _ in range(3)]
        di.line(pts, fill=int(rng.integers(0, 50)), width=3)
        de.line(pts, fill=0, width=2)
    return img, edge


def _mask(rng):
    """White = hole, 15-45 % of the image."""
    m = Image.new('L', (RES, RES), 0)
    d = ImageDraw.Draw(m)
    side = int(RES * np.sqrt(rng.uniform(0.15, 0.45)))
    x, y = rng.integers(0, RES - side, 2)
    d.rectangle([int(x), int(y), int(x) + side, int(y) + side], fill=255)
    return m


@pytest.fixture(scope='session')
def data_root(tmp_path_factory):
    """<root>/{train,val}/{images,edges}/*.jpg and <root>/masks/*.png (same layout as the real data)."""
    root = tmp_path_factory.mktemp('data')
    rng = np.random.default_rng(0)
    for split, n in (('train', 4), ('val', 3)):
        for sub in ('images', 'edges'):
            (root / split / sub).mkdir(parents=True)
        for i in range(n):
            img, edge = _sample(rng)
            img.save(root / split / 'images' / f'{split}{i:02d}.jpg', quality=95)
            edge.save(root / split / 'edges' / f'{split}{i:02d}.jpg', quality=95)    # JPEG -> soft edges
    (root / 'masks').mkdir()
    for i in range(6):
        _mask(rng).save(root / 'masks' / f'{i:05d}.png')
    return root


def make_generator(name):
    import dnnlib
    cls, extra = MODELS[name]
    return dnnlib.util.construct_class_by_name(class_name=cls, img_resolution=RES, **extra)


@pytest.fixture(scope='session')
def checkpoints(tmp_path_factory):
    """name -> path of a checkpoint-000000.pth with random weights + training_options.json."""
    import torch
    out = {}
    for name, (cls, extra) in MODELS.items():
        d = tmp_path_factory.mktemp(f'run_{name}')
        torch.manual_seed(0)
        G = make_generator(name)
        torch.save({'G_ema_state': G.state_dict()}, d / 'checkpoint-000000.pth')
        (d / 'training_options.json').write_text(
            json.dumps({'G_kwargs': dict(class_name=cls, img_resolution=RES, **extra)}))
        out[name] = d / 'checkpoint-000000.pth'
    return out

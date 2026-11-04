# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""
eval_core.py  –  Checkpoint evaluation engine for MAT sketch / edge inpainting.

Used by eval_checkpoints_gui.py (GUI and --headless mode). No tkinter here.

Pipeline
--------
  EvalSettings            one dataclass with every path / threshold / boundary,
                          shared by the GUI and the worker, written to the
                          Settings sheet.
  discover_samples()      pairs images <-> edges <-> masks (see "Mask pairing").
  SampleFilter            rejects samples whose hole contains no useful edges.
                          Runs once per (data, filter settings), independent of
                          any checkpoint, cached on disk in
                          <results>/.filter_cache/<key>.json.
  MaskGrouper             splits accepted samples into Small / Medium / Large
                          (fixed ranges or tertiles) + 'Out of range'.
  CheckpointEvaluator     runs every checkpoint on the SAME accepted samples,
                          per-image metrics + per-group FID.
  export_results()        Excel workbook, CSVs, per-group comparison grids.

Conventions
-----------
  * mask : 1 = hole, 0 = valid (after applying `mask_white_is_hole`).
  * edge : 1 = edge, 0 = background (after applying `edge_black_on_white`).
  * mask_ratio = hole_area / (res * res) at the evaluation resolution.
  * Prediction composite = GT edge outside the hole, prediction inside.

Mask pairing (assumption, see `mask_pairing`)
---------------------------------------------
  'by_name' : a mask belongs to an image if its stem equals the image stem or
              starts with '<image stem>' + one of '_', '-', '.'
              (e.g. img01.png <- img01.png, img01_0.png, img01-mask3.png).
              Several masks per image -> several samples. A mask that matches
              several image stems goes to the longest one.
  'pool'    : masks are a shared pool; each image gets `masks_per_image`
              masks drawn with a per-image seeded RNG (stable when images are
              added / removed).
  'auto'    : 'by_name' if at least one mask matches an image by name,
              otherwise 'pool'.
"""

import os
import sys
import csv
import json
import math
import time
import random
import hashlib
import dataclasses
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)      # networks/, dnnlib/, inference/ live here
for _p in (_HERE, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

GROUPS = ('Small', 'Medium', 'Large')
ALL = 'All'
OUT_OF_RANGE = 'Out of range'
SUMMARY_GROUPS = (ALL,) + GROUPS          # groups that get a summary table

# (key, label, higher_is_better)
METRICS = [
    ('psnr',  'PSNR',   True),
    ('ssim',  'SSIM',   True),
    ('l1',    'L1',     False),
    ('lpips', 'LPIPS',  False),
    ('fid',   'FID',    False),
    ('f1',    'F1',     True),
    ('f1_t2', 'F1@2px', True),
    ('f1_t5', 'F1@5px', True),
    # Structure metrics (same definitions as metrics/metrics_edge.py used in training):
    #   hd95 : symmetric 95th-percentile Hausdorff distance (px) between predicted and GT
    #          edge pixels inside the hole; NaN when either set is empty (excluded from means)
    #   conn : fraction of 8-connected GT strokes in the hole with >= 50 % of their pixels
    #          within 2 px of a predicted edge; NaN when the hole has no GT strokes
    ('hd95',  'HD95',   False),
    ('conn',  'Conn',   True),
]
CONN_TOL = 2
CONN_FRAC = 0.5
METRIC_LABEL = {k: lbl for k, lbl, _ in METRICS}
HIGHER_BETTER = {k: hb for k, _, hb in METRICS}
PER_IMAGE_METRICS = [k for k, _, _ in METRICS if k != 'fid']

REJECT_MISSING = 'missing file'
REJECT_EMPTY = 'empty hole'
REJECT_FEW = 'too few edges'
REJECT_DENSITY = 'low edge density'
REJECT_UNCONNECTED = 'edges not connected'
REJECT_REASONS = (REJECT_MISSING, REJECT_EMPTY, REJECT_FEW, REJECT_DENSITY, REJECT_UNCONNECTED)

PSNR_CAP = 100.0      # PSNR for a pixel-perfect sample (keeps means finite)
FILTER_CACHE_VERSION = 1

INCEPTION_URL = ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/'
                 'pretrained/metrics/inception-2015-12-05.pt')


class Cancelled(Exception):
    pass


class LoadError(RuntimeError):
    """A checkpoint could not be loaded into the selected network class."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class EvalSettings:
    # Paths
    image_dir: str = ''
    edge_dir: str = ''
    mask_dir: str = ''
    results_dir: str = ''
    checkpoints: list = field(default_factory=list)
    checkpoint_networks: list = field(default_factory=list)   # parallel to checkpoints; '' = auto
    manifest_path: str = ''              # fixed sample list (.json/.csv); '' = discover + filter

    # Client / server (GUI on the host -> eval_server.py in the container)
    server_addr: str = 'localhost:9998'
    path_map: str = ''                   # 'host_prefix=container_prefix[;...]'

    # Model / inference
    network: str = 'networks.new_mat.GeneratorSketch'
    resolution: int = 512
    batch_size: int = 8
    force_cpu: bool = False

    # Data conventions
    edge_black_on_white: bool = True     # edge files: black strokes on white
    mask_white_is_hole: bool = True      # mask files: white = hole
    fit_mode: str = 'center_crop'        # 'center_crop' | 'resize'
    mask_pairing: str = 'auto'           # 'auto' | 'by_name' | 'pool'
    masks_per_image: int = 1             # pool mode only
    seed: int = 0                        # pool mask assignment

    # Sample filter
    edge_threshold: float = 0.5
    min_edge_pixels: int = 20
    min_edge_density: float = 0.002
    require_connected: bool = False

    # Mask-ratio groups (percent)
    grouping_mode: str = 'fixed'         # 'fixed' | 'tertiles'
    small_lo: float = 10.0
    small_hi: float = 30.0
    medium_lo: float = 30.0
    medium_hi: float = 50.0
    large_lo: float = 50.0
    large_hi: float = 70.0

    # Metrics
    metric_region: str = 'hole'          # PSNR/SSIM/L1 over 'hole' or 'full'
    binarize_pred: bool = True           # threshold prediction at 0.5 in the composite
    compute_lpips: bool = True
    compute_fid: bool = True
    fid_min_group_size: int = 50

    # Comparison grids
    grid_per_group: int = 8
    grid_seed: int = 0
    grid_tile: int = 256

    # -- serialisation --
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # Fields that change which samples are accepted (and their stats).
    FILTER_FIELDS = ('image_dir', 'edge_dir', 'mask_dir', 'manifest_path', 'resolution',
                     'edge_black_on_white', 'mask_white_is_hole', 'fit_mode',
                     'mask_pairing', 'masks_per_image', 'seed',
                     'edge_threshold', 'min_edge_pixels', 'min_edge_density',
                     'require_connected')

    def filter_settings(self):
        return {k: getattr(self, k) for k in self.FILTER_FIELDS}

    def ckpt_specs(self):
        """[(checkpoint path, network override)]; override '' = auto (see resolve_network)."""
        nets = list(self.checkpoint_networks) + [''] * len(self.checkpoints)
        return [(p, nets[i] or '') for i, p in enumerate(self.checkpoints)]

    def validate(self, need_checkpoints=True):
        errs = []
        for name in ('image_dir', 'edge_dir', 'mask_dir'):
            p = getattr(self, name)
            if self.manifest_path and not p:
                continue                 # a sample list may use absolute paths only
            if not p or not os.path.isdir(p):
                errs.append(f'{name}: folder not found ({p!r})')
        if self.manifest_path and not os.path.isfile(self.manifest_path):
            errs.append(f'sample list not found: {self.manifest_path}')
        if not self.results_dir:
            errs.append('results_dir is empty')
        if need_checkpoints:
            if not self.checkpoints:
                errs.append('no checkpoints selected')
            for c in self.checkpoints:
                if not os.path.isfile(c):
                    errs.append(f'checkpoint not found: {c}')
        if self.resolution < 16:
            errs.append('resolution must be >= 16')
        if self.grouping_mode == 'fixed':
            try:
                group_bounds(self)
            except ValueError as e:
                errs.append(str(e))
        return errs


# ---------------------------------------------------------------------------
# Client <-> server protocol (same framing as inference_server.py)
#   Client -> Server : {'cmd': 'ping'|'filter'|'preview'|'evaluate'|'sample', 'settings': dict,
#                       'sample': accepted-sample dict (only for 'sample')}
#                      then optionally {'cmd': 'cancel'} while the job runs.
#   Server -> Client : any number of {'type': 'log', 'msg'} / {'type': 'progress', 'frac', 'text'},
#                      then exactly one {'type': 'result', ...} | {'type': 'error', 'msg', 'tb'}
#                      | {'type': 'cancelled'}.
# ---------------------------------------------------------------------------
import pickle
import struct

_HEADER = struct.Struct('>I')
DEFAULT_EVAL_PORT = 9998


def _recv_all(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('Connection closed')
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(_HEADER.pack(len(data)) + data)


def recv_msg(sock):
    (length,) = _HEADER.unpack(_recv_all(sock, _HEADER.size))
    return pickle.loads(_recv_all(sock, length))


def parse_addr(addr, default_port=DEFAULT_EVAL_PORT):
    if ':' in addr:
        host, port = addr.rsplit(':', 1)
        return host, int(port)
    return addr, default_port


# ---------------------------------------------------------------------------
# host <-> container path mapping
# ---------------------------------------------------------------------------
PATH_FIELDS = ('image_dir', 'edge_dir', 'mask_dir', 'results_dir', 'manifest_path')


def parse_path_map(spec):
    """'a=b;c=d' -> [(a, b), (c, d)] (prefixes without trailing '/')."""
    pairs = []
    for part in (spec or '').split(';'):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f'Path map entry must look like HOST_PREFIX=CONTAINER_PREFIX: {part!r}')
        a, b = part.split('=', 1)
        pairs.append((a.strip().rstrip('/') or '/', b.strip().rstrip('/') or '/'))
    return pairs


def map_path(p, pairs, reverse=False):
    """Translate host -> container (or back with reverse=True); longest prefix wins."""
    if not p:
        return p
    cands = [(b, a) if reverse else (a, b) for a, b in pairs]
    for src, dst in sorted(cands, key=lambda c: len(c[0]), reverse=True):
        if p == src or p.startswith(src.rstrip('/') + '/'):
            return dst + p[len(src):]
    return p


def settings_for_server(s):
    """Copy of the settings with every path translated to the container side."""
    pairs = parse_path_map(s.path_map)
    d = s.to_dict()
    for k in PATH_FIELDS:
        d[k] = map_path(d[k], pairs)
    d['checkpoints'] = [map_path(c, pairs) for c in d['checkpoints']]
    return EvalSettings.from_dict(d)


# ---------------------------------------------------------------------------
# Network class per checkpoint
# ---------------------------------------------------------------------------
_G_KWARG_SKIP = ('class_name', 'img_resolution')


def training_options_for(ckpt_path):
    """training_options.json written by train_tsr.py next to the checkpoints (or None)."""
    p = os.path.join(os.path.dirname(ckpt_path), 'training_options.json')
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def resolve_network(ckpt_path, override, default):
    """-> (class_name, extra_kwargs, source).

    override (per checkpoint)    -> used as is ('manual')
    else training_options.json   -> G_kwargs.class_name ('training_options.json')
    else settings.network        -> ('default')
    extra_kwargs (e.g. dec_norm) come from training_options.json, only when its
    class is the one being built.
    """
    g = (training_options_for(ckpt_path) or {}).get('G_kwargs') or {}
    trained = g.get('class_name')
    if override:
        cls, src = override, 'manual'
    elif trained:
        cls, src = trained, 'training_options.json'
    else:
        cls, src = default, 'default'
    extra = {k: v for k, v in g.items() if k not in _G_KWARG_SKIP} if trained == cls else {}
    return cls, extra, src


def checkpoint_names(paths):
    """Short unique display names = run id of the training folder.

    .../00073-edges_512crops-.../checkpoint-001000.pth  ->  '00073'
    several checkpoints of the same run                 ->  '00073_1', '00073_2', … (list order)
    folder without a leading number                     ->  '<folder>/<file stem>'
    """
    import re
    bases = []
    for p in paths:
        parent = os.path.basename(os.path.dirname(os.path.normpath(p)))
        m = re.match(r'\d+', parent)
        bases.append(m.group() if m else f'{parent}/{os.path.splitext(os.path.basename(p))[0]}')
    total = {b: bases.count(b) for b in bases}
    seen, out = {}, []
    for b in bases:
        seen[b] = seen.get(b, 0) + 1
        out.append(b if total[b] == 1 else f'{b}_{seen[b]}')
    return out


def checkpoint_file_label(path):
    """'<run folder>/<file stem>' — shown next to the short name."""
    return f'{os.path.basename(os.path.dirname(os.path.normpath(path)))}/' \
           f'{os.path.splitext(os.path.basename(path))[0]}'


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------
def _list_images(d):
    if not d or not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                  and os.path.isfile(os.path.join(d, f)))


def _stem(f):
    return os.path.splitext(f)[0]


def _match_masks_by_name(image_stems, mask_files):
    """Return {image_stem: [mask files]} using the longest matching image stem."""
    stems = sorted(set(image_stems), key=len, reverse=True)
    stem_set = set(stems)
    out = {}
    for mf in mask_files:
        ms = _stem(mf)
        owner = ms if ms in stem_set else None
        if owner is None:
            for s in stems:                      # longest first
                if len(ms) > len(s) and ms.startswith(s) and ms[len(s)] in '_-.':
                    owner = s
                    break
        if owner is not None:
            out.setdefault(owner, []).append(mf)
    return out


def discover_samples(s):
    """Pair images, edges and masks (or take them from the sample list).

    Returns (samples, pairing_used). Each sample is a dict:
      id, image_name, mask_name, image, edge, mask (abs paths or None), missing (str)
    """
    if s.manifest_path:
        return manifest_samples(s), 'manifest'
    images = _list_images(s.image_dir)
    edges = {_stem(f): f for f in _list_images(s.edge_dir)}
    masks = _list_images(s.mask_dir)

    by_name = _match_masks_by_name([_stem(f) for f in images], masks)
    pairing = s.mask_pairing
    if pairing == 'auto':
        pairing = 'by_name' if by_name else 'pool'

    samples = []
    for img in images:
        st = _stem(img)
        edge_f = edges.get(st)
        if pairing == 'by_name':
            mlist = sorted(by_name.get(st, []))
        else:
            k = max(1, int(s.masks_per_image))
            rng = random.Random(f'{s.seed}:{st}')
            mlist = sorted(rng.sample(masks, min(k, len(masks)))) if masks else []

        base = dict(image_name=img,
                    image=os.path.join(s.image_dir, img),
                    edge=os.path.join(s.edge_dir, edge_f) if edge_f else None)
        if not mlist:
            samples.append(dict(base, id=st, mask_name='', mask=None,
                                missing='no matching mask'))
            continue
        for mf in mlist:
            samples.append(dict(base, id=f'{st}__{_stem(mf)}', mask_name=mf,
                                mask=os.path.join(s.mask_dir, mf),
                                missing='' if edge_f else 'no edge map'))
    return samples, pairing


# ---------------------------------------------------------------------------
# Sample lists (manifests): fixed image / edge / mask triplets
# ---------------------------------------------------------------------------
#   JSON (primary): {'format', 'version', 'created', 'image_dir', 'edge_dir', 'mask_dir',
#                    'processed', 'settings': {...}, 'group_bounds': {...},
#                    'samples': [{'id', 'image', 'edge', 'mask',            <- relative to the dirs
#                                 'image_path', 'edge_path', 'mask_path',   <- absolute (fallback)
#                                 'mask_ratio', 'group', 'edge_pixels_in_hole', 'edge_density'}]}
#   CSV (for other tools, also loadable): same per-sample fields as columns.
MANIFEST_FORMAT = 'mat_sketch_eval_samples'
MANIFEST_VERSION = 1
# Settings stored with a sample list and restored when it is loaded.
MANIFEST_SETTINGS = ('resolution', 'fit_mode', 'edge_black_on_white', 'mask_white_is_hole',
                     'edge_threshold', 'min_edge_pixels', 'min_edge_density', 'require_connected',
                     'mask_pairing', 'masks_per_image', 'seed', 'grouping_mode',
                     'small_lo', 'small_hi', 'medium_lo', 'medium_hi', 'large_lo', 'large_hi')
MANIFEST_CSV_COLUMNS = ['sample_id', 'group', 'mask_ratio', 'edge_pixels_in_hole', 'edge_density',
                        'image', 'edge', 'mask', 'image_path', 'edge_path', 'mask_path']


def read_manifest(path):
    """Read a .json sample list (written by this tool) or a .csv with columns
    image/edge/mask (relative to the folders) and/or image_path/edge_path/mask_path."""
    if path.lower().endswith('.json'):
        with open(path) as f:
            m = json.load(f)
        if m.get('format') != MANIFEST_FORMAT:
            raise ValueError(f'{path}: not a sample list written by this tool')
        return m
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path}: no rows')
    for k in ('image', 'edge', 'mask'):
        if k not in rows[0] and f'{k}_path' not in rows[0]:
            raise ValueError(f'{path}: needs a column "{k}" or "{k}_path"')
    samples = []
    for r in rows:
        e = {k: (r.get(k) or '').strip() for k in ('image', 'edge', 'mask',
                                                   'image_path', 'edge_path', 'mask_path')}
        img, msk = e['image'] or e['image_path'], e['mask'] or e['mask_path']
        e['id'] = (r.get('sample_id') or '').strip() or \
            f'{_stem(os.path.basename(img))}__{_stem(os.path.basename(msk))}'
        samples.append(e)
    return {'format': MANIFEST_FORMAT, 'version': MANIFEST_VERSION, 'samples': samples,
            'settings': {}, 'image_dir': '', 'edge_dir': '', 'mask_dir': '', 'processed': False}


def manifest_samples(s):
    """Samples of s.manifest_path. Relative entries are joined with the current folders,
    absolute ones are translated with s.path_map (host -> container on the server)."""
    m = read_manifest(s.manifest_path)
    pairs = parse_path_map(s.path_map)
    bases = {'image': s.image_dir, 'edge': s.edge_dir, 'mask': s.mask_dir}
    out, seen = [], {}
    for e in m['samples']:
        smp, missing = {}, []
        for k in ('image', 'edge', 'mask'):
            cands = []
            for v in (e.get(k), e.get(f'{k}_path')):
                if v:
                    cands.append(map_path(v, pairs) if os.path.isabs(v) else os.path.join(bases[k], v))
            p = next((c for c in cands if os.path.isfile(c)), cands[0] if cands else None)
            smp[k] = p
            if not p or not os.path.isfile(p):
                missing.append(f'no {k}')
        sid = str(e.get('id') or len(out))
        seen[sid] = seen.get(sid, 0) + 1
        smp['id'] = sid if seen[sid] == 1 else f'{sid}#{seen[sid]}'
        smp['image_name'] = os.path.basename(smp['image'] or '')
        smp['mask_name'] = os.path.basename(smp['mask'] or '')
        smp['missing'] = ', '.join(missing)
        out.append(smp)
    return out


def _rel_or_abs(p, base):
    if p and base:
        try:
            r = os.path.relpath(p, base)
            if not r.startswith('..'):
                return r
        except ValueError:
            pass
    return p or ''


def _write_manifest(json_path, settings_dict, dirs, rows, processed, group_bounds):
    meta = {'format': MANIFEST_FORMAT, 'version': MANIFEST_VERSION,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'processed': processed,
            'image_dir': dirs['image'], 'edge_dir': dirs['edge'], 'mask_dir': dirs['mask'],
            'settings': {k: settings_dict[k] for k in MANIFEST_SETTINGS},
            'group_bounds': {g: [b[0] * 100, b[1] * 100] for g, b in (group_bounds or {}).items()},
            'samples': rows}
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=1)
    csv_path = os.path.splitext(json_path)[0] + '.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_CSV_COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, sample_id=r['id']))
    return json_path, csv_path


def save_sample_list(json_path, s, fr, group_info, to_local=lambda p: p):
    """Write the accepted samples of `fr` as <json_path> + <same name>.csv.

    to_local translates the sample paths in `fr` (container side when the filter ran
    on the server) to the paths of the machine writing the list."""
    dirs = {'image': s.image_dir, 'edge': s.edge_dir, 'mask': s.mask_dir}
    rows = []
    for i, a in enumerate(fr.accepted):
        r = {'id': a['id'], 'group': group_info.labels[i] if group_info else '',
             'mask_ratio': a['mask_ratio'], 'edge_pixels_in_hole': a['edge_pixels_in_hole'],
             'edge_density': a['edge_density']}
        for k in ('image', 'edge', 'mask'):
            p = to_local(a[k])
            r[f'{k}_path'] = p
            r[k] = _rel_or_abs(p, dirs[k])
        rows.append(r)
    return _write_manifest(json_path, s.to_dict(), dirs, rows, processed=False,
                           group_bounds=group_info.bounds if group_info else None)


def _fit_rgb(img, s):
    arr = np.array(img.convert('RGB'))
    return np.stack([_fit(arr[..., c], s.resolution, s.fit_mode, Image.BICUBIC) for c in range(3)], -1)


def export_fixed_test_set(out_dir, s, fr, group_info, to_local=lambda p: p, progress=None, cancel=None):
    """Write the accepted samples as ready-to-use files at the evaluation resolution:

        out_dir/images/<id>.png   RGB crop (same crop/resize as the evaluation)
        out_dir/edges/<id>.png    edges, black strokes on white
        out_dir/masks/<id>.png    white (255) = hole, black (0) = valid
        out_dir/samples.json + samples.csv

    One triplet per sample (an image with several masks is written once per mask), so any
    other model can be run on exactly the same inputs. Returns the samples.json path."""
    dirs = {k: os.path.join(out_dir, f'{k}s') for k in ('image', 'edge', 'mask')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    rows = []
    n = len(fr.accepted)
    for i, a in enumerate(fr.accepted):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        smp = dict(a, image=to_local(a['image']), edge=to_local(a['edge']), mask=to_local(a['mask']))
        _, edge, hole = load_sample(smp, s)
        rgb = _fit_rgb(Image.open(smp['image']), s)
        name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in a['id']) + '.png'
        Image.fromarray(rgb).save(os.path.join(dirs['image'], name))
        Image.fromarray(((1.0 - edge) * 255).round().astype(np.uint8)).save(os.path.join(dirs['edge'], name))
        Image.fromarray((hole * 255).astype(np.uint8)).save(os.path.join(dirs['mask'], name))
        rows.append({'id': a['id'], 'group': group_info.labels[i] if group_info else '',
                     'mask_ratio': a['mask_ratio'], 'edge_pixels_in_hole': a['edge_pixels_in_hole'],
                     'edge_density': a['edge_density'],
                     'image': name, 'edge': name, 'mask': name,
                     'image_path': os.path.join(dirs['image'], name),
                     'edge_path': os.path.join(dirs['edge'], name),
                     'mask_path': os.path.join(dirs['mask'], name),
                     'source_image': a['image_name'], 'source_mask': a['mask_name']})
        if progress and (i % 10 == 0 or i == n - 1):
            progress((i + 1) / max(n, 1), f'Exporting {i + 1}/{n}')
    # Files are already at the evaluation resolution and in the default conventions.
    sd = dict(s.to_dict(), fit_mode='center_crop', edge_black_on_white=True, mask_white_is_hole=True)
    json_path, _ = _write_manifest(os.path.join(out_dir, 'samples.json'), sd, dirs, rows,
                                   processed=True,
                                   group_bounds=group_info.bounds if group_info else None)
    return json_path


def _fingerprint(samples):
    h = hashlib.sha1()
    seen = set()
    for smp in samples:
        for key in ('image', 'edge', 'mask'):
            p = smp.get(key)
            if not p or p in seen:
                continue
            seen.add(p)
            try:
                st = os.stat(p)
                h.update(f'{p}|{st.st_size}|{int(st.st_mtime)}\n'.encode())
            except OSError:
                h.update(f'{p}|missing\n'.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Sample loading (identical for filter, evaluation and grids)
# ---------------------------------------------------------------------------
def _fit(arr, res, mode, resample):
    """Bring an HxW uint8 array to res x res (center crop w/ reflect pad, or resize)."""
    if arr.shape == (res, res):
        return arr
    if mode == 'resize':
        return np.array(Image.fromarray(arr).resize((res, res), resample))
    H, W = arr.shape
    ph, pw = max(0, res - H), max(0, res - W)
    if ph or pw:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), mode='symmetric')
        H, W = arr.shape
    y0, x0 = (H - res) // 2, (W - res) // 2
    return arr[y0:y0 + res, x0:x0 + res]


def load_sample(smp, s):
    """Return (gray uint8 HxW, edge float32 HxW in [0,1] with 1=edge, hole uint8 HxW 0/1)."""
    res = s.resolution
    img = Image.open(smp['image']).convert('L')
    edge = Image.open(smp['edge']).convert('L')
    if edge.size != img.size:
        edge = edge.resize(img.size, Image.BILINEAR)
    gray = _fit(np.array(img), res, s.fit_mode, Image.BICUBIC)
    edge = _fit(np.array(edge), res, s.fit_mode, Image.BILINEAR).astype(np.float32) / 255.0
    if s.edge_black_on_white:
        edge = 1.0 - edge

    m = Image.open(smp['mask']).convert('L')
    if m.size != (res, res):
        m = m.resize((res, res), Image.NEAREST)
    white = np.array(m) >= 128
    hole = (white if s.mask_white_is_hole else ~white).astype(np.uint8)
    return gray, edge, hole


# ---------------------------------------------------------------------------
# Connectivity helper (strict filter check)
# ---------------------------------------------------------------------------
def _label8(binary):
    try:
        import cv2
        _, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return lab
    except ImportError:
        pass
    try:
        from scipy.ndimage import label
        lab, _ = label(binary, structure=np.ones((3, 3), dtype=bool))
        return lab
    except ImportError:
        return None


def _edt_to_true(b):
    """Euclidean distance from every pixel to the nearest True pixel of b."""
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~b)
    except ImportError:
        import cv2
        return cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def structure_metrics(pr, gt, conn_tol=CONN_TOL, conn_frac=CONN_FRAC):
    """HD95 and connectivity for ONE image. pr, gt: bool HxW, already restricted to the hole."""
    n_pr, n_gt = int(pr.sum()), int(gt.sum())
    hd95 = conn = float('nan')
    if n_gt == 0:
        return hd95, conn
    dt_pr = _edt_to_true(pr) if n_pr else None
    if n_pr:
        dt_gt = _edt_to_true(gt)
        hd95 = float(max(np.percentile(dt_pr[gt], 95), np.percentile(dt_gt[pr], 95)))
    lab = _label8(gt)
    if lab is None:
        return hd95, conn
    lab_g = lab[gt]
    num = int(lab_g.max()) if lab_g.size else 0
    if num == 0:
        return hd95, conn
    sizes = np.bincount(lab_g, minlength=num + 1)[1:]
    if dt_pr is None:
        hits = np.zeros(num)
    else:
        near = (dt_pr[gt] <= conn_tol).astype(np.float64)
        hits = np.bincount(lab_g, weights=near, minlength=num + 1)[1:]
    valid = sizes > 0
    conn = float((hits[valid] >= conn_frac * sizes[valid]).mean()) if valid.any() else float('nan')
    return hd95, conn


def connected_hole_edge_pixels(edge_bin, hole_b):
    """Number of in-hole edge pixels 8-connected to an edge pixel in the valid region."""
    valid_edges = edge_bin & ~hole_b
    in_hole = edge_bin & hole_b
    if not valid_edges.any() or not in_hole.any():
        return 0
    lab = _label8(edge_bin)
    if lab is not None:
        good = np.unique(lab[valid_edges])
        good = good[good > 0]
        return int(np.isin(lab[in_hole], good).sum())
    # Fallback: geodesic reconstruction by repeated 3x3 dilation inside the edge map.
    reached = valid_edges.copy()
    while True:
        d = reached.copy()
        d[1:, :] |= reached[:-1, :]
        d[:-1, :] |= reached[1:, :]
        d2 = d.copy()
        d2[:, 1:] |= d[:, :-1]
        d2[:, :-1] |= d[:, 1:]
        new = d2 & edge_bin
        if np.array_equal(new, reached):
            break
        reached = new
    return int((reached & hole_b).sum())


# ---------------------------------------------------------------------------
# SampleFilter
# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    key: str
    pairing: str
    total: int
    accepted: list          # dicts: sample fields + stats
    rejected: list          # dicts: sample fields + stats + reason, detail
    settings: dict
    created: str = ''
    from_cache: bool = False

    @property
    def reason_counts(self):
        counts = {r: 0 for r in REJECT_REASONS}
        for r in self.rejected:
            counts[r['reason']] = counts.get(r['reason'], 0) + 1
        return counts

    def to_json(self):
        d = asdict(self)
        d.pop('from_cache')
        d['version'] = FILTER_CACHE_VERSION
        return d

    @classmethod
    def from_json(cls, d, from_cache=True):
        d = dict(d)
        d.pop('version', None)
        return cls(**d, from_cache=from_cache)


class SampleFilter:
    """Decide once per sample whether its hole holds enough edges to be evaluated."""

    def __init__(self, settings, log=print):
        self.s = settings
        self.log = log

    def cache_key(self, samples):
        payload = json.dumps(self.s.filter_settings(), sort_keys=True) + _fingerprint(samples)
        payload += '|' + '|'.join(smp['id'] for smp in samples)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _cache_path(self, key):
        return os.path.join(self.s.results_dir, '.filter_cache', f'{key}.json')

    def run(self, progress=None, cancel=None, use_cache=True):
        s = self.s
        samples, pairing = discover_samples(s)
        key = self.cache_key(samples)
        cpath = self._cache_path(key)
        if use_cache and os.path.isfile(cpath):
            try:
                with open(cpath) as f:
                    d = json.load(f)
                if d.get('version') == FILTER_CACHE_VERSION:
                    res = FilterResult.from_json(d)
                    self.log(f'[Filter] loaded from cache ({key}): '
                             f'{len(res.accepted)} accepted / {res.total} total')
                    return res
            except Exception as e:
                self.log(f'[Filter] cache unreadable, recomputing: {e}')

        self.log(f'[Filter] checking {len(samples)} samples (mask pairing: {pairing})')
        accepted, rejected = [], []
        total_px = float(s.resolution * s.resolution)
        fixed_list = (pairing == 'manifest')

        def _check(smp):
            stats = dict(mask_ratio=float('nan'), hole_area=0, edge_pixels_in_hole=0,
                         edge_density=float('nan'), connected_edge_pixels=None)
            if smp['missing']:
                return smp, stats, REJECT_MISSING, smp['missing']
            try:
                _, edge, hole = load_sample(smp, s)
            except Exception as e:
                return smp, stats, REJECT_MISSING, f'unreadable: {e}'
            hole_b = hole.astype(bool)
            edge_bin = edge > s.edge_threshold
            area = int(hole_b.sum())
            n_edge = int((edge_bin & hole_b).sum())
            stats.update(mask_ratio=area / total_px, hole_area=area,
                         edge_pixels_in_hole=n_edge,
                         edge_density=(n_edge / area) if area else 0.0)
            if area == 0:
                return smp, stats, REJECT_EMPTY, 'hole area is 0'
            if fixed_list:               # sample list: keep every sample, thresholds not re-applied
                return smp, stats, None, ''
            if n_edge < s.min_edge_pixels:
                return smp, stats, REJECT_FEW, f'{n_edge} < {s.min_edge_pixels} px'
            if stats['edge_density'] < s.min_edge_density:
                return smp, stats, REJECT_DENSITY, \
                    f'{stats["edge_density"]:.5f} < {s.min_edge_density}'
            if s.require_connected:
                n_conn = connected_hole_edge_pixels(edge_bin, hole_b)
                stats['connected_edge_pixels'] = n_conn
                if n_conn < s.min_edge_pixels:
                    return smp, stats, REJECT_UNCONNECTED, \
                        f'{n_conn} connected px < {s.min_edge_pixels}'
            return smp, stats, None, ''

        n = len(samples)
        with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2))) as ex:
            for i, (smp, stats, reason, detail) in enumerate(ex.map(_check, samples)):
                if cancel is not None and cancel.is_set():
                    ex.shutdown(cancel_futures=True)
                    raise Cancelled()
                row = {k: smp[k] for k in ('id', 'image_name', 'mask_name', 'image', 'edge', 'mask')}
                row.update(stats)
                if reason is None:
                    accepted.append(row)
                else:
                    row.update(reason=reason, detail=detail)
                    rejected.append(row)
                if progress and (i % 25 == 0 or i == n - 1):
                    progress((i + 1) / max(n, 1), f'Filtering {i + 1}/{n}')

        res = FilterResult(key=key, pairing=pairing, total=n, accepted=accepted,
                           rejected=rejected, settings=s.filter_settings(),
                           created=time.strftime('%Y-%m-%d %H:%M:%S'))
        try:
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            with open(cpath, 'w') as f:
                json.dump(res.to_json(), f)
        except OSError as e:
            self.log(f'[Filter] could not write cache: {e}')
        self.log(f'[Filter] {len(accepted)} accepted, {len(rejected)} rejected '
                 f'{ {k: v for k, v in res.reason_counts.items() if v} }')
        return res


# ---------------------------------------------------------------------------
# MaskGrouper
# ---------------------------------------------------------------------------
def group_bounds(s):
    """Fixed-mode bounds as fractions {group: (lo, hi)}; raises ValueError if invalid."""
    b = {'Small': (s.small_lo, s.small_hi),
         'Medium': (s.medium_lo, s.medium_hi),
         'Large': (s.large_lo, s.large_hi)}
    prev_hi = -1.0
    for g in GROUPS:
        lo, hi = b[g]
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(f'{g} range must satisfy 0 <= lo < hi <= 100 (got {lo}-{hi})')
        if lo < prev_hi:
            raise ValueError(f'{g} range overlaps the previous group')
        prev_hi = hi
    return {g: (lo / 100.0, hi / 100.0) for g, (lo, hi) in b.items()}


@dataclass
class GroupInfo:
    mode: str
    bounds: dict            # group -> (lo, hi) fractions (tertiles: actual min/max)
    members: dict           # group name (incl. All, Out of range) -> [indices into accepted]
    labels: list            # per accepted sample: group name
    warnings: list

    def count(self, g):
        return len(self.members.get(g, []))

    def ratio_stats(self, g, accepted):
        r = [accepted[i]['mask_ratio'] for i in self.members.get(g, [])]
        if not r:
            return dict(mean=float('nan'), min=float('nan'), max=float('nan'))
        return dict(mean=float(np.mean(r)), min=float(np.min(r)), max=float(np.max(r)))


class MaskGrouper:
    def __init__(self, settings):
        self.s = settings

    def assign(self, accepted):
        s = self.s
        n = len(accepted)
        ratios = [a['mask_ratio'] for a in accepted]
        labels = [OUT_OF_RANGE] * n

        if s.grouping_mode == 'tertiles':
            order = sorted(range(n), key=lambda i: (ratios[i], accepted[i]['id']))
            c1, c2 = round(n / 3), round(2 * n / 3)
            for rank, i in enumerate(order):
                labels[i] = GROUPS[0] if rank < c1 else GROUPS[1] if rank < c2 else GROUPS[2]
            bounds = {}
            for g in GROUPS:
                r = [ratios[i] for i in range(n) if labels[i] == g]
                bounds[g] = (min(r), max(r)) if r else (float('nan'), float('nan'))
        else:
            bounds = group_bounds(s)
            top = GROUPS[-1]
            for i, r in enumerate(ratios):
                for g in GROUPS:
                    lo, hi = bounds[g]
                    if lo <= r < hi or (g == top and r == hi):
                        labels[i] = g
                        break

        members = {ALL: list(range(n))}
        for g in GROUPS + (OUT_OF_RANGE,):
            members[g] = [i for i in range(n) if labels[i] == g]

        warnings = []
        for g in GROUPS:
            c = len(members[g])
            if c < s.fid_min_group_size:
                warnings.append(f'{g}: only {c} samples (< {s.fid_min_group_size}); '
                                f'FID is unreliable on small groups (still computed).')
        return GroupInfo(mode=s.grouping_mode, bounds=bounds, members=members,
                         labels=labels, warnings=warnings)


def select_grid_samples(group_info, s):
    """Fixed-seed random subset of each group: {group: [indices into accepted]}."""
    out = {}
    for g in GROUPS:
        m = sorted(group_info.members[g])
        rng = random.Random(f'{s.grid_seed}:{g}')
        k = min(max(0, int(s.grid_per_group)), len(m))
        out[g] = sorted(rng.sample(m, k)) if k else []
    return out


# ---------------------------------------------------------------------------
# Metric helpers (torch)
# ---------------------------------------------------------------------------
def _gauss_window(torch, device, size=11, sigma=1.5):
    xs = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def _ssim_map(torch, F, x, y, win, C1=0.01 ** 2, C2=0.03 ** 2):
    pad = win.shape[-1] // 2
    xp = F.pad(x, (pad,) * 4, mode='reflect')
    yp = F.pad(y, (pad,) * 4, mode='reflect')
    mu_x, mu_y = F.conv2d(xp, win), F.conv2d(yp, win)
    sxx = F.conv2d(xp * xp, win) - mu_x ** 2
    syy = F.conv2d(yp * yp, win) - mu_y ** 2
    sxy = F.conv2d(xp * yp, win) - mu_x * mu_y
    return ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))


def _disk(torch, r, device):
    yy, xx = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing='ij')
    return ((xx ** 2 + yy ** 2) <= r * r).float().to(device)[None, None]


def fid_from_features(f_fake, f_real):
    """Frechet distance between two feature sets (no scipy needed).

    With A, B the centred features (n1 x d, n2 x d): S1 = A'A/n1, S2 = B'B/n2 and the
    non-zero eigenvalues of S1 S2 are the squared singular values of A B' / sqrt(n1 n2), so
        trace sqrt(S1 S2) = nuclear_norm(A B') / sqrt(n1 n2)      (exact).
    When n <= d (typical: groups of tens/hundreds vs d=2048) this is an SVD of a small
    n1 x n2 matrix instead of square roots of d x d matrices (~1000x faster)."""
    if len(f_fake) < 2 or len(f_real) < 2:
        return float('nan')
    f1 = f_fake.astype(np.float64)
    f2 = f_real.astype(np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    A, B = f1 - mu1, f2 - mu2
    n1, n2, d = len(A), len(B), A.shape[1]
    tr1, tr2 = (A * A).sum() / n1, (B * B).sum() / n2
    if max(n1, n2) <= d:
        tr_sqrt = np.linalg.svd(A @ B.T, compute_uv=False).sum() / math.sqrt(n1 * n2)
    else:
        s1, s2 = A.T @ A / n1, B.T @ B / n2
        w, v = np.linalg.eigh(s1)
        sqrt_s1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
        m = sqrt_s1 @ s2 @ sqrt_s1
        tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(np.square(mu1 - mu2).sum() + tr1 + tr2 - 2 * tr_sqrt)


# ---------------------------------------------------------------------------
# Rendering helpers (grids)
# ---------------------------------------------------------------------------
HOLE_TINT = np.array([100, 150, 255], dtype=np.float32)


def _render_edges(edge01, hole=None, tint=0.25):
    """Black-on-white RGB uint8; optional blue tint over the hole."""
    g = ((1.0 - np.clip(edge01, 0, 1)) * 255.0).astype(np.float32)
    rgb = np.repeat(g[..., None], 3, axis=2)
    if hole is not None:
        h = hole.astype(np.float32)[..., None] * tint
        rgb = rgb * (1 - h) + h * HOLE_TINT
    return rgb.clip(0, 255).astype(np.uint8)


def _render_input(gray, hole):
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    hb = hole.astype(bool)
    rgb[hb, 0] = np.clip(rgb[hb, 0] + 80, 0, 255)
    rgb[hb, 1] *= 0.3
    rgb[hb, 2] *= 0.3
    return rgb.astype(np.uint8)


def _render_masked_edges(edge01, hole):
    rgb = _render_edges(edge01)
    rgb[hole.astype(bool)] = 200
    return rgb


def _to_tile(rgb, tile):
    """Downscale keeping thin dark strokes (block-min for integer factors)."""
    H = rgb.shape[0]
    if H == tile:
        return rgb
    if H > tile and H % tile == 0 and rgb.shape[1] == H:
        f = H // tile
        return rgb.reshape(tile, f, tile, f, 3).min(axis=(1, 3))
    return np.array(Image.fromarray(rgb).resize((tile, tile), Image.LANCZOS))


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def build_grid(rows, col_names, tile, label_w=170, header_h=34):
    """rows: [{'label': str, 'tiles': [HxWx3 uint8, ...]}] -> PIL RGB image."""
    ncol = len(col_names)
    W = label_w + ncol * tile
    H = header_h + max(1, len(rows)) * tile
    img = Image.new('RGB', (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f_head, f_lab = _font(13), _font(13)
    for c, name in enumerate(col_names):
        txt = name if len(name) <= 30 else '…' + name[-29:]
        dr.text((label_w + c * tile + 4, 8), txt, fill=(0, 0, 0), font=f_head)
    for r, row in enumerate(rows):
        y = header_h + r * tile
        dr.multiline_text((6, y + 8), row['label'], fill=(0, 0, 0), font=f_lab, spacing=4)
        for c, t in enumerate(row['tiles']):
            img.paste(Image.fromarray(t), (label_w + c * tile, y))
        dr.line([(0, y), (W, y)], fill=(180, 180, 180))
    for c in range(ncol + 1):
        x = label_w + c * tile
        dr.line([(x, 0), (x, H)], fill=(180, 180, 180))
    return img


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
@dataclass
class EvalResults:
    ckpt_names: list
    ckpt_paths: list
    per_image: dict          # ckpt name -> {metric: np.ndarray aligned with accepted}
    summary: dict            # group -> ckpt name -> {metric: float, 'n': int}
    grid_images: dict        # group -> PIL image
    notes: list
    device: str = ''
    ckpt_networks: list = field(default_factory=list)   # resolved class per checkpoint


MODEL_CACHE_SIZE = 8


class CheckpointEvaluator:
    def __init__(self, settings, filter_result, group_info, log=print, progress=None, cancel=None,
                 model_cache=None):
        self.s = settings
        self.fr = filter_result
        self.gi = group_info
        self.log = log
        self.progress = progress or (lambda f, t: None)
        self.cancel = cancel
        self.notes = []
        self.model_cache = model_cache       # OrderedDict -> keep models loaded between calls

    # -- setup --
    def _setup(self, need_fid=True):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        # device_count() first: avoids the CUDA "Error 2: out of memory" init bug (see train_tsr.py)
        have_cuda = torch.cuda.device_count() > 0 and torch.cuda.is_available()
        self.device = torch.device('cuda' if have_cuda and not self.s.force_cpu else 'cpu')
        self.log(f'[Eval] device: {self.device}')
        self.win = _gauss_window(torch, self.device)
        self.disks = {r: _disk(torch, r, self.device) for r in (2, 5)}

        self.lpips_fn = None
        if self.s.compute_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()
            except Exception as e:
                self._note(f'LPIPS disabled ({type(e).__name__}: {e}). Install with: pip install lpips')

        self.detector = None
        if self.s.compute_fid and need_fid:
            try:
                import dnnlib
                with dnnlib.util.open_url(INCEPTION_URL, verbose=False) as f:
                    self.detector = torch.jit.load(f).eval().to(self.device)
            except Exception as e:
                self._note(f'FID disabled: cannot load Inception detector ({type(e).__name__}: {e})')

    def _note(self, msg):
        if msg in self.notes:
            return
        self.notes.append(msg)
        self.log('[Eval] WARNING: ' + msg)

    def _check_cancel(self):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()

    def _batches(self, idx_list):
        """Yield (indices, gray, edge, hole) numpy batches with threaded loading."""
        bs = max(1, int(self.s.batch_size))
        acc = self.fr.accepted
        chunks = [idx_list[i:i + bs] for i in range(0, len(idx_list), bs)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = None
            for ci, chunk in enumerate(chunks):
                if futs is None:
                    futs = [ex.submit(load_sample, acc[i], self.s) for i in chunk]
                nxt = None
                if ci + 1 < len(chunks):        # prefetch next batch
                    nxt = [ex.submit(load_sample, acc[i], self.s) for i in chunks[ci + 1]]
                data = [f.result() for f in futs]
                yield (chunk,
                       np.stack([d[0] for d in data]),
                       np.stack([d[1] for d in data]),
                       np.stack([d[2] for d in data]))
                futs = nxt

    def _to_tensors(self, gray, edge, hole):
        t = self.torch
        img = t.from_numpy(gray).float().to(self.device)[:, None] / 127.5 - 1.0
        e = t.from_numpy(edge).float().to(self.device)[:, None]
        m = t.from_numpy(hole).float().to(self.device)[:, None]
        return img, e, m

    def _features(self, x01):
        """Inception features of black-on-white renderings of x01 (B,1,H,W) in [0,1]."""
        u8 = ((1.0 - x01.clamp(0, 1)) * 255.0).round().to(self.torch.uint8).repeat(1, 3, 1, 1)
        return self.detector(u8, return_features=True).float().cpu().numpy()

    def _load_generator(self, path, override=''):
        from inference.inference_server import load_generator
        cls, extra, src = resolve_network(path, override, self.s.network)
        cache = self.model_cache
        key = (path, cls, json.dumps(extra, sort_keys=True), self.s.resolution,
               os.path.getmtime(path), str(self.device))
        if cache is not None and key in cache:
            cache.move_to_end(key)
            return cache[key]
        self.log(f'[Eval]   network: {cls}  ({src})')
        name = os.path.basename(os.path.dirname(path))[:40] + '/' + os.path.basename(path)
        try:
            G, _ = load_generator(path, self.device, self.s.resolution, cls, extra)
        except RuntimeError as e:
            lines = [l.strip() for l in str(e).splitlines() if l.strip()]
            detail = '\n'.join(lines[:3]) + (f'\n… ({len(lines) - 3} more lines in the log)' if len(lines) > 3 else '')
            self.log(f'[Eval] full load error for {name}:\n{e}')
            raise LoadError(
                f'Checkpoint does not fit the network class.\n\n'
                f'Checkpoint: {name}\nNetwork:    {cls}  ({src})\n\n'
                f'Fix: double-click this checkpoint in tab 1 and choose the network it was '
                f'trained with.\n\n{detail}') from None
        missing, unexpected = getattr(G, 'load_report', ([], []))
        if missing or unexpected:
            self._note(f'{name}: {len(missing)} missing / {len(unexpected)} unexpected weights with {cls} '
                       f'(e.g. {(missing + unexpected)[:2]}) -> results are probably WRONG; '
                       f'choose the network class this checkpoint was trained with.')
        if cache is not None:
            cache[key] = G
            while len(cache) > MODEL_CACHE_SIZE:
                cache.popitem(last=False)
        return G

    def _forward(self, G, img, e, m):
        """-> (raw sigmoid prediction, composite)."""
        pred = self.torch.sigmoid(G(img, e, m))
        p = (pred > 0.5).float() if self.s.binarize_pred else pred
        return pred, e * (1 - m) + p * m

    def _predict(self, G, img, e, m):
        return self._forward(G, img, e, m)[1]

    # -- per-image metrics --
    def _metrics(self, comp, gt, m):
        t, F = self.torch, self.F
        region = m if self.s.metric_region == 'hole' else t.ones_like(m)
        n_reg = region.sum(dim=[1, 2, 3]).clamp(min=1.0)
        diff = (comp - gt) * region
        l1 = diff.abs().sum(dim=[1, 2, 3]) / n_reg
        mse = (diff ** 2).sum(dim=[1, 2, 3]) / n_reg
        psnr = t.where(mse > 0, 10.0 * t.log10(1.0 / mse.clamp(min=1e-12)),
                       t.full_like(mse, PSNR_CAP))
        ssim = (_ssim_map(t, F, comp, gt, self.win) * region).sum(dim=[1, 2, 3]) / n_reg

        thr = self.s.edge_threshold
        pb = ((comp > 0.5) & (m > 0.5)).float()
        tb = ((gt > thr) & (m > 0.5)).float()
        out = dict(psnr=psnr, ssim=ssim, l1=l1)
        n_p = pb.sum(dim=[1, 2, 3])
        n_t = tb.sum(dim=[1, 2, 3])

        def _f1(p, r):
            return t.where(p + r > 0, 2 * p * r / (p + r).clamp(min=1e-12), t.zeros_like(p))

        tp = (pb * tb).sum(dim=[1, 2, 3])
        p0 = t.where(n_p > 0, tp / n_p.clamp(min=1), t.ones_like(tp))
        r0 = t.where(n_t > 0, tp / n_t.clamp(min=1), t.ones_like(tp))
        out['f1'] = _f1(p0, r0)
        for r, key in ((2, 'f1_t2'), (5, 'f1_t5')):
            k = self.disks[r]
            dil_t = (F.conv2d(tb, k, padding=r) > 0).float()
            dil_p = (F.conv2d(pb, k, padding=r) > 0).float()
            pr = t.where(n_p > 0, (pb * dil_t).sum(dim=[1, 2, 3]) / n_p.clamp(min=1), t.ones_like(tp))
            rc = t.where(n_t > 0, (tb * dil_p).sum(dim=[1, 2, 3]) / n_t.clamp(min=1), t.ones_like(tp))
            out[key] = _f1(pr, rc)

        # HD95 / connectivity (CPU, exact Euclidean distance transform)
        pb_np = pb[:, 0].bool().cpu().numpy()
        tb_np = tb[:, 0].bool().cpu().numpy()
        hd, cn = zip(*(structure_metrics(pb_np[b], tb_np[b]) for b in range(pb_np.shape[0])))
        out['hd95'] = t.tensor(hd, dtype=t.float64)
        out['conn'] = t.tensor(cn, dtype=t.float64)

        if self.lpips_fn is not None:
            a = ((1.0 - comp) * 2 - 1).repeat(1, 3, 1, 1)
            b = ((1.0 - gt) * 2 - 1).repeat(1, 3, 1, 1)
            out['lpips'] = self.lpips_fn(a, b).flatten()
        else:
            out['lpips'] = t.full_like(l1, float('nan'))
        return {k: v.detach().float().cpu().numpy() for k, v in out.items()}

    # -- grids --
    def _grid_rows_base(self, grid_idx):
        """Fixed columns (input, masked edges, GT) for the grid samples."""
        acc, tile = self.fr.accepted, self.s.grid_tile
        base = {}
        for g, idxs in grid_idx.items():
            for i in idxs:
                gray, edge, hole = load_sample(acc[i], self.s)
                base[i] = dict(
                    label=f'{g}\nmask {acc[i]["mask_ratio"] * 100:.1f}%\n'
                          f'{acc[i]["image_name"][:20]}\n{acc[i]["mask_name"][:20]}',
                    tiles=[_to_tile(_render_input(gray, hole), tile),
                           _to_tile(_render_masked_edges(edge, hole), tile),
                           _to_tile(_render_edges(edge, hole), tile)])
        return base

    def _compose_grids(self, grid_idx, base, preds, names):
        cols = ['Input (hole red)', 'Masked edges', 'Ground truth'] + list(names)
        out = {}
        for g, idxs in grid_idx.items():
            if not idxs:
                continue
            rows = [dict(label=base[i]['label'],
                         tiles=base[i]['tiles'] + [preds[n][i] for n in names]) for i in idxs]
            out[g] = build_grid(rows, cols, self.s.grid_tile)
        return out

    def preview(self):
        """Run every checkpoint on the grid samples only -> {group: PIL image}."""
        self._setup()
        grid_idx = select_grid_samples(self.gi, self.s)
        all_idx = sorted(i for v in grid_idx.values() for i in v)
        if not all_idx:
            raise ValueError('No grid samples: groups are empty or "grid samples per group" is 0.')
        base = self._grid_rows_base(grid_idx)
        names = checkpoint_names(self.s.checkpoints)
        preds = {}
        for ci, ((path, override), name) in enumerate(zip(self.s.ckpt_specs(), names)):
            self._check_cancel()
            self.progress(ci / len(names), f'Preview: {name}')
            G = self._load_generator(path, override)
            preds[name] = {}
            with self.torch.no_grad():
                for chunk, gray, edge, hole in self._batches(all_idx):
                    img, e, m = self._to_tensors(gray, edge, hole)
                    comp = self._predict(G, img, e, m)
                    for j, i in enumerate(chunk):
                        preds[name][i] = _to_tile(_render_edges(comp[j, 0].cpu().numpy(), hole[j]),
                                                  self.s.grid_tile)
            del G
            self._free()
        self.progress(1.0, 'Preview ready')
        return self._compose_grids(grid_idx, base, preds, names)

    def _free(self):
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()

    # -- single sample (sample viewer) --
    def sample(self, smp):
        """Run every checkpoint on one sample.

        Returns {'input', 'masked', 'gt': HxWx3 uint8 renders,
                 'preds': [{'name', 'path', 'network', 'source', 'comp', 'raw', 'metrics'} or
                           {'name', 'path', 'network', 'source', 'error'}]}.
        A checkpoint that fails to load reports its error; the others still run.
        """
        gray, edge, hole = load_sample(smp, self.s)
        out = {'input': _render_input(gray, hole),
               'masked': _render_masked_edges(edge, hole),
               'gt': _render_edges(edge, hole),
               'preds': []}
        img, e, m = self._to_tensors(gray[None], edge[None], hole[None])
        names = checkpoint_names(self.s.checkpoints)
        for ci, ((path, override), name) in enumerate(zip(self.s.ckpt_specs(), names)):
            self._check_cancel()
            self.progress(ci / max(1, len(names)), f'Sample: {name}')
            cls, _, src = resolve_network(path, override, self.s.network)
            entry = {'name': name, 'path': path, 'network': cls, 'source': src}
            try:
                G = self._load_generator(path, override)
                miss, unexp = getattr(G, 'load_report', ([], []))
                if miss or unexp:
                    entry['warning'] = (f'⚠ weights do not match this network: {len(miss)} missing, '
                                        f'{len(unexp)} unexpected — choose the right class')
                with self.torch.no_grad():
                    pred, comp = self._forward(G, img, e, m)
                    met = self._metrics(comp, e, m)
                entry.update(metrics={k: float(v[0]) for k, v in met.items()},
                             comp=_render_edges(comp[0, 0].cpu().numpy(), hole),
                             raw=_render_edges(pred[0, 0].cpu().numpy(), hole))
            except Cancelled:
                raise
            except LoadError as ex:
                entry['error'] = str(ex).split('\n\n')[0] + '  ' + str(ex).split('\n\n')[2]
                self.log(f'[Sample] {name}: does not fit {cls}')
            except Exception as ex:
                entry['error'] = f'{type(ex).__name__}: {ex}'
                self.log(f'[Sample] {name}: {entry["error"]}')
            out['preds'].append(entry)
        self.progress(1.0, 'Sample ready')
        return out

    # -- full evaluation --
    def run(self):
        self._setup()
        s, acc = self.s, self.fr.accepted
        N = len(acc)
        if N == 0:
            raise ValueError('No accepted samples to evaluate.')
        names = checkpoint_names(s.checkpoints)
        all_idx = list(range(N))
        n_batches = math.ceil(N / max(1, s.batch_size))
        passes = len(names) + (1 if self.detector is not None else 0)
        step = [0]

        def _tick(text):
            step[0] += 1
            self.progress(step[0] / max(1, passes * n_batches), text)

        # Pre-flight: every checkpoint must load before any time is spent on metrics.
        failures = []
        for ci, ((path, override), name) in enumerate(zip(s.ckpt_specs(), names)):
            self._check_cancel()
            self.progress(ci / len(names), f'Checking {name}')
            try:
                G = self._load_generator(path, override)
                del G
            except LoadError as e:
                failures.append(str(e).split('\n\n')[1])
            self._free()
        if failures:
            raise LoadError(f'{len(failures)} checkpoint(s) do not fit their network class:\n\n'
                            + '\n\n'.join(failures) +
                            '\n\nFix: double-click each one in tab 1 and choose the network it was '
                            'trained with. Details are in the log.')

        # Real (GT) Inception features: computed ONCE, shared by all checkpoints.
        real_feats = None
        if self.detector is not None:
            self.log('[Eval] Inception features of ground-truth edges…')
            real_feats = np.zeros((N, 2048), dtype=np.float32)
            with self.torch.no_grad():
                for chunk, gray, edge, hole in self._batches(all_idx):
                    self._check_cancel()
                    _, e, _ = self._to_tensors(gray, edge, hole)
                    real_feats[chunk] = self._features(e)
                    _tick('GT features')

        grid_idx = select_grid_samples(self.gi, s)
        grid_set = {i for v in grid_idx.values() for i in v}
        base = self._grid_rows_base(grid_idx) if grid_set else {}
        grid_preds = {}
        per_image, fake_feats = {}, {}

        for ci, ((path, override), name) in enumerate(zip(s.ckpt_specs(), names)):
            self._check_cancel()
            self.log(f'[Eval] ({ci + 1}/{len(names)}) {name}')
            G = self._load_generator(path, override)
            pm = {k: np.full(N, np.nan, dtype=np.float64) for k in PER_IMAGE_METRICS}
            ff = np.zeros((N, 2048), dtype=np.float32) if self.detector is not None else None
            grid_preds[name] = {}
            t0 = time.time()
            with self.torch.no_grad():
                for chunk, gray, edge, hole in self._batches(all_idx):
                    self._check_cancel()
                    img, e, m = self._to_tensors(gray, edge, hole)
                    comp = self._predict(G, img, e, m)
                    for k, v in self._metrics(comp, e, m).items():
                        pm[k][chunk] = v
                    if ff is not None:
                        ff[chunk] = self._features(comp)
                    for j, i in enumerate(chunk):
                        if i in grid_set:
                            grid_preds[name][i] = _to_tile(
                                _render_edges(comp[j, 0].cpu().numpy(), hole[j]), s.grid_tile)
                    _tick(f'{name}: {min(chunk[-1] + 1, N)}/{N}')
            per_image[name] = pm
            fake_feats[name] = ff
            self.log(f'[Eval]   done in {time.time() - t0:.1f}s  '
                     f'PSNR={np.nanmean(pm["psnr"]):.3f}  F1={np.nanmean(pm["f1"]):.4f}')
            del G
            self._free()

        # Per-group summaries
        summary = {}
        for g in SUMMARY_GROUPS:
            idx = self.gi.members.get(g, [])
            summary[g] = {}
            for name in names:
                row = {'n': len(idx)}
                for k in PER_IMAGE_METRICS:
                    vals = per_image[name][k][idx] if idx else np.array([])
                    row[k] = float(np.nanmean(vals)) if len(vals) and np.isfinite(vals).any() else float('nan')
                if real_feats is not None and len(idx) >= 2:
                    row['fid'] = fid_from_features(fake_feats[name][idx], real_feats[idx])
                else:
                    row['fid'] = float('nan')
                summary[g][name] = row

        grids = self._compose_grids(grid_idx, base, grid_preds, names) if grid_set else {}
        self.progress(1.0, 'Evaluation finished')
        return EvalResults(ckpt_names=names, ckpt_paths=list(s.checkpoints), per_image=per_image,
                           summary=summary, grid_images=grids, notes=self.notes,
                           device=str(self.device),
                           ckpt_networks=[resolve_network(p, o, s.network)[0]
                                          for p, o in s.ckpt_specs()])


# ---------------------------------------------------------------------------
# Best-value helper (GUI table + Excel)
# ---------------------------------------------------------------------------
def best_index(values, metric):
    """Index of the best finite value respecting the metric direction (None if none)."""
    finite = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if not finite:
        return None
    pick = max if HIGHER_BETTER[metric] else min
    return pick(finite, key=lambda iv: iv[1])[0]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
REJECT_COLUMNS = ['sample_id', 'image', 'mask', 'mask_ratio', 'edge_pixels_in_hole',
                  'edge_density', 'reason', 'detail']


def _num(v):
    """NaN/inf -> None (empty cell); numpy scalars -> python."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _rejected_rows(fr):
    return [[r['id'], r['image_name'], r['mask_name'], _num(r['mask_ratio']),
             r['edge_pixels_in_hole'], _num(r['edge_density']), r['reason'], r['detail']]
            for r in fr.rejected]


def write_rejected_csv(fr, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(REJECT_COLUMNS)
        w.writerows(_rejected_rows(fr))


def _write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(['' if v is None else v for v in r])


def summary_table(results, group, fid_min):
    metric_keys = [k for k, _, _ in METRICS]
    header = ['checkpoint', 'n'] + [METRIC_LABEL[k] for k in metric_keys] + ['note', 'file', 'network']
    nets = results.ckpt_networks or [''] * len(results.ckpt_names)
    rows = []
    for name, path, net in zip(results.ckpt_names, results.ckpt_paths, nets):
        r = results.summary[group][name]
        note = (f'FID on n={r["n"]} < {fid_min}: unreliable' if 0 < r['n'] < fid_min else '')
        rows.append([name, r['n']] + [_num(r[k]) for k in metric_keys] +
                    [note, checkpoint_file_label(path), net])
    return header, rows, metric_keys


def group_info_table(fr, gi, s):
    header = ['group', 'mode', 'lower_bound_%', 'upper_bound_%', 'count',
              'mean_mask_ratio_%', 'min_mask_ratio_%', 'max_mask_ratio_%', 'note']
    rows = []
    for g in GROUPS + (OUT_OF_RANGE, ALL):
        st = gi.ratio_stats(g, fr.accepted)
        lo, hi = gi.bounds.get(g, (None, None))
        note = ''
        if g in GROUPS and gi.count(g) < s.fid_min_group_size:
            note = f'< {s.fid_min_group_size} samples: FID unreliable'
        if g == OUT_OF_RANGE:
            note = 'reported only; excluded from Small/Medium/Large, included in All'
        if g == ALL:
            note = 'all accepted samples'
        pct = lambda v: None if v is None or not math.isfinite(v) else round(v * 100, 3)
        rows.append([g, gi.mode, pct(lo), pct(hi), gi.count(g),
                     pct(st['mean']), pct(st['min']), pct(st['max']), note])
    return header, rows


def per_image_table(fr, gi, results):
    header = ['sample_id', 'image', 'mask', 'mask_ratio', 'group',
              'edge_pixels_in_hole', 'edge_density']
    cols = [(n, k) for n in results.ckpt_names for k in PER_IMAGE_METRICS]
    header += [f'{n} | {METRIC_LABEL[k]}' for n, k in cols]
    rows = []
    for i, a in enumerate(fr.accepted):
        row = [a['id'], a['image_name'], a['mask_name'], _num(a['mask_ratio']), gi.labels[i],
               a['edge_pixels_in_hole'], _num(a['edge_density'])]
        row += [_num(results.per_image[n][k][i]) for n, k in cols]
        rows.append(row)
    return header, rows


def settings_rows(s, fr, results, pairing):
    rows = [['timestamp', time.strftime('%Y-%m-%d %H:%M:%S')],
            ['device', results.device],
            ['filter_cache_key', fr.key],
            ['mask_pairing_used', pairing],
            ['total_samples', fr.total],
            ['accepted', len(fr.accepted)],
            ['rejected', len(fr.rejected)]]
    for k, v in s.to_dict().items():
        if k == 'checkpoints':
            continue
        rows.append([k, v])
    for (path, override), name in zip(s.ckpt_specs(), results.ckpt_names):
        cls, extra, src = resolve_network(path, override, s.network)
        rows.append([f'checkpoint: {name}', path])
        rows.append([f'network: {name}', f'{cls}  ({src})' + (f'  {extra}' if extra else '')])
    for n in results.notes:
        rows.append(['note', n])
    return rows


def export_results(s, fr, gi, results, log=print):
    """Write workbook + CSVs + grids into a timestamped folder; returns that folder."""
    run_dir = os.path.join(s.results_dir, time.strftime('eval_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    sheets = []                                          # (name, header, rows, best_cols)
    for g in SUMMARY_GROUPS:
        header, rows, mkeys = summary_table(results, g, s.fid_min_group_size)
        _write_csv(os.path.join(run_dir, f'summary_{g.lower()}.csv'), header, rows)
        best = {2 + j: k for j, k in enumerate(mkeys)}   # column index -> metric
        sheets.append((f'Summary_{g}', header, rows, best))

    gh, grows = group_info_table(fr, gi, s)
    _write_csv(os.path.join(run_dir, 'group_info.csv'), gh, grows)
    sheets.append(('Group_Info', gh, grows, {}))

    ph, prows = per_image_table(fr, gi, results)
    _write_csv(os.path.join(run_dir, 'per_image.csv'), ph, prows)
    sheets.append(('PerImage', ph, prows, {}))

    write_rejected_csv(fr, os.path.join(run_dir, 'rejected_samples.csv'))
    sheets.append(('Rejected', REJECT_COLUMNS, _rejected_rows(fr), {}))
    # The exact evaluated samples, reusable via "Load sample list…" (paths relative to the folders).
    save_sample_list(os.path.join(run_dir, 'samples.json'), s, fr, gi)

    srows = settings_rows(s, fr, results, fr.pairing)
    sheets.append(('Settings', ['setting', 'value'], [[k, json.dumps(v) if isinstance(v, (list, dict)) else v]
                                                      for k, v in srows], {}))
    s.save_json(os.path.join(run_dir, 'settings.json'))

    try:
        _write_xlsx(os.path.join(run_dir, 'evaluation.xlsx'), sheets)
    except ImportError:
        log('[Export] WARNING: openpyxl not installed -> no .xlsx (CSVs written). pip install openpyxl')

    for g, img in results.grid_images.items():
        d = os.path.join(run_dir, 'grids', g.lower())
        os.makedirs(d, exist_ok=True)
        img.save(os.path.join(d, f'grid_{g.lower()}.png'))

    log(f'[Export] results written to {run_dir}')
    return run_dir


def _write_xlsx(path, sheets):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)
    best_fill = PatternFill('solid', fgColor='C6EFCE')
    for name, header, rows, best_cols in sheets:
        ws = wb.create_sheet(name[:31])
        ws.append(header)
        for c in ws[1]:
            c.font = bold
        for r in rows:
            ws.append([_num(v) for v in r])
        for col, metric in best_cols.items():
            b = best_index([r[col] for r in rows], metric)
            if b is not None:
                cell = ws.cell(row=b + 2, column=col + 1)
                cell.fill = best_fill
                cell.font = bold
        for j, h in enumerate(header, start=1):
            ws.column_dimensions[get_column_letter(j)].width = min(45, max(10, len(str(h)) + 2))
        ws.freeze_panes = 'B2'
    wb.save(path)

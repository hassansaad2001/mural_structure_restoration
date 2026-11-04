# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified by Hassan Saad (2026) for mural structure (edge) restoration.
# Distributed as part of a derivative of MAT under CC BY-NC 4.0 (see LICENSE, NOTICE.md).

import cv2
import os
import numpy as np
import PIL.Image
import json
import torch
import dnnlib
import random
import zipfile
import pickle

try:
    import pyspng
except ImportError:
    pyspng = None

from datasets.mask_generator_512 import RandomMask  # 512-resolution mask generator

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_edge(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        # Override to ensure consistency potentially or rely on subclass
        pass

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


#----------------------------------------------------------------------------


class TSRDataset(Dataset):
    """TSR Dataset for edge inpainting (512-resolution variant).

    Modified: Now only loads edges and images, without line data.
    Edges are the source of truth - only edge/image pairs where both exist are included.
    Returns: image, edge, mask, idx

    Optional mask_path: when provided, masks are sampled at random from that
    folder instead of being generated on-the-fly by RandomMask.
    """
    def __init__(self,
        edge_path,              # Path to edges directory.
        data_path=None,         # Path to images directory (Gray/RGB).
        mask_path=None,         # Optional: path to pre-made mask images.
        resolution      = None, # Ensure specific resolution, None = highest available.
        hole_range=[0,0.2],
        fix_crops=False,        # Deterministic crops and masks
        mask_aug=False,         # Random flip / rot90 / zoom augmentation of pre-made masks
        face_boxes=None,        # Optional JSON {crop_basename: [[x0,y0,x1,y1], ...]} of face/hand boxes
        face_mask_prob=0.0,     # Prob. of a face/part hole on images that have boxes (training)
        face_oversample=1,      # Repeat crops that have boxes this many times per epoch (training)
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._mask_aug = mask_aug
        self._face_mask_prob = float(face_mask_prob)
        self._face_oversample = max(1, int(face_oversample))
        self._face_boxes = {}
        if face_boxes:
            with open(face_boxes, 'r') as f:
                raw = json.load(f)
            for k, v in raw.items():
                uniq = []
                for b in v:                                   # drop accidental duplicate boxes
                    b = [float(t) for t in b]
                    if b not in uniq:
                        uniq.append(b)
                if uniq:
                    self._face_boxes[os.path.splitext(os.path.basename(k))[0]] = uniq
            print(f"[TSRDataset] Face/part boxes for {len(self._face_boxes)} images from: {face_boxes}")
        self._last_crop = (0, 0)
        self._edge_path = edge_path
        self._data_path = data_path
        self._mask_path = mask_path
        self._zipfile = None
        self._data_zipfile = None
        self._hole_range = hole_range
        self._resolution = resolution
        self._fix_crops = fix_crops

        # Define image file extensions
        self._image_extensions = PIL.Image.EXTENSION

        # Determine type based on edge_path
        if os.path.isdir(self._edge_path):
            self._type = 'dir'
        elif os.path.splitext(self._edge_path)[1].lower() == '.zip':
            self._type = 'zip'
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_extensions = set(k.lower() for k in PIL.Image.EXTENSION.keys())

        # Collect edge files (relative paths) - edges are the source of truth
        all_edge_fnames = self._find_image_files(self._edge_path, self._image_extensions)

        if len(all_edge_fnames) == 0:
            raise IOError(f"No edge files found in: {self._edge_path}")

        # If data_path is provided, filter to only include edge/image pairs where both exist
        if self._data_path is not None:
            self._edge_fnames, self._data_fnames = self._find_matching_pairs(
                all_edge_fnames, self._edge_path, self._data_path, self._image_extensions
            )

            if len(self._edge_fnames) == 0:
                raise IOError(f"No matching edge/image pairs found! "
                             f"Edges: {self._edge_path}, Images: {self._data_path}")
        else:
            # No image path provided, just use edges
            self._edge_fnames = all_edge_fnames
            self._data_fnames = []

        # Collect pre-made mask files (if mask_path was given)
        if self._mask_path is not None:
            self._mask_fnames = self._find_image_files(self._mask_path, self._image_extensions)
            if len(self._mask_fnames) == 0:
                raise IOError(f"No mask files found in: {self._mask_path}")
            print(f"[TSRDataset] Using {len(self._mask_fnames)} pre-made masks from: {self._mask_path}")
        else:
            self._mask_fnames = []  # empty → use RandomMask

        # Final setup
        name = os.path.splitext(os.path.basename(self._edge_path))[0]
        raw_shape = [len(self._edge_fnames)] + list(self._load_raw_edge(0).shape)
        if resolution is not None:
             raw_shape[2] = resolution
             raw_shape[3] = resolution

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

        # Oversample crops that have face/hand boxes (training only), so face
        # holes are seen often enough to matter. Applied AFTER max_size/xflip;
        # each extra copy keeps the flip flag of the entry it duplicates.
        if self._face_boxes and self._face_mask_prob > 0 and self._face_oversample > 1 and not fix_crops:
            keys = [os.path.splitext(os.path.basename(f))[0] for f in self._edge_fnames]
            sel = np.array([i for i, r in enumerate(self._raw_idx) if keys[r] in self._face_boxes], dtype=np.int64)
            if sel.size:
                rep = self._face_oversample - 1
                self._raw_idx = np.concatenate([self._raw_idx, np.tile(self._raw_idx[sel], rep)])
                self._xflip = np.concatenate([self._xflip, np.tile(self._xflip[sel], rep)])
                print(f"[TSRDataset] Face oversampling x{self._face_oversample}: "
                      f"{sel.size} boxed entries -> {sel.size * self._face_oversample} of {self._raw_idx.size} "
                      f"({100.0 * sel.size * self._face_oversample / self._raw_idx.size:.1f}% of samples)")

    def _find_image_files(self, path, extensions):
        if os.path.isdir(path):
            return sorted([
                os.path.relpath(os.path.join(root, fname), start=path)
                for root, _dirs, files in os.walk(path)
                for fname in files if os.path.splitext(fname)[1].lower() in extensions
            ])
        elif os.path.splitext(path)[1].lower() == ".zip":
            self._type = "zip"
            self._zipfile = zipfile.ZipFile(path, "r")
            return sorted([
                name for name in self._zipfile.namelist()
                if os.path.splitext(name)[1].lower() in extensions
            ])
        else:
            raise IOError("Image path must be a directory or zip.")

    def _find_matching_pairs(self, edge_fnames, edge_base_path, data_base_path, extensions):
        """Find edge/image pairs where both files exist.
        
        Edges are the source of truth. For each edge, we look for a matching image.
        Only pairs where both exist are included. Missing images are logged.
        
        Args:
            edge_fnames: List of edge file paths (relative to edge_base_path)
            edge_base_path: Base path for edge files
            data_base_path: Base path for image files
            extensions: Valid image extensions
            
        Returns:
            Tuple of (filtered_edge_fnames, matching_data_fnames)
        """
        # Build a map of image basenames to their full paths
        if os.path.isdir(data_base_path):
            data_files_map = {
                os.path.splitext(os.path.basename(fname))[0]: os.path.relpath(os.path.join(root, fname), start=data_base_path)
                for root, _dirs, files in os.walk(data_base_path)
                for fname in files if os.path.splitext(fname)[1].lower() in extensions
            }
        elif os.path.splitext(data_base_path)[1].lower() == ".zip":
            zf = zipfile.ZipFile(data_base_path, "r")
            data_files_map = {
                os.path.splitext(os.path.basename(fname))[0]: fname
                for fname in zf.namelist() if os.path.splitext(fname)[1].lower() in extensions
            }
        else:
            raise IOError("Data path must be a directory or zip.")

        # Filter: only keep edges that have matching images
        filtered_edges = []
        matching_images = []
        missing_images = []
        
        for edge_fname in edge_fnames:
            edge_basename = os.path.splitext(os.path.basename(edge_fname))[0]
            
            if edge_basename in data_files_map:
                filtered_edges.append(edge_fname)
                matching_images.append(data_files_map[edge_basename])
            else:
                missing_images.append(edge_fname)
        
        # Log statistics
        total_edges = len(edge_fnames)
        matched = len(filtered_edges)
        missing = len(missing_images)
        
        print(f"[TSRDataset] Edge/Image matching:")
        print(f"  - Total edges found: {total_edges}")
        print(f"  - Matching pairs: {matched}")
        print(f"  - Edges without matching images (skipped): {missing}")
        
        if missing > 0 and missing <= 20:
            print(f"  - Missing images for edges: {missing_images}")
        elif missing > 20:
            print(f"  - First 20 missing: {missing_images[:20]}...")
        
        return filtered_edges, matching_images

    def _find_corresponding_files(self, image_paths, image_base_path, corresponding_base_path, extensions):
        """Legacy method - kept for compatibility but not used when filtering is enabled."""
        if os.path.isdir(corresponding_base_path):
            files_map = {
                os.path.splitext(os.path.basename(fname))[0]: os.path.relpath(os.path.join(root, fname), start=corresponding_base_path)
                for root, _dirs, files in os.walk(corresponding_base_path)
                for fname in files if os.path.splitext(fname)[1].lower() in extensions
            }
        elif os.path.splitext(corresponding_base_path)[1].lower() == ".zip":
            zf = zipfile.ZipFile(corresponding_base_path, "r")
            files_map = {
                os.path.splitext(os.path.basename(fname))[0]: fname
                for fname in zf.namelist() if os.path.splitext(fname)[1].lower() in extensions
            }
        else:
            raise IOError("Corresponding path must be a directory or zip.")

        out_list = []
        for img in image_paths:
            base = os.path.splitext(os.path.basename(img))[0]
            if base in files_map:
                out_list.append(files_map[base])
            else:
                pass
                
        return out_list

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._edge_path)
        return self._zipfile
        
    def _get_data_zipfile(self):
        assert self._type == 'zip'
        if self._data_zipfile is None and self._data_path:
             self._data_zipfile = zipfile.ZipFile(self._data_path)
        return self._data_zipfile

    def _open_file(self, fname, source='edge'):
        if source == 'data':
            base = self._data_path
            zf_getter = self._get_data_zipfile
        else:
            base = self._edge_path
            zf_getter = self._get_zipfile

        if self._type == 'dir':
            return open(os.path.join(base, fname), 'rb')
        if self._type == 'zip':
            return zf_getter().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
            if self._data_zipfile is not None:
                 self._data_zipfile.close()
        finally:
            self._zipfile = None
            self._data_zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None, _data_zipfile=None)

    def _process_image(self, image, rng, is_rgb=False):
        # Image processing with random crop consistency.
        # uses a *local* random.Random instance passed in by the
        # caller. The old code called random.seed(seed) on the GLOBAL RNG, which
        # turned the per-item seed into a deterministic chain that collapsed onto
        # a few short cycles (~400 states), so only a few hundred distinct masks
        # were ever sampled during training.
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        
        H, W, C = image.shape
        res = self._resolution if self._resolution else 512  # default 512 for this variant
        
        if H < res or W < res:
            top = 0; bottom = max(0, res - H)
            left = 0; right = max(0, res - W)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
            if image.ndim == 2:
                 image = image[:, :, np.newaxis]
        
        H, W, C = image.shape
        h = rng.randint(0, H - res)
        w = rng.randint(0, W - res)
        self._last_crop = (h, w)          # used to shift face boxes into crop coords
        image = image[h:h+res, w:w+res, :]

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return image
        
    def _load_raw_edge(self, raw_idx, rng=None):
        fname = self._edge_fnames[raw_idx]
        with self._open_file(fname, source='edge') as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                edge = pyspng.load(f.read())
            else:
                edge = np.array(PIL.Image.open(f))
        
        # Ensure single channel
        if edge.ndim == 3:
            edge = edge[:, :, 0] # Take first channel if RGB
            
        if rng is None:
            rng = random.Random(0)
        return self._process_image(edge, rng=rng)
    


    def _load_raw_image(self, raw_idx, rng=None):
        if not self._data_fnames:
            # Return dummy if no data path provided
            return np.zeros((1, self._resolution or 512, self._resolution or 512), dtype=np.uint8)

        fname = self._data_fnames[raw_idx]
        with self._open_file(fname, source='data') as f:
            image = np.array(PIL.Image.open(f).convert('L'))

        if rng is None:
            rng = random.Random(0)
        return self._process_image(image, rng=rng, is_rgb=False)

    def _load_mask_from_folder(self, res, seed=None, augment=None):
        """Load a random mask file from self._mask_path, resize to (res × res).

        Convention: white pixel (255) = hole, black pixel (0) = valid.
        Returns float32 array [1, res, res] where 1 = hole, 0 = valid.

        augment: None -> use self._mask_aug. When enabled, applies random
        h/v flip, rot90 and a random zoom-crop (1.0-1.5x) using a LOCAL rng,
        so the effective mask pool is many times larger than the file count.
        """
        if augment is None:
            augment = self._mask_aug
        rng = random.Random(seed) if seed is not None else random.Random(random.getrandbits(32))
        fname = rng.choice(self._mask_fnames)

        mask_full_path = os.path.join(self._mask_path, fname)
        mask_img = PIL.Image.open(mask_full_path).convert('L')  # grayscale

        # Resize to target resolution (nearest-neighbour keeps hard edges)
        if mask_img.size != (res, res):
            mask_img = mask_img.resize((res, res), PIL.Image.NEAREST)

        mask_np = np.array(mask_img, dtype=np.float32) / 255.0  # [0, 1]
        # Binarise: values >= 0.5 → hole (1), otherwise valid (0)
        mask_np = (mask_np >= 0.5).astype(np.float32)

        if augment:
            if rng.random() < 0.5:
                mask_np = mask_np[:, ::-1]
            if rng.random() < 0.5:
                mask_np = mask_np[::-1, :]
            k = rng.randint(0, 3)
            if k:
                mask_np = np.rot90(mask_np, k)
            zoom = rng.uniform(1.0, 1.5)
            if zoom > 1.01:
                crop = int(round(res / zoom))
                y0 = rng.randint(0, res - crop)
                x0 = rng.randint(0, res - crop)
                mask_np = np.ascontiguousarray(mask_np[y0:y0 + crop, x0:x0 + crop])
                mask_np = cv2.resize(mask_np, (res, res), interpolation=cv2.INTER_NEAREST)
            mask_np = np.ascontiguousarray(mask_np)

        return mask_np[np.newaxis, ...].astype(np.float32)  # [1, res, res]

    def _load_raw_labels(self):
        return None

    # ── Face / semantic-part holes ────────────────────────────────────────
    @property
    def has_face_boxes(self):
        return len(self._face_boxes) > 0

    def _boxes_for(self, raw_idx, res, crop_hw):
        key = os.path.splitext(os.path.basename(self._edge_fnames[raw_idx]))[0]
        out = []
        h0, w0 = crop_hw
        for x0, y0, x1, y1 in self._face_boxes.get(key, []):
            x0, x1 = sorted((x0 - w0, x1 - w0)); y0, y1 = sorted((y0 - h0, y1 - h0))
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(res), x1), min(float(res), y1)
            if x1 - x0 >= 8 and y1 - y0 >= 8:          # keep boxes that survive the crop
                out.append((x0, y0, x1, y1))
        return out

    @staticmethod
    def _face_hole(res, boxes, rng):
        """Irregular hole over a whole annotated region or a random part of it
        (eye / mouth / hand-sized). Returns float32 [res, res], 1 = hole."""
        x0, y0, x1, y1 = boxes[rng.randrange(len(boxes))]
        bw, bh = x1 - x0, y1 - y0
        hole = np.zeros((res, res), np.uint8)
        if rng.random() < 0.5:
            # whole region, slightly enlarged
            s = rng.uniform(1.0, 1.25)
            cx, cy, ax, ay = (x0 + x1) / 2, (y0 + y1) / 2, s * bw / 2, s * bh / 2
        else:
            # a part: 25-55 % of the box, anywhere inside it
            f = rng.uniform(0.25, 0.55)
            ax, ay = f * bw / 2, f * bh * rng.uniform(0.6, 1.2) / 2
            cx = rng.uniform(x0 + ax, max(x0 + ax, x1 - ax))
            cy = rng.uniform(y0 + ay, max(y0 + ay, y1 - ay))
        # jittered polygon around an ellipse -> irregular damage-like outline
        n = 16
        pts = []
        for i in range(n):
            t = 2 * np.pi * i / n
            r = rng.uniform(0.8, 1.15)
            pts.append((cx + r * ax * np.cos(t), cy + r * ay * np.sin(t)))
        cv2.fillPoly(hole, [np.round(np.array(pts)).astype(np.int32)], 1)
        return hole.astype(np.float32)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Returns:
            image: Grayscale image [1, H, W] in range [0, 255]
            edge: Edge map [1, H, W] in range [0, 255] (255=edge)
            mask: Hole mask [1, H, W] float32 (1=Hole, 0=Valid)
            raw_idx: Original index in dataset
        """
        raw_idx = self._raw_idx[idx]
        fix = getattr(self, '_fix_crops', False)

        # never reseed the global RNG. Draw a fresh 31-bit seed and
        # use LOCAL random.Random instances so edge and image get the same crop.
        if fix:
            seed = int(raw_idx)
            np.random.seed(seed)   # only affects the RandomMask fallback below
        else:
            seed = random.getrandbits(31)

        edge  = self._load_raw_edge(raw_idx, rng=random.Random(seed))
        image = self._load_raw_image(raw_idx, rng=random.Random(seed))  # same crop as edge

        assert isinstance(edge, np.ndarray)
        assert edge.dtype == np.uint8
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8

        # X-Flip
        if self._xflip[idx]:
            edge  = edge[:, :, ::-1]
            image = image[:, :, ::-1]

        # Invert Edge: black background (0) → signal (255)
        edge = 255 - edge

        # ── Mask ─────────────────────────────────────────────────────────
        res = edge.shape[-1]  # spatial size after crop
        if self._mask_fnames:
            # Sample a pre-made mask; use a deterministic-ish seed so that
            # different epochs still sample varied masks.
            mask_seed = seed + 1234 if not fix else seed + 999
            mask = self._load_mask_from_folder(res, seed=mask_seed)
            # mask is already float32 [1, H, W] with 1=hole
        else:
            # Fall back to on-the-fly random mask generation
            if fix:
                # RandomMask uses the global RNGs; save/restore so we don't
                # leak deterministic state into the rest of the worker.
                _py_state, _np_state = random.getstate(), np.random.get_state()
                random.seed(seed + 999)
                np.random.seed(seed + 999)
                mask = RandomMask(res, hole_range=self._hole_range)
                random.setstate(_py_state); np.random.set_state(_np_state)
            else:
                mask = RandomMask(res, hole_range=self._hole_range)
            mask = 1.0 - mask  # flip: RandomMask returns 0=hole → we want 1=hole

        # ── Face / part holes (training only) ───────────────────────────
        if self._face_mask_prob > 0 and self._face_boxes and not fix:
            frng = random.Random(seed + 777)
            if frng.random() < self._face_mask_prob:
                boxes = self._boxes_for(raw_idx, res, self._last_crop)
                if boxes:
                    fh = self._face_hole(res, boxes, frng)
                    if self._xflip[idx]:
                        fh = fh[:, ::-1]                    # boxes are in un-flipped coords
                    if frng.random() < 0.5:
                        mask = fh[np.newaxis]               # face/part hole only
                    else:
                        mask = np.maximum(mask, fh[np.newaxis])   # plus the usual damage mask
                    mask = np.ascontiguousarray(mask, dtype=np.float32)

        # Image is Gray uint8 (1, H, W), Edge is Gray uint8 (1, H, W) 255=Edge
        return image.copy(), edge.copy(), mask, raw_idx

    def get_eval_sample(self, idx, k=0, face=False):
        """Deterministic (image, edge, mask) for validation.

        Crop depends only on raw_idx; mask k is a deterministic draw from the
        mask pool WITHOUT augmentation, so several masks per image can be
        evaluated and the set is identical across runs / checkpoints.

        face=True: deterministic face/part hole from the box annotations
        instead (returns None if this image has no usable box).
        """
        raw_idx = int(self._raw_idx[idx])
        edge  = self._load_raw_edge(raw_idx, rng=random.Random(raw_idx))
        image = self._load_raw_image(raw_idx, rng=random.Random(raw_idx))
        if self._xflip[idx]:
            edge, image = edge[:, :, ::-1], image[:, :, ::-1]
        edge = 255 - edge
        res = edge.shape[-1]
        mseed = 1_000_003 * (k + 1) + raw_idx
        if face:
            boxes = self._boxes_for(raw_idx, res, self._last_crop)
            if not boxes:
                return None
            fh = self._face_hole(res, boxes, random.Random(mseed))
            if self._xflip[idx]:
                fh = fh[:, ::-1]
            return image.copy(), edge.copy(), np.ascontiguousarray(fh[np.newaxis]), raw_idx
        if self._mask_fnames:
            mask = self._load_mask_from_folder(res, seed=mseed, augment=False)
        else:
            _py_state, _np_state = random.getstate(), np.random.get_state()
            random.seed(mseed); np.random.seed(mseed % (2**32))
            mask = 1.0 - RandomMask(res, hole_range=self._hole_range)
            random.setstate(_py_state); np.random.set_state(_np_state)
        return image.copy(), edge.copy(), mask.astype(np.float32), raw_idx


#----------------------------------------------------------------------------
# SlottedSampler: Infinite sampler that yields batches with a fixed ratio
# of primary vs secondary indices. Replicates InfiniteSampler's resume logic.
#----------------------------------------------------------------------------

class SlottedSampler(torch.utils.data.Sampler):
    """Infinite sampler that ensures each batch has a fixed ratio of primary/secondary samples.
    
    Like InfiniteSampler but yields indices in slot-groups: for each batch,
    `n_primary` indices come from [0, primary_size) and `n_secondary` from
    [primary_size, primary_size + secondary_size).
    
    Resume logic is consistent with InfiniteSampler: a local RNG seeded with
    `seed` is fast-forwarded to `resume_idx` so that the sequence is identical
    whether training is continuous or resumed.
    """
    def __init__(self, primary_size, secondary_size, batch_size,
                 slot_ratio=0.75, rank=0, num_replicas=1, shuffle=True,
                 seed=0, window_size=0.5, resume_idx=0):
        assert primary_size > 0 and secondary_size > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 < slot_ratio < 1
        super().__init__(None)
        
        self.primary_size = primary_size
        self.secondary_size = secondary_size
        self.batch_size = batch_size
        self.slot_ratio = slot_ratio
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.resume_idx = resume_idx
        
        # Compute per-batch slot counts
        per_gpu_batch = batch_size // num_replicas
        self.n_primary = max(1, int(round(per_gpu_batch * slot_ratio)))
        self.n_secondary = per_gpu_batch - self.n_primary
        if self.n_secondary < 1:
            self.n_secondary = 1
            self.n_primary = per_gpu_batch - 1

    def __iter__(self):
        # Two independent index orders + RNGs (same seed structure as InfiniteSampler)
        order_p = np.arange(self.primary_size)
        order_s = np.arange(self.secondary_size)
        
        rnd_p = None
        rnd_s = None
        window_p = 0
        window_s = 0
        
        if self.shuffle:
            rnd_p = np.random.RandomState(self.seed)
            rnd_s = np.random.RandomState(self.seed + 1)  # Different seed for secondary
            rnd_p.shuffle(order_p)
            rnd_s.shuffle(order_s)
            window_p = int(np.rint(order_p.size * self.window_size))
            window_s = int(np.rint(order_s.size * self.window_size))
            
            # Fast-forward RNG states for resume (same logic as InfiniteSampler)
            if self.resume_idx > 0:
                # Each "step" the sampler yielded n_primary + n_secondary indices
                # We need to figure out how many primary/secondary draws happened
                total_per_step = self.n_primary + self.n_secondary
                n_steps_done = self.resume_idx // total_per_step
                
                if window_p >= 2:
                    for _ in range(n_steps_done * self.n_primary):
                        rnd_p.randint(window_p)
                if window_s >= 2:
                    for _ in range(n_steps_done * self.n_secondary):
                        rnd_s.randint(window_s)
        
        idx_p = 0
        idx_s = 0
        
        # Fast-forward index counters
        if self.resume_idx > 0:
            total_per_step = self.n_primary + self.n_secondary
            n_steps_done = self.resume_idx // total_per_step
            idx_p = n_steps_done * self.n_primary
            idx_s = n_steps_done * self.n_secondary
        
        while True:
            # Yield n_primary indices from primary range
            for _ in range(self.n_primary):
                i = idx_p % order_p.size
                yield int(order_p[i])  # Primary indices: [0, primary_size)
                if window_p >= 2:
                    j = (i - rnd_p.randint(window_p)) % order_p.size
                    order_p[i], order_p[j] = order_p[j], order_p[i]
                idx_p += 1
            
            # Yield n_secondary indices from secondary range (offset by primary_size)
            for _ in range(self.n_secondary):
                i = idx_s % order_s.size
                yield int(order_s[i]) + self.primary_size  # Offset into secondary range
                if window_s >= 2:
                    j = (i - rnd_s.randint(window_s)) % order_s.size
                    order_s[i], order_s[j] = order_s[j], order_s[i]
                idx_s += 1


#----------------------------------------------------------------------------


class SlottedTSRDataset(Dataset):
    """TSR Dataset that merges two data sources with slot-based batch mixing (512-resolution variant).

    Wraps two TSRDataset instances (primary + secondary). Indices [0..N1-1]
    map to primary, [N1..N1+N2-1] map to secondary. Use with SlottedSampler
    to ensure each batch respects the slot ratio.
    """
    def __init__(self,
        edge_path,              # Primary edge path
        data_path=None,         # Primary image path
        mask_path=None,         # Optional: pre-made masks for primary source
        edge_path_2=None,       # Secondary edge path
        data_path_2=None,       # Secondary image path
        mask_path_2=None,       # Optional: pre-made masks for secondary source
        slot_ratio=0.5,         # Fraction of batch from primary
        resolution=None,
        hole_range=[0, 0.2],
        fix_crops=False,
        mask_aug=False,
        **super_kwargs,
    ):
        assert edge_path_2 is not None, "edge_path_2 is required for SlottedTSRDataset"

        self._slot_ratio = slot_ratio

        # Build primary dataset
        self.ds_primary = TSRDataset(
            edge_path=edge_path,
            data_path=data_path,
            mask_path=mask_path,
            resolution=resolution,
            hole_range=hole_range,
            fix_crops=fix_crops,
            mask_aug=mask_aug,
            **super_kwargs,
        )

        # Build secondary dataset (same kwargs, separate mask pool)
        self.ds_secondary = TSRDataset(
            edge_path=edge_path_2,
            data_path=data_path_2,
            mask_path=mask_path_2,
            resolution=resolution,
            hole_range=hole_range,
            fix_crops=fix_crops,
            mask_aug=mask_aug,
            **super_kwargs,
        )

        self._primary_size = len(self.ds_primary)
        self._secondary_size = len(self.ds_secondary)
        total_size = self._primary_size + self._secondary_size

        # Use primary dataset's shape for the base class
        name = f"{self.ds_primary.name}+{self.ds_secondary.name}"
        raw_shape = [total_size] + list(self.ds_primary.image_shape)

        # Initialize base Dataset (skip TSRDataset.__init__, go straight to Dataset)
        Dataset.__init__(self, name=name, raw_shape=raw_shape, **super_kwargs)
        
        # Store resolution from primary
        self._resolution = resolution
        self._hole_range = hole_range
        
        print(f"[SlottedTSRDataset] Primary: {self._primary_size}, Secondary: {self._secondary_size}")
        print(f"[SlottedTSRDataset] Total: {total_size}, Slot ratio: {slot_ratio}")

    def __len__(self):
        return self._primary_size + self._secondary_size

    def __getitem__(self, idx):
        """Route to primary or secondary dataset based on index."""
        if idx < self._primary_size:
            return self.ds_primary[idx]
        else:
            return self.ds_secondary[idx - self._primary_size]

    @property
    def primary_size(self):
        return self._primary_size

    @property
    def secondary_size(self):
        return self._secondary_size

    @property
    def slot_ratio(self):
        return self._slot_ratio

    def close(self):
        self.ds_primary.close()
        self.ds_secondary.close()

    # Forward resolution/shape from primary
    @property
    def resolution(self):
        return self.ds_primary.resolution

    @property
    def image_shape(self):
        return self.ds_primary.image_shape

    def _load_raw_image(self, raw_idx):
        pass  # Not used directly — __getitem__ routes to sub-datasets

    def _load_raw_edge(self, raw_idx):
        pass  # Not used directly — __getitem__ routes to sub-datasets

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

"""Training loop for Sketch Inpainting with GAN (Generator + Discriminator).

This module implements the main training loop for edge/sketch inpainting using
a GAN architecture. Key features:

- **Generator + Discriminator Training**: Alternates between G and D optimization
- **Cosine LR Decay**: Learning rate decay with configurable warmup period
- **Separate G/D Learning Rates**: Supports different LR for generator and discriminator
- **Checkpoint Resume**: Full state restoration including optimizer, RNG states, and LR config
- **Snapshot Visualization**: Multiple output types including:
  - Full edge predictions
  - Masked composites
  - Highlighted mask regions (RGB)
  - Real-fake comparison grids
- **Validation Metrics**: Precision, Recall, F1, IoU, SSIM computed on validation set

Usage:
    Called by train_tsr.py via dnnlib.util.construct_class_by_name()
"""

import os
import math
import random
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

from metrics.metrics_edge import compute_edge_metrics, EdgeMetricAccumulator
from datasets.dataset_tsr_512 import SlottedSampler

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0, max_images=None):
    """Setup grid for visualization snapshots.
    
    Grid count is capped so the total pixel budget stays roughly constant
    regardless of resolution (calibrated to ~120 images at 256px).
    This avoids OOM and slow snapshot saves at higher resolutions (e.g. 512px).
    
    Args:
        training_set: Dataset to draw images from
        random_seed: Seed for reproducible grid selection
        max_images: If set, further cap the grid to this many images
    """
    rnd = np.random.RandomState(random_seed)
    res = training_set.image_shape[1]  # spatial resolution (H)

    # Base grid at 256px → 30×16 = 480, but cap at 120 for speed.
    # Scale inversely with (res/256)^2 to keep pixel budget constant.
    pixel_budget = 120 * (256 ** 2)  # ~7.8M pixels (≈120 images at 256px)
    max_from_budget = max(1, int(pixel_budget / (res * res)))

    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)
    total_grid = gw * gh

    # Apply pixel-budget cap first
    total_grid = min(total_grid, max_from_budget)

    # Then apply caller-provided cap
    if max_images is not None and max_images < total_grid:
        total_grid = max_images

    # Refit gw/gh to total_grid (keep aspect ratio close to original)
    gw = min(gw, total_grid)
    gh = max(1, total_grid // gw)
    total_grid = gw * gh

    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(total_grid)]

    # Load data: img, edge, mask, idx
    batch_data = [training_set[i] for i in grid_indices]
    imgs, edges, masks, _ = zip(*batch_data)

    return (gw, gh), np.stack(imgs), np.stack(edges), np.stack(masks)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    """Save image grid to file.
    
    Args:
        img: Image array of shape (N, C, H, W)
        fname: Output filename
        drange: Dynamic range (lo, hi) of input values
        grid_size: Tuple (gw, gh) for grid layout
    """
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    elif C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def save_masked_with_transparent(fake_edges, masks, fname, grid_size, overlay_color=(100, 150, 255), alpha=0.3):
    """Save predicted edges with transparent color overlay on mask region.
    
    Shows the predicted edges clearly, with a semi-transparent color tint 
    on the mask region to indicate where inpainting occurred.
    
    Args:
        fake_edges: Predicted edge array of shape (N, 1, H, W) in range [0, 1]
        masks: Mask array of shape (N, 1, H, W), 1=hole (inpainted region)
        fname: Output filename
        grid_size: Tuple (gw, gh) for grid layout
        overlay_color: RGB color for mask region tint (default: light blue)
        alpha: Transparency of overlay (0=invisible, 1=opaque), default 0.3
    """
    N, C, H, W = fake_edges.shape
    
    # Create RGB output
    output = np.zeros((N, 3, H, W), dtype=np.float32)
    
    for i in range(N):
        mask = masks[i, 0]  # (H, W), 1=hole
        fake = fake_edges[i, 0]  # (H, W), predicted edges
        
        # Convert grayscale edge to RGB (white edge on black background)
        edge_rgb = fake * 255  # (H, W)
        
        for c in range(3):
            # Base: predicted edge as grayscale
            output[i, c] = edge_rgb
            # Add transparent color overlay on mask region
            # Formula: output = edge * (1 - alpha) + color * alpha (only in mask)
            output[i, c] = output[i, c] * (1 - mask * alpha) + mask * alpha * overlay_color[c]
    
    output = output.clip(0, 255).astype(np.uint8)
    
    # Reshape to grid
    gw, gh = grid_size
    output = output.reshape(gh, gw, 3, H, W)
    output = output.transpose(0, 3, 1, 4, 2)
    output = output.reshape(gh * H, gw * W, 3)
    
    PIL.Image.fromarray(output, 'RGB').save(fname)

#----------------------------------------------------------------------------

def cosine_decay_lr(base_lr, cur_kimg, warmup_kimg, decay_kimg, min_ratio=0.01, absolute_min_lr=0.0):
    """Compute learning rate with cosine decay after warmup.
    
    Args:
        base_lr: Initial/base learning rate
        cur_kimg: Current training progress in kimg
        warmup_kimg: kimg to use constant base_lr (no decay during warmup)
        decay_kimg: Total kimg from warmup_kimg to end of decay
        min_ratio: Minimum LR ratio at end of decay (default 0.01 = 1% of base_lr)
        absolute_min_lr: Hard floor for learning rate (default 0.0)
    
    Returns:
        Current learning rate
    
    Schedule:
        [0, warmup_kimg): lr = base_lr (constant)
        [warmup_kimg, warmup_kimg + decay_kimg]: lr decays from base_lr to base_lr * min_ratio
        > warmup_kimg + decay_kimg: lr = base_lr * min_ratio (constant minimum)
    """
    import math
    
    if cur_kimg < warmup_kimg:
        # Warmup phase: constant LR
        return base_lr
    
    # Progress through decay phase (0 to 1)
    decay_progress = (cur_kimg - warmup_kimg) / max(decay_kimg, 1e-8)
    decay_progress = min(decay_progress, 1.0)  # Cap at 1.0
    
    # Cosine decay from 1 to min_ratio
    # cos(0) = 1, cos(pi) = -1, so (1 + cos(pi*t))/2 goes from 1 to 0
    cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
    
    # Scale from [1, 0] to [1, min_ratio]
    lr_ratio = min_ratio + (1 - min_ratio) * cosine_factor
    
    new_lr = base_lr * lr_ratio
    return max(new_lr, absolute_min_lr)

#----------------------------------------------------------------------------
# Validation bank
#   * whole val set (not the 30-image snapshot grid)
#   * deterministic crops + K deterministic masks per image (identical every eval)
#   * metrics reported overall AND per hole-ratio bucket
#----------------------------------------------------------------------------

VAL_BUCKETS = [            # (name, lo, hi) on hole ratio = mean(mask)
    ('h00_10',  0.00, 0.10),
    ('h10_30',  0.10, 0.30),
    ('h30_50',  0.30, 0.50),
    ('h50_100', 0.50, 1.01),
]
VAL_METRIC_NAMES = ['precision', 'recall', 'f1', 'iou', 'ssim', 'psnr', 'mae', 'connectivity', 'hausdorff',
                    'f1_img', 'f1_t2', 'f1_t5', 'precision_t2', 'recall_t2', 'hd_valid_frac']
VAL_BUCKET_METRICS = ['precision', 'recall', 'f1', 'iou', 'hausdorff', 'f1_t2', 'f1_t5', 'connectivity']


def build_val_bank(val_set, masks_per_image=3, max_samples=1500, face=False):
    """Load a fixed validation bank as uint8 CPU tensors.
    face=True: face/part holes from box annotations (images without boxes skipped)."""
    has_eval = hasattr(val_set, 'get_eval_sample')
    if face and not (has_eval and getattr(val_set, 'has_face_boxes', False)):
        return None
    K = max(1, int(masks_per_image)) if has_eval else 1
    if not has_eval:
        print('  [Val] WARNING: dataset has no get_eval_sample(); using __getitem__ '
              '(masks are NOT deterministic across evaluations).')
    imgs, edges, masks = [], [], []
    for i in range(len(val_set)):
        for k in range(K):
            if len(imgs) >= max_samples:
                break
            if face:
                s = val_set.get_eval_sample(i, k, face=True)
                if s is None:
                    break
                img, edge, mask, _ = s
            else:
                img, edge, mask, _ = val_set.get_eval_sample(i, k) if has_eval else val_set[i]
            imgs.append(np.asarray(img, dtype=np.uint8))
            edges.append(np.asarray(edge, dtype=np.uint8))
            masks.append((np.asarray(mask) > 0.5).astype(np.uint8))
    if not imgs:
        return None
    bank = (torch.from_numpy(np.stack(imgs)),
            torch.from_numpy(np.stack(edges)),
            torch.from_numpy(np.stack(masks)))
    ratios = bank[2].float().mean(dim=[1, 2, 3])
    counts = {name: int(((ratios >= lo) & (ratios < hi)).sum()) for name, lo, hi in VAL_BUCKETS}
    print(f'  [Val{"-face" if face else ""}] bank: {len(imgs)} samples ({len(val_set)} images x {K} masks). '
          f'Hole-ratio buckets: {counts}')
    return bank


@torch.no_grad()
def evaluate_val_bank(G_eval, bank, device, batch_gpu):
    """Streaming evaluation: one EdgeMetricAccumulator per group ('all' + buckets).
    Nothing is concatenated on the GPU, so memory is independent of bank size."""
    imgs_u8, edges_u8, masks_u8 = bank
    N = imgs_u8.shape[0]
    ratios = masks_u8.float().mean(dim=[1, 2, 3])
    groups = {'all': (0.0, 2.0)}
    groups.update({name: (lo, hi) for name, lo, hi in VAL_BUCKETS})
    accs = {name: EdgeMetricAccumulator() for name in groups}
    counts = {name: 0 for name in groups}
    for s in range(0, N, batch_gpu):
        img  = imgs_u8[s:s + batch_gpu].to(device).float() / 127.5 - 1.0
        edge = edges_u8[s:s + batch_gpu].to(device).float() / 255.0
        mask = masks_u8[s:s + batch_gpu].to(device).float()
        pred = torch.sigmoid(G_eval(img, edge, mask))
        r = ratios[s:s + batch_gpu]
        for name, (lo, hi) in groups.items():
            sel = ((r >= lo) & (r < hi)).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            sel = sel.to(device)
            accs[name].update(pred[sel], edge[sel], mask[sel])
            counts[name] += int(sel.numel())
    return {name: (dict(accs[name].compute(), n=counts[name]) if counts[name] > 0 else None)
            for name in groups}

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',
    training_set_kwargs     = {},
    val_set_kwargs          = {},
    data_loader_kwargs      = {},
    G_kwargs                = {},
    D_kwargs                = {},
    G_opt_kwargs            = {},
    D_opt_kwargs            = {},
    use_discriminator       = True,
    augment_kwargs          = None,
    loss_kwargs             = {},
    metrics                 = [],
    random_seed             = 42,
    num_gpus                = 1,
    rank                    = 0,
    batch_size              = 4,
    batch_gpu               = 4,
    ema_kimg                = 10,
    ema_rampup              = None,
    G_reg_interval          = None,
    D_reg_interval          = 16,
    augment_p               = 0,
    ada_target              = None,
    ada_interval            = 4,
    ada_kimg                = 500,
    total_kimg              = 25000,
    kimg_per_tick           = 2,
    image_snapshot_ticks    = 2,
    network_snapshot_ticks  = 5,
    resume_pkl              = None,
    resume_ckpt             = None,
    cudnn_benchmark         = True,
    allow_tf32              = False,
    lr_decay_start_kimg     = 200,      # Start LR decay after this many kimg (0 = decay from start)
    lr_min_ratio            = 0.01,     # Minimum LR ratio at end of decay (0.01 = 1% of base LR)
    min_lr                  = 0.0,      # Absolute floor for G
    min_lr_d                = 0.0,      # Absolute floor for D
    lr_warmup_kimg          = 0,        # Linear LR ramp from 0 over this many kimg, per iteration (0 = off)
    grad_clip               = None,     # Clip the gradient norm of each phase module (None = off)
    lr_user_set             = True,     # Whether --lr was explicitly set by user
    lr_d_user_set           = True,     # Whether --lr-d was explicitly set by user
    sync_bn                 = True,     # Convert BatchNorm -> SyncBatchNorm when num_gpus > 1
    val_masks_per_image     = 3,        # Deterministic masks per val image
    val_max_samples         = 1500,     # Cap on val bank size (memory)
    abort_fn                = None,
    progress_fn             = None,
):
    """Main training loop for Sketch Inpainting GAN."""

    # Initialize
    start_time = time.time()
    
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * 2 + rank)
    torch.manual_seed(random_seed * 2 + rank)
    random.seed(random_seed * 2 + rank)  # ADDED: Seed Python random for dataset crop consistency
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    # Load checkpoint data
    cur_nimg = 0
    batch_idx = 0
    cur_tick = 0
    resume_checkpoint_data = None
    if resume_ckpt is not None: 
        if os.path.exists(resume_ckpt):
            if rank == 0: 
                print(f'Loading full resume checkpoint: {resume_ckpt}')
            
            try:   # checkpoints hold numpy RNG states -> torch>=2.6 needs weights_only=False
                resume_checkpoint_data = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
            except TypeError:   # torch < 1.13 has no weights_only argument
                resume_checkpoint_data = torch.load(resume_ckpt, map_location='cpu')
            cur_nimg = int(resume_checkpoint_data.get('cur_nimg', 0))
            batch_idx = int(resume_checkpoint_data.get('batch_idx', 0))
            cur_tick = int(resume_checkpoint_data.get('cur_tick', 0))
            
            # IMPORTANT: Increment cur_tick by 1 since checkpoint was saved at end of tick
            # This prevents re-logging the same tick and ensures fresh start
            cur_tick += 1
            
            if rank == 0: 
                print(f'Resuming from kimg {cur_nimg / 1e3:.1f}, batch index {batch_idx}, next tick {cur_tick}.')

        else:
            if rank == 0: 
                print(f'Warning: resume_ckpt not found: {resume_ckpt}')

    # Load training set
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    val_set = dnnlib.util.construct_class_by_name(**val_set_kwargs)

    # Detect slots mode (SlottedTSRDataset has ds_secondary attribute)
    is_slots = hasattr(training_set, 'ds_secondary')
    
    if is_slots:
        if rank == 0:
            print(f'[Slots] Mode enabled: primary={training_set.primary_size}, secondary={training_set.secondary_size}, ratio={training_set.slot_ratio}')
        training_set_sampler = SlottedSampler(
            primary_size=training_set.primary_size,
            secondary_size=training_set.secondary_size,
            batch_size=batch_size,
            slot_ratio=training_set.slot_ratio,
            rank=rank,
            num_replicas=num_gpus,
            seed=random_seed,
            resume_idx=batch_idx * batch_size,
        )
    else:
        training_set_sampler = misc.InfiniteSampler(
            dataset=training_set,
            rank=rank,
            num_replicas=num_gpus,
            seed=random_seed,
            resume_idx=batch_idx * batch_size
        )
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set,
        sampler=training_set_sampler,
        batch_size=batch_size // num_gpus,
        **data_loader_kwargs
    ))
    
    if rank == 0:
        print()
        print('Num images:', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Resolution:', training_set.resolution)
        print()
    
    # Construct networks
    if rank == 0:
        print('Constructing networks...')
    
    # Generator
    G = dnnlib.util.construct_class_by_name(**G_kwargs).train().requires_grad_(False).to(device)
    
    # Discriminator
    if use_discriminator:
        D = dnnlib.util.construct_class_by_name(**D_kwargs).train().requires_grad_(False).to(device)
    else:
        D = None
    
    # Broadcast params for multi-GPU
    if num_gpus > 1:
        for param in G.parameters():
            torch.distributed.broadcast(param, src=0)
        for buffer in G.buffers():
            torch.distributed.broadcast(buffer, src=0)
        if D is not None:
            for param in D.parameters():
                torch.distributed.broadcast(param, src=0)
            for buffer in D.buffers():
                torch.distributed.broadcast(buffer, src=0)
            
    # EMA copy
    G_ema = copy.deepcopy(G).eval()

    # Resume from checkpoint
    if resume_checkpoint_data is not None:
        # FIX: strict=False silently left any renamed/new layer at random init
        # (e.g. after swapping FFC <-> MDTA blocks or adding mask_embed / changing
        # D's input channels). Load non-strictly but REPORT every mismatch, and
        # skip tensors whose shape changed instead of crashing.
        def _load(module, state, tag):
            own = module.state_dict()
            ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
            bad_shape = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
            res = module.load_state_dict(ok, strict=False)
            if rank == 0:
                missing = [k for k in res.missing_keys if k not in bad_shape]
                if missing or res.unexpected_keys or bad_shape:
                    print(f'  [Resume:{tag}] WARNING  missing={len(missing)}  '
                          f'unexpected={len(res.unexpected_keys)}  shape-mismatch={len(bad_shape)}')
                    for k in (missing + bad_shape)[:20]:
                        print(f'      not loaded (random init): {k}')
                else:
                    print(f'  [Resume:{tag}] all {len(ok)} tensors loaded')
        if 'G_state' in resume_checkpoint_data:
            _load(G, resume_checkpoint_data['G_state'], 'G')
        if 'G_ema_state' in resume_checkpoint_data:
            _load(G_ema, resume_checkpoint_data['G_ema_state'], 'G_ema')
        if D is not None and 'D_state' in resume_checkpoint_data:
            _load(D, resume_checkpoint_data['D_state'], 'D')

    # Print network summary
    if rank == 0:
        dummy_img = torch.empty([batch_gpu, 1, training_set.resolution, training_set.resolution], device=device)
        dummy_edge = torch.empty([batch_gpu, 1, training_set.resolution, training_set.resolution], device=device)
        dummy_mask = torch.empty([batch_gpu, 1, training_set.resolution, training_set.resolution], device=device)
        
        print('Generator:')
        try:
             misc.print_module_summary(G, [dummy_img, dummy_edge, dummy_mask])
        except Exception as e:
             print(f"Failed to print summary: {e}")
        
        if D is not None:
            print('Discriminator:')
            try:
                 misc.print_module_summary(D, [dummy_edge, dummy_mask])
            except Exception as e:
                 print(f"Failed to print summary: {e}")

    # Export sample images
    grid_size = None
    # Slot2 snapshot variables (initialized to None, set if slots active)
    grid_size_s2 = None
    grid_imgs_s2_split = None
    grid_edges_s2_split = None
    grid_masks_s2_split = None
    grid_edges_s2_t = None
    grid_masks_s2_t = None
    
    if rank == 0:
        print('Exporting sample images...')
        
        rng_state_backup = np.random.get_state()
        torch_state_backup = torch.get_rng_state()
        random_state_backup = random.getstate()
        
        np.random.seed(random_seed * 2 + rank)
        torch.manual_seed(random_seed * 2 + rank)
        random.seed(random_seed * 2 + rank)

        grid_size, raw_imgs, raw_edges, raw_masks = setup_snapshot_image_grid(training_set=val_set, random_seed=random_seed)
        
        np.random.set_state(rng_state_backup)
        torch.set_rng_state(torch_state_backup)
        random.setstate(random_state_backup)
        
        # Preprocess
        # Image (Gray): 0..255 -> -1..1
        grid_imgs_t = (torch.from_numpy(raw_imgs).float().to(device) / 127.5) - 1.0
        
        # Edge/Mask: 0..255 -> 0..1
        grid_edges_t = torch.from_numpy(raw_edges).float().to(device) / 255.0
        grid_masks_t = torch.from_numpy(raw_masks).float().to(device)
        
        save_image_grid(grid_imgs_t.cpu().numpy(), os.path.join(run_dir, 'reals_imgs.png'), drange=[-1, 1], grid_size=grid_size)
        save_image_grid(grid_edges_t.cpu().numpy() * 255.0, os.path.join(run_dir, 'reals_edges.png'), drange=[0, 255], grid_size=grid_size)
        save_image_grid(grid_masks_t.cpu().numpy() * 255.0, os.path.join(run_dir, 'reals_masks.png'), drange=[0, 255], grid_size=grid_size)

        grid_imgs_split = grid_imgs_t.split(batch_gpu)
        grid_edges_split = grid_edges_t.split(batch_gpu)
        grid_masks_split = grid_masks_t.split(batch_gpu)
        
        # Initial fakes (torch.no_grad: no computation graph needed here)
        fakes_edge_out = []
        with torch.no_grad():
            for img, edge, mask in zip(grid_imgs_split, grid_edges_split, grid_masks_split):
                edge_l = G_ema(img, edge, mask)
                fakes_edge_out.append(torch.sigmoid(edge_l).cpu())

        fakes_edge_out = torch.cat(fakes_edge_out).numpy()
        save_image_grid(fakes_edge_out * 255.0, os.path.join(run_dir, 'fakes_edges_init.png'), drange=[0, 255], grid_size=grid_size)
        
        # ---- Slot2 snapshot grid (only if slots mode) ----
        if is_slots:
            print('[Slots] Building secondary snapshot grid (n=50)...')
            
            rng_state_backup2 = np.random.get_state()
            torch_state_backup2 = torch.get_rng_state()
            random_state_backup2 = random.getstate()
            
            np.random.seed(random_seed * 3 + rank)
            torch.manual_seed(random_seed * 3 + rank)
            random.seed(random_seed * 3 + rank)
            
            grid_size_s2, raw_imgs_s2, raw_edges_s2, raw_masks_s2 = setup_snapshot_image_grid(
                training_set=training_set.ds_secondary, random_seed=random_seed + 100, max_images=50
            )
            
            np.random.set_state(rng_state_backup2)
            torch.set_rng_state(torch_state_backup2)
            random.setstate(random_state_backup2)
            
            # Preprocess slot2
            grid_imgs_s2_t = (torch.from_numpy(raw_imgs_s2).float().to(device) / 127.5) - 1.0
            grid_edges_s2_t = torch.from_numpy(raw_edges_s2).float().to(device) / 255.0
            grid_masks_s2_t = torch.from_numpy(raw_masks_s2).float().to(device)
            
            save_image_grid(grid_imgs_s2_t.cpu().numpy(), os.path.join(run_dir, 'reals_imgs_slot2.png'), drange=[-1, 1], grid_size=grid_size_s2)
            save_image_grid(grid_edges_s2_t.cpu().numpy() * 255.0, os.path.join(run_dir, 'reals_edges_slot2.png'), drange=[0, 255], grid_size=grid_size_s2)
            save_image_grid(grid_masks_s2_t.cpu().numpy() * 255.0, os.path.join(run_dir, 'reals_masks_slot2.png'), drange=[0, 255], grid_size=grid_size_s2)
            
            grid_imgs_s2_split = grid_imgs_s2_t.split(batch_gpu)
            grid_edges_s2_split = grid_edges_s2_t.split(batch_gpu)
            grid_masks_s2_split = grid_masks_s2_t.split(batch_gpu)
            
            # Initial fakes slot2 (torch.no_grad: no computation graph needed here)
            fakes_edge_s2_out = []
            with torch.no_grad():
                for img, edge, mask in zip(grid_imgs_s2_split, grid_edges_s2_split, grid_masks_s2_split):
                    edge_l = G_ema(img, edge, mask)
                    fakes_edge_s2_out.append(torch.sigmoid(edge_l).cpu())
            fakes_edge_s2_out = torch.cat(fakes_edge_s2_out).numpy()
            save_image_grid(fakes_edge_s2_out * 255.0, os.path.join(run_dir, 'fakes_edges_init_slot2.png'), drange=[0, 255], grid_size=grid_size_s2)
            
            print(f'[Slots] Secondary snapshot grid: {grid_size_s2[0]}x{grid_size_s2[1]} = {grid_size_s2[0]*grid_size_s2[1]} images')

    # Fixed validation bank (rank 0 only)
    val_bank = None
    face_bank = None
    if rank == 0:
        val_bank = build_val_bank(val_set, masks_per_image=val_masks_per_image,
                                  max_samples=val_max_samples)
        face_bank = build_val_bank(val_set, masks_per_image=val_masks_per_image,
                                   max_samples=val_max_samples, face=True)
    # every rank must agree on whether face metrics are reported (report0 names)
    has_face_bank = torch.tensor([1.0 if face_bank is not None else 0.0], device=device)
    if num_gpus > 1:
        torch.distributed.broadcast(has_face_bank, src=0)
    has_face_bank = bool(has_face_bank.item() > 0)

    # Restore RNG states from checkpoint
    if resume_checkpoint_data is not None:
        if rank == 0:
            print('Restoring RNG states from checkpoint...')
        if 'rng_numpy' in resume_checkpoint_data:
            np.random.set_state(resume_checkpoint_data['rng_numpy'])
        if 'rng_python' in resume_checkpoint_data:
            random.setstate(resume_checkpoint_data['rng_python'])
        if 'rng_torch' in resume_checkpoint_data:
            torch.set_rng_state(resume_checkpoint_data['rng_torch'])
        if 'rng_cuda' in resume_checkpoint_data and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(resume_checkpoint_data['rng_cuda'][rank], device)
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to restore CUDA RNG state: {e}")

    # Distribute across GPUs
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()

    # FFC blocks use BatchNorm with only batch_gpu samples per GPU.
    # SyncBatchNorm computes batch statistics across all GPUs (8 samples with
    # 2x4). Done AFTER print_module_summary (rank-0-only forward would hang
    # with SyncBN) and after G_ema was copied (G_ema keeps plain BN; it only
    # runs in eval mode, where both are identical; state_dict keys match).
    if num_gpus > 1 and sync_bn:
        n_bn_G = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in G.modules())
        G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
        if D is not None:
            n_bn_D = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in D.modules())
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
        else:
            n_bn_D = 0
        if rank == 0:
            print(f'  SyncBatchNorm: converted {n_bn_G} BN layers in G, {n_bn_D} in D')
    
    for name, module in [('G', G), ('D', D)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(
                module, device_ids=[device], broadcast_buffers=False
            )
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases
    if rank == 0:
        print('Setting up training phases...')
        print(f'  G learning rate: {G_opt_kwargs.get("lr", "not set")}')
        if D_opt_kwargs is not None:
            print(f'  D learning rate: {D_opt_kwargs.get("lr", "not set")}')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs)
    
    # ================================================================
    # Resolve base learning rates:
    #   - If user explicitly set --lr / --lr-d, use those values
    #   - If resuming without explicit --lr, read from checkpoint
    #   - Otherwise, use the default from G_opt_kwargs / D_opt_kwargs
    # ================================================================
    base_lr_G = G_opt_kwargs.get('lr', 0.0001) if G_opt_kwargs is not None else 0.0001
    base_lr_D = D_opt_kwargs.get('lr', 0.0001) if D_opt_kwargs is not None else 0.0001
    
    if resume_checkpoint_data is not None:
        if not lr_user_set and 'base_lr_G' in resume_checkpoint_data:
            base_lr_G = resume_checkpoint_data['base_lr_G']
            if rank == 0:
                # NOTE: training_options.json still shows the CLI default (1e-4),
                # but THIS value is what is used. Pass --lr to override.
                print(f'  [Resume] G base LR restored from checkpoint: {base_lr_G} '
                      f'(training_options.json lr={G_opt_kwargs.get("lr")} is ignored)')
        elif lr_user_set and rank == 0:
            print(f'  [Resume] G base LR overridden by user: {base_lr_G}')
        
        if not lr_d_user_set and 'base_lr_D' in resume_checkpoint_data:
            base_lr_D = resume_checkpoint_data['base_lr_D']
            if rank == 0:
                print(f'  [Resume] D base LR restored from checkpoint: {base_lr_D}')
        elif lr_d_user_set and rank == 0:
            print(f'  [Resume] D base LR overridden by user: {base_lr_D}')
    
    # Generator phase
    # When resuming without explicit --lr, construct optimizer with checkpoint base_lr
    g_opt_kwargs_actual = dict(G_opt_kwargs)
    g_opt_kwargs_actual['lr'] = base_lr_G
    G_opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **g_opt_kwargs_actual)
    if resume_checkpoint_data and 'G_opt_state' in resume_checkpoint_data:
        G_opt.load_state_dict(resume_checkpoint_data['G_opt_state'])
        # Override LR with resolved base_lr (user-specified or checkpoint-restored)
        for param_group in G_opt.param_groups:
            param_group['lr'] = base_lr_G
        if rank == 0:
            print(f'  G optimizer LR set to {base_lr_G}')
    
    # Discriminator phase
    if use_discriminator:
        d_opt_kwargs_actual = dict(D_opt_kwargs)
        d_opt_kwargs_actual['lr'] = base_lr_D
        D_opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **d_opt_kwargs_actual)
        if resume_checkpoint_data and 'D_opt_state' in resume_checkpoint_data:
            D_opt.load_state_dict(resume_checkpoint_data['D_opt_state'])
            # Override LR with resolved base_lr (user-specified or checkpoint-restored)
            for param_group in D_opt.param_groups:
                param_group['lr'] = base_lr_D
            if rank == 0:
                print(f'  D optimizer LR set to {base_lr_D}')
    else:
        D_opt = None
    
    phases = []
    
    # Generator phase
    phases.append(dnnlib.EasyDict(name='Gmain', module=G, opt=G_opt, interval=1))
    
    # Discriminator phases
    if use_discriminator:
        phases.append(dnnlib.EasyDict(name='Dmain', module=D, opt=D_opt, interval=1))
        if D_reg_interval is not None and D_reg_interval > 0:
            phases.append(dnnlib.EasyDict(name='Dreg', module=D, opt=D_opt, interval=D_reg_interval))
    
    # Calculate decay duration (from start_kimg to total_kimg)
    decay_kimg = max(total_kimg - lr_decay_start_kimg, 0)
    
    if rank == 0:
        print(f'  LR decay: base_lr_G={base_lr_G}, base_lr_D={base_lr_D}')
        print(f'  LR decay: starts at {lr_decay_start_kimg} kimg, duration {decay_kimg} kimg')
        print(f'  LR decay: min ratio {lr_min_ratio} (G: {base_lr_G * lr_min_ratio:.6f}, D: {base_lr_D * lr_min_ratio:.6f})')
        if min_lr > 0 or min_lr_d > 0:
            print(f'  LR decay: absolute floors enabled (G_min: {min_lr:.6f}, D_min: {min_lr_d:.6f})')
        if lr_warmup_kimg > 0:
            print(f'  LR warm-up: linear from 0 over the first {lr_warmup_kimg} kimg')
        if grad_clip:
            print(f'  Gradient clipping: max norm {grad_clip}')
    
    # Event timers
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Logs
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_jsonl = None
    stats_tfevents = None
    
    if rank == 0:
        # Use append mode if resuming, write mode otherwise
        stats_file_mode = 'at' if resume_checkpoint_data is not None else 'wt'
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), stats_file_mode)
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        if resume_checkpoint_data is not None:
            print(f'  Resuming from: cur_nimg={cur_nimg}, cur_tick={cur_tick}, batch_idx={batch_idx}')
            print(f'  First log step will be: {int(cur_nimg / 1e3)}')
        print()
    
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
    
    while True:
        # Fetch training data
        with torch.autograd.profiler.record_function('data_fetch'):
            # Dataset returns: img, edge, mask, idx
            phase_img, phase_edge, phase_mask, phase_indices = next(training_set_iterator)
            
            # Normalize inputs
            # Image (Gray): 0..255 -> -1..1
            phase_img = ((phase_img.to(device).to(torch.float32) / 127.5) - 1.0).split(batch_gpu)
            
            # Edge: 0..255 -> 0..1
            phase_edge = (phase_edge.to(device).to(torch.float32) / 255.0).split(batch_gpu)
            
            # Mask: 0..1 Float
            phase_mask = phase_mask.to(device).to(torch.float32).split(batch_gpu)
                        
        # Execute training phases
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue

            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients
            for round_idx, (img, edge, mask) in enumerate(zip(phase_img, phase_edge, phase_mask)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                
                loss.accumulate_gradients(
                    phase=phase.name,
                    full_image=img,
                    full_edge=edge,
                    mask=mask,
                    sync=sync,
                    gain=gain
                )

            # Update weights
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(phase.module.parameters(), grad_clip)
                if lr_warmup_kimg > 0 and cur_nimg < lr_warmup_kimg * 1000:
                    # Linear warm-up (ZITS): lr = base * iteration / warmup_iterations
                    warm = cur_nimg / (lr_warmup_kimg * 1000)
                    base = base_lr_G if phase.opt is G_opt else base_lr_D
                    for param_group in phase.opt.param_groups:
                        param_group['lr'] = base * warm
                phase.opt.step()
                
                # Log learning rate
                current_lr = phase.opt.param_groups[0]['lr']
                training_stats.report0(f'Progress/lr_{phase.name}', current_lr)
            
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # ================================================================
        # Update learning rate with cosine decay (after warmup period)
        # ================================================================
        cur_kimg = cur_nimg / 1000
        
        # Compute decayed LRs
        new_lr_G = cosine_decay_lr(base_lr_G, cur_kimg, lr_decay_start_kimg, decay_kimg, lr_min_ratio, min_lr)
        new_lr_D = cosine_decay_lr(base_lr_D, cur_kimg, lr_decay_start_kimg, decay_kimg, lr_min_ratio, min_lr_d)
        
        # Update G optimizer LR
        for param_group in G_opt.param_groups:
            param_group['lr'] = new_lr_G
        
        # Update D optimizer LR
        if D_opt is not None:
            for param_group in D_opt.param_groups:
                param_group['lr'] = new_lr_D
        
        # Log current LRs
        training_stats.report0('Progress/lr_G', new_lr_G)
        if D_opt is not None:
            training_stats.report0('Progress/lr_D', new_lr_D)

        # Print status line
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print('Aborting...')

        # Image snapshot and metrics
        snapshot_metrics = None
        should_snapshot = (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0)
        
        if should_snapshot:
            if rank == 0:
                fakes_edge_out = []
                fakes_edge_masked_out = []
                
                with torch.no_grad():
                    for img, edge, mask in zip(grid_imgs_split, grid_edges_split, grid_masks_split):
                        edge_l = G_ema(img, edge, mask)
                        
                        fake_edge = (torch.sigmoid(edge_l) > 0.5).float()
                        fakes_edge_out.append(fake_edge.cpu())
                        
                        fake_edge_composite = edge * (1 - mask) + fake_edge * mask
                        fakes_edge_masked_out.append(fake_edge_composite.cpu())
                
                fakes_edge_out = torch.cat(fakes_edge_out).numpy()
                fakes_edge_masked_out = torch.cat(fakes_edge_masked_out).numpy()
                
                # Standard snapshots
                save_image_grid(fakes_edge_out * 255.0, os.path.join(run_dir, f'fakes_edges_full_{cur_nimg//1000:06d}.png'), drange=[0, 255], grid_size=grid_size)
                save_image_grid(fakes_edge_masked_out * 255.0, os.path.join(run_dir, f'fakes_edges_masked_{cur_nimg//1000:06d}.png'), drange=[0, 255], grid_size=grid_size)
                
                # NEW: Masked with transparent overlay (RGB, mask region tinted)
                masks_np = grid_masks_t.cpu().numpy()
                save_masked_with_transparent(
                    fake_edges=fakes_edge_out,
                    masks=masks_np,
                    fname=os.path.join(run_dir, f'fakes_edges_highlight_{cur_nimg//1000:06d}.png'),
                    grid_size=grid_size,
                    overlay_color=(100, 150, 255),  # Light blue for mask region
                    alpha=0.3  # 30% transparency
                )
                
                # ---- Slot2 snapshots (only if slots mode) ----
                if is_slots and grid_imgs_s2_split is not None:
                    fakes_edge_s2_out = []
                    fakes_edge_s2_masked_out = []
                    
                    with torch.no_grad():
                        for img, edge, mask in zip(grid_imgs_s2_split, grid_edges_s2_split, grid_masks_s2_split):
                            edge_l = G_ema(img, edge, mask)
                            fake_edge = (torch.sigmoid(edge_l) > 0.5).float()
                            fakes_edge_s2_out.append(fake_edge.cpu())
                            fake_edge_composite = edge * (1 - mask) + fake_edge * mask
                            fakes_edge_s2_masked_out.append(fake_edge_composite.cpu())
                    
                    fakes_edge_s2_out = torch.cat(fakes_edge_s2_out).numpy()
                    fakes_edge_s2_masked_out = torch.cat(fakes_edge_s2_masked_out).numpy()
                    
                    save_image_grid(fakes_edge_s2_out * 255.0, os.path.join(run_dir, f'fakes_edges_full_slot2_{cur_nimg//1000:06d}.png'), drange=[0, 255], grid_size=grid_size_s2)
                    save_image_grid(fakes_edge_s2_masked_out * 255.0, os.path.join(run_dir, f'fakes_edges_masked_slot2_{cur_nimg//1000:06d}.png'), drange=[0, 255], grid_size=grid_size_s2)
                    
                    masks_s2_np = grid_masks_s2_t.cpu().numpy()
                    save_masked_with_transparent(
                        fake_edges=fakes_edge_s2_out,
                        masks=masks_s2_np,
                        fname=os.path.join(run_dir, f'fakes_edges_highlight_slot2_{cur_nimg//1000:06d}.png'),
                        grid_size=grid_size_s2,
                        overlay_color=(255, 150, 100),  # Orange for slot2 (distinguish from primary)
                        alpha=0.3
                    )
                
                # Validation metrics on the fixed bank
                val_results = evaluate_val_bank(G_ema, val_bank, device, batch_gpu)
                snapshot_metrics = val_results['all']

                def _fmt(v):
                    return f'{v:.4f}' if (v is not None and math.isfinite(v)) else 'nan'
                psnr_display = snapshot_metrics['psnr']
                if not math.isfinite(psnr_display):
                    psnr_display = 99.99  # cap inf (pixel-perfect) for display
                print(
                    f"  [Metrics all n={snapshot_metrics['n']}] "
                    f"P:{snapshot_metrics['precision']:.4f}  "
                    f"R:{snapshot_metrics['recall']:.4f}  "
                    f"F1:{snapshot_metrics['f1']:.4f}  "
                    f"IoU:{snapshot_metrics['iou']:.4f}  "
                    f"SSIM:{snapshot_metrics['ssim']:.4f}  "
                    f"PSNR:{psnr_display:.2f}dB  "
                    f"MAE:{snapshot_metrics['mae']:.4f}  "
                    f"Conn:{snapshot_metrics['connectivity']:.4f}  "
                    f"HD95:{snapshot_metrics['hausdorff']:.2f}px  "
                    f"F1@2px:{snapshot_metrics['f1_t2']:.4f}  F1@5px:{snapshot_metrics['f1_t5']:.4f}  "
                    f"F1img:{_fmt(snapshot_metrics['f1_img'])}"
                )
                for bname, _, _ in VAL_BUCKETS:
                    r = val_results[bname]
                    if r is None:
                        print(f"  [Metrics {bname}] (no samples)")
                    else:
                        print(f"  [Metrics {bname} n={r['n']}] "
                              f"P:{_fmt(r['precision'])}  R:{_fmt(r['recall'])}  "
                              f"F1:{_fmt(r['f1'])}  F1@2px:{_fmt(r['f1_t2'])}  F1@5px:{_fmt(r['f1_t5'])}  "
                              f"IoU:{_fmt(r['iou'])}  Conn:{_fmt(r['connectivity'])}  "
                              f"HD95:{_fmt(r['hausdorff'])}px")
            
            # ALL ranks must call report0 with the same set of names
            for metric_name in VAL_METRIC_NAMES:
                if snapshot_metrics is not None:
                    val = snapshot_metrics[metric_name]
                    # Replace non-finite values (e.g. inf PSNR) with a sentinel
                    # so TensorBoard / stats_jsonl don't choke on them.
                    if not math.isfinite(val):
                        val = 99.99 if metric_name == 'psnr' else []   # NaN -> not logged (0.0 would look perfect for HD)
                    training_stats.report0(f'Metrics/val_{metric_name}', val)
                else:
                    training_stats.report0(f'Metrics/val_{metric_name}', [])
            for bname, _, _ in VAL_BUCKETS:
                r = val_results[bname] if (rank == 0 and snapshot_metrics is not None) else None
                for metric_name in VAL_BUCKET_METRICS:
                    val = r[metric_name] if r is not None else []
                    if r is not None and not math.isfinite(val):
                        val = []
                    training_stats.report0(f'Metrics/val_{metric_name}_{bname}', val)
                training_stats.report0(f'Metrics/val_n_{bname}', r['n'] if r is not None else [])

            # Face/part-hole bank (only if --val-face-boxes was given)
            if has_face_bank:
                fr = None
                if rank == 0:
                    fr = evaluate_val_bank(G_ema, face_bank, device, batch_gpu)['all']
                    print(f"  [Metrics FACE n={fr['n']}] "
                          f"P:{_fmt(fr['precision'])}  R:{_fmt(fr['recall'])}  F1:{_fmt(fr['f1'])}  "
                          f"F1@2px:{_fmt(fr['f1_t2'])}  F1@5px:{_fmt(fr['f1_t5'])}  "
                          f"Conn:{_fmt(fr['connectivity'])}  HD95:{_fmt(fr['hausdorff'])}px")
                for metric_name in VAL_BUCKET_METRICS:
                    val = fr[metric_name] if fr is not None else []
                    if fr is not None and not math.isfinite(val):
                        val = []
                    training_stats.report0(f'Metrics/valface_{metric_name}', val)

        # Save network snapshot
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            if rank == 0:
                ckpt = {
                    'cur_nimg': cur_nimg,
                    'batch_idx': batch_idx,
                    'cur_tick': cur_tick,
                    'total_kimg': total_kimg,
                    'G_state': G.state_dict(),
                    'G_ema_state': G_ema.state_dict(),
                    'G_opt_state': G_opt.state_dict(),
                    'rng_numpy': np.random.get_state(),
                    'rng_python': random.getstate(),
                    'rng_torch': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    # LR decay config for proper resume
                    'base_lr_G': base_lr_G,
                    'base_lr_D': base_lr_D,
                    'lr_decay_start_kimg': lr_decay_start_kimg,
                    'lr_min_ratio': lr_min_ratio,
                    'min_lr': min_lr,
                    'min_lr_d': min_lr_d,
                }
                
                if D is not None:
                    ckpt['D_state'] = D.state_dict()
                if D_opt is not None:
                    ckpt['D_opt_state'] = D_opt.state_dict()

                # ── Atomic write: write to temp file then rename.
                # This prevents a half-written (corrupt) checkpoint if the
                # process is killed or the container is stopped during the save.
                # os.replace() is atomic on POSIX — the old file is only
                # replaced once the new one is fully flushed.

                # 1. Always overwrite the rolling "latest" checkpoint.
                #    This is the primary resume target and stays small on disk.
                latest_path = os.path.join(run_dir, 'checkpoint-latest.pth')
                tmp_latest  = latest_path + '.tmp'
                torch.save(ckpt, tmp_latest)
                os.replace(tmp_latest, latest_path)

                # 2. Save a milestone checkpoint every snap ticks
                #    (same frequency, but kept permanently for rollback).
                milestone_path = os.path.join(run_dir, f'checkpoint-{cur_nimg//1000:06d}.pth')
                tmp_milestone  = milestone_path + '.tmp'
                torch.save(ckpt, tmp_milestone)
                os.replace(tmp_milestone, milestone_path)

                print(f'  [Checkpoint] Saved: {os.path.basename(milestone_path)}')

        # Collect statistics
        if num_gpus > 1:
            torch.distributed.barrier()
        
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                try:
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                except RuntimeError:
                    pass
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
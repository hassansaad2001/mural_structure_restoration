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

"""Train a TSR (Transformer Structure Restoration) model."""

# IMPORTANT: Set these environment variables BEFORE importing torch
# to prevent CUDA initialization issues with multiprocessing.spawn
import os
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')  # Lazy CUDA module loading
import click
import re
import json
import tempfile
import torch
import dnnlib

from training import training_loop_tsr as training_loop
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options
    gpus       = None,
    snap       = None,
    metrics    = None,
    seed       = None,

    # Dataset (removed line paths)
    data_edge      = None,
    data_image     = None,
    data_val_edge  = None,
    data_val_image = None,
    mask_path      = None,  # Optional: pre-made mask folder (instead of random masks)
    val_mask_path  = None,  # Held-out mask folder for validation
    val_masks_per_image = None,  # Deterministic masks per val image
    mask_aug       = None,  # Flip/rot90/zoom augmentation of training masks
    dataloader = None,
    subset     = None,
    mirror     = None,

    # Slots mode
    slots      = None,
    data_edge_2    = None,
    data_image_2   = None,
    mask_path_2    = None,  # Optional: pre-made masks for secondary slot source
    slot_ratio     = None,

    # Base config
    cfg        = None,
    generator  = None,
    loss       = None,
    gamma      = None,
    kimg       = None,
    batch      = None,
    batch_gpu  = None,
    ema        = None,
    lr         = None,
    lr_d       = None,  # Separate learning rate for discriminator
    lr_decay_start = None,  # Start LR decay after this many kimg (default 200)
    lr_min_ratio   = None,  # Minimum LR ratio at end of decay (default 0.01)
    min_lr         = None,  # Explicit absolute minimum learning rate for G
    min_lr_d       = None,  # Explicit absolute minimum learning rate for D

    # Loss mode and weights
    loss_mode = None,
    bce_weight = None,
    dice_weight = None,
    hd_weight = None,
    # Adversarial weights
    adv_weight = None,
    fm_weight = None,
    use_discriminator = None,
    discriminator = None,
    d_norm = None,          # DiscriminatorLama: 'sn' | 'bn' | 'none'
    d_layers = None,        # DiscriminatorLama: n_layers
    d_blur = None,          # Gaussian sigma applied to real & fake D inputs
    d_mask_input = None,    # DiscriminatorLama: feed mask channel to D (old behaviour)
    d_fake_mode = None,     # 'lama' | 'legacy' fake-label normalisation
    rec_tau = None,         # boundary-weighted BCE (px); 0 = off
    focal = None,           # use sigmoid focal loss instead of BCE
    focal_alpha = None,
    focal_gamma = None,
    bce_pos_weight = None,  # positive-class weight for BCE
    face_boxes = None,      # JSON of face/hand boxes for TRAINING crops
    face_mask_prob = None,  # prob. of a face/part hole per sample
    face_oversample = None, # repeat boxed crops this many times per epoch
    val_face_boxes = None,  # JSON of face/hand boxes for VALIDATION crops
    swin_after = None,      # e.g. '1,3' -> Swin@H/8 after FFC blocks 1 and 3
    swin_dim = None,
    bottleneck = None,      # 'ffc' | 'conv' | 'swin'
    n_btn = None,           # number of bottleneck blocks
    dec_attn = None,        # MDTA in the decoder (True) or not
    dec_norm = None,        # Generator decoder norm: 'none' | 'in' | 'gn'
    g_beta1 = None,         # Adam beta1 for G
    d_beta1 = None,         # Adam beta1 for D
    g_beta2 = None,         # Adam beta2 for G
    d_beta2 = None,         # Adam beta2 for D
    optimizer = None,       # 'adam' | 'adamw'
    weight_decay = None,    # optimizer weight decay
    grad_clip = None,       # max gradient norm (None = off)
    lr_warmup = None,       # linear LR warm-up length in kimg (None/0 = off)
    preset = None,          # 'ours' | 'edgeconnect' | 'sgtb_srn' | 'zits_tsr' (only used for the run name)
    sync_bn = None,         # SyncBatchNorm for multi-GPU

    # Transfer learning
    resume     = None,
    resume_ckpt= None,

    # Performance options
    fp32       = None,
    nhwc       = None,
    allow_tf32 = None,
    nobench    = None,
    workers    = None,
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options
    # ------------------------------------------
    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        # raise UserError('--gpus must be a power of two')
        pass # Relaxing check for now
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = []
    assert isinstance(metrics, list)
    args.metrics = metrics

    if seed is None:
        seed = 42
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset
    # -----------------------------------
    if dataloader is None:
        dataloader = 'datasets.dataset_tsr_512.TSRDataset'

    args.training_set_kwargs = dnnlib.EasyDict(
        class_name=dataloader,
        edge_path=data_edge,
        data_path=data_image,
        max_size=None,
        xflip=False
    )
    # Only inject mask_path when provided — avoids passing unknown kwargs
    # to custom dataset classes that don't accept it
    if mask_path is not None:
        args.training_set_kwargs.mask_path = mask_path
    _is_512_loader = 'dataset_tsr_512' in dataloader
    if mask_aug:
        if not _is_512_loader:
            raise UserError('--mask-aug requires --dataloader=datasets.dataset_tsr_512.TSRDataset')
        args.training_set_kwargs.mask_aug = True
    if face_boxes is not None:
        if not _is_512_loader:
            raise UserError('--face-boxes requires --dataloader=datasets.dataset_tsr_512.TSRDataset')
        args.training_set_kwargs.face_boxes = face_boxes
        args.training_set_kwargs.face_mask_prob = face_mask_prob if face_mask_prob is not None else 0.3
        args.training_set_kwargs.face_oversample = face_oversample if face_oversample is not None else 1

    # Slots mode: switch to SlottedTSRDataset and add secondary paths
    if slots:
        if data_edge_2 is None:
            raise UserError('--data-edge-2 is required when --slots is enabled')
        args.training_set_kwargs.class_name = 'datasets.dataset_tsr_512.SlottedTSRDataset'
        args.training_set_kwargs.edge_path_2 = data_edge_2
        args.training_set_kwargs.data_path_2 = data_image_2
        if mask_path_2 is not None:
            args.training_set_kwargs.mask_path_2 = mask_path_2
        args.training_set_kwargs.slot_ratio = slot_ratio if slot_ratio is not None else 0.75

    # Use specific val paths if provided, else default to training paths (or subset)
    args.val_set_kwargs = dnnlib.EasyDict(
        class_name=dataloader,
        edge_path=data_val_edge if data_val_edge else data_edge,
        data_path=data_val_image if data_val_image else data_image,
        max_size=None,
        xflip=False
    )
    # Validation masks: held-out pool if given (recommended), else training pool.
    if val_mask_path is not None:
        args.val_set_kwargs.mask_path = val_mask_path
    elif mask_path is not None:
        print('WARNING: --val-mask-path not set; validation uses the TRAINING mask pool.')
        args.val_set_kwargs.mask_path = mask_path
    if _is_512_loader:
        args.val_set_kwargs.fix_crops = True   # deterministic val crops/masks
        if val_face_boxes is not None:
            args.val_set_kwargs.face_boxes = val_face_boxes   # extra FACE val bank
    
    args.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=False, 
        num_workers=3, 
        prefetch_factor=2
    )

    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
        args.training_set_kwargs.resolution = training_set.resolution
        args.training_set_kwargs.max_size = len(training_set)
        desc = training_set.name
        if preset is not None and preset != 'ours':
            desc = f'{preset}-{desc}'
        if slots:
            desc += '-slots'
        
        val_set = dnnlib.util.construct_class_by_name(**args.val_set_kwargs)
        args.val_set_kwargs.resolution = val_set.resolution
        args.val_set_kwargs.max_size = len(val_set)

        del training_set, val_set
    except IOError as err:
        raise UserError(f'--data: {err}')

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config
    # ------------------------------------
    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=1000, mb=-1, fmaps=-1, lrate=-1, gamma=-1, ema=-1, ramp=0.05),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.0001
        spec.gamma = 0
        spec.ema = spec.mb * 10 / 32

    # Network setup
    if generator is None:
        generator = 'networks.new_mat.GeneratorSketch'
    elif preset in (None, 'ours'):
        desc += '-' + generator.split('.')[-2]
    
    # Generator kwargs
    args.G_kwargs = dnnlib.EasyDict(
        class_name=generator,
        img_resolution=args.training_set_kwargs.resolution,
    )
    if dec_norm is not None:
        args.G_kwargs.dec_norm = dec_norm
        desc += f'-dec{dec_norm}'
    if swin_after:
        idx = [int(s) for s in str(swin_after).split(',') if s.strip() != '']
        args.G_kwargs.swin_after = idx
        if swin_dim is not None:
            args.G_kwargs.swin_dim = swin_dim
        desc += '-swin' + ''.join(str(i) for i in idx)
    if bottleneck is not None and bottleneck != 'ffc':
        args.G_kwargs.bottleneck = bottleneck
        desc += f'-btn{bottleneck}'
        if bottleneck == 'swin' and swin_dim is not None:
            args.G_kwargs.swin_dim = swin_dim
    if n_btn is not None:
        args.G_kwargs.n_btn = n_btn
        desc += f'x{n_btn}'
    if dec_attn is not None and not dec_attn:
        args.G_kwargs.dec_attn = False
        desc += '-nomdta'
    
    # Discriminator kwargs (for GAN)
    if discriminator is None or discriminator == 'sketch':
        args.D_kwargs = dnnlib.EasyDict(
            class_name='networks.new_mat.DiscriminatorSketch',
            img_resolution=args.training_set_kwargs.resolution,
            img_channels=1,
            c_dim=0,
            mbstd_group_size=4,
            use_spec_norm=False
        )
    elif discriminator == 'lama':
        args.D_kwargs = dnnlib.EasyDict(
            class_name='networks.new_mat.DiscriminatorLama',
            img_resolution=args.training_set_kwargs.resolution,
            img_channels=1,
            c_dim=0,
            ndf=64,
            n_layers=d_layers if d_layers is not None else 4,
            mbstd_group_size=4,
            use_spec_norm=False,
            norm=d_norm if d_norm is not None else 'sn',
            mask_input=bool(d_mask_input) if d_mask_input is not None else False,
        )
        desc += f'-lamaD{args.D_kwargs.norm}{args.D_kwargs.n_layers}'
        if args.D_kwargs.mask_input:
            desc += 'm'
    elif discriminator == 'projected':
        # Projected GAN discriminator: pretrained EfficientNet-B0 backbone (frozen),
        # 4 frozen random projections, 4 patch discriminator heads.
        # Edge (1ch) is repeated to 3ch to match EfficientNet input.
        args.D_kwargs = dnnlib.EasyDict(
            class_name='networks.projected_discriminator.ProjectedDiscriminator',
            img_resolution=args.training_set_kwargs.resolution,
            proj_dim=256,
            ndf=64,
            pretrained=True,
        )
        desc += '-projectedD'
    elif discriminator == 'edgeconnect':
        # EdgeConnect 70x70 PatchGAN on (gray image, edge); needs losses.loss_baselines.EdgeConnectLoss
        args.D_kwargs = dnnlib.EasyDict(
            class_name='networks.baselines.edgeconnect.EdgeConnectDiscriminator',
            img_resolution=args.training_set_kwargs.resolution,
        )
    else:
        raise UserError(f'Unknown discriminator type: {discriminator}')

    # Remember whether the projected D was chosen; we set the loss flag
    # after args.loss_kwargs is created (further below).
    _use_projected_d = (discriminator == 'projected')


    # Optimizer setup
    if lr is not None:
        assert isinstance(lr, float)
        spec.lrate = lr
        desc += f'-lr{lr:g}'
        
    if gamma is not None:
        assert isinstance(gamma, float)
        spec.gamma = gamma
        desc += f'-gamma{gamma:g}'

    opt_class = {None: 'torch.optim.Adam', 'adam': 'torch.optim.Adam', 'adamw': 'torch.optim.AdamW'}[optimizer]
    args.G_opt_kwargs = dnnlib.EasyDict(
        class_name=opt_class,
        lr=spec.lrate,
        betas=[g_beta1 if g_beta1 is not None else 0.0, g_beta2 if g_beta2 is not None else 0.99],
        eps=1e-8
    )
    if weight_decay is not None:
        args.G_opt_kwargs.weight_decay = weight_decay

    # Discriminator optimizer (for GAN)
    # Use lr_d if specified, otherwise use same lr as G
    d_lrate = lr_d if lr_d is not None else spec.lrate
    if lr_d is not None:
        desc += f'-lrd{lr_d:g}'

    args.D_opt_kwargs = dnnlib.EasyDict(
        class_name=opt_class,
        lr=d_lrate,
        betas=[d_beta1 if d_beta1 is not None else 0.0, d_beta2 if d_beta2 is not None else 0.99],
        eps=1e-8
    )
    if weight_decay is not None:
        args.D_opt_kwargs.weight_decay = weight_decay

    # Loss setup
    if loss is None:
        loss = 'losses.loss_line.SketchGANLoss'
    elif preset in (None, 'ours'):
        desc += '-' + loss.split('.')[-1]
    
    args.loss_kwargs = dnnlib.EasyDict(
        class_name=loss,
        r1_gamma=spec.gamma,  # R1 regularization
    )
    # Projected discriminator: disable R1 (random projections already stabilise
    # training — Sauer et al. 2021). Must be set AFTER loss_kwargs is created.
    if _use_projected_d:
        args.loss_kwargs.use_projected_d = True
    if d_blur is not None and d_blur > 0:
        args.loss_kwargs.d_blur_sigma = d_blur
        desc += f'-dblur{d_blur:g}'
    args.loss_kwargs.d_fake_mode = d_fake_mode if d_fake_mode is not None else 'lama'
    if loss.endswith('SketchGANLoss'):
        desc += f'-dfake{args.loss_kwargs.d_fake_mode}'
    if rec_tau is not None and rec_tau > 0:
        args.loss_kwargs.rec_boundary_tau = rec_tau
        desc += f'-rectau{rec_tau:g}'
    if focal:
        args.loss_kwargs.use_focal_loss = True
        if focal_alpha is not None:
            args.loss_kwargs.focal_alpha = focal_alpha
        if focal_gamma is not None:
            args.loss_kwargs.focal_gamma = focal_gamma
        desc += '-focal'
    elif bce_pos_weight is not None:
        args.loss_kwargs.bce_pos_weight = bce_pos_weight
        desc += f'-pw{bce_pos_weight:g}'
    if face_boxes is not None:
        desc += f'-face{args.training_set_kwargs.face_mask_prob:g}x{args.training_set_kwargs.face_oversample}'

    # Loss mode and weights
    if loss_mode is not None:
        args.loss_kwargs.loss_mode = loss_mode
        desc += f'-{loss_mode}'
    else:
        args.loss_kwargs.loss_mode = 'bce_dice'  # Default
    
    if bce_weight is not None:
        args.loss_kwargs.bce_weight = bce_weight
        desc += f'-bce{bce_weight:g}'
    
    if dice_weight is not None:
        args.loss_kwargs.dice_weight = dice_weight
        desc += f'-dice{dice_weight:g}'
    
    if hd_weight is not None:
        args.loss_kwargs.hd_weight = hd_weight
        desc += f'-hd{hd_weight:g}'
    
    # Adversarial weights
    if adv_weight is not None:
        args.loss_kwargs.adv_weight = adv_weight
        desc += f'-adv{adv_weight:g}'
    
    if fm_weight is not None:
        args.loss_kwargs.fm_weight = fm_weight
        desc += f'-fm{fm_weight:g}'

    if use_discriminator is None:
        use_discriminator = True
    args.use_discriminator = use_discriminator
    if not use_discriminator:
        desc += '-nodisc'
        args.D_kwargs = None
        args.D_opt_kwargs = None
        args.loss_kwargs.adv_weight = 0.0
        args.loss_kwargs.fm_weight = 0.0

    # Baseline losses (losses.loss_baselines) only use these; keep training_options.json honest
    if not loss.endswith('SketchGANLoss'):
        args.loss_kwargs = dnnlib.EasyDict({k: v for k, v in args.loss_kwargs.items()
                                            if k in ('class_name', 'adv_weight', 'fm_weight')})

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if kimg is not None:
        assert isinstance(kimg, int)
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        if batch_gpu is not None:
            assert isinstance(batch_gpu, int)
            desc += f'-batchgpu{batch_gpu}'
            args.batch_gpu = batch_gpu
        else:
            args.batch_gpu = batch // gpus

    if ema is not None:
        assert isinstance(ema, int)
        desc += '-ema' + str(ema)
        args.ema_kimg = ema

    # ----------------------------------
    # LR user-set flags (for resume logic)
    # ----------------------------------
    args.sync_bn = True if sync_bn is None else bool(sync_bn)
    args.val_masks_per_image = val_masks_per_image if val_masks_per_image is not None else 3
    args.lr_user_set = (lr is not None)
    args.lr_d_user_set = (lr_d is not None)

    # ----------------------------------
    # LR Decay settings
    # ----------------------------------
    if lr_decay_start is not None:
        assert isinstance(lr_decay_start, int) and lr_decay_start >= 0
        args.lr_decay_start_kimg = lr_decay_start
    else:
        args.lr_decay_start_kimg = 200  # Default: start decay after 200 kimg
    
    if lr_min_ratio is not None:
        assert isinstance(lr_min_ratio, float) and 0 < lr_min_ratio <= 1
        args.lr_min_ratio = lr_min_ratio
    else:
        args.lr_min_ratio = 0.01  # Default: decay to 1% of base LR

    if min_lr is not None:
        assert isinstance(min_lr, float) and min_lr >= 0
        args.min_lr = min_lr
    else:
        args.min_lr = 0.0

    if min_lr_d is not None:
        assert isinstance(min_lr_d, float) and min_lr_d >= 0
        args.min_lr_d = min_lr_d
    else:
        args.min_lr_d = 0.0

    if lr_warmup:
        args.lr_warmup_kimg = float(lr_warmup)
    if grad_clip:
        args.grad_clip = float(grad_clip)

    # ----------------------------------
    # Transfer learning
    # ----------------------------------
    args.resume_ckpt = resume_ckpt
    args.resume_pkl = None

    if resume is not None and resume != 'noresume':
        desc += '-resumecustom'

    # -------------------------------------------------
    # Performance options
    # -------------------------------------------------
    if fp32 is None:
        fp32 = False
    if fp32:
        desc += '-fp32'

    if nobench is None:
        nobench = False
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    # CRITICAL WORKAROUND: Call device_count() BEFORE any other CUDA operations
    # There's a PyTorch bug where calling is_available() or set_device() first
    # causes "Error 2: out of memory" during CUDA initialization.
    # Calling device_count() first somehow initializes CUDA correctly.
    _ = torch.cuda.device_count()
    
    # Now set the CUDA device for this process
    torch.cuda.set_device(rank)
    
    dnnlib.util.Logger(
        file_name=os.path.join(args.run_dir, 'log.txt'),
        file_mode='a',
        should_flush=True
    )

    # Init torch.distributed
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(
                backend='gloo',
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus
            )
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus
            )

    # Init torch_utils
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

# Baseline recipes (docs/BASELINES.md). A preset fills every option it lists unless that
# option was given on the command line, so any value can still be overridden.
PRESETS = {
    'ours': {},
    # EdgeConnect edge model (Nazeri et al. 2019), config.yml defaults:
    # LR 1e-4, D2G_LR 0.1, BETA1 0, BETA2 0.9, BATCH_SIZE 8, GAN_LOSS nsgan, FM_LOSS_WEIGHT 10,
    # constant learning rate, no R1.
    'edgeconnect': dict(
        generator='networks.baselines.edgeconnect.EdgeConnectEdgeGenerator',
        discriminator='edgeconnect', use_discriminator=True,
        loss='losses.loss_baselines.EdgeConnectLoss',
        lr=1e-4, lr_d=1e-5, g_beta1=0.0, d_beta1=0.0, g_beta2=0.9, d_beta2=0.9,
        gamma=0.0, adv_weight=1.0, fm_weight=10.0, batch=8, lr_min_ratio=1.0,
    ),
    # SGTB structure reconstruction network (Deng & Yu, Heritage Science 2023), "Training settings"
    # and "Loss function": Adam lr 1e-4 (G) / 1e-5 (D), betas (0.5, 0.9), batch 4,
    # lambda_s_adv 1, lambda_FM 10; EdgeConnect's 70x70 PatchGAN (paper Table 1). No schedule given.
    'sgtb_srn': dict(
        generator='networks.baselines.sgtb_srn.SGTBStructureGenerator',
        discriminator='edgeconnect', use_discriminator=True,
        loss='losses.loss_baselines.EdgeConnectLoss',
        lr=1e-4, lr_d=1e-5, g_beta1=0.5, d_beta1=0.5, g_beta2=0.9, d_beta2=0.9,
        gamma=0.0, adv_weight=1.0, fm_weight=10.0, batch=4, lr_min_ratio=1.0,
    ),
    # ZITS TSR (Dong et al. 2022), TSR_train.py / TSR_trainer.py defaults:
    # AdamW lr 4.24e-4, betas (0.9, 0.95), weight decay 0, batch 32, grad-norm clip 1.0,
    # linear warm-up 1500 iterations (= 48 kimg at batch 32), then cosine decay to 0.1x.
    'zits_tsr': dict(
        generator='networks.baselines.zits_tsr.ZITSTSREdge',
        use_discriminator=False,
        loss='losses.loss_baselines.ZITSTSRLoss',
        optimizer='adamw', lr=4.24e-4, g_beta1=0.9, g_beta2=0.95, weight_decay=0.0,
        batch=32, grad_clip=1.0, lr_warmup=48.0, lr_decay_start=48, lr_min_ratio=0.1,
    ),
}

def apply_preset(ctx, preset, config_kwargs):
    from click.core import ParameterSource
    for key, value in PRESETS[preset].items():
        if ctx.get_parameter_source(key) in (ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT):
            print(f'  preset {preset}: --{key.replace("_", "-")} kept from command line ({config_kwargs[key]})')
            continue
        config_kwargs[key] = value

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 42]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
@click.option('--preset', help='Model + training recipe: ours, or a baseline (docs/BASELINES.md); explicit options override it',
              type=click.Choice(sorted(PRESETS)), default='ours', show_default=True)

# Dataset (removed line paths)
@click.option('--data-edge',       help='Training data edge (directory)', metavar='PATH', required=True)
@click.option('--data-image',      help='Training data image (directory)', metavar='PATH', required=False)
@click.option('--data-val-edge',   help='Validation data edge (directory)', metavar='PATH', required=False)
@click.option('--data-val-image',  help='Validation data image (directory)', metavar='PATH', required=False)
@click.option('--mask-path',       help='Pre-made mask folder (PNG/images); sampled randomly instead of generating masks', metavar='PATH', required=False)
@click.option('--val-mask-path',   help='Held-out mask folder for validation (default: --mask-path)', metavar='PATH', required=False)
@click.option('--val-masks-per-image', help='Deterministic masks per validation image [default: 3]', type=int, metavar='INT')
@click.option('--mask-aug',        help='Random flip/rot90/zoom augmentation of training masks', type=bool, metavar='BOOL')
@click.option('--dataloader',      help='Dataloader class', type=str, metavar='STRING')
@click.option('--subset',          help='Train with only N images', type=int, metavar='INT')
@click.option('--mirror',          help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Slots mode
@click.option('--slots',           help='Enable dual-path slot mode', type=bool, metavar='BOOL')
@click.option('--data-edge-2',     help='Secondary edge path (required with --slots)', metavar='PATH')
@click.option('--data-image-2',    help='Secondary image path (optional with --slots)', metavar='PATH')
@click.option('--mask-path-2',     help='Pre-made mask folder for secondary slot source', metavar='PATH', required=False)
@click.option('--slot-ratio',      help='Fraction of batch from primary path [default: 0.5]', type=float, metavar='FLOAT')

# Base config
@click.option('--cfg', help='Base config', type=click.Choice(['auto']))
@click.option('--generator', help='Generator class path', type=str, metavar='STRING')
@click.option('--loss', help='Loss class path', type=str, metavar='STRING')
@click.option('--batch', help='Batch size', type=int, metavar='INT')
@click.option('--batch-gpu', help='Samples per GPU', type=int, metavar='INT')
@click.option('--kimg', help='kimg', type=int, metavar='INT')

@click.option('--ema', help='EMA half-life', type=int, metavar='INT')
@click.option('--lr', help='Learning rate (for Generator, and D if --lr-d not set)', type=float)
@click.option('--lr-d', help='Discriminator learning rate (optional, defaults to --lr)', type=float)
@click.option('--lr-decay-start', help='Start LR decay after this many kimg [default: 200]', type=int, metavar='INT')
@click.option('--lr-min-ratio', help='Minimum LR ratio at end of decay [default: 0.01]', type=float, metavar='FLOAT')
@click.option('--min-lr', help='Approximate absolute minimum LR for Generator (clamps decay)', type=float, metavar='FLOAT')
@click.option('--min-lr-d', help='Approximate absolute minimum LR for Discriminator (clamps decay)', type=float, metavar='FLOAT')
@click.option('--gamma', help='R1 regularization weight', type=float)

@click.option('--loss-mode', help='Loss mode: bce, bce_dice, bce_dice_hd', 
              type=click.Choice(['bce', 'bce_dice', 'bce_dice_hd']))
@click.option('--bce-weight', help='BCE loss weight', type=float)
@click.option('--dice-weight', help='Dice loss weight', type=float)
@click.option('--hd-weight', help='Hausdorff loss weight', type=float)
@click.option('--adv-weight', help='Adversarial loss weight', type=float)
@click.option('--fm-weight', help='Feature matching loss weight', type=float)
@click.option('--use-discriminator/--no-discriminator', default=True, help='Enable/disable discriminator and adversarial training', show_default=True)
@click.option('--discriminator', help='Discriminator architecture', type=click.Choice(['sketch', 'lama', 'projected', 'edgeconnect']), default='sketch', show_default=True)
@click.option('--d-norm', help='DiscriminatorLama normalisation [default: sn]', type=click.Choice(['sn', 'bn', 'none']))
@click.option('--d-layers', help='DiscriminatorLama n_layers [default: 4]', type=int, metavar='INT')
@click.option('--d-blur', help='Gaussian sigma on real & fake D inputs [default: 0]', type=float, metavar='FLOAT')
@click.option('--d-mask-input', help='Feed the mask to DiscriminatorLama (old behaviour) [default: false]', type=bool, metavar='BOOL')
@click.option('--d-fake-mode', help='Fake-label normalisation [default: lama]', type=click.Choice(['lama', 'legacy']))
@click.option('--rec-tau', help='Boundary-weighted BCE decay length in px, 0 = off [default: 0]', type=float, metavar='FLOAT')
@click.option('--focal', help='Sigmoid focal loss instead of BCE [default: false]', type=bool, metavar='BOOL')
@click.option('--focal-alpha', help='Focal alpha (positive-class weight) [default: 0.95]', type=float, metavar='FLOAT')
@click.option('--focal-gamma', help='Focal gamma [default: 2]', type=float, metavar='FLOAT')
@click.option('--bce-pos-weight', help='BCE positive-class weight (ignored with --focal) [default: 1]', type=float, metavar='FLOAT')
@click.option('--face-boxes', help='JSON {crop_name: [[x0,y0,x1,y1],...]} of face/hand boxes (training)', metavar='PATH')
@click.option('--face-mask-prob', help='Probability of a face/part hole per sample [default: 0.3]', type=float, metavar='FLOAT')
@click.option('--face-oversample', help='Repeat crops with face boxes this many times per epoch [default: 1]', type=int, metavar='INT')
@click.option('--val-face-boxes', help='JSON of face/hand boxes for validation crops (adds FACE metrics)', metavar='PATH')
@click.option('--swin-after', help='Comma list of FFC block indices followed by a Swin@H/8 stage, e.g. 1,3', type=str, metavar='LIST')
@click.option('--swin-dim', help='Swin stage width [default: 256]', type=int, metavar='INT')
@click.option('--bottleneck', help='Bottleneck type [default: ffc]', type=click.Choice(['ffc', 'conv', 'swin']))
@click.option('--n-btn', help='Number of bottleneck blocks [default: 6]', type=int, metavar='INT')
@click.option('--dec-attn', help='MDTA attention in the decoder [default: true]', type=bool, metavar='BOOL')
@click.option('--dec-norm', help='Generator decoder norm [default: none]', type=click.Choice(['none', 'in', 'gn']))
@click.option('--g-beta1', help='Adam beta1 for G [default: 0.0]', type=float, metavar='FLOAT')
@click.option('--d-beta1', help='Adam beta1 for D [default: 0.0]', type=float, metavar='FLOAT')
@click.option('--g-beta2', help='Adam beta2 for G [default: 0.99]', type=float, metavar='FLOAT')
@click.option('--d-beta2', help='Adam beta2 for D [default: 0.99]', type=float, metavar='FLOAT')
@click.option('--optimizer', help='Optimizer [default: adam]', type=click.Choice(['adam', 'adamw']))
@click.option('--weight-decay', help='Optimizer weight decay [default: 0]', type=float, metavar='FLOAT')
@click.option('--grad-clip', help='Clip gradient norm to this value [default: off]', type=float, metavar='FLOAT')
@click.option('--lr-warmup', help='Linear LR warm-up from 0 over this many kimg [default: off]', type=float, metavar='KIMG')
@click.option('--sync-bn', help='SyncBatchNorm across GPUs [default: true]', type=bool, metavar='BOOL')

# Transfer learning
@click.option('--resume', help='Resume training', metavar='PKL')
@click.option('--resume-ckpt', help='Resume from checkpoint', metavar='PATH')

# Performance options
@click.option('--fp32', help='Disable mixed-precision', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow TF32', type=bool, metavar='BOOL')
@click.option('--workers', help='DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, dry_run, gpus, snap, metrics, seed, **config_kwargs):
    """Train a TSR (Transformer Structure Restoration) model.

    Examples:
    
    python train_tsr.py --outdir=./runs --data-edge=./edges --data-image=./images
    """
    print('Starting TSR training...')
    dnnlib.util.Logger(should_flush=True)
    if config_kwargs['preset'] != 'ours':
        print(f"Preset: {config_kwargs['preset']}")
        apply_preset(ctx, config_kwargs['preset'], config_kwargs)

    # Setup training options
    try:
        run_desc, args = setup_training_loop_kwargs(
            gpus=gpus, snap=snap, metrics=metrics, seed=seed, **config_kwargs
        )
    except UserError as err:
        ctx.fail(err)

    print(f"DEBUG: Selected num_gpus: {args.num_gpus}")
    
    # Validate GPU count WITHOUT initializing CUDA (which would break multiprocessing.spawn)
    # Use nvidia-smi instead of torch.cuda.device_count() to avoid CUDA context creation
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_indices = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            # Account for CUDA_VISIBLE_DEVICES if set
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible is not None:
                available_gpus = len(cuda_visible.split(','))
            else:
                available_gpus = len(gpu_indices)
            
            if available_gpus == 0:
                ctx.fail('No CUDA devices available. Check your CUDA installation.')
            if args.num_gpus > available_gpus:
                ctx.fail(f'Requested --gpus={args.num_gpus} but only {available_gpus} CUDA device(s) available. '
                         f'CUDA_VISIBLE_DEVICES={cuda_visible or "not set"}. '
                         f'Set --gpus={available_gpus} or adjust CUDA_VISIBLE_DEVICES.')
            print(f"DEBUG: Validated {available_gpus} GPU(s) available via nvidia-smi")
        else:
            print(f"WARNING: nvidia-smi check failed, proceeding anyway...")
    except Exception as e:
        print(f"WARNING: Could not validate GPU count: {e}")

    # Pick output directory
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:    {args.run_dir}')
    print(f'Training data:       {args.training_set_kwargs.edge_path}')
    print(f'Batch size:          {args.batch_size}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(
                fn=subprocess_fn,
                args=(args, temp_dir),
                nprocs=args.num_gpus
            )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

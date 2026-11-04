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

"""Loss functions for Sketch Inpainting GAN.

Combines reconstruction losses (BCE, Dice, Hausdorff)
with adversarial loss for GAN training.

Input/Output ranges:
- Image: [-1, 1] (RGB)
- Edge: [0, 1] (binary probability)
- Mask: [0, 1] (0=valid, 1=hole)
- Generator output: logits (before sigmoid)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix


# ---------------------------------------------------------------------------
# VGG-based perceptual / style losses
# ---------------------------------------------------------------------------

class VGGFeatureExtractor(nn.Module):
    """
    Frozen VGG-16 feature extractor for perceptual and style losses.

    Extracts activations after relu1_2, relu2_2, relu3_3, relu4_3
    (indices 4, 9, 16, 23 in torchvision's vgg16.features).

    Input  : (B, 3, H, W)  in [0, 1]  (images are normalised internally)
    Output : list of 4 feature maps, from shallow to deep
    """

    # ImageNet mean / std  (same normalisation as torchvision pre-trained)
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    # Indices of the layers we want to tap (inclusive)
    _SLICE_ENDS = [4, 9, 16, 23]   # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self):
        super().__init__()
        try:
            import torchvision
            vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1')
        except TypeError:
            import torchvision
            vgg = torchvision.models.vgg16(pretrained=True)

        features = vgg.features

        # Build sequential slices between tap points
        prev = 0
        self.slices = nn.ModuleList()
        for end in self._SLICE_ENDS:
            self.slices.append(nn.Sequential(*list(features.children())[prev:end + 1]))
            prev = end + 1

        # Freeze — VGG weights must not receive gradients
        for p in self.parameters():
            p.requires_grad_(False)

        # Register normalisation buffers
        self.register_buffer('mean', torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(self._STD ).view(1, 3, 1, 1))

    def _preprocess(self, x):
        """Normalise from [0, 1] to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, x):
        """
        Args:
            x: (B, 1 or 3, H, W) in [0, 1]
        Returns:
            list of 4 feature tensors
        """
        # Edge images are 1-channel — replicate to 3 for VGG
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = self._preprocess(x.float())
        feats = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats


def _gram_matrix(feat):
    """Compute normalised Gram matrix for a single feature map.

    Args:
        feat: (B, C, H, W)
    Returns:
        gram: (B, C, C)  normalised by 1 / (C * H * W)
    """
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)          # (B, C, HW)
    gram = torch.bmm(f, f.transpose(1, 2))  # (B, C, C)
    return gram / (C * H * W)

#----------------------------------------------------------------------------
# Reconstruction loss components

class SigmoidFocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance in edge detection."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(torch.nn.Module):
    """Dice Loss for binary segmentation."""
    def __init__(self, smooth=1.0, reduction='none'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)
        
        if mask is not None:
            probs = probs * mask
            targets = targets * mask
        
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class HausdorffDTLoss(torch.nn.Module):
    """Hausdorff Distance Loss using GPU-only Distance Transform.
    
    Uses iterative min-pooling to approximate EDT entirely on GPU,
    avoiding CPU↔GPU transfers that stall the training pipeline.
    """
    def __init__(self, alpha=2.0, reduction='mean', dt_iterations=64):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.dt_iterations = dt_iterations  # max distance to compute
    
    @torch.no_grad()
    def _distance_transform_gpu(self, binary_mask):
        """Approximate Euclidean DT using iterative 3×3 min-pooling on GPU.
        
        For a binary mask (1=foreground, 0=background), computes approximate
        distance from each background pixel to the nearest foreground pixel.
        
        Args:
            binary_mask: (B, 1, H, W) float tensor, 1=object, 0=background
        Returns:
            dt: (B, 1, H, W) distance transform (approx), 0 at foreground
        """
        # We want distance from each pixel to the nearest 1-pixel.
        # Start with: 0 where foreground, large where background
        INF = float(self.dt_iterations + 1)
        dt = torch.where(binary_mask > 0.5, torch.zeros_like(binary_mask), 
                         torch.full_like(binary_mask, INF))
        
        # Iteratively propagate: each iteration extends distance by 1
        for _ in range(self.dt_iterations):
            # 3×3 min-pool via: min(x) = -max(-x)
            dt_padded = F.pad(dt, (1, 1, 1, 1), mode='replicate')
            dt = -F.max_pool2d(-dt_padded, kernel_size=3, stride=1, padding=0) + 1.0
            # Restore foreground pixels to 0
            dt = dt * (1 - binary_mask)
        
        # pixels with no foreground within dt_iterations kept
        # growing (up to INF + dt_iterations = 129). Cap them at the max
        # measurable distance so empty predictions don't produce outliers.
        # NOTE: 3x3 min-pooling yields a chessboard (L-inf) distance, not EDT.
        return dt.clamp_(max=float(self.dt_iterations))
    
    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)
        target_binary = (targets > 0.5).float()
        target_dt = self._distance_transform_gpu(target_binary)
        
        # build the predicted DT from the COMPOSITE (GT outside the
        # hole, prediction inside). The raw prediction is unsupervised in the
        # known region and full of spurious edges, which shrank pred_dt near
        # the hole boundary and corrupted the backward term.
        if mask is not None:
            comp = targets * (1 - mask) + probs.detach() * mask
        else:
            comp = probs.detach()
        pred_binary = (comp > 0.5).float()
        pred_dt = self._distance_transform_gpu(pred_binary)
        
        # the old 'backward' term, targets * pred_dt, contains no
        # trainable tensor (targets is data, pred_dt is computed under no_grad),
        # so it was logged but produced ZERO gradient: missed GT edges far from
        # any prediction were never penalised. Use the Karimi & Salcudean (2019)
        # form, where both DT maps weight the squared error:
        #   loss = (p - t)^2 * (dt_target^a + dt_pred^a)
        # GT edge pixels that were missed get weight = distance to the nearest
        # predicted edge, i.e. a real gradient that grows with the miss distance.
        err = (probs - targets) ** 2
        forward_loss = err * (target_dt ** self.alpha)
        backward_loss = err * (pred_dt ** self.alpha)
        
        if mask is not None:
            forward_loss = forward_loss * mask
            backward_loss = backward_loss * mask
            num_pixels = mask.sum().clamp(min=1.0)
        else:
            num_pixels = logits.numel()
        
        total_loss = (forward_loss.sum() + backward_loss.sum()) / (2 * num_pixels)
        
        if self.reduction == "mean":
            return total_loss
        elif self.reduction == "sum":
            return total_loss * num_pixels
        else:
            return forward_loss + backward_loss


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, sync, gain):
        raise NotImplementedError()

#----------------------------------------------------------------------------

class SketchGANLoss(Loss):
    """Loss for Sketch Inpainting GAN.
    
    Combines:
    - Reconstruction losses (BCE/Dice/Hausdorff)
    - Adversarial loss (GAN)
    - Feature matching loss
    
    Supports loss modes:
        - 'bce': BCE/Focal loss only
        - 'bce_dice': BCE + Dice loss (default)
        - 'bce_dice_hd': BCE + Dice + Hausdorff distance loss
    
    Args:
        use_projected_d : set True when using ProjectedDiscriminator.
                          Disables R1 gradient penalty (random projections
                          already stabilise training — see Sauer et al. 2021).
                          All other losses are unchanged.
    """
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10,
                 # Loss mode and weights
                 loss_mode='bce_dice',
                 bce_weight=1.0, dice_weight=1.0, hd_weight=0.1,
                 use_focal_loss=False, focal_alpha=0.95, focal_gamma=2.0,
                 hd_alpha=1.0,
                 # Adversarial weights
                 adv_weight=1.0, fm_weight=10.0,
                 # VGG-based style and perceptual losses
                 use_vgg_losses=False,
                 style_weight=50.0, perc_weight=0.1,
                 # Projected discriminator flag
                 use_projected_d=False,
                 # Gaussian blur applied to BOTH real and fake D inputs.
                 # Removes the shortcut "real edges are exactly 0/1, fakes are soft".
                 d_blur_sigma=0.0,
                 # FIX (D shortcut): 'lama' = LaMa NonSaturatingWithR1 with
                 # mask_as_fake_target: per-cell target w*fake + (1-w)*real and a
                 # plain .mean() over ALL cells (same normaliser as the real
                 # term). 'legacy' = old behaviour (fake term divided by sum(w),
                 # i.e. ~1/hole_ratio times heavier than the real term).
                 d_fake_mode='lama',
                 # Boundary-weighted pixel losses: inside the hole, BCE/HD weight
                 # = max(floor, exp(-d / tau)), d = distance (px) to the nearest
                 # known pixel. Pixel losses then supervise continuation near the
                 # boundary and leave the ambiguous interior to adv/FM/perceptual
                 # terms. tau=0 disables (old behaviour).
                 rec_boundary_tau=0.0, rec_interior_floor=0.1,
                 # Positive-class weight for BCE (ignored when use_focal_loss=True).
                 bce_pos_weight=None):
        super().__init__()
        assert d_fake_mode in ('lama', 'legacy'), d_fake_mode
        self.d_fake_mode = d_fake_mode
        self.rec_boundary_tau = float(rec_boundary_tau)
        self.rec_interior_floor = float(rec_interior_floor)
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.use_projected_d = use_projected_d
        self.d_blur_sigma = float(d_blur_sigma)
        self._d_blur_kernel = None
        if self.d_blur_sigma > 0:
            radius = max(1, int(math.ceil(2.5 * self.d_blur_sigma)))
            xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
            g = torch.exp(-xs ** 2 / (2 * self.d_blur_sigma ** 2))
            g = g / g.sum()
            self._d_blur_kernel = (g[:, None] * g[None, :])[None, None].to(device)
        
        if use_projected_d:
            # Projected GAN: R1 is incompatible with frozen random projections.
            # Override r1_gamma to 0 regardless of what was passed in.
            self.r1_gamma = 0
            print("[SketchGANLoss] ProjectedDiscriminator mode: R1 regularization DISABLED")
        
        # Loss mode
        valid_modes = ['bce', 'bce_dice', 'bce_dice_hd']
        if loss_mode not in valid_modes:
            raise ValueError(f"loss_mode must be one of {valid_modes}, got: {loss_mode}")
        self.loss_mode = loss_mode
        
        # BCE/Focal Loss
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.bce_fn = SigmoidFocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
        elif bce_pos_weight is not None and bce_pos_weight != 1.0:
            # Weighted BCE: edge pixels are ~9-10 % of the hole, so up-weight
            # positives. 2-4 is a good range; the full inverse ratio (~10)
            # tends to thicken strokes.
            self.bce_fn = torch.nn.BCEWithLogitsLoss(
                reduction='none', pos_weight=torch.tensor([float(bce_pos_weight)], device=device))
        else:
            self.bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        # Dice Loss
        self.dice_fn = DiceLoss(smooth=1.0, reduction='mean')
        
        # Hausdorff Loss
        self.hd_fn = HausdorffDTLoss(alpha=hd_alpha, reduction='mean')
        
        # VGG-based perceptual and style losses
        self.use_vgg_losses = use_vgg_losses
        self.style_weight   = style_weight
        self.perc_weight    = perc_weight
        if use_vgg_losses:
            self.vgg = VGGFeatureExtractor().to(device).eval()
        else:
            self.vgg = None
        
        # Weights
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.hd_weight = hd_weight
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        
        # Determine which losses to use
        self.use_dice = loss_mode in ['bce_dice', 'bce_dice_hd']
        self.use_hd = loss_mode == 'bce_dice_hd'
        
        # L1 loss for feature matching
        self.l1_loss = torch.nn.L1Loss()
        
        print(f"[SketchGANLoss] Mode: {loss_mode}")
        if use_projected_d:
            print(f"  Discriminator: ProjectedDiscriminator (EfficientNet-B0 backbone)")
            print(f"  R1 regularization: DISABLED (incompatible with random projections)")
        print(f"  BCE weight: {bce_weight}  "
              f"({'focal a=%g g=%g' % (focal_alpha, focal_gamma) if use_focal_loss else 'pos_weight=%s' % bce_pos_weight})")
        if self.use_dice:
            print(f"  Dice weight: {dice_weight}")
        if self.use_hd:
            print(f"  Hausdorff weight: {hd_weight}")
        print(f"  Adversarial weight: {adv_weight}")
        print(f"  Feature matching weight: {fm_weight}")
        print(f"  VGG losses: {'enabled' if use_vgg_losses else 'disabled'}")
        print(f"  D input blur sigma: {self.d_blur_sigma}")
        print(f"  D fake-label mode: {self.d_fake_mode}")
        print(f"  Boundary-weighted recon: tau={self.rec_boundary_tau} floor={self.rec_interior_floor}")
        if use_vgg_losses:
            print(f"    Style weight: {style_weight}")
            print(f"    Perceptual weight: {perc_weight}")

    # ------------------------------------------------------------------
    # VGG-based loss helpers
    # ------------------------------------------------------------------

    # Maximum spatial size fed to VGG.  Keeping this at 256 avoids OOM on
    # 11 GB GPUs while still capturing meaningful texture / feature statistics.
    _VGG_MAX_SIZE = 256

    def _vgg_losses(self, pred_edge, target_edge, mask):
        """Compute style (Gram) and perceptual (feature L1) losses together.

        Runs VGG exactly **twice** — once for pred (with grad), once for
        target (inside no_grad) — to minimise peak memory.  Inputs are
        downsampled to at most ``_VGG_MAX_SIZE`` px on the longest side
        before entering VGG, which is standard practice: Gram matrices and
        feature distributions are resolution-agnostic.

        Args:
            pred_edge   : (B, 1, H, W)  sigmoid probabilities in [0, 1]
            target_edge : (B, 1, H, W)  ground-truth edges   in [0, 1]
            mask        : (B, 1, H, W)  1 = hole
        Returns:
            (loss_style, loss_perc)  — two scalar tensors
        """
        # ── Downsample to _VGG_MAX_SIZE ──────────────────────────────────
        H, W = pred_edge.shape[-2:]
        scale = self._VGG_MAX_SIZE / max(H, W)
        if scale < 1.0:
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            pred_s   = F.interpolate(pred_edge,   size=(new_h, new_w),
                                     mode='bilinear', align_corners=False)
            target_s = F.interpolate(target_edge, size=(new_h, new_w),
                                     mode='bilinear', align_corners=False)
            mask_s   = F.interpolate(mask,         size=(new_h, new_w),
                                     mode='nearest')
        else:
            pred_s, target_s, mask_s = pred_edge, target_edge, mask

        # ── Single VGG pass for target (no gradient needed) ──────────────
        with torch.no_grad():
            target_feats = self.vgg(target_s * mask_s)   # masked for style
            target_feats_unmasked = self.vgg(target_s)   # unmasked for perc

        # ── Single VGG pass for pred (gradient flows through here) ───────
        pred_feats         = self.vgg(pred_s * mask_s)   # masked for style
        pred_feats_unmasked = self.vgg(pred_s)           # unmasked for perc

        # ── Style loss: Gram matrices on masked features ──────────────────
        loss_style = torch.tensor(0.0, device=self.device)
        for pf, tf in zip(pred_feats, target_feats):
            gp = _gram_matrix(pf)
            gt = _gram_matrix(tf)
            loss_style = loss_style + F.l1_loss(gp, gt)
        loss_style = loss_style / len(pred_feats)

        # ── Perceptual loss: masked feature L1 ───────────────────────────
        loss_perc = torch.tensor(0.0, device=self.device)
        for pf, tf in zip(pred_feats_unmasked, target_feats_unmasked):
            mf  = F.adaptive_avg_pool2d(mask_s, pf.shape[-2:])
            num = mf.sum().clamp(min=1.0) * pf.shape[1]   # C × masked_px
            loss_perc = loss_perc + ((pf - tf).abs() * mf).sum() / num
        loss_perc = loss_perc / len(pred_feats_unmasked)

        return loss_style, loss_perc

    # ------------------------------------------------------------------

    def run_G(self, full_image, full_edge, mask, sync):
        """Run generator forward pass.
        
        Args:
            full_image: RGB image [-1, 1]
            full_edge: Edge map [0, 1]
            mask: Binary mask [0, 1] (0=valid, 1=hole)
            sync: Whether to sync DDP
            
        Returns:
            edge_logits: Raw logits (before sigmoid)
        """
        with misc.ddp_sync(self.G, sync):
            edge_logits = self.G(full_image, full_edge, mask)
        return edge_logits
    
    def _blur_for_D(self, edge):
        if self._d_blur_kernel is None:
            return edge
        k = self._d_blur_kernel.to(edge.dtype)
        pad = k.shape[-1] // 2
        return F.conv2d(F.pad(edge, (pad, pad, pad, pad), mode='reflect'), k)

    @staticmethod
    def _hole_weight(mask, logits):
        """Hole-coverage weight per D output.

        Global D (B,1): returns None -> plain mean (old behaviour).
        Patch D (B,1,h,w): fraction of each output cell covered by the hole.
        Fake labels are applied only in proportion to hole coverage
        (LaMa-style 'mask as fake target'); fully-known patches of a fake
        composite are identical to real and receive no fake label.
        """
        if logits.ndim != 4:
            return None
        return F.adaptive_avg_pool2d(mask, logits.shape[-2:]).detach()

    @staticmethod
    def _weighted_mean(x, w):
        if w is None:
            return x.mean()
        return (x * w).sum() / w.sum().clamp(min=1e-6)

    def _d_fake_loss(self, gen_logits, w_hole):
        """D loss on the fake composite."""
        if w_hole is None or self.d_fake_mode == 'legacy':
            return self._weighted_mean(F.softplus(gen_logits), w_hole)
        # LaMa mask_as_fake_target: hole cells -> fake, known cells -> real,
        # normalised by ALL cells (matches the real term's .mean()).
        return (w_hole * F.softplus(gen_logits) + (1 - w_hole) * F.softplus(-gen_logits)).mean()

    def _g_adv_loss(self, gen_logits, w_hole):
        if w_hole is None or self.d_fake_mode == 'legacy':
            return self._weighted_mean(F.softplus(-gen_logits), w_hole)
        return F.softplus(-gen_logits).mean()   # LaMa default (use_unmasked_for_gen=True)

    def _rec_weight(self, hole_mask):
        """Per-pixel weight for pixel losses inside the hole (1 = old behaviour)."""
        if self.rec_boundary_tau <= 0:
            return hole_mask
        # distance from each pixel to the nearest KNOWN pixel (chessboard, capped at 64)
        d = self.hd_fn._distance_transform_gpu((hole_mask < 0.5).float())
        w = torch.exp(-d / self.rec_boundary_tau).clamp(min=self.rec_interior_floor)
        return (w * hole_mask).detach()

    def run_D(self, edge, mask, sync, return_features=False):
        """Run discriminator forward pass.
        
        Args:
            edge: Edge map [0, 1] (probabilities)
            mask: Binary mask [0, 1]
            sync: Whether to sync DDP
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Discriminator logits
            features: List of intermediate features (if return_features=True)
        """
        edge = self._blur_for_D(edge)
        with misc.ddp_sync(self.D, sync):
            if return_features:
                logits, features = self.D(edge, mask, return_features=True)
            else:
                logits = self.D(edge, mask)
                features = None
        return logits, features

    def accumulate_gradients(self, phase, full_image, full_edge, mask, sync, gain):
        """Accumulate gradients for training.
        
        Args:
            phase: 'Gmain', 'Dmain', or 'Dreg'
            full_image: RGB image [-1, 1]
            full_edge: Edge map [0, 1]
            mask: Binary mask [0, 1] (0=valid, 1=hole)
            sync: Whether to sync DDP
            gain: Gradient scaling factor
        """
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Generator training
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # Generate fake edges
                edge_logits = self.run_G(full_image, full_edge, mask, sync=sync)
                edge_probs = torch.sigmoid(edge_logits)
                
                # Compute composite (for discriminator)
                hole_mask = mask  # 1 = hole
                fake_edge_composite = full_edge * (1 - hole_mask) + edge_probs * hole_mask
                
                # --- Adversarial Loss ---
                if self.D is not None:
                    gen_logits, gen_features = self.run_D(fake_edge_composite, mask, sync=False, return_features=True)
                    w_hole = self._hole_weight(mask, gen_logits)
                    _rep = gen_logits if w_hole is None else gen_logits[w_hole > 0.5]
                    training_stats.report('Loss/scores/fake', _rep)
                    training_stats.report('Loss/signs/fake', _rep.sign())
                    
                    loss_Gadv = self._g_adv_loss(gen_logits, w_hole)
                    training_stats.report('Loss/G/adv', loss_Gadv)
                    
                    # --- Feature Matching Loss ---
                    with torch.no_grad():
                        _, real_features = self.run_D(full_edge, mask, sync=False, return_features=True)
                    
                    loss_fm = torch.tensor(0.0, device=self.device)
                    if real_features and gen_features:
                        for r_f, g_f in zip(real_features, gen_features):
                            loss_fm += F.l1_loss(g_f, r_f)
                        loss_fm = loss_fm / len(real_features)
                    training_stats.report('Loss/G/fm', loss_fm)
                else:
                    loss_Gadv = torch.tensor(0.0, device=self.device)
                    loss_fm = torch.tensor(0.0, device=self.device)
                    training_stats.report('Loss/G/adv', loss_Gadv)
                    training_stats.report('Loss/G/fm', loss_fm)
                
                # --- Reconstruction Losses (masked region only) ---
                # BCE/Focal Loss
                raw_bce_loss = self.bce_fn(edge_logits, full_edge)
                rec_w = self._rec_weight(hole_mask)          # == hole_mask when tau == 0
                bce_loss = (raw_bce_loss * rec_w).sum() / rec_w.sum().clamp(min=1.0)
                training_stats.report('Loss/G/bce', bce_loss)
                
                loss_recon = self.bce_weight * bce_loss
                
                # Dice Loss
                if self.use_dice:
                    dice_loss = self.dice_fn(edge_logits, full_edge, mask=hole_mask)
                    loss_recon = loss_recon + self.dice_weight * dice_loss
                    training_stats.report('Loss/G/dice', dice_loss)
                else:
                    training_stats.report('Loss/G/dice', torch.tensor(0.0))
                
                # Hausdorff Loss
                if self.use_hd:
                    hd_loss = self.hd_fn(edge_logits, full_edge, mask=hole_mask)
                    loss_recon = loss_recon + self.hd_weight * hd_loss
                    training_stats.report('Loss/G/hd', hd_loss)
                else:
                    training_stats.report('Loss/G/hd', torch.tensor(0.0))
                
                training_stats.report('Loss/G/recon', loss_recon)
                
                # --- Debug Statistics ---
                with torch.no_grad():
                    masked_probs = edge_probs * hole_mask
                    masked_probs_flat = masked_probs[hole_mask > 0]
                    
                    if masked_probs_flat.numel() > 0:
                        training_stats.report('Debug/probs_masked_min', masked_probs_flat.min())
                        training_stats.report('Debug/probs_masked_max', masked_probs_flat.max())
                        training_stats.report('Debug/probs_masked_mean', masked_probs_flat.mean())
                        
                        edge_ratio = (masked_probs_flat > 0.5).float().mean()
                        training_stats.report('Debug/edge_activation_ratio', edge_ratio)
                    
                    target_masked = full_edge * hole_mask
                    target_flat = target_masked[hole_mask > 0]
                    if target_flat.numel() > 0:
                        training_stats.report('Debug/target_edge_ratio', target_flat.mean())
                
                # --- VGG Style and Perceptual Losses ---
                if self.use_vgg_losses and self.vgg is not None:
                    # edge_probs already computed above as sigmoid(edge_logits)
                    loss_style, loss_perc = self._vgg_losses(edge_probs, full_edge, hole_mask)
                    training_stats.report('Loss/G/style', loss_style)
                    training_stats.report('Loss/G/perc',  loss_perc)
                else:
                    loss_style = torch.tensor(0.0, device=self.device)
                    loss_perc  = torch.tensor(0.0, device=self.device)
                    training_stats.report('Loss/G/style', loss_style)
                    training_stats.report('Loss/G/perc',  loss_perc)

                # --- Total Generator Loss ---
                loss_Gall = (self.adv_weight  * loss_Gadv  +
                             self.fm_weight   * loss_fm    +
                             loss_recon                    +
                             self.style_weight * loss_style +
                             self.perc_weight  * loss_perc)
                
                training_stats.report('Loss/G/total', loss_Gall)
                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gall.mul(gain).backward()

        # Dmain: Discriminator training on fake images
        loss_Dgen = 0
        fake_hole_scores = None
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # Generate fake edges (no gradient for G)
                with torch.no_grad():
                    edge_logits = self.run_G(full_image, full_edge, mask, sync=False)
                    edge_probs = torch.sigmoid(edge_logits)
                    fake_edge_composite = full_edge * (1 - mask) + edge_probs * mask
                
                gen_logits, _ = self.run_D(fake_edge_composite, mask, sync=False)
                w_hole = self._hole_weight(mask, gen_logits)
                _rep = gen_logits if w_hole is None else gen_logits[w_hole > 0.5]
                # Separate tag from the Gmain report (that one doubled the counts).
                training_stats.report('Loss/scores/fake_D', _rep)
                training_stats.report('Loss/signs/fake_D', _rep.sign())
                fake_hole_scores = _rep.detach()
                loss_Dgen = self._d_fake_loss(gen_logits, w_hole)
                
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mul(gain).backward()

        # Dmain + Dr1: Discriminator on real images + R1 regularization
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                full_edge_tmp = full_edge.detach().requires_grad_(do_Dr1)
                mask_tmp = mask.detach().requires_grad_(do_Dr1)
                real_logits, _ = self.run_D(full_edge_tmp, mask_tmp, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                # --- Diagnostic: can D tell real from fake INSIDE the hole? ---
                # If real_hole ~= fake_hole, D is judging the mask, not the content,
                # and the adversarial/FM gradient to G is ~0 in the hole.
                if do_Dmain and real_logits.ndim == 4:
                    with torch.no_grad():
                        w_r = self._hole_weight(mask, real_logits)
                        real_hole = real_logits.detach()[w_r > 0.5]
                        training_stats.report('Diag/D/real_hole_score', real_hole)
                        training_stats.report('Diag/D/real_known_score', real_logits.detach()[w_r < 0.05])
                        if real_hole.numel() > 0 and fake_hole_scores is not None and fake_hole_scores.numel() > 0:
                            training_stats.report('Diag/D/hole_gap', real_hole.mean() - fake_hole_scores.mean())
                            acc = 0.5 * ((real_hole > 0).float().mean() + (fake_hole_scores < 0).float().mean())
                            training_stats.report('Diag/D/hole_acc', acc)   # 0.5 = blind, 1.0 = perfect

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = F.softplus(-real_logits).mean()  # -log(sigmoid(real_logits))
                    _dgen = loss_Dgen.detach() if torch.is_tensor(loss_Dgen) else loss_Dgen
                    training_stats.report('Loss/D/loss', _dgen + loss_Dreal.detach())

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()], 
                            inputs=[full_edge_tmp], 
                            create_graph=True, 
                            only_inputs=True
                        )[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = (r1_penalty * (self.r1_gamma / 2)).mean()
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits.mean() * 0 + loss_Dreal + loss_Dr1).mul(gain).backward()

#----------------------------------------------------------------------------
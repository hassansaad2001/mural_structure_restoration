# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""
Projected Discriminator for Sketch Inpainting GAN.

Based on:
  Sauer et al., "Projected GANs Converge Faster" (NeurIPS 2021)
  https://github.com/autonomousvision/projected-gan

  Applied to mural inpainting in:
  St-Diffnet: Diffusion-based inpainting of Dunhuang murals with
  structural and textural guidance.

Architecture:
  ┌─ ProjectedDiscriminator ─────────────────────────────────────────┐
  │                                                                   │
  │  edge (B,1,H,W) ──repeat─▶ (B,3,H,W) ──┐                        │
  │  mask (B,1,H,W) ──────────────────────┐  │                        │
  │                                        ▼  ▼                       │
  │            EfficientNet-B0 backbone (FROZEN, pretrained)          │
  │               4 feature levels: H/4, H/8, H/16, H/32             │
  │               channels:          24,  40,  112,  320              │
  │                        │                                          │
  │           RandomProjection × 4  (FROZEN, random init)            │
  │              projects each scale → proj_dim=256                  │
  │                        │                                          │
  │           PatchDisc × 4  (TRAINED)                               │
  │              3-layer strided conv → 1-ch logit map               │
  │                        │                                          │
  │           sum of mean(logit_i) over 4 scales ──▶ scalar          │
  └───────────────────────────────────────────────────────────────────┘

Input format:
    edge : (B, 1, H, W)  in [0, 1]  — edge probability map
    mask : (B, 1, H, W)  in [0, 1]  — 1=hole, 0=valid

    The backbone receives: edge repeated to 3 channels.
    The mask is passed to each PatchDisc as an extra channel
    (concatenated after random projection).

Output:
    logits (scalar) — sum of mean patch-disc logits across 4 scales
    features (list)  — 4 projected feature maps, for feature matching
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.diffaug import DiffAugment


# ---------------------------------------------------------------------------
# EfficientNet-B0 feature channels at each level (from timm docs)
# levels 1-4 correspond to strides 4, 8, 16, 32
# level 0 (stride 2, 16ch) is skipped — too high-res for patch disc
# ---------------------------------------------------------------------------
_EFFNET_B0_CHANNELS = [24, 40, 112, 320]   # backbone output channels
_BACKBONE_LEVEL_INDICES = [1, 2, 3, 4]     # timm feature_info indices to use


class _EfficientNetBackbone(nn.Module):
    """
    Frozen pretrained EfficientNet-B0 feature extractor.

    Accepts 3-channel RGB-normalised input (edge repeated × 3).
    Returns 4 feature maps at H/4, H/8, H/16, H/32.
    """
    # ImageNet mean/std for EfficientNet normalisation
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for ProjectedDiscriminator. "
                "Install with: pip install timm"
            )

        # features_only=True → returns list of feature maps, no classifier
        self.net = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=_BACKBONE_LEVEL_INDICES,   # 4 scales
        )

        # Register normalisation constants as buffers (moved with .to(device))
        self.register_buffer(
            'mean', torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            'std',  torch.tensor(self.STD ).view(1, 3, 1, 1), persistent=False
        )

        # Freeze all backbone parameters — never trained
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x_3ch: torch.Tensor):
        """
        Args:
            x_3ch: (B, 3, H, W) in [0, 1]
        Returns:
            list of 4 feature tensors at H/4, H/8, H/16, H/32
        """
        # Normalise to ImageNet statistics
        x = (x_3ch - self.mean) / self.std
        return self.net(x)   # list of 4 tensors


# ---------------------------------------------------------------------------
# Random Projection  (frozen)
# ---------------------------------------------------------------------------

class _RandomProjection(nn.Module):
    """
    Frozen 1×1 convolution that projects c_in → proj_dim.

    Initialised with a random normal weight (Glorot-scaled) and
    immediately frozen — weights are never updated.

    Following the projected-GAN paper, the mask is concatenated as
    an extra spatial channel before projection so each scale is
    conditioned on the inpainting mask.
    """

    def __init__(self, c_in: int, proj_dim: int = 256):
        super().__init__()
        # +1 for mask channel concatenated before projection
        self.proj = nn.Conv2d(c_in + 1, proj_dim, kernel_size=1, bias=False)
        # Glorot uniform init (matches original paper)
        nn.init.xavier_uniform_(self.proj.weight)
        # Freeze — random projections are never trained
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat : (B, C, H, W)  backbone feature at this scale
            mask : (B, 1, H, W)  hole mask, resized to match feat spatial dims
        Returns:
            (B, proj_dim, H, W)
        """
        # Resize mask to match feature spatial dimensions
        if mask.shape[2:] != feat.shape[2:]:
            mask_rs = F.interpolate(
                mask, size=feat.shape[2:], mode='nearest'
            )
        else:
            mask_rs = mask

        x = torch.cat([feat, mask_rs], dim=1)   # (B, C+1, H, W)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Patch Discriminator Head  (trained)
# ---------------------------------------------------------------------------

class _PatchDisc(nn.Module):
    """
    Lightweight patch discriminator on a single projected feature scale.

    3-layer stack:
        proj_dim → ndf → ndf//2 → 1  (all 3×3, LeakyReLU 0.2)

    Returns a spatial logit map (B, 1, H', W') — not a scalar.
    The caller takes .mean() to get the per-sample score.
    """

    def __init__(self, proj_dim: int = 256, ndf: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            # Layer 0: proj_dim → ndf
            nn.Conv2d(proj_dim, ndf,    kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 1: ndf → ndf//2  (strided to further reduce spatial)
            nn.Conv2d(ndf,     ndf // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: ndf//2 → 1  (logit map)
            nn.Conv2d(ndf // 2, 1,      kernel_size=3, stride=1, padding=1, bias=True),
        )
        # Spectral norm on the first two conv layers for stability
        self.layers[0] = nn.utils.spectral_norm(self.layers[0])
        self.layers[2] = nn.utils.spectral_norm(self.layers[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1, H', W') logit map."""
        return self.layers(x)


# ---------------------------------------------------------------------------
# ProjectedDiscriminator  (public API)
# ---------------------------------------------------------------------------

class ProjectedDiscriminator(nn.Module):
    """
    Projected GAN Discriminator for grayscale edge inpainting.

    Works as a drop-in replacement for DiscriminatorSketch /
    DiscriminatorLama in the training loop.  The signature of
    ``forward`` matches the existing discriminators:

        forward(edge, mask, return_features=False)
            → logits                   (return_features=False)
            → (logits, features)       (return_features=True)

    Architecture summary:
        1. edge (1ch) → repeat × 3 → EfficientNet-B0 (FROZEN, pretrained)
           → 4 feature maps at H/4, H/8, H/16, H/32
        2. Each feature + mask → RandomProjection (FROZEN) → proj_dim=256
        3. Each projected feature → PatchDisc (TRAINED) → logit map
        4. logits = sum of mean(logit_i) across 4 scales

    Args:
        img_resolution : spatial resolution of input (unused directly, kept
                         for API compatibility with the training loop)
        proj_dim       : projection dimension for random projections (default 256)
        ndf            : base channel width for patch discriminator heads (default 64)
        pretrained     : whether to load pretrained EfficientNet-B0 weights (default True)
        use_diffaug    : whether to use DiffAugment (color, translation, cutout) as in the original paper
    """

    def __init__(
        self,
        img_resolution: int = 256,
        proj_dim: int = 256,
        ndf: int = 64,
        pretrained: bool = True,
        use_diffaug: bool = True,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.proj_dim = proj_dim
        self.num_scales = len(_BACKBONE_LEVEL_INDICES)   # 4
        self.use_diffaug = use_diffaug

        # ── Backbone (frozen pretrained EfficientNet-B0) ─────────────────
        self.backbone = _EfficientNetBackbone(pretrained=pretrained)

        # ── Random projections (one per scale, frozen) ────────────────────
        self.rand_proj = nn.ModuleList([
            _RandomProjection(c_in, proj_dim)
            for c_in in _EFFNET_B0_CHANNELS
        ])

        # ── Patch discriminator heads (one per scale, trained) ────────────
        self.patch_disc = nn.ModuleList([
            _PatchDisc(proj_dim, ndf)
            for _ in range(self.num_scales)
        ])

    # ------------------------------------------------------------------
    def _to_3ch(self, x: torch.Tensor) -> torch.Tensor:
        """Expand (B,1,H,W) to (B,3,H,W) by repeating the channel."""
        return x.repeat(1, 3, 1, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        edge: torch.Tensor,
        mask: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Args:
            edge            : (B, 1, H, W) in [0, 1]
            mask            : (B, 1, H, W) in [0, 1]  (1=hole)
            return_features : if True, also return projected feature list

        Returns:
            logits   : (B, 1) scalar score per sample
            features : list of 4 projected tensors (only if return_features=True)
        """
        if self.use_diffaug and self.training:
            # Apply Color augment only to the edge map so it doesn't corrupt the binary mask
            edge = DiffAugment(edge, policy='color')
            # Apply Spatial augments (Translation, Cutout) to both jointly to keep them aligned
            x = torch.cat([edge, mask], dim=1)
            x = DiffAugment(x, policy='translation,cutout')
            edge, mask = x[:, :1], x[:, 1:]

        # 1. Expand edge to 3 channels and extract multi-scale backbone features
        edge_3ch  = self._to_3ch(edge.float())          # (B, 3, H, W) in [0,1]
        bb_feats  = self.backbone(edge_3ch)              # list of 4 feature tensors

        # 2. Random project + patch discriminate at each scale
        logit_sum   = None
        proj_feats  = []

        for i, (feat, rand_proj, disc) in enumerate(
            zip(bb_feats, self.rand_proj, self.patch_disc)
        ):
            # Random projection (frozen) — injects mask conditioning
            pf = rand_proj(feat, mask)       # (B, proj_dim, H_i, W_i)
            proj_feats.append(pf)

            # Patch discriminator (trained) → (B, 1, H_i', W_i')
            logit_map = disc(pf)

            # Global-average over spatial dims → (B, 1)
            logit_scale = logit_map.flatten(2).mean(dim=2, keepdim=True)  # (B, 1, 1)
            logit_scale = logit_scale.squeeze(-1)                          # (B, 1)

            logit_sum = logit_scale if logit_sum is None else logit_sum + logit_scale

        if return_features:
            return logit_sum, proj_feats
        return logit_sum


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D = ProjectedDiscriminator(img_resolution=256, pretrained=False).to(device)

    B, H, W = 2, 256, 256
    edge = torch.rand(B, 1, H, W, device=device)
    mask = torch.rand(B, 1, H, W, device=device)

    # Forward: logits only
    logits = D(edge, mask)
    print(f'logits shape : {logits.shape}')    # (2, 1)

    # Forward: with features
    logits, feats = D(edge, mask, return_features=True)
    print(f'logits shape : {logits.shape}')
    print(f'num feat scales: {len(feats)}')
    for i, f in enumerate(feats):
        print(f'  scale {i}: {tuple(f.shape)}')

    # Only patch_disc and rand_proj (non-backbone) should have gradients
    trained   = sum(p.numel() for p in D.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in D.parameters() if not p.requires_grad)
    print(f'Trained params : {trained:,}')
    print(f'Frozen  params : {frozen:,}')

# Structure Reconstruction Network (SRN) of the SGTB mural-inpainting model, as a baseline.
#
# Re-implemented from the paper (no official code has been released):
#   X. Deng and Y. Yu, "Ancient mural inpainting via structure information guided two-branch
#   model", Heritage Science 11:131 (2023), https://doi.org/10.1186/s40494-023-00972-x
#   (article under CC BY 4.0).
# FourierUnit, SpectralTransform, FFC, FFC_BN_ACT, FFCResnetBlock and ConcatTupleLayer follow
# LaMa (Suvorov et al., WACV 2022, https://github.com/advimman/lama,
# saicinpainting/training/modules/ffc.py, Apache-2.0); the gated convolution follows
# DeepFill v2 (Yu et al., ICCV 2019).
#
# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""SGTB structure reconstruction generator (G_s) with this repo's interfaces.

    G(img, edge, mask) -> edge logits          (img in [-1, 1], edge in [0, 1], mask 1 = hole)

Architecture (paper, section "Structure reconstruction network" and Fig. 1b), at width 64:
    encoder  GatedConv 7x7 s1 ->  64 | GatedConv 5x5 s2 -> 128 | GatedConv 3x3 s2 -> 256
             each followed by InstanceNorm + ReLU
    middle   8 FFC residual blocks (LaMa FFCResnetBlock, 256 channels, 3/4 global, BN + ReLU)
    decoder  up x2 + GatedConv 3x3 -> 128 | up x2 + GatedConv 5x5 -> 64   (InstanceNorm + ReLU)
             GatedConv 7x7 -> 1 (logits)
Input, as in the paper: cat(M, gray * (1 - M), S * (1 - M)), gray in [0, 1], S = binary edge map.
The edge map is binarised at 0.5 first (the dataset's edge maps are soft JPEG maps; the paper
uses binary Canny edges). The paper's discriminator (Table 1) and stage-1 loss (adversarial +
10 x feature matching) are EdgeConnect's, so EdgeConnectDiscriminator and EdgeConnectLoss are reused.
Choices the paper leaves open are listed in docs/BASELINES.md.
"""

import torch
import torch.nn as nn

from networks.baselines import binarize_edge
from networks.baselines.edgeconnect import BaseNetwork, to_gray01


# =============================================================================
# Fast Fourier Convolution (LaMa, saicinpainting/training/modules/ffc.py)
# =============================================================================

class FourierUnit(nn.Module):
    """rfft2 -> 1x1 conv + BN + ReLU on stacked (real, imag) -> irfft2 (global receptive field)."""

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, stride=1,
                                    padding=0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x.float(), dim=(-2, -1), norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)                  # (B, C, H, W/2+1, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()                      # (B, C, 2, H, W/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.relu(self.bn(self.conv_layer(ffted)))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)


class SpectralTransform(nn.Module):
    """Global branch of FFC: 1x1 conv-BN-ReLU -> FourierUnit (+ optional LFU) -> residual -> 1x1 conv."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2) if stride == 2 else nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        return self.conv2(x + output + xs)


class FFC(nn.Module):
    """Local (conv) and global (spectral) branches with the four cross paths l2l, l2g, g2l, g2g."""

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=False, padding_type='reflect'):
        super().__init__()
        assert stride in (1, 2), 'Stride should be 1 or 2.'
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        def conv(cin, cout):
            if cin == 0 or cout == 0:
                return nn.Identity()
            return nn.Conv2d(cin, cout, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)

        self.convl2l = conv(in_cl, out_cl)
        self.convl2g = conv(in_cl, out_cg)
        self.convg2l = conv(in_cg, out_cl)
        self.convg2g = nn.Identity() if in_cg == 0 or out_cg == 0 else \
            SpectralTransform(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    """FFC followed by BatchNorm + activation on each branch."""

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                 padding_type='reflect', enable_lfu=False):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type)
        out_cg = int(out_channels * ratio_gout)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(out_channels - out_cg)
        self.bn_g = gnorm(out_cg)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


class FFCResnetBlock(nn.Module):
    """Two FFC_BN_ACT layers (3x3) with a residual connection on both branches (paper Fig. 2)."""

    def __init__(self, dim, ratio_g=0.75, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, dilation=1, enable_lfu=False):
        super().__init__()
        kwargs = dict(kernel_size=3, ratio_gin=ratio_g, ratio_gout=ratio_g, padding=dilation, dilation=dilation,
                      norm_layer=norm_layer, activation_layer=activation_layer, padding_type=padding_type,
                      enable_lfu=enable_lfu)
        self.conv1 = FFC_BN_ACT(dim, dim, **kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, **kwargs)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        return id_l + x_l, id_g + x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


# =============================================================================
# Gated convolution (DeepFill v2) and the SRN
# =============================================================================

class GatedConv2d(nn.Module):
    """Gated convolution: one conv gives a feature and a gate; output = feature * sigmoid(gate).

    Reflection padding keeps the size ('same' for stride 1, half for stride 2).
    Normalisation and activation are applied by the caller (paper: GatedConv -> IN -> ReLU).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(dilation * (kernel_size - 1) // 2)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride=stride, padding=0, dilation=dilation)

    def forward(self, x):
        feature, gate = self.conv(self.pad(x)).chunk(2, dim=1)
        return feature * torch.sigmoid(gate)


def _gated_block(cin, cout, kernel_size, stride=1, upsample=False):
    layers = [nn.Upsample(scale_factor=2, mode='nearest')] if upsample else []
    layers += [GatedConv2d(cin, cout, kernel_size, stride=stride),
               nn.InstanceNorm2d(cout, track_running_stats=False),
               nn.ReLU(True)]
    return nn.Sequential(*layers)


class StructureReconstructionNetwork(BaseNetwork):
    """G_s: 3 gated-conv encoder layers, 8 FFC residual blocks, 3 gated-conv decoder layers. Returns logits."""

    def __init__(self, in_channels=3, ngf=64, residual_blocks=8, ratio_g=0.75, enable_lfu=False, init_weights=True):
        super().__init__()
        dim = ngf * 4
        self.encoder = nn.Sequential(
            _gated_block(in_channels, ngf, 7),                       # H     x  64
            _gated_block(ngf, ngf * 2, 5, stride=2),                 # H/2   x 128
            _gated_block(ngf * 2, dim, 3, stride=2),                 # H/4   x 256
        )
        self.global_num = int(dim * ratio_g)
        self.middle = nn.Sequential(
            *[FFCResnetBlock(dim, ratio_g=ratio_g, enable_lfu=enable_lfu) for _ in range(residual_blocks)],
            ConcatTupleLayer(),
        )
        self.decoder = nn.Sequential(
            _gated_block(dim, ngf * 2, 3, upsample=True),            # H/2   x 128
            _gated_block(ngf * 2, ngf, 5, upsample=True),            # H     x  64
            GatedConv2d(ngf, 1, 7),                                  # H     x   1 (logits)
        )
        if init_weights:
            self.init_weights()                                      # EdgeConnect init: N(0, 0.02)

    def forward(self, x):
        x = self.encoder(x)
        x = (x[:, :-self.global_num], x[:, -self.global_num:])      # (local, global), LaMa channel order
        x = self.middle(x)
        return self.decoder(x)


# =============================================================================
# Wrapper for this framework
# =============================================================================

class SGTBStructureGenerator(nn.Module):
    """SGTB structure reconstruction generator; returns logits. Resolution independent.

    binarize_edges: the input edge map is binarised at this threshold (the paper uses binary
    Canny edges); None keeps it soft.
    """

    def __init__(self, img_resolution=512, ngf=64, residual_blocks=8, ratio_g=0.75, enable_lfu=False,
                 binarize_edges=0.5):
        super().__init__()
        self.img_resolution = img_resolution
        self.binarize_edges = binarize_edges
        self.net = StructureReconstructionNetwork(in_channels=3, ngf=ngf, residual_blocks=residual_blocks,
                                                  ratio_g=ratio_g, enable_lfu=enable_lfu)

    def forward(self, img, edge, mask):
        gray = to_gray01(img)
        edge = binarize_edge(edge, self.binarize_edges)
        mask = mask.to(torch.float32)
        gray_masked = gray * (1 - mask)                              # paper: gray_masked = I_gray ⊙ (1 − M)
        edge_masked = edge * (1 - mask)                              #        S_masked   = S_gray ⊙ (1 − M)
        return self.net(torch.cat((mask, gray_masked, edge_masked), dim=1))

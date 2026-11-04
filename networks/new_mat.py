# Based on networks/mat.py of MAT: Mask-Aware Transformer for Large Hole Image
# Inpainting (Li et al., CVPR 2022, https://github.com/fenglinglwb/MAT), which
# builds on NVIDIA StyleGAN2-ADA. Licensed under CC BY-NC 4.0 (see LICENSE).
# Modified by Hassan Saad (2026) for mural structure (edge) restoration.
"""Networks for mural structure (edge) restoration.

Main classes
------------
GeneratorSketch      Edge-inpainting generator used for all experiments:
                     mask-aware partial-conv encoder -> global-context
                     bottleneck (LaMa FFC by default; plain conv or MAT-style
                     mask-aware Swin as ablations) -> depthwise-separable
                     decoder with multi-dilation transposed attention -> edge logits.
DiscriminatorSketch  StyleGAN-style discriminator on (edge, mask).
DiscriminatorLama    PatchGAN discriminator (LaMa style).

Building blocks kept from MAT: Conv2dLayerPartial (mask-aware conv),
WindowAttention / SwinTransformerBlock / BasicLayer (mask-aware Swin).
Ideas adapted from other work: FFC / SpectralTransform (LaMa, Suvorov et al.
2022), multi-Dconv transposed attention (Restormer, Zamir et al. 2022).

Other classes in this file (FirstStage_Sketch, TEP*, ParallelFFCMDTABlock,
EdgeRefinementHead, ...) are earlier architecture variants. They are kept so
that class names stay stable for old checkpoints, but GeneratorSketch does not
use them.

Class and parameter names must not be renamed: checkpoints store weights by
attribute name, and training_options.json stores 'networks.new_mat.<Class>'.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample, bias_act
from networks.basic_module import FullyConnectedLayer, Conv2dLayer, MappingNet, MinibatchStdLayer, DisFromRGB, DisBlock, StyleConv, ToRGB, get_style_code, GatedConv2d


@misc.profiled_function
def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2 ** stage]


@persistence.persistent_class
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(in_features=in_features, out_features=hidden_features, activation='lrelu')
        self.fc2 = FullyConnectedLayer(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# =============================================================================
# TEP Module Components (from Bai et al., CMC 2024, DOI:10.32604/cmc.2023.044612)
# =============================================================================

@persistence.persistent_class
class TEPMlp(nn.Module):
    """MLP as per TEP paper: FC(D→4D) → GELU → Dropout → FC(4D→D) → Dropout."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@persistence.persistent_class
class AxialAttentionLayer(nn.Module):
    """
    Axial attention with relative position encoding (RPE) — Eq. 5-6 of the paper.

    Computes row-wise (width-axis) and column-wise (height-axis) attention
    independently, each with its own learnable RPE table, then sums both
    results with a residual connection.

    Args:
        dim: Channel dimension C
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for RPE table (per axis)
        attn_drop: Attention weight dropout
        proj_drop: Output projection dropout
    """
    def __init__(self, dim, num_heads=6, max_seq_len=128, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Width-axis (row) Q/K/V
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

        # Height-axis (column) Q/K/V
        self.hq = nn.Linear(dim, dim)
        self.hk = nn.Linear(dim, dim)
        self.hv = nn.Linear(dim, dim)

        # Learnable relative position encoding tables (1D per axis)
        self.rpe_width = nn.Parameter(torch.zeros(2 * max_seq_len - 1, num_heads))
        self.rpe_height = nn.Parameter(torch.zeros(2 * max_seq_len - 1, num_heads))
        trunc_normal_(self.rpe_width, std=0.02)
        trunc_normal_(self.rpe_height, std=0.02)
        self.max_seq_len = max_seq_len

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rpe_bias(self, seq_len, rpe_table):
        """Get relative position bias matrix of shape (num_heads, seq_len, seq_len)."""
        coords = torch.arange(seq_len, device=rpe_table.device)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # (L, L)
        relative_coords = relative_coords + self.max_seq_len - 1  # shift to [0, 2*max-2]
        relative_coords = relative_coords.clamp(0, 2 * self.max_seq_len - 2)
        bias = rpe_table[relative_coords]  # (L, L, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, L, L)

    def _axial_attn(self, x, q_proj, k_proj, v_proj, rpe_table):
        """
        Standard MHA along last-but-one dim with RPE.
        x: (Batch, SeqLen, C)
        """
        B, N, C = x.shape
        nh = self.num_heads
        hd = C // nh

        q = q_proj(x).reshape(B, N, nh, hd).permute(0, 2, 1, 3)  # (B, nh, N, hd)
        k = k_proj(x).reshape(B, N, nh, hd).permute(0, 2, 3, 1)  # (B, nh, hd, N)
        v = v_proj(x).reshape(B, N, nh, hd).permute(0, 2, 1, 3)  # (B, nh, N, hd)

        attn = (q @ k) * self.scale  # (B, nh, N, N)

        # Add relative position bias
        rpe_bias = self._get_rpe_bias(N, rpe_table)  # (nh, N, N)
        attn = attn + rpe_bias.unsqueeze(0)

        attn = attn.clamp(-50, 50)  # prevent overflow in softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        return out

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C) — token sequence
            H, W: spatial dimensions
        Returns:
            (B, H*W, C)
        """
        B, N, C = x.shape
        assert N == H * W

        # --- Width-axis (row) attention ---
        # Reshape to (B*H, W, C) — each row is an independent sequence
        x_w = x.view(B, H, W, C).reshape(B * H, W, C)
        out_w = self._axial_attn(x_w, self.wq, self.wk, self.wv, self.rpe_width)
        out_w = out_w.reshape(B, H, W, C).reshape(B, N, C)

        # --- Height-axis (column) attention ---
        # Reshape to (B*W, H, C) — each column is an independent sequence
        x_h = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B * W, H, C)
        out_h = self._axial_attn(x_h, self.hq, self.hk, self.hv, self.rpe_height)
        out_h = out_h.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, N, C)

        # Sum both axes + project (Eq. 6)
        out = self.proj(out_w + out_h)
        out = self.proj_drop(out)
        return out


@persistence.persistent_class
class StandardAttentionLayer(nn.Module):
    """
    Standard full global multi-head self-attention (Eq. 8 of the paper).
    Operates on flattened spatial tokens (B, D, C) where D=H*W.
    """
    def __init__(self, dim, num_heads=6, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: (B, N, C) — token sequence
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        nh = self.num_heads
        hd = C // nh

        q = self.q(x).reshape(B, N, nh, hd).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, nh, hd).permute(0, 2, 3, 1)
        v = self.v(x).reshape(B, N, nh, hd).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale
        attn = attn.clamp(-50, 50)  # prevent overflow in softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


@persistence.persistent_class
class TEPBlock(nn.Module):
    """
    Single TEP block from the paper (Eqs. 5-9):
      1. LN → AxialAttention (RPE) → + Residual
      2. LN → MLP (GELU) → + Residual
      3. LN → StandardAttention → + Residual
      4. LN → MLP (GELU) → + Residual

    Interface matches SwinTransformerBlock:
        forward(x, x_size, mask=None) → (x, mask)
        x: (B, H*W, C), mask: (B, H*W, 1) or None
    """
    def __init__(self, dim, input_resolution, num_heads=6, mlp_ratio=4.,
                 attn_drop=0., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 max_seq_len=128):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        mlp_hidden = int(dim * mlp_ratio)

        # Sub-block 1: Axial Attention
        self.norm1 = norm_layer(dim)
        self.axial_attn = AxialAttentionLayer(
            dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Sub-block 2: MLP after axial attention
        self.norm2 = norm_layer(dim)
        self.mlp1 = TEPMlp(in_features=dim, hidden_features=mlp_hidden, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Sub-block 3: Standard Attention
        self.norm3 = norm_layer(dim)
        self.std_attn = StandardAttentionLayer(
            dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Sub-block 4: MLP after standard attention
        self.norm4 = norm_layer(dim)
        self.mlp2 = TEPMlp(in_features=dim, hidden_features=mlp_hidden, drop=drop)
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, x_size, mask=None):
        """
        Args:
            x: (B, H*W, C)
            x_size: (H, W)
            mask: (B, H*W, 1) or None — passed through unchanged
        Returns:
            (x, mask)
        """
        H, W = x_size

        # 1. Axial Attention + Residual (Eq. 5-6)
        x = x + self.drop_path1(self.axial_attn(self.norm1(x), H, W))

        # 2. MLP + Residual (Eq. 7)
        x = x + self.drop_path2(self.mlp1(self.norm2(x)))

        # 3. Standard Attention + Residual (Eq. 8)
        x = x + self.drop_path3(self.std_attn(self.norm3(x)))

        # 4. MLP + Residual (Eq. 9)
        x = x + self.drop_path4(self.mlp2(self.norm4(x)))

        return x, mask


# =============================================================================
# TEP-aware BasicLayer
# =============================================================================

@persistence.persistent_class
class TEPBasicLayer(nn.Module):
    """
    BasicLayer variant that uses TEPBlock instead of SwinTransformerBlock.
    Same interface as BasicLayer for drop-in replacement.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4.,
                 attn_drop=0., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, max_seq_len=128):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = None

        self.blocks = nn.ModuleList([
            TEPBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, max_seq_len=max_seq_len
            )
            for i in range(depth)
        ])

        self.conv = Conv2dLayerPartial(
            in_channels=dim, out_channels=dim, kernel_size=3, activation='lrelu'
        )

    def forward(self, x, x_size, mask=None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@misc.profiled_function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@misc.profiled_function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    # Avoid int() which triggers CPU sync — use pure integer math
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = windows.shape[0] // (num_windows_h * num_windows_w)
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


@persistence.persistent_class
class Conv2dLayerPartial(nn.Module):
    """Mask-aware (partial) convolution built on MAT's Conv2dLayer.

    FIX. The original MAT version
        out = act(conv(x) + b) * (k^2 / n_valid)
    rescaled AFTER bias and activation, so boundary pixels with 1 valid
    neighbour got their activated output (bias included) amplified up to 9x.
    It also updated the mask with a plain 3x3/stride conv while the feature
    conv used the [1,3,3,1] low-pass filter when down=2, so the renormalisation
    did not match the real footprint. Now:
        out = act( conv(x) * (full / coverage) + b ) * update_mask
    where `coverage` is computed with the SAME conv2d_resample op (same filter,
    stride, padding) applied to the mask, and `full` is the interior footprint
    mass. Holes stay exactly zero after the activation.
    Same parameters / state_dict keys as before.
    """
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 bias            = True,         # Apply additive bias before the activation function?
                 activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
                 conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
                 trainable       = True,         # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter,
                                conv_clamp, trainable)
        for _attr in ('weight', 'weight_gain', 'resample_filter', 'up', 'down', 'padding',
                      'activation', 'act_gain', 'conv_clamp'):
            assert hasattr(self.conv, _attr), f'Conv2dLayer is missing attribute {_attr!r}'
        self.kernel_size = kernel_size
        self.max_ratio = float(kernel_size ** 2)
        self.min_coverage = 1e-3

    def _resample(self, x, w):
        return conv2d_resample.conv2d_resample(x=x, w=w, f=self.conv.resample_filter,
                                               up=self.conv.up, down=self.conv.down,
                                               padding=self.conv.padding)

    def forward(self, x, mask=None):
        if mask is None:
            x = self.conv(x)
            return x, None

        with torch.no_grad():
            k = self.kernel_size
            ones_w = torch.ones([1, 1, k, k], device=x.device, dtype=x.dtype)
            m = mask.to(x.dtype)
            cov = self._resample(m, ones_w)                                     # footprint mass on valid pixels
            full = self._resample(torch.ones_like(m[:1]), ones_w).amax()        # interior footprint mass
            frac = cov / full
            update_mask = (frac > self.min_coverage).to(x.dtype)
            ratio = (1.0 / frac.clamp(min=1.0 / self.max_ratio)) * update_mask

        w = self.conv.weight * self.conv.weight_gain
        x = self._resample(x, w.to(x.dtype))
        x = x * ratio
        x = bias_act.bias_act(x, self.conv.bias, act=self.conv.activation,
                              gain=self.conv.act_gain, clamp=self.conv.conv_clamp)
        x = x * update_mask
        return x, update_mask


@persistence.persistent_class
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, down_ratio=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_windows=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = self.q(norm_x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(norm_x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(attn_mask_windows == 0, float(-100.0)).masked_fill(
                attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(torch.sum(mask_windows, dim=1, keepdim=True), 0, 1).repeat(1, N, 1)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows


@persistence.persistent_class
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, down_ratio=1, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    down_ratio=down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                    proj_drop=drop)

        self.fuse = FullyConnectedLayer(in_features=dim * 2, out_features=dim, activation='lrelu')

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size, device=None):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, mask=None):
        # H, W = self.input_resolution
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.attn_mask)
        else:
            # Dynamic resolution: compute mask directly on GPU (avoid CPU→GPU transfer)
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.calculate_mask(x_size, device=x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            if mask is not None:
                mask = torch.roll(shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask


@persistence.persistent_class
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       down=down,
                                       )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       up=up,
                                       )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask



@persistence.persistent_class
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, down_ratio=1,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, down_ratio=down_ratio, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, activation='lrelu')

    def forward(self, x, x_size, mask=None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class ToToken(nn.Module):
    def __init__(self, in_channels=3, dim=128, kernel_size=5, stride=1):
        super().__init__()

        self.proj = Conv2dLayerPartial(in_channels=in_channels, out_channels=dim, kernel_size=kernel_size, activation='lrelu')

    def forward(self, x, mask):
        x, mask = self.proj(x, mask)

        return x, mask

#----------------------------------------------------------------------------

@persistence.persistent_class
class EncFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                activation=activation,
                                )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                activation=activation,
                                )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x

@persistence.persistent_class
class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(self, res_log2, img_channels, activation, patch_size=5, channels=16, drop_path_rate=0.1):
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i+1), nf(i), activation)
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x

        return out


@persistence.persistent_class
class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(in_features=in_channels,
                                      out_features=out_channels,
                                      activation=activation)
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        # x = self.dropout(x)

        return x


@persistence.persistent_class
class DecBlockFirstV2(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                activation=activation,
                                )
        self.conv1 = StyleConv(in_channels=in_channels,
                              out_channels=out_channels,
                              style_dim=style_dim,
                              resolution=2**res,
                              kernel_size=3,
                              use_noise=use_noise,
                              activation=activation,
                              demodulate=demodulate,
                              )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistence.persistent_class
class Decoder(nn.Module):
    def __init__(self, res_log2, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels)
        for res in range(5, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), activation, style_dim, use_noise, demodulate, img_channels))
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img


@persistence.persistent_class
class DecStyleBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, style, skip, noise_mode='random'):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img



# The code below assumes the following helper classes/functions are available
# from the original file and will be imported at module-level when integrated:
# - Conv2dLayerPartial
# - Conv2dLayer
# - DecStyleBlock (not used here, but left in original file)
# - token2feature, feature2token
# - BasicLayer, PatchMerging, PatchUpsampling
# ------------------------------
# Replacement FirstStage class
# ------------------------------
class FirstStage_Sketch(nn.Module):
    """
    Sketch-optimized FirstStage for MAT with Distance Transform support.
    """

    def __init__(self, img_resolution=256, dim=180, use_noise=False, 
                 demodulate=True, activation='lrelu', preserve_high_res=True):
        super().__init__()

        res = 64
        
        # Input: [mask, sketch*mask, dt*mask, gray_image*mask]
        self.conv_first = Conv2dLayerPartial(
            in_channels=4, 
            out_channels=dim,
            kernel_size=3,
            activation=activation
        )

        # Encoder conv-down blocks
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        if preserve_high_res and down_time >= 1:
            down_time = max(0, down_time - 1)

        for i in range(down_time):
            self.enc_conv.append(
                Conv2dLayerPartial(
                    in_channels=dim, 
                    out_channels=dim,
                    kernel_size=3, 
                    down=2, 
                    activation=activation
                )
            )

        # Transformer stack
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1/2, 1/2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        cur_res = res
        dp_ptr = 0
        for i, depth in enumerate(depths):
            cur_res = int(cur_res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1/ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None

            dp_rates = dpr[dp_ptr: dp_ptr + depth] if isinstance(dpr, list) else dpr
            dp_ptr += depth

            layer = BasicLayer(
                dim=dim, 
                input_resolution=[cur_res, cur_res], 
                depth=depth,
                num_heads=num_heads, 
                window_size=window_sizes[i], 
                down_ratio=1,
                mlp_ratio=2., 
                qkv_bias=True, 
                qk_scale=None, 
                drop=0., 
                attn_drop=0.,
                drop_path=dp_rates, 
                norm_layer=nn.LayerNorm, 
                downsample=merge
            )
            self.tran.append(layer)

        # Decoder with upsampling blocks
        self.decoder_blocks = nn.ModuleList()
        num_ups = max(1, down_time)
        for i in range(num_ups):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    # Input channels = dim * 2 (because of concat)
                    Conv2dLayer(
                        in_channels=dim * 2, 
                        out_channels=dim, 
                        kernel_size=3, 
                        activation=activation
                    )
                )
            )

        # Separate heads for sketch and distance transform
        self.to_sketch = nn.Sequential(
            Conv2dLayer(in_channels=dim, out_channels=dim//2, kernel_size=3, activation=activation),
            Conv2dLayer(in_channels=dim//2, out_channels=1, kernel_size=1, activation='linear')
        )
        
        self.to_dt = nn.Sequential(
            Conv2dLayer(in_channels=dim, out_channels=dim//2, kernel_size=3, activation=activation),
            Conv2dLayer(in_channels=dim//2, out_channels=1, kernel_size=1, activation='linear')
        )

        # Refinement convs for both outputs
        self.refine_sketch = Conv2dLayer(
            in_channels=1, 
            out_channels=1, 
            kernel_size=3, 
            activation='linear'
        )
        self.refine_dt = Conv2dLayer(
            in_channels=1, 
            out_channels=1, 
            kernel_size=3, 
            activation='linear'
        )

        # Add a projection layer for the skips
        self.skip_proj = nn.ModuleList()
        # Projections for Transformer skips (dim*2 -> dim)
        for i in range(len(self.tran) // 2): 
            # FIXED: Changed Conv2dLayer to Conv2dLayerPartial to handle (x, mask=None) call
            self.skip_proj.append(Conv2dLayerPartial(dim * 2, dim, 1)) 
        
        self.use_noise = use_noise
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, Conv2dLayer)):
            pass
        
        # Initialize output bias
        last_conv = self.to_sketch[-1] 
        if hasattr(last_conv, 'bias') and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, 0.5) 
            
        if hasattr(self.refine_sketch, 'bias') and self.refine_sketch.bias is not None:
             nn.init.constant_(self.refine_sketch.bias, 0.5)

    def forward(self, gray_in, sketch_in, mask_in, dt_in, ws=None, noise_mode='random'):
        
        gray_in = gray_in.to(dtype=torch.float32)
        sketch_in = sketch_in.to(dtype=torch.float32)
        mask_in = mask_in.to(dtype=torch.float32)
        dt_in = dt_in.to(dtype=torch.float32)

        # Concatenate 4 channels
        x = torch.cat([
            mask_in - 0.5,           
            sketch_in * mask_in,     
            dt_in * mask_in,         
            gray_in * mask_in        
        ], dim=1)

        skips = []
        x, mask = self.conv_first(x, mask_in)
        skips.append(x)

        # Encoder
        for i, block in enumerate(self.enc_conv):
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)
            else:
                pass

        # Transformer
        x_size = x.size()[-2:]
        x_tok = feature2token(x)
        mask_tok = feature2token(mask)

        mid = len(self.tran) // 2
        
        for i, block in enumerate(self.tran):
            if i < mid:
                # Downsampling part
                x_tok, x_size, mask_tok = block(x_tok, x_size, mask_tok)
                skips.append(x_tok)
                
            elif i > mid:
                # Upsampling part
                x_tok, x_size, _ = block(x_tok, x_size, None)
                
                if skips:
                    skip_feat = skips.pop()
                    
                    # 1. Concatenate features
                    x_tok = torch.cat([x_tok, skip_feat], dim=2)
                    
                    # 2. Project back to C (B, L, 2C) -> (B, L, C)
                    B, N, C2 = x_tok.shape
                    H, W = x_size
                    x_spatial = x_tok.transpose(1, 2).view(B, C2, H, W)
                    
                    # Find correct skip projection layer index
                    # Mapping: i=3 -> p_idx=1; i=4 -> p_idx=0
                    p_idx = (len(self.tran) - 1) - i
                    
                    # Call now works because Conv2dLayerPartial accepts (x, None)
                    x_spatial, _ = self.skip_proj[p_idx](x_spatial, None)
                    
                    x_tok = feature2token(x_spatial)

            else:
                # Middle part
                x_tok, x_size, mask_tok = block(x_tok, x_size, None)

        # Back to spatial features
        x = token2feature(x_tok, x_size)

        # Decoder
        for up_block in self.decoder_blocks:
            if skips:
                skip = skips.pop() 
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='nearest')
                x = torch.cat([x, skip], dim=1) 
            
            x = up_block(x)

        # Dual heads
        sketch_out = self.to_sketch(x)
        dt_out = self.to_dt(x)

        # Refinement
        sketch_out = self.refine_sketch(sketch_out)
        dt_out = self.refine_dt(dt_out)

        # Final Activations
        sketch_out = torch.tanh(sketch_out)
        dt_out = torch.sigmoid(dt_out)

        # Ensemble with known pixels
        sketch_out = sketch_out * (1 - mask_in) + sketch_in * mask_in
        dt_out = dt_out * (1 - mask_in) + dt_in * mask_in
        
        return sketch_out, dt_out

class ResidualPartialBlock(nn.Module):
    """
    Residual Block with Partial Convolutions for mask-aware synthesis.
    Eliminates checkerboard artifacts and preserves structural detail.
    """
    def __init__(self, in_channels, out_channels, activation='lrelu'):
        super().__init__()
        self.conv1 = Conv2dLayerPartial(in_channels, out_channels, kernel_size=3, activation=activation)
        self.conv2 = Conv2dLayerPartial(out_channels, out_channels, kernel_size=3, activation='linear')
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation='linear')

    def forward(self, x, mask):
        identity = x
        # Synthesis branch with mask tracking
        out, updated_mask = self.conv1(x, mask)
        out, updated_mask = self.conv2(out, updated_mask)
        
        # Residual branch
        if not isinstance(self.shortcut, nn.Identity):
            identity = self.shortcut(identity)
            
        return F.leaky_relu(out + identity, negative_slope=0.2), updated_mask

# =============================================================================
# LaMa FFC — Fast Fourier Convolution (NeurIPS 2020)
# Ported from https://github.com/advimman/lama/blob/main/saicinpainting/training/modules/ffc.py
# Self-contained: no saicinpainting imports required.
# =============================================================================

class FourierUnit(nn.Module):
    """
    Applies a 1×1 convolution in the frequency domain (global receptive field).

    Pipeline:
        x  →  rfftn  →  stack(real, imag)  →  reshape  →  Conv2d(1×1)+BN+ReLU
           →  unstack  →  irfftn  →  output

    Args:
        in_channels   : number of input channels
        out_channels  : number of output channels
        groups        : conv groups (default 1)
        fft_norm      : normalization for torch.fft (default 'ortho')
    """
    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0,
            groups=self.groups, bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]

        # rfftn over last two spatial dims → complex tensor (B, C, H, W//2+1)
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        # stack real/imag as extra dim, then flatten into channel axis
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)          # (B,C,H,W',2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()              # (B,C,2,H,W')
        ffted = ffted.view((batch, -1) + ffted.size()[3:])              # (B,2C,H,W')

        ffted = self.conv_layer(ffted)                                  # (B,2C_out,H,W')
        ffted = self.relu(self.bn(ffted))

        # unstack channels back to real+imag, then irfftn
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(  # (B,C,H,W',2)
            0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        return output


class SpectralTransform(nn.Module):
    """
    LaMa's SpectralTransform: a richer global branch.

    Pipeline:
        x  →  [AvgPool(stride=2)]  →  conv1(C→C/2)+BN+ReLU
           →  FourierUnit  →  [LFU on quarter tiles]  →  conv2(C/2→C)

    The Local Fourier Unit (LFU, enable_lfu=True) splits the feature map
    into 2×2 quarter-patches, applies a shared FourierUnit, and tiles the
    result back — capturing mid-frequency patterns.

    Args:
        in_channels  : input channel count
        out_channels : output channel count
        stride       : 1 or 2 (strided variant uses AvgPool)
        groups       : conv groups
        enable_lfu   : whether to enable Local Fourier Unit (default True)
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.stride = stride

        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            # The LFU tiles (half_c // 4) channels into 2×2 spatial quarter-patches,
            # so the actual input channel count to the LFU FourierUnit is
            # (half_c // 4) * 4, which may be less than half_c when half_c is not
            # divisible by 4 (e.g., dim=180, ratio_g=0.75 → half_c=67, lfu_in=64).
            # We project from lfu_in_c → half_c so the residual `x + output + xs`
            # has consistent channels. When half_c % 4 == 0, lfu_in_c == half_c
            # and the behaviour is identical to the original LaMa code.
            _half_c = out_channels // 2
            _lfu_in_c = (_half_c // 4) * 4
            self.lfu = FourierUnit(_lfu_in_c, _half_c, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels,
                               kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s  = h // split_no
            # tile quarter-patches along channel axis, apply LFU, repeat back
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    """
    Fast Fourier Convolution (NeurIPS 2020).

    Splits channels into a *local* branch (regular Conv2d) and a *global*
    branch (SpectralTransform).  Four cross-branch paths are learned:

        local_in  →  conv_l2l  →  local_out
        local_in  →  conv_l2g  →  global_out
        global_in →  conv_g2l  →  local_out
        global_in →  conv_g2g (SpectralTransform) → global_out

    Args:
        in_channels   : total input channels
        out_channels  : total output channels
        kernel_size   : conv kernel size for local branches
        ratio_gin     : fraction of *input*  channels that are global
        ratio_gout    : fraction of *output* channels that are global
        stride        : 1 or 2
        padding       : conv padding
        dilation      : conv dilation
        groups        : conv groups
        bias          : whether to add bias
        enable_lfu    : pass-through to SpectralTransform
        padding_type  : padding mode for Conv2d ('reflect', 'zeros', etc.)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.75, ratio_gout=0.75,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 enable_lfu=True, padding_type='reflect', **spectral_kwargs):
        super().__init__()
        assert stride in (1, 2), "FFC stride must be 1 or 2"
        self.stride = stride

        in_cg  = int(in_channels  * ratio_gin)
        in_cl  = in_channels  - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin     = ratio_gin
        self.ratio_gout    = ratio_gout
        self.global_in_num = in_cg

        def _conv(cin, cout):
            return nn.Conv2d(cin, cout, kernel_size, stride, padding,
                             dilation, groups, bias, padding_mode=padding_type)

        # local → local
        self.convl2l = nn.Identity() if (in_cl == 0 or out_cl == 0) else _conv(in_cl, out_cl)
        # local → global
        self.convl2g = nn.Identity() if (in_cl == 0 or out_cg == 0) else _conv(in_cl, out_cg)
        # global → local
        self.convg2l = nn.Identity() if (in_cg == 0 or out_cl == 0) else _conv(in_cg, out_cl)
        # global → global  (via SpectralTransform)
        self.convg2g = nn.Identity() if (in_cg == 0 or out_cg == 0) else SpectralTransform(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2,
            enable_lfu, **spectral_kwargs)

    def forward(self, x):
        # x may be a tensor (ratio_gin == 0) or a (local, global) tuple
        if type(x) is tuple:
            x_l, x_g = x
        else:
            # split according to ratio_gin
            split = self.global_in_num
            x_l = x if split == 0 else x[:, :-split]
            x_g = 0  if split == 0 else x[:, -split:]

        # four cross-branch paths
        out_xl  = (0 if isinstance(self.convl2l, nn.Identity) else self.convl2l(x_l)) + \
                  (0 if isinstance(self.convg2l, nn.Identity) else self.convg2l(x_g))
        out_xg  = (0 if isinstance(self.convl2g, nn.Identity) else self.convl2g(x_l)) + \
                  (0 if isinstance(self.convg2g, nn.Identity) else self.convg2g(x_g))
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    """
    FFC followed by per-branch BatchNorm + activation.

    Wraps FFC so each branch gets its own normalisation, matching LaMa's
    original design.  Local branch uses lnorm/lact; global branch uses gnorm/gact.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.75, ratio_gout=0.75,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                 padding_type='reflect', enable_lfu=True, **kwargs):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                       stride=stride, padding=padding, dilation=dilation,
                       groups=groups, bias=bias,
                       enable_lfu=enable_lfu, padding_type=padding_type, **kwargs)

        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        # use Identity when a branch has zero channels
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l  = lnorm(out_cl)
        self.bn_g  = gnorm(out_cg)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    """
    LaMa's FFCResnetBlock — two FFC_BN_ACT layers with residual connection.

    When ``inline=True`` the module accepts/returns a plain concatenated tensor
    (suitable for use in a Sequential or plain loop), splitting off the global
    channels internally.  When ``inline=False`` it accepts/returns a
    (local, global) tuple.

    Args:
        dim          : total channel width
        ratio_g      : fraction of channels treated as global (default 0.75)
        padding_type : padding mode for internal convolutions
        norm_layer   : normalisation class (default BatchNorm2d)
        activation_layer : activation class (default ReLU)
        dilation     : conv dilation
        enable_lfu   : pass-through to SpectralTransform / FourierUnit
        inline       : if True, concatenate local+global before returning
    """
    def __init__(self, dim, ratio_g=0.75,
                 padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, dilation=1,
                 enable_lfu=True, inline=True):
        super().__init__()
        conv_kwargs = dict(
            ratio_gin=ratio_g, ratio_gout=ratio_g,
            padding_type=padding_type, enable_lfu=enable_lfu,
        )
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3,
                                padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3,
                                padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                **conv_kwargs)
        self.inline = inline
        self._global_num = int(dim * ratio_g)

    def forward(self, x):
        if self.inline:
            # split concatenated tensor into (local, global)
            x_l = x[:, :-self._global_num]
            x_g = x[:,  -self._global_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        # residual
        x_l = id_l + x_l
        if torch.is_tensor(id_g):
            x_g = id_g + x_g

        if self.inline:
            return torch.cat([x_l, x_g], dim=1)
        return x_l, x_g


# =============================================================================
# Multi-Dilation Transposed Attention (MDTA)
# Inspired by Restormer (CVPR 2022) with per-head dilated DW-conv for
# multi-scale spatial bias — especially effective for sketch/edge restoration.
# =============================================================================

class MultiDilationTransposedAttention(nn.Module):
    """
    Transposed Multi-head Self-Attention with per-head dilated depth-wise convolutions.

    Key properties
    --------------
    - Complexity O(C² · HW) vs O(HW² · C) for standard spatial attention —
      practical at the H/8 bottleneck resolution.
    - Each head i uses a different dilation rate d_i for its Q/K depth-wise conv,
      giving an implicit multi-scale spatial receptive field without extra FLOPs.
    - Residual connection is applied internally.
    - No mask input: at bottleneck depth, partial-conv has already folded mask
      validity into the feature statistics.

    Args
    ----
    dim        : number of input channels (C_mdta).
    num_heads  : number of attention heads — must equal len(dilations).
    dilations  : tuple of dilation rates, one per head.
                 For H/8=32 (256-px images) use (1, 2, 4, 6).
                 For H/8=64 (512-px images) use (1, 2, 4, 8).
    bias       : whether conv layers use bias (default False; GroupNorm handles shift).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dilations: tuple = (1, 2, 4, 6),
        bias: bool = False,
    ):
        super().__init__()
        assert num_heads == len(dilations), (
            f"num_heads ({num_heads}) must match len(dilations) ({len(dilations)})"
        )
        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        # Learnable per-head temperature (scalar scale on attention logits)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Shared 1×1 projections for Q, K, V
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Per-head dilated DW-conv applied to Q and K chunks
        # Each head gets a different dilation → multi-scale spatial context
        self.qk_dw = nn.ModuleList([
            nn.Conv2d(
                self.head_dim, self.head_dim,
                kernel_size=3, dilation=d, padding=d,
                groups=self.head_dim, bias=bias,
            )
            for d in dilations
        ])

        # Output 1×1 projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # GroupNorm: channel-wise normaliser, robust to small batch sizes.
        # Ensure divisibility by walking down from 32.
        num_groups = min(32, dim)
        while dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)

    # ------------------------------------------------------------------
    def _apply_dw(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Split feat into per-head channel chunks, apply the corresponding
        dilated DW-conv to each chunk, then concatenate.

        Args:
            feat: (B, C, H, W)
        Returns:
            (B, C, H, W) — same shape, richer multi-scale context per head
        """
        chunks = feat.chunk(self.num_heads, dim=1)          # list of (B, head_dim, H, W)
        return torch.cat(
            [self.qk_dw[i](c) for i, c in enumerate(chunks)],
            dim=1,
        )                                                    # (B, C, H, W)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)  — residual-added output
        """
        B, C, H, W = x.shape
        nh = self.num_heads
        hd = self.head_dim

        # Normalise input
        x_n = self.norm(x)

        # Project to Q, K, V
        Q = self.q_proj(x_n)   # (B, C, H, W)
        K = self.k_proj(x_n)
        V = self.v_proj(x_n)

        # Enrich Q and K with per-head dilated spatial context
        Q = self._apply_dw(Q)  # (B, C, H, W)
        K = self._apply_dw(K)

        # Reshape to (B*nh, hd, HW) for batched transposed attention
        Q = Q.reshape(B * nh, hd, H * W)
        K = K.reshape(B * nh, hd, H * W)
        V = V.reshape(B * nh, hd, H * W)

        # L2-normalise Q and K (stabilises transposed attention training)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # Transposed attention: attn is (B*nh, hd, hd) — C×C not HW×HW
        # temperature broadcast: (nh,1,1) → (B*nh,1,1)
        temp = self.temperature.repeat(B, 1, 1).reshape(B * nh, 1, 1)
        attn = torch.bmm(Q, K.transpose(-2, -1)) * temp
        attn = F.softmax(attn, dim=-1)

        # Aggregate: (B*nh, hd, HW)
        out = torch.bmm(attn, V)
        out = out.reshape(B, C, H, W)

        # Output projection + residual
        return x + self.out_proj(out)


# =============================================================================
# ParallelFFCMDTABlock
# Runs FFC and MDTA on complementary channel splits in parallel, then fuses.
# =============================================================================

class ParallelFFCMDTABlock(nn.Module):
    """
    Bottleneck block that processes two channel splits in parallel:

        x (C) ─┬─► x_ffc  (C_ffc  = C * ffc_ratio) ─► FFCResnetBlock ─►─┐
               │                                                            ├─► concat ─► fuse ─► + x
               └─► x_mdta (C_mdta = C - C_ffc)      ─► MDTA            ─►─┘

    The fusion is a 1×1 Conv → GroupNorm → GELU followed by a global residual,
    so the block is a strict residual module (identity at initialization).

    Channel split rationale
    -----------------------
    ffc_ratio=0.75: FFC gets 75% of channels, mirroring its own internal
    local/global split (ratio_g=0.75).  MDTA gets the remaining 25%, which
    is enough for 4 heads at the bottleneck resolution.

    Args
    ----
    dim           : total channel count (e.g. dim*8 = 256 with dim=32).
    ffc_ratio     : fraction of channels routed to the FFC branch.
    ratio_g       : FFC internal global fraction (passed to FFCResnetBlock).
    mdta_heads    : number of MDTA attention heads.
    mdta_dilations: per-head dilation rates for MDTA.
    enable_lfu    : whether to enable Local Fourier Unit inside FFC.
    """

    def __init__(
        self,
        dim: int,
        ffc_ratio: float = 0.75,
        ratio_g: float = 0.75,
        mdta_heads: int = 4,
        mdta_dilations: tuple = (1, 2, 4, 6),
        enable_lfu: bool = True,
    ):
        super().__init__()

        self.C_ffc  = int(dim * ffc_ratio)
        self.C_mdta = dim - self.C_ffc

        assert self.C_mdta > 0, "ffc_ratio must be < 1 to leave channels for MDTA"
        assert self.C_mdta % mdta_heads == 0, (
            f"C_mdta ({self.C_mdta}) must be divisible by mdta_heads ({mdta_heads})"
        )

        # # ── FFC branch (inline=True: takes / returns concatenated tensor) ──
        # self.ffc_branch = FFCResnetBlock(
        #     dim=self.C_ffc,
        #     ratio_g=ratio_g,
        #     enable_lfu=enable_lfu,
        #     inline=True,
        # )

        # ── MDTA branch ─────────────────────────────────────────────────────
        self.mdta_branch = MultiDilationTransposedAttention(
            dim=dim,
            num_heads=mdta_heads,
            dilations=mdta_dilations,
        )

        # # ── Fusion: concat(C_ffc + C_mdta = dim) → 1×1 → GroupNorm → GELU ─
        # num_groups = min(32, dim)
        # while dim % num_groups != 0 and num_groups > 1:
        #     num_groups -= 1
        # self.fuse = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        #     nn.GroupNorm(num_groups=num_groups, num_channels=dim),
        #     nn.GELU(),
        # )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — concatenated feature tensor
        Returns:
            (B, C, H, W) — same shape, residual connection applied
        """
        identity = x

        # # Split channels for each branch
        # x_ffc  = x[:, :self.C_ffc]    # (B, C_ffc,  H, W)
        # x_mdta = x[:, self.C_ffc:]    # (B, C_mdta, H, W)

        # Process in parallel (no sequential dependency between branches)
        # out_ffc  = self.ffc_branch(x_ffc)    # (B, C_ffc,  H, W)
        out = self.mdta_branch(x)  # (B, C_mdta, H, W)

        # Fuse branches and apply global residual
        # out = torch.cat([out_ffc, out_mdta], dim=1)  # (B, C, H, W)
        # out = self.fuse(out)
        return out + identity


# =============================================================================
# DepthwiseSeparableConv — lightweight drop-in for decoder projection layers
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-Separable Convolution with Instance Normalization.
    
    Optimized for STRUCTURE synthesis (not style transfer):
    - IN after depthwise: stabilizes per-channel spatial features
    - NO normalization after pointwise: preserves cross-channel correlations
      built by the 1x1 projection — critical for coherent structures.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'lrelu',
        use_in: bool = True,  # legacy toggle (kept for compatibility)
        norm: str = None,     # 'in' | 'gn' | 'none'  (overrides use_in when given)
    ):
        super().__init__()
        if norm is None:
            norm = 'in' if use_in else 'none'

        # -- Depthwise 3×3: spatial filtering per channel --
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels,
            bias=False
        )
        
        # IN after depthwise: normalizes spatial structure, keeps channels independent
        # InstanceNorm statistics here are computed over feature
        # maps whose hole region is zero-filled (skip features), so they depend
        # on hole size (mean-shift problem). Default is now no normalisation;
        # 'gn' is offered for ablation but shares the same spatial-stats issue.
        if norm == 'in':
            self.norm_dw = nn.InstanceNorm2d(in_channels, affine=True)
        elif norm == 'gn':
            g = min(32, in_channels)
            while in_channels % g != 0 and g > 1:
                g -= 1
            self.norm_dw = nn.GroupNorm(g, in_channels)
        elif norm == 'none':
            self.norm_dw = nn.Identity()
        else:
            raise ValueError(f'Unknown decoder norm: {norm}')

        # -- Pointwise 1×1: cross-channel mixing --
        # CRITICAL: No normalization here. The correlations built by this conv
        # must survive to the next layer for structure synthesis.
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, padding=0,
            bias=True  # bias needed since no norm follows
        )

        # -- Activation --
        self.act = nn.LeakyReLU(0.2, inplace=True) if activation == 'lrelu' else nn.Identity()

    def forward(self, x):
        # Step 1: Depthwise spatial filtering + IN
        x = self.depthwise(x)
        x = self.norm_dw(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        # Step 2: Pointwise channel projection (NO NORM)
        # The cross-channel correlations created here are preserved
        x = self.pointwise(x)
        
        # Final activation only
        x = self.act(x)
        return x


class ResDepthwiseBlock(nn.Module):
    """
    Two-stage Depthwise-Separable Conv block with a residual connection.
    Refines concatenated skip-features prior to MDTA attention while preserving gradients.
    """
    def __init__(self, in_channels, out_channels, activation='lrelu', norm='none'):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, activation=activation, norm=norm)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, activation=activation, norm=norm)
        
        # # 1x1 projection shortcut if channel dimensions change
        # self.shortcut = nn.Identity()
        # if in_channels != out_channels:
        #     self.shortcut = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation='linear')

    def forward(self, x):
        # residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out #+ residual

class EdgeRefinementHead(nn.Module):
    """
    Enhanced edge projection head with multi-scale dilation and residual fusion.
    Captures both fine stroke details (dilation=1) and broader boundary context (dilation=2).
    """
    def __init__(self, in_channels, hidden_dim, activation='lrelu'):
        super().__init__()
        half_dim = hidden_dim // 2
        
        # Branch 1: Standard 3x3 Conv for fine, localized stroke details
        self.branch_fine = Conv2dLayer(in_channels, half_dim, kernel_size=3, activation=activation)
        
        # Branch 2: Dilated 3x3 Conv (d=2) for broader structural continuity
        self.branch_context = nn.Sequential(
            nn.Conv2d(in_channels, half_dim, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True) if activation == 'lrelu' else (nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity())
        )
        
        # Feature fusion block
        self.fuse = Conv2dLayer(hidden_dim, hidden_dim, kernel_size=3, activation=activation)
        
        # Output projection layer
        self.to_logits = Conv2dLayer(hidden_dim, 1, kernel_size=1, activation='linear')

    def forward(self, x):
        fine = self.branch_fine(x)
        context = self.branch_context(x)
        feat = torch.cat([fine, context], dim=1)
        feat = self.fuse(feat)
        return self.to_logits(feat)

class ConvResnetBlock(nn.Module):
    """FFCResnetBlock with the Fourier/global branch removed (ratio_g = 0):
    two (3x3 reflect conv -> BN -> ReLU) layers + residual. Used as the
    'no global-context module' bottleneck baseline in the architecture ablation."""
    def __init__(self, dim):
        super().__init__()
        def layer():
            return nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect', bias=False),
                                 nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.conv1 = layer()
        self.conv2 = layer()

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class SwinBottleneck(nn.Module):
    """Mask-aware Swin stage at the bottleneck resolution (64x64 for 512 input).

    1x1 down-projection (C -> swin_dim), MAT's mask-aware BasicLayer (shifted
    windows; hole tokens are excluded from attention keys until the partial
    mask has grown over them), 1x1 up-projection with a ZERO-initialised
    residual, so the block starts as an identity and can be added to a
    trained FFC bottleneck without disturbing it.

    Gives the content-dependent spatial (token-to-token) attention that FFC
    (global linear filtering) and MDTA (channel attention) do not provide.
    """
    def __init__(self, dim, resolution, swin_dim=256, depth=2, num_heads=8, window_size=8):
        super().__init__()
        self.resolution = resolution
        self.proj_in = nn.Conv2d(dim, swin_dim, kernel_size=1)
        self.layer = BasicLayer(dim=swin_dim, input_resolution=(resolution, resolution),
                                depth=depth, num_heads=num_heads, window_size=window_size)
        self.proj_out = nn.Conv2d(swin_dim, dim, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, hole):
        """x: (B,C,h,w) features; hole: (B,1,H,W) full-res hole mask (1 = hole)."""
        B, C, h, w = x.shape
        hole_s = F.adaptive_avg_pool2d(hole, (h, w))
        valid = (hole_s < 0.99).to(x.dtype)                 # token has some known pixels
        t = feature2token(self.proj_in(x))                  # (B, h*w, swin_dim)
        m = feature2token(valid)                            # (B, h*w, 1)
        t, _, _ = self.layer(t, (h, w), m)
        return x + self.proj_out(token2feature(t, (h, w)))


class GeneratorSketch(nn.Module):
    """
    Generator for mural structure (edge) inpainting.

    Architecture (H = img_resolution, e.g. 512):
      Encoder   : 4 × Conv2dLayerPartial    H → H/2 → H/4 → H/8   (dim … dim*8 ch)
      Mask embed: hole-coverage map added at H/8 and concatenated at every
                  decoder stage (mask_to_decoder=True)
      Bottleneck: n_btn blocks @ H/8, type set by `bottleneck`:
                    'ffc'  FFCResnetBlock (LaMa Fourier conv, ratio_g=0.75, LFU)  [default]
                    'conv' ConvResnetBlock (same block, no global branch)        [ablation]
                    'swin' SwinBottleneck (MAT mask-aware window attention)      [ablation]
                  optional extra Swin stages after the blocks listed in `swin_after`
      Decoder   : 3 × (bilinear up + skip concat + ResDepthwiseBlock
                  + MultiDilationTransposedAttention)   H/8 → H
                  (dec_attn=False replaces the attention with Identity)
      Head      : Conv 3×3 → Conv 1×1 → edge logits

    FFC Bottleneck:
      Each FFCResnetBlock performs two passes of LaMa's 4-way cross-branch mixing:
        local_in  → conv_l2l → local_out          (standard Conv2d)
        local_in  → conv_l2g → global_out         (standard Conv2d)
        global_in → conv_g2l → local_out          (standard Conv2d)
        global_in → conv_g2g → global_out         (SpectralTransform w/ LFU)
      ratio_g=0.75: 75% of channels use the frequency-domain global branch.

    Inputs
    ------
      img  : (B, C, H, W)  image   in [-1, 1]
      edge : (B, 1, H, W)  edge    in [ 0, 1]
      mask : (B, 1, H, W)  1=hole  in [ 0, 1]

    Output
    ------
      edge_logits : (B, 1, H, W)  raw logits (sigmoid → probs)
    """

    def __init__(self, img_resolution=512, dim=64, use_noise=False,
                 demodulate=True, activation='lrelu', in_channels=3,
                 down_time_override=1, dec_norm='none', mask_to_decoder=True,
                 swin_after=(), swin_dim=256, swin_depth=2, swin_heads=8, swin_window=8,
                 bottleneck='ffc', n_btn=6, dec_attn=True):
        # swin_after: indices of FFC blocks AFTER which a mask-aware Swin stage
        # (at H/8, i.e. 64x64 for 512 input) is inserted, e.g. (1, 3) gives
        # FFC FFC SWIN FFC FFC SWIN FFC FFC. Empty tuple = old architecture
        # (state_dict-compatible with old checkpoints).
        super().__init__()

        self.in_channels = in_channels
        self.mask_to_decoder = mask_to_decoder
        _mc = 1 if mask_to_decoder else 0   # extra mask channel per decoder stage

        # ── Encoder: 4 stages, pure PartialConv, NO FFC ──────────────────
        #   enc0 : in_channels → dim    @ H     (no down)    → skip_0
        #   enc1 : dim   → dim*2        @ H→H/2  (down=2)    → skip_1
        #   enc2 : dim*2 → dim*4        @ H/2→H/4 (down=2)   → skip_2
        #   enc3 : dim*4 → dim*8        @ H/4→H/8 (down=2)   → bottleneck
        self.enc0 = Conv2dLayerPartial(in_channels, dim,
                                       kernel_size=3, activation=activation)
        self.enc1 = Conv2dLayerPartial(dim, dim*2,
                                       kernel_size=3, down=2, activation=activation)
        self.enc2 = Conv2dLayerPartial(dim*2, dim*4,
                                       kernel_size=3, down=2, activation=activation)
        self.enc3 = Conv2dLayerPartial(dim*4, dim*8,
                                       kernel_size=3, down=2, activation=activation)

        # ── Bottleneck @ H/8, dim*8 channels ─────────────────────────────
        _btn_dim = dim * 8

        def _ffc_block():
            return FFCResnetBlock(
                dim=_btn_dim,
                ratio_g=0.75,
                enable_lfu=True,
                inline=True,
            )

        # Bottleneck type (architecture ablation):
        #   'ffc'  : n_btn x FFCResnetBlock (LaMa Fourier convs)      [default, old runs]
        #   'conv' : n_btn x ConvResnetBlock = the same block with NO Fourier/global
        #            branch (plain 3x3 convs) -> "no global context module" baseline
        #   'swin' : n_btn x SwinBottleneck (MAT-style mask-aware window attention,
        #            depth swin_depth, width swin_dim) -> Swin INSTEAD of FFC
        _btn_res = img_resolution // 8
        assert bottleneck in ('ffc', 'conv', 'swin'), bottleneck
        self.bottleneck_type = bottleneck
        if bottleneck == 'ffc':
            blocks = [_ffc_block() for _ in range(n_btn)]
        elif bottleneck == 'conv':
            blocks = [ConvResnetBlock(_btn_dim) for _ in range(n_btn)]
        else:
            blocks = [SwinBottleneck(_btn_dim, _btn_res, swin_dim=swin_dim, depth=swin_depth,
                                     num_heads=swin_heads, window_size=swin_window)
                      for _ in range(n_btn)]
            swin_after = ()          # already all-Swin
        self.bottleneck = nn.ModuleList(blocks)

        # ── Optional mask-aware Swin stages at H/8 (keys 'swin.<idx>') ──────
        self.swin = nn.ModuleDict({
            str(int(i)): SwinBottleneck(_btn_dim, _btn_res, swin_dim=swin_dim, depth=swin_depth,
                                        num_heads=swin_heads, window_size=swin_window)
            for i in swin_after
        })

        # ── Decoder: 3 stages, pure Conv, NO FFC ─────────────────────────
        #   dec_up[i]  : bilinear ×2 upsample (no learnable params)
        #   dec_proj[i]: channel projection after skip-cat (see below)
        #
        #   Stage 0: up H/8→H/4,  cat skip2 (dim*4) → in=dim*12, out=dim*4
        #   Stage 1: up H/4→H/2,  cat skip1 (dim*2) → in=dim*6,  out=dim*2
        #   Stage 2: up H/2→H,    cat skip0 (dim)   → in=dim*3,  out=dim
        #   (+1 input channel per stage for the hole map when mask_to_decoder)
        #
        # ResDepthwiseBlock (depthwise-separable) is ~8-9x cheaper than a
        # plain 3x3 conv at the same width.
        self.dec_up = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        ])
        
        # dec_attn=False replaces each MDTA with Identity (architecture ablation);
        # the ResDepthwiseBlock keys (dec_proj.N.0.*) are unchanged.
        def _attn(c, dil):
            return (MultiDilationTransposedAttention(dim=c, num_heads=4, dilations=dil)
                    if dec_attn else nn.Identity())

        self.dec_proj = nn.ModuleList([
            # Stage 0: H/8 -> H/4 (Fuses skip2: dim*12 -> dim*4)
            nn.Sequential(
                ResDepthwiseBlock(dim * 12 + _mc, dim * 4, activation=activation, norm=dec_norm),
                _attn(dim * 4, (1, 2, 3, 4)),
            ),
            # Stage 1: H/4 -> H/2 (Fuses skip1: dim*6 -> dim*2)
            nn.Sequential(
                ResDepthwiseBlock(dim * 6 + _mc, dim * 2, activation=activation, norm=dec_norm),
                _attn(dim * 2, (1, 1, 2, 2)),
            ),
            # Stage 2: H/2 -> H (Fuses skip0: dim*3 -> dim)
            nn.Sequential(
                ResDepthwiseBlock(dim * 3 + _mc, dim, activation=activation, norm=dec_norm),
                _attn(dim, (1, 1, 1, 1)),
            ),
        ])
        
        # ── Output head ──────────────────────────────────────────────────
        self.to_edge = nn.Sequential(
            Conv2dLayer(dim, dim // 2, kernel_size=3, activation=activation),
            Conv2dLayer(dim // 2, 1,   kernel_size=1, activation='linear'),
        )

        # The partial-conv mask is lost after enc3, so wide holes would reach
        # the bottleneck as unlabeled zeros. Inject the hole-coverage map (area-pooled) into the bottleneck (zero-init 1x1,
        # starts as a no-op) and concatenate it at every decoder stage.
        if mask_to_decoder:
            self.mask_embed = nn.Conv2d(1, _btn_dim, kernel_size=1)
            nn.init.zeros_(self.mask_embed.weight)
            nn.init.zeros_(self.mask_embed.bias)

        self.use_noise = use_noise
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, Conv2dLayer)):
            pass

    # ------------------------------------------------------------------
    def forward(self, img, edge, mask, noise_mode='random'):
        """
        Args
        ----
          img  : (B, C, H, W)  image in [-1, 1]
          edge : (B, 1, H, W)  edge map in [0, 1]
          mask : (B, 1, H, W)  1 = hole, 0 = valid

        Returns
        -------
          edge_logits : (B, 1, H, W)
        """
        img  = img.to(dtype=torch.float32)
        edge = edge.to(dtype=torch.float32)
        mask = mask.to(dtype=torch.float32)

        mask_valid = 1.0 - mask          # 1 = known pixel, 0 = hole

        # Collapse to grayscale when the network expects 1 image channel
        if self.in_channels == 3 and img.shape[1] == 3:
            img = (img[:, 0:1] * 0.299
                   + img[:, 1:2] * 0.587
                   + img[:, 2:3] * 0.114)

        # 3-channel input: [mask_bias | visible_edge | visible_image]
        x = torch.cat([
            mask - 0.5,           # (B,1,H,W)
            edge * mask_valid,    # (B,1,H,W)
            img  * mask_valid,    # (B,1,H,W)
        ], dim=1)                 # → (B, 3, H, W)

        # ── Encoder ───────────────────────────────────────────────────────
        x, m  = self.enc0(x, mask_valid)   # (B, dim,   H,   W  )
        skip0 = x                           # skip_0 @ full res

        x, m  = self.enc1(x, m)            # (B, dim*2, H/2, W/2)
        skip1 = x                           # skip_1 @ H/2

        x, m  = self.enc2(x, m)            # (B, dim*4, H/4, W/4)
        skip2 = x                           # skip_2 @ H/4

        x, _  = self.enc3(x, m)            # (B, dim*8, H/8, W/8)

        def _hole_at(t):
            return F.adaptive_avg_pool2d(mask, t.shape[-2:])

        if self.mask_to_decoder:
            x = x + self.mask_embed(_hole_at(x))

        # ── Bottleneck @ H/8 ──────────────────────────────────────────────
        for i, blk in enumerate(self.bottleneck):
            if isinstance(blk, SwinBottleneck):
                x = blk(x, mask)            # mask-aware attention needs the hole map
            else:
                x = blk(x)                  # (B, dim*8, H/8, W/8)
            if str(i) in self.swin:
                x = self.swin[str(i)](x, mask)

        # ── Decoder ───────────────────────────────────────────────────────
        # Stage 0: H/8 → H/4, fuse skip2  (dim*8 + dim*4 = dim*12 → dim*4)
        x = self.dec_up[0](x)              # (B, dim*8,  H/4, W/4)
        x = torch.cat([x, skip2] + ([_hole_at(skip2)] if self.mask_to_decoder else []), dim=1)   # (B, dim*12, H/4, W/4)
        x = self.dec_proj[0](x)            # (B, dim*4,  H/4, W/4)

        # Stage 1: H/4 → H/2, fuse skip1  (dim*4 + dim*2 = dim*6 → dim*2)
        x = self.dec_up[1](x)              # (B, dim*4,  H/2, W/2)
        x = torch.cat([x, skip1] + ([_hole_at(skip1)] if self.mask_to_decoder else []), dim=1)   # (B, dim*6,  H/2, W/2)
        x = self.dec_proj[1](x)            # (B, dim*2,  H/2, W/2)

        # Stage 2: H/2 → H, fuse skip0  (dim*2 + dim = dim*3 → dim)
        x = self.dec_up[2](x)              # (B, dim*2,  H,   W  )
        x = torch.cat([x, skip0] + ([_hole_at(skip0)] if self.mask_to_decoder else []), dim=1)   # (B, dim*3,  H,   W  )
        x = self.dec_proj[2](x)            # (B, dim,    H,   W  )

        # ── Output ────────────────────────────────────────────────────────
        edge_logits = self.to_edge(x)      # (B, 1, H, W)
        return edge_logits



class SpectralNormConv2dLayer(nn.Module):
    """Conv2dLayer wrapper with spectral normalization (pickle-compatible)."""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, **kwargs)
        
        # Initialize u vector for power iteration
        weight = self.conv.weight
        with torch.no_grad():
            # Flatten weight to (out_channels, in_channels * k * k)
            w_shape = weight.shape
            w_flat = weight.view(w_shape[0], -1)
            u = torch.randn(w_shape[0])
            u = u / u.norm()
            self.register_buffer('_u', u)
            self.register_buffer('_v', torch.zeros(w_flat.shape[1]))
    
    def _compute_spectral_norm(self):
        """Compute spectral normalization factor."""
        weight = self.conv.weight
        w_shape = weight.shape
        w_flat = weight.view(w_shape[0], -1)
        
        # Power iteration
        with torch.no_grad():
            v = torch.mv(w_flat.t(), self._u)
            v = v / (v.norm() + 1e-12)
            u = torch.mv(w_flat, v)
            u = u / (u.norm() + 1e-12)
            self._u.copy_(u)
            self._v.copy_(v)
        
        # Compute sigma
        sigma = torch.dot(self._u, torch.mv(w_flat, self._v))
        return sigma
    
    def forward(self, x, gain=1):
        # Apply spectral norm scaling
        sigma = self._compute_spectral_norm()
        
        # Scale weight during forward
        original_weight = self.conv.weight
        
        # We want the final effective weight (after Conv2dLayer's internal scaling) to have SN=1.
        # Conv2dLayer applies: w = weight * weight_gain
        # So we set weight = (original / sigma) / weight_gain
        # Then w = (original / sigma) / weight_gain * weight_gain = original / sigma
        w_normalized = original_weight / (sigma + 1e-12) / self.conv.weight_gain
        
        # Temporarily replace weight with normalized tensor (preserving gradient)
        # We manipulate _parameters directly to preserve OrderedDict key order
        self.conv._parameters['weight'] = w_normalized 
        
        try:
            out = self.conv(x, gain=gain)
        finally:
            # Restore original parameter
            self.conv._parameters['weight'] = original_weight
            
        return out


class SpectralNormFC(nn.Module):
    """FullyConnectedLayer wrapper with spectral normalization (pickle-compatible)."""
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.fc = FullyConnectedLayer(in_features, out_features, **kwargs)
        
        # Initialize u vector for power iteration
        weight = self.fc.weight
        with torch.no_grad():
            u = torch.randn(weight.shape[0])
            u = u / u.norm()
            self.register_buffer('_u', u)
            self.register_buffer('_v', torch.zeros(weight.shape[1]))
    
    def _compute_spectral_norm(self):
        """Compute spectral normalization factor."""
        weight = self.fc.weight
        
        # Power iteration
        with torch.no_grad():
            v = torch.mv(weight.t(), self._u)
            v = v / (v.norm() + 1e-12)
            u = torch.mv(weight, v)
            u = u / (u.norm() + 1e-12)
            self._u.copy_(u)
            self._v.copy_(v)
        
        # Compute sigma
        sigma = torch.dot(self._u, torch.mv(weight, self._v))
        return sigma
    
    def forward(self, x):
        # Apply spectral norm scaling during forward
        sigma = self._compute_spectral_norm()
        
        # Scale weight during forward
        original_weight = self.fc.weight
        
        # We want the final effective weight to have SN=1.
        # FullyConnectedLayer applies: w = weight * weight_gain
        # So we set weight = (original / sigma) / weight_gain
        # Then w = (original / sigma) / weight_gain * weight_gain = original / sigma
        w_normalized = original_weight / (sigma + 1e-12) / self.fc.weight_gain
        
        # Temporarily replace weight
        self.fc._parameters['weight'] = w_normalized
        
        try:
            out = self.fc(x)
        finally:
            # Restore original parameter
            self.fc._parameters['weight'] = original_weight
            
        return out


class SNDisFromRGB(nn.Module):
    """DisFromRGB with spectral normalization (pickle-compatible)."""
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.conv = SpectralNormConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )

    def forward(self, x):
        return self.conv(x)


class SNDisBlock(nn.Module):
    """DisBlock with spectral normalization (pickle-compatible)."""
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.conv0 = SpectralNormConv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = SpectralNormConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            down=2,
            activation=activation,
        )
        self.skip = SpectralNormConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            down=2,
            bias=False,
        )

    def forward(self, x):
        skip = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        out = skip + x
        return out


class DiscriminatorSketch(nn.Module):
    """
    Discriminator for sketch inpainting with optional spectral normalization.
    
    Input: edge (B,1,H,W) [0,1], mask (B,1,H,W) [0,1]
    Output: logits (B,1)
    
    Args:
        use_spec_norm: If True, uses spectral normalization on all layers (pickle-compatible).
    """
    def __init__(self, img_resolution=256, img_channels=1, c_dim=0,
                 channel_base=32768, channel_max=512, activation='lrelu',
                 mbstd_group_size=4, mbstd_num_channels=1, use_spec_norm=True):
        super().__init__()
        resolution_log2 = int(math.log2(img_resolution))
        self.resolution_log2 = resolution_log2
        self.use_spec_norm = use_spec_norm
        
        if use_spec_norm:
            # Use spectral normalized layers
            Dis = [SNDisFromRGB(img_channels + 1, nf(resolution_log2), activation)]
            for res in range(resolution_log2, 2, -1):
                Dis.append(SNDisBlock(nf(res), nf(res - 1), activation))

            if mbstd_num_channels > 0:
                Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
            Dis.append(SpectralNormConv2dLayer(nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation))
            self.Dis = nn.Sequential(*Dis)
            self.fc0 = SpectralNormFC(nf(2) * 4 ** 2, nf(2), activation=activation)
            self.fc1 = SpectralNormFC(nf(2), 1)
        else:
            # Use regular layers (no SN)
            Dis = [DisFromRGB(img_channels + 1, nf(resolution_log2), activation)]
            for res in range(resolution_log2, 2, -1):
                Dis.append(DisBlock(nf(res), nf(res - 1), activation))

            if mbstd_num_channels > 0:
                Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
            Dis.append(Conv2dLayer(nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation))
            self.Dis = nn.Sequential(*Dis)
            self.fc0 = FullyConnectedLayer(nf(2) * 4 ** 2, nf(2), activation=activation)
            self.fc1 = FullyConnectedLayer(nf(2), 1)

    def forward(self, edge, mask, return_features=False):
        """
        Args:
            edge: Edge map (B,1,H,W) in range [0, 1]
            mask: Binary mask (B,1,H,W), 1=hole, 0=valid
            
        Returns:
            logits: Discriminator output
            features: List of intermediate features (if return_features=True)
        """
        features = []
        x = torch.cat([mask - 0.5, edge], dim=1)
        for layer in self.Dis:
            x = layer(x)
            if isinstance(layer, (DisFromRGB, DisBlock, Conv2dLayer, SNDisFromRGB, SNDisBlock, SpectralNormConv2dLayer)):
                # do NOT detach. Detaching here cut the gradient of the
                # feature-matching loss to G (the fake features were constants).
                # Real features are already computed under torch.no_grad() in the loss.
                features.append(x)
        
        x = x.flatten(start_dim=1)
        x = self.fc0(x)
        x = self.fc1(x)
        
        if return_features:
            return x, features
        return x


class DiscriminatorLama(nn.Module):
    """
    PatchGAN discriminator (LaMa's NLayerDiscriminator), adapted for edge maps.

    Changes vs. the previous port:
      * norm='sn' (default): spectral norm on every conv, NO BatchNorm.
        BatchNorm in D is a poor fit here: real and fake batches are forwarded
        separately (D can separate them by batch statistics), batch_gpu=4 gives
        noisy stats, and BN couples samples inside the R1 gradient.
        norm='bn' reproduces the original LaMa layout; norm='none' = plain convs.
      * Output is a logit MAP (B,1,h,w). SketchGANLoss detects ndim==4 and
        applies the fake label only in proportion to hole coverage of each
        output cell (LaMa 'mask as fake target'), so fully-known patches of a
        fake composite (identical to real) are not labelled fake.
      * Intermediate activations are returned WITHOUT detach for feature matching.
      * Optional cond_channels: extra conditioning input (e.g. masked gray image).
        Not wired in the loss yet; keep 0.

    Receptive field per output logit (k=4): n_layers=3 -> 70 px,
    n_layers=4 -> 142 px (output 34x34 at 512), n_layers=5 -> 286 px
    (output 18x18 at 512; judges face/hand-scale coherence).
    """
    def __init__(self, img_resolution=None, img_channels=1, c_dim=0, ndf=64, n_layers=4,
                 mbstd_group_size=4, use_spec_norm=False, norm='sn', cond_channels=0,
                 mask_input=False):
        super().__init__()
        # img_resolution, c_dim, mbstd_group_size are ignored (API compatibility).
        # Legacy flag: use_spec_norm=True forces norm='sn'.
        if use_spec_norm:
            norm = 'sn'
        assert norm in ('sn', 'bn', 'none'), f'Unknown D norm: {norm}'
        self.n_layers = n_layers
        self.norm = norm
        # FIX (D shortcut): LaMa's NLayerDiscriminator does NOT see the mask.
        # With the mask as an input channel AND hole-weighted fake labels, the
        # loss-optimal D is a function of the mask alone ("hole => fake"), i.e.
        # it outputs ~log(mean hole ratio) on every hole cell, real or fake, and
        # gives G (almost) no content gradient. Default is now mask_input=False.
        # Set True only to reproduce old runs / load old D checkpoints.
        self.mask_input = mask_input

        input_nc = img_channels + (1 if mask_input else 0) + cond_channels   # edge (+ mask) (+ cond)

        def conv(cin, cout, stride):
            c = nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=2)
            return nn.utils.spectral_norm(c) if norm == 'sn' else c

        def norm_layer(c):
            return nn.BatchNorm2d(c) if norm == 'bn' else nn.Identity()

        sequence = [[conv(input_nc, ndf, 2), nn.LeakyReLU(0.2, True)]]

        nf_ = ndf
        for n in range(1, n_layers):
            nf_prev = nf_
            nf_ = min(nf_ * 2, 512)
            sequence.append([conv(nf_prev, nf_, 2), norm_layer(nf_), nn.LeakyReLU(0.2, True)])

        nf_prev = nf_
        nf_ = min(nf_ * 2, 512)
        sequence.append([conv(nf_prev, nf_, 1), norm_layer(nf_), nn.LeakyReLU(0.2, True)])
        sequence.append([conv(nf_, 1, 1)])

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, edge, mask, return_features=False, cond=None):
        parts = ([mask - 0.5] if self.mask_input else []) + [edge]
        if cond is not None:
            parts.append(cond)
        x = torch.cat(parts, dim=1)
        act = self.get_all_activations(x)
        if return_features:
            return act[-1], act[:-1]
        return act[-1]

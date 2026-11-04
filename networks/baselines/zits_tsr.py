# ZITS Transformer Structure Restorer (TSR) + Simple Structure Upsampler (SSU) as a baseline.
#
# The model classes below (gelu, GELU, GELU2, SelfAttention, AxialAttention, BlockAxial,
# CausalSelfAttention, my_Block_2, EdgeLineGPTConfig, EdgeLineGPT256RelBCE,
# StructureUpsampling) are copied from ZITS
# (Dong et al., CVPR 2022, https://github.com/DQiaole/ZITS_inpainting:
#  src/models/transformer.py, src/models/TSR_model.py, src/models/upsample.py),
# Copyright the ZITS authors, licensed under the Apache License 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Changes to the copied code:
#   * EdgeLineGPT256RelBCE.configure_optimizers removed (the optimizer is built by
#     the training loop; ZITS trains TSR with weight_decay=0, so the param groups
#     made no difference).
# Everything else in this file (wrapper, edge-only sampling) is
# Copyright (c) 2026 Hassan Saad, CC BY-NC 4.0.
"""ZITS TSR, edge-only variant, with this repo's generator interface.

    G(img, edge, mask) -> edge logits at img_resolution

Pipeline (as ZITS, see docs/BASELINES.md for the deviations):
  0. The edge map is binarised at 0.5 (ZITS uses binary Canny edges; the dataset's edge
     maps are soft JPEG maps). Inputs and both training targets use this binary map.
  1. Inputs are resized to 256 (TSR is fixed to a 32x32 token grid):
       image : area resize, gray copied to 3 channels, in [-1, 1] (ZITS normalises to [-1, 1])
       edge  : max-pool (keeps 1-px strokes);  mask : max-pool (a 256-px cell is a hole if any
               of its pixels is)
       line  : all zeros (edge-only variant: no wireframe detector)
  2. Training: one TSR pass -> edge logits at 256 (BCE inside the hole).
     Evaluation: ZITS's iterative sampling (SampleEdgeLineLogits, 5 steps, edge threshold
     0.25), ranking by edge confidence only.
  3. SSU upsamples the 256 edge map to img_resolution. ZITS did not release SSU training
     code; here it is trained alongside TSR to upsample the (binary) ground-truth edge map
     at 256 to the ground truth at img_resolution, i.e. on the same kind of input it
     receives at test time. It never back-propagates into TSR.
"""

import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from networks.baselines import binarize_edge

logger = logging.getLogger(__name__)


# =============================================================================
# ZITS src/models/transformer.py (unchanged)
# =============================================================================

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None, rel_pos=None, return_att=False):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if rel_pos is not None:
            att += rel_pos
        if mask is not None:  # maybe we don't need mask in axial-transformer
            # mask:[B,1,L(1),L]
            att = att.masked_fill(mask == 1, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if return_att:
            return y, att
        else:
            return y


class AxialAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()

        self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.add_rel_pos = add_rel_pos
        # self.rel_pos_bins = rel_pos_bins
        self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        if row:
            rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        rel_pos = rel_pos.contiguous()
        return rel_pos

    def forward(self, x, return_att=False, debug=False):  # x:[B,C,H,W], mask:[B,1,H,W]
        [b, c, h, w] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)
        mask_row = None
        mask_col = None

        # ROW ATTENTION
        x = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        if self.add_rel_pos:
            row_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=h, row=True)
        else:
            row_rel_pos = None
        x_row = self.attn_row(self.rln1(x), mask_row, row_rel_pos, return_att=return_att)
        if return_att:
            x_row, att_row = x_row
        else:
            att_row = None
        x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

        # COL ATTENTION
        x = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
        if self.add_rel_pos:
            col_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=w, row=False)
        else:
            col_rel_pos = None
        x_col = self.attn_col(self.cln1(x), mask_col, col_rel_pos, return_att=return_att)
        if return_att:
            x_col, att_col = x_col
        else:
            att_col = None
        x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)

        # [B,HW,C]
        x = x0 + x_row + x_col
        x = x + self.ff(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x.contiguous()

        if return_att:
            # att_row:[BW,head,H,H]
            att_row = torch.mean(att_row, dim=1).reshape(b, w, h, h)
            att_row = torch.sum(att_row, dim=2).permute(0, 2, 1)  # [b,h,w]
            # att_col:[BH,head,W,W]
            att_col = torch.mean(att_col, dim=1).reshape(b, h, w, w)
            att_col = torch.sum(att_col, dim=2)
            att_score = att_row * att_col
            return x, att_score
        else:
            return x


class BlockAxial(AxialAttention):

    def __init__(self, config):
        super().__init__(config.n_embd, config.n_head, config.attn_pdrop, config.resid_pdrop, 32, 32)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

        self.config = config

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class my_Block_2(nn.Module):
    """ Transformer block with original GELU2 """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU2(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        [b, c, h, w] = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x.contiguous()
        return x


# =============================================================================
# ZITS src/models/TSR_model.py (unchanged except configure_optimizers removed)
# =============================================================================

class EdgeLineGPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class EdgeLineGPT256RelBCE(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer, input: 32*32*config.n_embd
        self.blocks = []
        for _ in range(config.n_layer // 2):
            self.blocks.append(BlockAxial(config))
            self.blocks.append(my_Block_2(config))
        self.blocks = nn.Sequential(*self.blocks)
        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(256)

        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0)

        self.act_last = nn.Sigmoid()

        self.block_size = 32
        self.config = config

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, img_idx, edge_idx, line_idx, edge_targets=None, line_targets=None, masks=None):
        img_idx = img_idx * (1 - masks)
        edge_idx = edge_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=1)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.act(x)

        [b, c, h, w] = x.shape
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()

        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)  # [b,hw,c]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln_f(x).permute(0, 3, 1, 2).contiguous()

        x = self.convt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)

        edge, line = torch.split(x, [1, 1], dim=1)

        if edge_targets is not None and line_targets is not None:
            loss = F.binary_cross_entropy_with_logits(edge.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      edge_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      reduction='none')
            loss = loss + F.binary_cross_entropy_with_logits(line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             reduction='none')
            masks_ = masks.view(-1, 1)

            loss *= masks_
            loss = torch.mean(loss)
        else:
            loss = 0
        edge, line = self.act_last(edge), self.act_last(line)
        return edge, line, loss

    def forward_with_logits(self, img_idx, edge_idx, line_idx, masks=None):
        img_idx = img_idx * (1 - masks)
        edge_idx = edge_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=1)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.act(x)

        [b, c, h, w] = x.shape
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()

        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)  # [b,hw,c]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln_f(x).permute(0, 3, 1, 2).contiguous()

        x = self.convt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)

        edge, line = torch.split(x, [1, 1], dim=1)

        return edge, line


# =============================================================================
# ZITS src/models/upsample.py (unchanged)
# =============================================================================

class StructureUpsampling(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.out = nn.Sequential(nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, line):
        x = line
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.convs(x)
        x2 = self.out(x)

        return x, x2


# =============================================================================
# Wrapper for this framework
# =============================================================================

class ZITSTSREdge(nn.Module):
    """ZITS TSR (edge only) + SSU with the interface G(img, edge, mask) -> edge logits.

    Args:
        img_resolution    : output resolution (multiple of 256; 512 for this repo)
        n_layer, n_embd, n_head : TSR size (ZITS defaults 16 / 256 / 8)
        sample_iterations : iterative sampling steps at evaluation (ZITS: 5)
        edge_threshold    : binarisation threshold inside sampling (ZITS: 0.25)
        binarize_edges    : the input/target edge map is binarised at this threshold before
                            anything else (ZITS uses binary Canny edges); None keeps it soft
    """

    TSR_RES = 256   # fixed by the 32x32 token grid (pos_emb has 1024 entries, BlockAxial is 32x32)

    def __init__(self, img_resolution=512, n_layer=16, n_embd=256, n_head=8,
                 sample_iterations=5, edge_threshold=0.25, binarize_edges=0.5):
        super().__init__()
        assert img_resolution >= self.TSR_RES, 'ZITS TSR needs img_resolution >= 256'
        self.img_resolution = img_resolution
        self.sample_iterations = sample_iterations
        self.edge_threshold = edge_threshold
        self.binarize_edges = binarize_edges
        # TSR_train.py: dropout 0, block_size 32
        config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                                   n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=32)
        self.transformer = EdgeLineGPT256RelBCE(config)
        self.structure_upsample = StructureUpsampling()

    # ------------------------------------------------------------------
    def _to_tsr(self, img, edge, mask):
        """Expects `edge` already binarised (see forward)."""
        r = self.TSR_RES
        img = img.to(torch.float32)
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        if img.shape[-1] != r:
            img = F.interpolate(img, size=(r, r), mode='area')
            edge = F.adaptive_max_pool2d(edge.to(torch.float32), r)
            mask = F.adaptive_max_pool2d(mask.to(torch.float32), r)
        return img, edge.to(torch.float32), torch.zeros_like(edge, dtype=torch.float32), mask.to(torch.float32)

    def upsample(self, edge256, return_unused=False):
        """SSU: 256 edge map in [0, 1] -> logits at img_resolution.

        StructureUpsampling also returns a second output (its `out` branch, back at the
        input resolution) that ZITS never uses at inference; with return_unused=True it is
        returned so the loss can keep it in the graph (DDP needs a gradient for every parameter).
        """
        x, logits, unused = edge256, None, None
        while x.shape[-1] < self.img_resolution:
            logits, unused = self.structure_upsample(x)
            x = torch.sigmoid(logits)
        if logits is None:                                   # img_resolution == 256
            logits = torch.logit(edge256.clamp(1e-4, 1 - 1e-4))
        elif logits.shape[-1] != self.img_resolution:
            logits = F.interpolate(logits, size=(self.img_resolution,) * 2, mode='bilinear', align_corners=False)
        return (logits, unused) if return_unused else logits

    @torch.no_grad()
    def sample(self, img, edge, line, mask):
        """ZITS SampleEdgeLineLogits, edge only: fill the hole in `sample_iterations` steps,
        each step fixing the most confident remaining hole pixels. Returns a binary edge map."""
        img = img * (1 - mask)
        edge = edge * (1 - mask)
        line = line * (1 - mask)
        mask = mask.clone()
        iterations = self.sample_iterations
        for i in range(iterations):
            edge_logits, _ = self.transformer.forward_with_logits(img, edge, line, masks=mask)
            edge_pred = torch.sigmoid(edge_logits)
            edge = edge + edge_pred * mask
            edge[edge >= self.edge_threshold] = 1
            edge[edge < self.edge_threshold] = 0

            b, _, h, w = edge_pred.shape
            edge_pred = edge_pred.reshape(b, -1, 1)
            mask = mask.reshape(b, -1)

            edge_probs = torch.cat([1 - edge_pred, edge_pred], dim=-1)
            edge_probs[:, :, 1] += 0.5
            edge_max_probs = edge_probs.max(dim=-1)[0] + (1 - mask) * (-100)

            indices = torch.sort(edge_max_probs, dim=-1, descending=True)[1]

            for ii in range(b):
                keep = int((i + 1) / iterations * torch.sum(mask[ii, ...]))
                mask[ii][indices[ii, :keep]] = 0

            mask = mask.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
        return edge

    # ------------------------------------------------------------------
    def forward(self, img, edge, mask, return_aux=False):
        edge = binarize_edge(edge, self.binarize_edges)          # binary at full resolution
        img_t, edge_t, line_t, mask_t = self._to_tsr(img, edge, mask)

        if self.training:
            tsr_logits, _ = self.transformer.forward_with_logits(img_t, edge_t, line_t, masks=mask_t)
            if return_aux:
                # SSU learns to upsample the ground-truth 256 edge map (its test-time input
                # is the binary TSR output), so TSR and SSU are trained independently.
                ssu_logits, ssu_unused = self.upsample(edge_t, return_unused=True)
                return None, dict(tsr_logits=tsr_logits, edge256=edge_t, mask256=mask_t,
                                  ssu_logits=ssu_logits, ssu_unused=ssu_unused, edge_full=edge)
            return self.upsample(torch.sigmoid(tsr_logits))

        edge256 = self.sample(img_t, edge_t, line_t, mask_t)
        return self.upsample(edge256)

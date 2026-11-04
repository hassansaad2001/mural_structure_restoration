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

"""
Edge / sketch inpainting evaluation metrics (hole region only).

Revision notes (vs. previous version)
-------------------------------------
* Streaming ``EdgeMetricAccumulator``: metrics are accumulated over any number
  of batches and computed once, so results no longer depend on batch size
  (the old ``compute_metrics_batch`` averaged per-batch pooled values).
* Hausdorff-95 uses an exact Euclidean distance transform (scipy EDT, O(HW))
  instead of an N x M pairwise-distance matrix, which needed ~2 GB RAM per
  image for 50 % holes and ~3.7 GB for 70 % holes.
  Images where exactly one of {gt, pred} is empty in the hole can no longer
  silently vanish: they are counted and exposed as ``hd_valid_frac``.
  An empty result is reported as NaN, not 0.0 (0.0 looked like "perfect").
* Tolerance ("relaxed") precision / recall / F1 at several pixel radii
  (Mnih & Hinton 2010 style; BSDS uses ~0.0075 x diagonal = 5.4 px at 512).
  Strict pixel F1 punishes a correct stroke shifted by 1 px as a full miss.
  Note: relaxed matching is many-to-one (no bipartite assignment), so it is
  slightly more lenient than the BSDS ``correspondPixels`` protocol.
* SSIM is now a real windowed SSIM (Gaussian 11x11, sigma 1.5) computed on the
  inpainting COMPOSITE and averaged over hole pixels. The previous value was
  a single global-statistics formula over hole pixels, i.e. not SSIM.
* Connectivity: 8-connected components (thin diagonal strokes were split into
  many 4-connected fragments), tolerance-aware coverage (a stroke counts as
  recovered if >= ``conn_frac`` of its pixels lie within ``conn_tol`` px of a
  predicted edge, instead of "any single pixel overlaps"), and a vectorised
  bincount instead of one full-image comparison per component.
* ``f1_img``: per-image (macro) F1, alongside the pooled (micro) F1.

Pooling conventions (state these in the paper)
----------------------------------------------
* precision / recall / f1 / iou / *_t{r} / psnr / mae / ssim : POOLED over all
  hole pixels of all images (micro average; large holes weigh more).
* f1_img / connectivity / hausdorff : mean over images (macro average).

All metric values are Python floats.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt as _edt
    from scipy.ndimage import label as _label
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

_STRUCT8 = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_window(size=11, sigma=1.5, device='cpu', dtype=torch.float32):
    xs = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]          # (1,1,k,k)


def _ssim_map(x, y, window, C1=0.01 ** 2, C2=0.03 ** 2):
    """Standard windowed SSIM map for (B,1,H,W) tensors in [0,1]."""
    pad = window.shape[-1] // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    yp = F.pad(y, (pad, pad, pad, pad), mode='reflect')
    mu_x = F.conv2d(xp, window)
    mu_y = F.conv2d(yp, window)
    sxx = F.conv2d(xp * xp, window) - mu_x ** 2
    syy = F.conv2d(yp * yp, window) - mu_y ** 2
    sxy = F.conv2d(xp * yp, window) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2)
    return num / den


def _safe_div(a, b, empty=1.0):
    return float(a) / float(b) if b > 0 else empty


def _f1(p, r):
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Streaming accumulator
# ---------------------------------------------------------------------------

class EdgeMetricAccumulator:
    """Accumulate edge-inpainting metrics over batches; call ``compute()`` once.

    Args:
        threshold     : binarisation threshold for pred and target.
        tolerances    : pixel radii for relaxed precision / recall / F1.
        conn_tol      : pixel radius used by the connectivity score.
        conn_frac     : fraction of a gt stroke that must be within conn_tol
                        of a prediction for the stroke to count as recovered.
        hd_percentile : percentile for the Hausdorff distance (95 = HD95).
    """

    def __init__(self, threshold=0.5, tolerances=(2, 5), conn_tol=2, conn_frac=0.5,
                 hd_percentile=95.0, ssim_window=11, ssim_sigma=1.5):
        self.threshold = threshold
        self.tolerances = tuple(int(t) for t in tolerances)
        self.conn_tol = conn_tol
        self.conn_frac = conn_frac
        self.hd_percentile = hd_percentile
        self.ssim_window = ssim_window
        self.ssim_sigma = ssim_sigma
        self._win = None
        self.reset()

    def reset(self):
        self.n_images = 0
        # strict (pooled)
        self.tp = self.fp = self.fn = 0.0
        # continuous (pooled over hole pixels)
        self.n_hole = 0.0
        self.sq_err = 0.0
        self.abs_err = 0.0
        self.ssim_sum = 0.0
        # relaxed (pooled)
        self.n_pred = 0
        self.n_gt = 0
        self.pred_hit = {r: 0 for r in self.tolerances}
        self.gt_hit = {r: 0 for r in self.tolerances}
        # per-image lists
        self.f1_img = []
        self.conn_img = []
        self.hd_img = []
        self.hd_invalid = 0     # exactly one of {gt, pred} empty in the hole

    @torch.no_grad()
    def update(self, pred, target, mask):
        """pred: probabilities [B,1,H,W] in [0,1]; target: [B,1,H,W] in [0,1];
        mask: [B,1,H,W], 1 = hole."""
        pred = pred.float()
        target = target.to(pred.device).float()
        mask = (mask.to(pred.device).float() > 0.5).float()
        thr = self.threshold

        pb = (pred > thr).float() * mask
        tb = (target > thr).float() * mask

        # ---- strict pooled counts
        self.tp += float((pb * tb).sum())
        self.fp += float((pb * (1 - tb)).sum())
        self.fn += float(((1 - pb) * tb * mask).sum())

        # ---- continuous metrics on hole pixels
        n_hole = float(mask.sum())
        self.n_hole += n_hole
        diff = (pred - target) * mask
        self.sq_err += float((diff ** 2).sum())
        self.abs_err += float(diff.abs().sum())

        # ---- windowed SSIM on the composite, averaged over hole pixels
        if self._win is None or self._win.device != pred.device:
            self._win = _gaussian_window(self.ssim_window, self.ssim_sigma, device=pred.device)
        comp = target * (1 - mask) + pred * mask
        self.ssim_sum += float((_ssim_map(comp, target, self._win) * mask).sum())

        # ---- per-image distance-based metrics (CPU, exact EDT)
        if not _HAS_SCIPY:
            self.n_images += pred.shape[0]
            return
        pb_np = pb.squeeze(1).bool().cpu().numpy()
        tb_np = tb.squeeze(1).bool().cpu().numpy()
        for b in range(pb_np.shape[0]):
            self._update_image(pb_np[b], tb_np[b])
        self.n_images += pred.shape[0]

    def _update_image(self, pr, gt):
        n_pr = int(pr.sum())
        n_gt = int(gt.sum())
        self.n_pred += n_pr
        self.n_gt += n_gt

        # per-image strict F1 (only if something to evaluate)
        if n_pr + n_gt > 0:
            tp = int((pr & gt).sum())
            p = _safe_div(tp, n_pr, empty=1.0)
            r = _safe_div(tp, n_gt, empty=1.0)
            self.f1_img.append(_f1(p, r))

        if n_pr == 0 and n_gt == 0:
            return

        # distance to nearest gt / pred edge pixel (inside hole). inf if empty.
        dt_gt = _edt(~gt) if n_gt > 0 else None
        dt_pr = _edt(~pr) if n_pr > 0 else None

        # relaxed precision / recall
        for r in self.tolerances:
            if n_pr > 0 and dt_gt is not None:
                self.pred_hit[r] += int((dt_gt[pr] <= r).sum())
            if n_gt > 0 and dt_pr is not None:
                self.gt_hit[r] += int((dt_pr[gt] <= r).sum())

        # Hausdorff (percentile, symmetric)
        if n_pr > 0 and n_gt > 0:
            d_gt2pr = dt_pr[gt]
            d_pr2gt = dt_gt[pr]
            self.hd_img.append(float(max(np.percentile(d_gt2pr, self.hd_percentile),
                                         np.percentile(d_pr2gt, self.hd_percentile))))
        else:
            self.hd_invalid += 1

        # Connectivity: fraction of 8-connected gt strokes recovered
        if n_gt > 0:
            labeled, num = _label(gt, structure=_STRUCT8)
            lab = labeled[gt]
            sizes = np.bincount(lab, minlength=num + 1)[1:]
            if dt_pr is not None:
                near = (dt_pr[gt] <= self.conn_tol).astype(np.float64)
                hits = np.bincount(lab, weights=near, minlength=num + 1)[1:]
            else:
                hits = np.zeros(num)
            covered = (hits >= self.conn_frac * sizes).sum()
            self.conn_img.append(covered / num)

    def compute(self):
        prec = _safe_div(self.tp, self.tp + self.fp, empty=1.0)
        rec = _safe_div(self.tp, self.tp + self.fn, empty=1.0)
        union = self.tp + self.fp + self.fn
        mse = self.sq_err / self.n_hole if self.n_hole > 0 else float('nan')
        if self.n_hole == 0:
            psnr = float('nan')
        elif mse == 0:
            psnr = float('inf')
        else:
            psnr = 10.0 * math.log10(1.0 / mse)

        out = {
            'precision':    prec,
            'recall':       rec,
            'f1':           _f1(prec, rec),
            'iou':          _safe_div(self.tp, union, empty=1.0),
            'f1_img':       float(np.mean(self.f1_img)) if self.f1_img else float('nan'),
            'ssim':         self.ssim_sum / self.n_hole if self.n_hole > 0 else float('nan'),
            'psnr':         psnr,
            'mae':          self.abs_err / self.n_hole if self.n_hole > 0 else float('nan'),
            'connectivity': float(np.mean(self.conn_img)) if self.conn_img else float('nan'),
            'hausdorff':    float(np.mean(self.hd_img)) if self.hd_img else float('nan'),
            'hd_valid_frac': _safe_div(len(self.hd_img), len(self.hd_img) + self.hd_invalid,
                                       empty=float('nan')),
            'n_images':     self.n_images,
        }
        for r in self.tolerances:
            p = _safe_div(self.pred_hit[r], self.n_pred, empty=1.0)
            rc = _safe_div(self.gt_hit[r], self.n_gt, empty=1.0)
            out[f'precision_t{r}'] = p
            out[f'recall_t{r}'] = rc
            out[f'f1_t{r}'] = _f1(p, rc)
        return out


# ---------------------------------------------------------------------------
# Backward-compatible entry points
# ---------------------------------------------------------------------------

def compute_edge_metrics(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor, threshold: float = 0.5, **acc_kwargs) -> dict:
    """One-shot metrics for a set of images (same signature as before).

    Returns all previous keys (precision, recall, f1, iou, ssim, psnr, mae,
    connectivity, hausdorff) plus f1_img, hd_valid_frac, n_images and
    precision_t{r} / recall_t{r} / f1_t{r} for each tolerance r.
    """
    acc = EdgeMetricAccumulator(threshold=threshold, **acc_kwargs)
    acc.update(pred, target, mask)
    return acc.compute()


def compute_metrics_batch(model, dataloader, device, max_samples: int = 100, **acc_kwargs) -> dict:
    """Metrics over a validation dataloader, accumulated across batches
    (independent of batch size)."""
    model.eval()
    acc = EdgeMetricAccumulator(**acc_kwargs)
    num_samples = 0
    with torch.no_grad():
        for img, edge, mask, _ in dataloader:
            if num_samples >= max_samples:
                break
            img = (img.float().to(device) / 127.5) - 1.0
            edge = edge.float().to(device) / 255.0
            mask = mask.float().to(device)
            pred = torch.sigmoid(model(img, edge, mask))
            acc.update(pred, edge, mask)
            num_samples += img.shape[0]
    return acc.compute()

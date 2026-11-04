"""Baseline structure generators re-trained in this framework for comparison (see docs/BASELINES.md)."""

import torch


def binarize_edge(edge, threshold=0.5):
    """Soft edge map in [0, 1] -> {0, 1}, same rule as metrics/metrics_edge.py (edge > threshold).

    EdgeConnect and ZITS were designed for binary Canny edges; the dataset's edge maps are
    soft JPEG maps. threshold=None returns the map unchanged.
    """
    edge = edge.to(torch.float32)
    if threshold is None:
        return edge
    return (edge > threshold).to(torch.float32)

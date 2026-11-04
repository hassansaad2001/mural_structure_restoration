# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Data conventions, model forward passes and one optimisation step of every loss."""
import numpy as np
import pytest
import torch

from conftest import RES, make_generator


def _batch(data_root, n=1):
    """A batch exactly as training_loop_tsr.py builds it: img [-1,1], edge [0,1] (1 = edge), mask {0,1} (1 = hole)."""
    from datasets.dataset_tsr_512 import TSRDataset
    ds = TSRDataset(edge_path=str(data_root / 'train' / 'edges'), data_path=str(data_root / 'train' / 'images'),
                    mask_path=str(data_root / 'masks'))
    s = [ds[i][:3] for i in range(n)]
    img, edge, mask = (torch.from_numpy(np.stack([np.asarray(x[k]) for x in s])) for k in range(3))
    return img.float() / 127.5 - 1.0, edge.float() / 255.0, mask.float()


def test_dataset_conventions(data_root):
    img, edge, mask = _batch(data_root, 2)
    assert img.shape == edge.shape == mask.shape == (2, 1, RES, RES)
    assert img.min() >= -1 and img.max() <= 1
    assert 0.0 < edge.mean() < 0.5, 'edges must be the minority with value 1 (files are black-on-white)'
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0} and 0.1 < mask.mean() < 0.5, 'mask: 1 = hole'


def test_binarize_matches_metric_rule(data_root):
    from networks.baselines import binarize_edge
    _, edge, _ = _batch(data_root)
    assert torch.unique(edge).numel() > 2                  # JPEG edges are soft
    assert torch.equal(binarize_edge(edge), (edge > 0.5).float())
    assert torch.equal(binarize_edge(edge, None), edge)


@pytest.mark.parametrize('name', ['ours', 'edgeconnect', 'zits_tsr'])
def test_generator_forward(data_root, name):
    img, edge, mask = _batch(data_root)
    G = make_generator(name)
    with torch.no_grad():
        for mode in (G.train, G.eval):                     # ZITS uses iterative sampling in eval mode
            mode()
            y = G(img, edge, mask)
            assert y.shape == (1, 1, RES, RES) and torch.isfinite(y).all()


def _step(loss, phases, modules, data_root):
    img, edge, mask = _batch(data_root)
    for phase, module in phases:
        for m in modules:
            m.requires_grad_(m is module)
            m.zero_grad(set_to_none=True)
        loss.accumulate_gradients(phase, img, edge, mask, sync=True, gain=1)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if phase != 'Dreg' or getattr(loss, 'r1_gamma', 0):
            assert grads and all(torch.isfinite(g).all() for g in grads), f'{phase}: no / non-finite gradients'


def test_loss_ours(data_root):
    from networks.new_mat import DiscriminatorLama
    from losses.loss_line import SketchGANLoss
    G = make_generator('ours').train()
    D = DiscriminatorLama(img_resolution=RES, img_channels=1, c_dim=0, ndf=64, n_layers=5, mbstd_group_size=4,
                          use_spec_norm=False, norm='sn', mask_input=True).train()
    loss = SketchGANLoss(torch.device('cpu'), G, D, r1_gamma=1.0, loss_mode='bce_dice_hd', hd_weight=0.5,
                         d_blur_sigma=0.7, d_fake_mode='legacy', bce_pos_weight=3.0)
    _step(loss, [('Gmain', G), ('Dmain', D), ('Dreg', D)], [G, D], data_root)


def test_loss_edgeconnect(data_root):
    from networks.baselines.edgeconnect import EdgeConnectDiscriminator
    from losses.loss_baselines import EdgeConnectLoss
    G = make_generator('edgeconnect').train()
    D = EdgeConnectDiscriminator(RES).train()
    _step(EdgeConnectLoss(torch.device('cpu'), G, D), [('Gmain', G), ('Dmain', D), ('Dreg', D)], [G, D], data_root)


def test_loss_zits(data_root):
    from losses.loss_baselines import ZITSTSRLoss
    G = make_generator('zits_tsr').train()
    _step(ZITSTSRLoss(torch.device('cpu'), G), [('Gmain', G)], [G], data_root)
    assert all(p.grad is not None for p in G.parameters()), 'every parameter needs a gradient (multi-GPU DDP)'

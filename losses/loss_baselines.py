# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Training losses of the baseline structure generators (see docs/BASELINES.md).

Both classes have the interface the training loop expects from SketchGANLoss:
    loss.accumulate_gradients(phase, full_image, full_edge, mask, sync, gain)
with full_image in [-1, 1], full_edge in [0, 1], mask 1 = hole.
Options that only SketchGANLoss understands (loss_mode, d_fake_mode, ...) are
accepted and ignored, so train_tsr.py can pass its usual loss kwargs.
"""

import torch
import torch.nn.functional as F

from torch_utils import misc
from torch_utils import training_stats
from networks.baselines import binarize_edge
from networks.baselines.edgeconnect import to_gray01


def _report_ignored(name, ignored):
    if ignored:
        print(f'[{name}] ignoring options not used by this baseline: {sorted(ignored)}')


#----------------------------------------------------------------------------

class EdgeConnectLoss:
    """EdgeConnect EdgeModel.process (src/models.py), unchanged in substance:

    G: non-saturating adversarial loss on D(gray, G output)
       + fm_weight * sum_i L1(D_i(gray, G output), D_i(gray, real edge))      (FM_LOSS_WEIGHT = 10)
       (no reconstruction loss; the edge model in EdgeConnect has none)
    D: (BCE(D(gray, real), 1) + BCE(D(gray, G output), 0)) / 2
    The discriminator sees the RAW generator output (not a composite) and the
    unmasked gray image, as in the original. No R1 penalty.
    Real edges shown to D are binarised at `binarize_edges` (EdgeConnect's real edges are
    binary Canny maps); the generator binarises its own input the same way.
    """

    def __init__(self, device, G, D, augment_pipe=None, adv_weight=1.0, fm_weight=10.0,
                 binarize_edges=0.5, **ignored):
        self.device = device
        self.G = G
        self.D = D
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        self.binarize_edges = binarize_edges
        ignored.pop('r1_gamma', None)
        _report_ignored('EdgeConnectLoss', ignored)
        print(f'[EdgeConnectLoss] adv_weight={adv_weight}  fm_weight={fm_weight}  '
              f'real edges binarised at {binarize_edges}')

    def run_G(self, img, edge, mask, sync):
        with misc.ddp_sync(self.G, sync):
            return torch.sigmoid(self.G(img, edge, mask))

    def run_D(self, edge, gray, sync, return_features=False):
        with misc.ddp_sync(self.D, sync):
            return self.D(edge, gray, return_features=return_features)

    def accumulate_gradients(self, phase, full_image, full_edge, mask, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        gray = to_gray01(full_image)
        real_edge = binarize_edge(full_edge, self.binarize_edges)

        if phase in ['Gmain', 'Gboth']:
            outputs = self.run_G(full_image, full_edge, mask, sync=sync)
            gen_fake, gen_fake_feat = self.run_D(outputs, gray, sync=False, return_features=True)
            gen_gan_loss = F.softplus(-gen_fake).mean()          # BCE(sigmoid(x), 1)
            with torch.no_grad():
                _, dis_real_feat = self.run_D(real_edge, gray, sync=False, return_features=True)
            gen_fm_loss = sum(F.l1_loss(g, r) for g, r in zip(gen_fake_feat, dis_real_feat))
            loss_G = self.adv_weight * gen_gan_loss + self.fm_weight * gen_fm_loss
            training_stats.report('Loss/G/adv', gen_gan_loss)
            training_stats.report('Loss/G/fm', gen_fm_loss)
            training_stats.report('Loss/G/total', loss_G)
            loss_G.mul(gain).backward()

        if phase in ['Dmain', 'Dboth']:
            with torch.no_grad():
                outputs = self.run_G(full_image, full_edge, mask, sync=False)
            dis_fake = self.run_D(outputs, gray, sync=False)
            dis_real = self.run_D(real_edge, gray, sync=sync)
            dis_loss = (F.softplus(-dis_real).mean() + F.softplus(dis_fake).mean()) / 2
            training_stats.report('Loss/scores/real', dis_real)
            training_stats.report('Loss/scores/fake_D', dis_fake)
            training_stats.report('Loss/D/loss', dis_loss)
            dis_loss.mul(gain).backward()
        # 'Dreg': EdgeConnect has no R1 penalty -> nothing to do

#----------------------------------------------------------------------------

class ZITSTSRLoss:
    """ZITS TSR loss (EdgeLineGPT256RelBCE.forward), edge term only, plus the SSU loss.

    TSR: mean over ALL 256x256 pixels of BCE(edge logits, GT edge) * hole mask
         (ZITS: 'loss *= masks; loss = torch.mean(loss)'); no discriminator.
    SSU: ssu_weight * BCE(SSU(GT edge at 256), GT edge at full resolution), whole image.
    Both targets are the binarised GT edge map produced by the generator wrapper.
    """

    def __init__(self, device, G, D=None, augment_pipe=None, ssu_weight=1.0, **ignored):
        self.device = device
        self.G = G
        self.ssu_weight = ssu_weight
        for k in ('r1_gamma', 'adv_weight', 'fm_weight'):
            ignored.pop(k, None)
        _report_ignored('ZITSTSRLoss', ignored)
        print(f'[ZITSTSRLoss] TSR hole BCE + {ssu_weight} x SSU BCE')

    def accumulate_gradients(self, phase, full_image, full_edge, mask, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if phase not in ['Gmain', 'Gboth']:
            return
        with misc.ddp_sync(self.G, sync):
            _, aux = self.G(full_image, full_edge, mask, return_aux=True)
        tsr_bce = F.binary_cross_entropy_with_logits(aux['tsr_logits'], aux['edge256'], reduction='none')
        tsr_loss = (tsr_bce * aux['mask256']).mean()
        ssu_loss = F.binary_cross_entropy_with_logits(aux['ssu_logits'], aux['edge_full'])   # binarised by G
        loss_G = tsr_loss + self.ssu_weight * ssu_loss
        if aux.get('ssu_unused') is not None:
            # SSU's second output is not used by ZITS at inference: zero weight, but kept in
            # the graph so multi-GPU DDP gets a (zero) gradient for those parameters.
            loss_G = loss_G + 0.0 * aux['ssu_unused'].sum()
        training_stats.report('Loss/G/bce', tsr_loss)
        training_stats.report('Loss/G/ssu', ssu_loss)
        training_stats.report('Loss/G/total', loss_G)
        loss_G.mul(gain).backward()

#----------------------------------------------------------------------------

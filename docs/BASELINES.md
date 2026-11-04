# Baselines: EdgeConnect and ZITS TSR

Two published structure-restoration models are re-trained **inside this framework**.
All models therefore use the same 512×512 mural crops, masks, mask augmentation, face/hand holes,
fixed validation bank, metrics, evaluation GUI and `generate_edges.py`.
Only the model and its training recipe change. `--preset` selects them:

```bash
DATA_ROOT=/path/to/dataset bash scripts/train_edgeconnect.sh
DATA_ROOT=/path/to/dataset bash scripts/train_zits_tsr.sh
python train_tsr.py --preset zits_tsr ... --dry-run      # print the resolved options
```

A preset fills every option listed below **unless you give that option on the command line**, so
`--preset edgeconnect --batch 16` changes only the batch size. `--preset ours` (the default) changes nothing.

ZITS++ is not included: its authors have not released training code (only test code and Places2 models).

### Edge maps: binarised for the baselines

The dataset's edge maps are soft JPEG maps:

| Edge value | Share of pixels |
|---|---|
| exactly 0 | 68 % |
| (0, 0.1], JPEG noise | 20 % |
| (0.1, 0.9), soft stroke borders | 8 % |
| ≥ 0.9 | 5 % |

EdgeConnect and ZITS were designed for binary Canny edges. Both baselines therefore binarise the edge map at
**0.5** (`edge > 0.5`, the same rule the metrics use to define a ground-truth edge) before anything else. This applies to their edge input, to the real edges EdgeConnect's discriminator sees, and to ZITS's training targets.
The data pipeline, the evaluation code and our own model are unchanged: our model keeps training on the soft maps, as in all earlier runs.
The behaviour is set by the `binarize_edges` argument of the generator (and of `EdgeConnectLoss`). Setting it to `None` keeps the edges soft.

---

## EdgeConnect – edge model (stage 1)

Nazeri et al., *EdgeConnect: Structure Guided Image Inpainting using Edge Prediction*, ICCV Workshops 2019.
Code: https://github.com/knazeri/edge-connect (CC BY-NC 4.0).

| | |
|---|---|
| Generator | `networks.baselines.edgeconnect.EdgeConnectEdgeGenerator`: original `EdgeGenerator` (spectral norm, 2 down-samplings, 8 dilated residual blocks) |
| Generator input | `cat(gray·(1−m)+m, edge·(1−m), m)` as in `EdgeModel.forward` |
| Discriminator | `EdgeConnectDiscriminator`: original 70×70 PatchGAN with spectral norm, on `cat(gray, edge)` |
| G loss | non-saturating GAN loss + 10 × feature matching (L1 summed over the 5 D layers); **no** reconstruction loss |
| D loss | (BCE(real) + BCE(fake)) / 2. The fake is the raw generator output, not a composite. No R1 |
| Optimiser | Adam, G lr 1e-4, D lr 1e-5 (`D2G_LR` 0.1), betas (0, 0.9), constant learning rate |
| Batch | 8 |

Preset: `--generator ...EdgeConnectEdgeGenerator --discriminator edgeconnect --loss losses.loss_baselines.EdgeConnectLoss --lr 1e-4 --lr-d 1e-5 --g-beta1 0 --d-beta1 0 --g-beta2 0.9 --d-beta2 0.9 --gamma 0 --adv-weight 1 --fm-weight 10 --batch 8 --lr-min-ratio 1`

**Deviations from the original**

- Trained at 512×512 (the original used 256). The network is fully convolutional, so the architecture is unchanged.
- Edges are the dataset's edge maps, binarised at 0.5, instead of Canny (σ = 2) edges, so all models learn the same structures.
- Original weights load into `.net` (`EdgeConnectEdgeGenerator.load_official('EdgeModel_gen.pth')`), for example to report a zero-shot row.

---

## ZITS – Transformer Structure Restorer (edge only)

Dong et al., *Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding*, CVPR 2022.
Code: https://github.com/DQiaole/ZITS_inpainting (Apache-2.0).

| | |
|---|---|
| Generator | `networks.baselines.zits_tsr.ZITSTSREdge`: original `EdgeLineGPT256RelBCE` (16 layers = 8 axial-attention + 8 full-attention blocks, width 256, 8 heads, 32×32 tokens, dropout 0), plus the original `StructureUpsampling` (SSU) |
| Input | image (gray copied to 3 channels, [−1, 1]), edge, line, mask, all at 256×256 |
| Loss | TSR: mean over all pixels of BCE(edge) × hole mask (as in `EdgeLineGPT256RelBCE.forward`); SSU: BCE, see below. No discriminator |
| Optimiser | AdamW, lr 4.24e-4, betas (0.9, 0.95), weight decay 0, gradient-norm clip 1.0 |
| Schedule | linear warm-up over 1500 iterations (48 kimg at batch 32), then cosine decay to 0.1 × lr |
| Batch | 32 |
| Inference | ZITS `SampleEdgeLineLogits`: 5 iterative steps, edge threshold 0.25, then SSU from 256 to 512 |

Preset: `--generator ...ZITSTSREdge --no-discriminator --loss losses.loss_baselines.ZITSTSRLoss --optimizer adamw --lr 4.24e-4 --g-beta1 0.9 --g-beta2 0.95 --weight-decay 0 --batch 32 --grad-clip 1 --lr-warmup 48 --lr-decay-start 48 --lr-min-ratio 0.1`

**Deviations from the original**

- **Edge only.** The line (wireframe) channel is all zeros, and the line output is neither trained nor used. The original obtains lines from the LSM-HAWP detector, which was trained on man-made scenes. The sampling ranks hole pixels by edge confidence only, instead of edge + line confidence.
- **256 ↔ 512.** TSR is fixed to 256 px, as in ZITS. The 512 inputs are reduced to 256 as follows:
  - image: area interpolation;
  - edge: max-pooling, which keeps 1-px strokes;
  - mask: max-pooling, so a 256-px cell counts as hole if any of its pixels is a hole.
- **SSU training.** ZITS did not release SSU training code. Here SSU is trained at the same time as TSR, using BCE to upsample the binary ground-truth edge map at 256 to the ground truth at 512, which is the same kind of input it receives at test time. It never back-propagates into TSR. Plain sigmoid is used on its output, instead of ZITS's `sigmoid((x + 2) * 2)`, which was calibrated for their own SSU weights. SSU's second output branch (`out`), which ZITS never uses at inference, gets zero loss weight.
- The "mask-and-predict" fine-tuning stage (`--MaP`) and mixed precision are not used.
- Edges are the dataset's edge maps, binarised at 0.5 at 512 px before the max-pooling, instead of Canny edges.

---

## Fairness notes for the comparison

- All models are evaluated with the EMA copy of the generator (`G_ema`), as in this framework. Neither original code base keeps an EMA.
- Train for the same number of images (`--kimg`) and compare checkpoints from the same point in training.
- The training/validation split, masks, metrics and evaluation code are identical for all models.
- The baselines see binarised edges and our model sees soft edges (see "Edge maps" above). The evaluation scores every model against the same ground truth, edge > 0.5.

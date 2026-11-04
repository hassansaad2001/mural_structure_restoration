# Notice: third-party code and modifications

This repository is a modified version of **MAT** (https://github.com/fenglinglwb/MAT,
© 2022 Wenbo Li et al., licensed **CC BY-NC 4.0**). MAT is built on **StyleGAN2-ADA**
(https://github.com/NVlabs/stylegan2-ada-pytorch, © 2021 NVIDIA CORPORATION,
NVIDIA Source Code License, non-commercial). The whole repository is therefore distributed
under **CC BY-NC 4.0** (see `LICENSE`) for non-commercial use only.

Modifications © 2026 Hassan Saad.

## Origin of each part

| Path | Origin | Status |
|---|---|---|
| `dnnlib/`, `torch_utils/` | StyleGAN2-ADA / MAT | unchanged except small compatibility fixes; NVIDIA headers kept |
| `networks/basic_module.py` | MAT | modified |
| `networks/new_mat.py` | MAT `networks/mat.py` | heavily modified: new GeneratorSketch, DiscriminatorSketch/Lama |
| `datasets/mask_generator_512.py` | MAT | modified |
| `datasets/dataset_tsr_512.py` | MAT dataset (StyleGAN2-ADA) | heavily modified; NVIDIA header kept |
| `training/training_loop_tsr.py` | MAT / StyleGAN2-ADA training loop | heavily modified; NVIDIA header kept |
| `train_tsr.py` | MAT `train.py` | heavily modified; NVIDIA header kept |
| `losses/loss_line.py` | MAT loss structure | rewritten for edge restoration; NVIDIA header kept |
| `metrics/metrics_edge.py` | new | written for this project; NVIDIA header kept from the template file |
| `networks/projected_discriminator.py` | new, after Sauer et al., *Projected GANs Converge Faster* (NeurIPS 2021) | own implementation |
| `networks/diffaug.py` | DiffAugment, Zhao et al. (NeurIPS 2020), https://github.com/mit-han-lab/data-efficient-gans (BSD 2-Clause) | reused |
| `evaluation/`, `inference/`, `tools/`, `generate_edges.py` | new | written for this project |
| `networks/baselines/edgeconnect.py` | EdgeConnect, Nazeri et al. (ICCVW 2019), https://github.com/knazeri/edge-connect (CC BY-NC 4.0) | network classes copied unchanged; wrappers new |
| `networks/baselines/zits_tsr.py` | ZITS, Dong et al. (CVPR 2022), https://github.com/DQiaole/ZITS_inpainting (Apache-2.0) | TSR / SSU classes copied (only `configure_optimizers` removed); wrapper and edge-only sampling new |
| `losses/loss_baselines.py` | re-implements the EdgeConnect and ZITS TSR training losses | written for this project |

## Ideas and building blocks from other work (re-implemented in `networks/new_mat.py`)

- Mask-aware partial convolution and mask-aware Swin Transformer blocks: MAT (Li et al., CVPR 2022)
- Swin Transformer window attention: Liu et al., ICCV 2021 (MIT licence)
- Fast Fourier Convolution / SpectralTransform: LaMa, Suvorov et al., WACV 2022 (Apache-2.0)
- PatchGAN discriminator (`DiscriminatorLama`): LaMa / pix2pix
- Multi-Dconv transposed attention: Restormer, Zamir et al., CVPR 2022
- Pretrained backbones from `timm` (Apache-2.0) are used by the projected discriminator

## Not included

Pretrained checkpoints, datasets and the publication PDF are not part of this repository.

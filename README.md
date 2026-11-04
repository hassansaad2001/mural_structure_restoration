# Mural Structure Restoration

Deep-learning restoration of the **structure (edge / line) maps of damaged murals**.
Given a mural image, its edge map and a mask of the damaged region, the network predicts the missing lines inside the hole.
The restored structure can then guide a second-stage colour/texture inpainting model.

This code is part of a master's thesis by **Hassan Saad**. It is a heavily modified version of
[MAT: Mask-Aware Transformer for Large Hole Image Inpainting](https://github.com/fenglinglwb/MAT) (CVPR 2022),
which is itself built on NVIDIA [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
See [NOTICE.md](NOTICE.md) for what was reused and what changed.

## Method in short

`networks/new_mat.py` → `GeneratorSketch`

- **Input:** gray image, visible edges and the hole mask (1 = hole).
- **Encoder:** 4 mask-aware partial-convolution stages (from MAT), 512 → 64 px.
- **Bottleneck at 1/8 resolution:** Fast Fourier Convolution blocks (LaMa) for global context. Optional mask-aware Swin Transformer stages (from MAT) go between them (`--swin-after`). Ablations: `--bottleneck conv|swin`.
- **Decoder:** depthwise-separable residual blocks with multi-dilation transposed attention. Skip connections and the hole map feed every stage.
- **Output:** edge logits. The final edge map is the ground truth outside the hole and the prediction inside it.
- **Training:** an adversarial setup with a PatchGAN (`DiscriminatorLama`) or StyleGAN-style (`DiscriminatorSketch`) discriminator. Losses are BCE, Dice, a distance-transform Hausdorff term, an adversarial loss and feature matching (`losses/loss_line.py`).
- **Metrics (inside the hole):** precision, recall, F1, IoU, tolerance-F1 @2/5 px, SSIM, PSNR, MAE, connectivity and HD95 (`metrics/metrics_edge.py`).

## Repository layout

```
train_tsr.py                 training entry point (click CLI)
generate_edges.py            test script: predict structure for a folder of samples
networks/new_mat.py          generator + discriminators          (main model)
datasets/dataset_tsr_512.py  512×512 dataset, masks, face/hand holes, validation bank
losses/loss_line.py          SketchGANLoss
metrics/metrics_edge.py      edge metrics (streaming accumulator)
training/training_loop_tsr.py training loop: EMA, cosine LR, resume, snapshots, validation
evaluation/                  checkpoint comparison: eval_core (engine), eval_server (container), eval_checkpoints_gui (Tk GUI)
inference/                   interactive inference: inference_server (container), inference_gui (Tk GUI)
tools/                       data preparation: cropping, mask drawing, face boxes, resizing, model complexity
dnnlib/, torch_utils/        utilities from StyleGAN2-ADA / MAT (unchanged)
docker/                      Dockerfile (pinned environment) and run.sh (start the container)
scripts/                     training / test commands (container) and GUI launchers (host)
tests/                       CPU tests on a synthetic dataset (pytest)
docs/                        INSTALL, TRAINING and BASELINES guides
```

## Installation (Ubuntu + Docker)

Training, testing and the evaluation/inference servers run in a Docker container with a pinned, known-good environment: PyTorch 1.13.1, CUDA 11.6, Python 3.10. The GUIs run on the host. It works the same on native Ubuntu 22.04/24.04 and in WSL 2.
**Full guide: [docs/INSTALL.md](docs/INSTALL.md)**. It covers the NVIDIA driver, Docker, the NVIDIA Container Toolkit, the GUIs and troubleshooting.

Quick start, after the host setup in INSTALL.md:

1. Start the container. The first run builds the image `mural-structure:latest`:

```bash
bash docker/run.sh
```

2. Open a shell in it, at the repository root:

```bash
docker exec -it msr bash
```

3. Inside the container, check the installation:

```bash
python3 -m pytest tests -q
```

4. On the host, set up the GUI environment once:

```bash
bash scripts/setup_gui_env.sh
```

The commands in the next sections run **inside the container**, except the GUI launchers, which run on the host.

## Data

The training data is organised as 512×512 crops with matching names in `images_512crops/` and `edges_512crops/`. Edge maps use black lines on a white background. Masks use white for the hole. The full layout is in [docs/TRAINING.md](docs/TRAINING.md).
`tools/` contains the helpers used to build it:

- `crop_by_density.py`: crop by edge density
- `crop_mask_tool.py`: interactive crop and mask drawing
- `annotate_face_boxes.py` and `export_face_masks.py`: face/hand holes
- `resize_dataset.py`: resize a dataset

The mural dataset itself is not part of this repository.

## Training

Final configuration (inside the container; `DATA_ROOT` defaults to `/master/DhMurals-inpainting-dataset`):

```bash
DATA_ROOT=/master/DhMurals-inpainting-dataset bash scripts/train_example.sh
```

All options:

```bash
python3 train_tsr.py --help
```

Runs are written to `outputs/tsr/<NNNNN>-<description>/` and include `training_options.json`, `checkpoint-*.pth`, snapshots and TensorBoard logs.

### Baselines

EdgeConnect (edge stage) and ZITS TSR (edge only) are re-trained in the same framework with `--preset`, on the same data, masks and metrics:

```bash
DATA_ROOT=/path/to/dataset bash scripts/train_edgeconnect.sh
DATA_ROOT=/path/to/dataset bash scripts/train_zits_tsr.sh
```

Their recipes and every deviation from the original code are in [docs/BASELINES.md](docs/BASELINES.md).

## Testing

```bash
python generate_edges.py --network outputs/tsr/<run>/checkpoint-001000.pth \
    --dpath test/images --epath test/edges --mpath test/masks --outdir results/
```

The script writes `results/` (composite edge map), `highlights/` (hole region tinted) and, with `--save-raw`, `raw/` (probabilities).
If you leave out `--mpath`, random masks are generated.

## Evaluation (compare checkpoints)

On the host, start the server in the container and open the GUI:

```bash
bash scripts/eval_gui.sh
```

Or, inside the container, run headless with settings saved from the GUI:

```bash
python3 evaluation/eval_server.py --headless --settings eval_settings.json
```

The tool filters samples whose hole contains no useful edges. It splits the masks into Small / Medium / Large groups, evaluates every checkpoint on the same samples and exports an Excel workbook, CSVs and comparison grids.

## Interactive inference

On the host (this starts the server with the checkpoint in the container):

```bash
bash scripts/inference_gui.sh outputs/tsr/<run>/checkpoint-001000.pth
```

Click a 512×512 crop, paint the damaged region and press **Run**.

## Licence

This repository is a derivative of MAT and StyleGAN2-ADA and is released under
**[CC BY-NC 4.0](LICENSE)**, for **non-commercial research and education only**.
Files taken from NVIDIA/MAT keep their original copyright notices. See [NOTICE.md](NOTICE.md).

## Citation

If you use this code, please cite the thesis/paper (to be added) and the original MAT paper:

```bibtex
@inproceedings{li2022mat,
  title     = {MAT: Mask-Aware Transformer for Large Hole Image Inpainting},
  author    = {Li, Wenbo and Lin, Zhe and Zhou, Kun and Qi, Lu and Wang, Yi and Jia, Jiaya},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
}
```

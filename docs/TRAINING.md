# Training guide

All commands are run from the repository root. `train_tsr.py --help` lists every option.

## Data layout

```
<DATA_ROOT>/
├── train/
│   ├── images_512crops/   mural crops, 512×512 (RGB or gray; converted to gray)
│   ├── edges_512crops/    structure maps with the SAME file names, black lines on white
│   └── masks/             optional pool of hole masks (white = hole); otherwise random masks
├── val/
│   ├── images_512crops/
│   ├── edges_512crops/
│   └── masks_heldout/     held-out validation masks
├── face_boxes_train.json  optional face/hand boxes (tools/annotate_face_boxes.py)
└── face_boxes_val.json
```

## Main options

| Group | Options |
|---|---|
| Data | `--data-edge`, `--data-image`, `--mask-path`, `--mask-aug`, `--data-val-edge`, `--data-val-image`, `--val-mask-path`, `--val-masks-per-image`, `--mirror` |
| Face/hand holes | `--face-boxes`, `--face-mask-prob`, `--face-oversample`, `--val-face-boxes` |
| Generator | `--bottleneck {ffc,conv,swin}`, `--n-btn`, `--swin-after 1,3`, `--swin-dim`, `--dec-attn`, `--dec-norm` |
| Discriminator | `--discriminator {sketch,lama,projected,edgeconnect}`, `--no-discriminator`, `--d-norm`, `--d-layers`, `--d-mask-input`, `--d-blur`, `--d-fake-mode` |
| Loss | `--loss-mode {bce,bce_dice,bce_dice_hd}`, `--bce-weight`, `--dice-weight`, `--hd-weight`, `--adv-weight`, `--fm-weight`, `--bce-pos-weight`, `--focal`, `--rec-tau`, `--gamma` (R1) |
| Optimiser | `--optimizer {adam,adamw}`, `--lr`, `--lr-d`, `--g-beta1`, `--d-beta1`, `--g-beta2`, `--d-beta2`, `--weight-decay`, `--grad-clip`, `--lr-warmup`, `--lr-decay-start`, `--lr-min-ratio`, `--min-lr`, `--min-lr-d`, `--ema` |
| Baselines | `--preset {ours,edgeconnect,sgtb_srn,zits_tsr}`, see [BASELINES.md](BASELINES.md) |
| Run | `--outdir`, `--gpus`, `--batch`, `--batch-gpu`, `--workers`, `--kimg`, `--snap`, `--seed`, `--resume-ckpt`, `--allow-tf32`, `-n/--dry-run` |

See [`scripts/train_example.sh`](../scripts/train_example.sh) for the configuration of the final model.

## Learning-rate schedule

The generator and discriminator can use different learning rates (`--lr`, `--lr-d`; D uses `--lr` when `--lr-d` is not given).
The rate is constant until `--lr-decay-start` kimg, then follows a cosine decay down to `--lr-min-ratio` × base LR at `--kimg`:

```
LR
│  ████████                     constant (warm-up)
│          ╲
│           ╲   cosine decay
│            ╲
│             ███████████       min LR
└─────────────────────────────── kimg
   0     lr-decay-start      kimg
```

Set `--lr-decay-start` equal to `--kimg` for a constant LR.
`Progress/lr_G` and `Progress/lr_D` are logged to TensorBoard.

## Resuming

`--resume-ckpt=<run_dir>/checkpoint-XXXXXX.pth` restores G, G_ema, D, optimisers, RNG states and the LR schedule.

- The decay continues from the stored `cur_nimg`.
- If you raise `--kimg`, the decay length is recomputed as `kimg - lr-decay-start`.
- `--lr` / `--lr-d` given on the command line override the stored rates.

## Outputs

Each run creates `<outdir>/<NNNNN>-<description>/` with:

- `training_options.json` – every option (the evaluation tools read the network class and its kwargs from here)
- `checkpoint-XXXXXX.pth` – full training state (`G_ema_state` is used for inference)
- snapshot grids (predictions, composites, highlighted holes) and TensorBoard logs
- validation metrics on a fixed bank of (image, mask) pairs: precision, recall, F1, IoU, SSIM, PSNR, MAE, connectivity, HD95, tolerance F1 @2/5 px, reported overall and per hole-size bucket

## Multi-GPU notes

- `pin_memory` is off in `train_tsr.py`; with only 1–2 GPUs turning it on can be faster.
- BatchNorm in the FFC blocks is converted to SyncBatchNorm when `--gpus > 1` (`--sync-bn`, default on).
- Start Docker with a large `--shm-size` (e.g. `64g`) for the DataLoader workers.

# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Test script: predict missing mural structure (edges) with a trained GeneratorSketch.

Loads a training checkpoint (.pth) with the same loader as the inference /
evaluation servers (architecture options are detected from the weights).

For each sample it writes:
    <outdir>/results/     composite edge map (GT outside the hole, prediction inside), black-on-white
    <outdir>/highlights/  the composite with the hole region tinted
    <outdir>/raw/         raw sigmoid output (only with --save-raw)

Usage:
    python generate_edges.py \
        --network checkpoint-000100.pth \
        --dpath ./images --epath ./edges [--mpath ./masks] \
        --outdir ./output
"""
import os
import glob
import numpy as np
import torch
import PIL.Image
import click

from datasets.mask_generator_512 import RandomMask
from inference.inference_server import load_generator
from evaluation.eval_core import resolve_network

#----------------------------------------------------------------------------

def save_image(img, fname):
    """Save image to file.
    img: (C, H, W) or (H, W, C) or (H, W) numpy array, uint8 0-255.
    """
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW -> HWC
        img = img.transpose(1, 2, 0)
    
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
        
    if img.ndim == 2:
        PIL.Image.fromarray(img, 'L').save(fname)
    else:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def save_masked_with_transparent(fake_edges, masks, fname, overlay_color=(100, 150, 255), alpha=0.3):
    """Save predicted edges with transparent color overlay on mask region.
    
    Args:
        fake_edges: (1, H, W) in range [0, 1]
        masks: (1, H, W), 1=hole
        fname: Output file path
    """
    fake = fake_edges[0]
    mask = masks[0]
    H, W = fake.shape
    
    output = np.zeros((3, H, W), dtype=np.float32)
    edge_rgb = fake * 255.0
    
    for c in range(3):
        output[c] = edge_rgb
        output[c] = output[c] * (1 - mask * alpha) + mask * alpha * overlay_color[c]
    
    output = output.clip(0, 255).astype(np.uint8)
    output = output.transpose(1, 2, 0)
    PIL.Image.fromarray(output, 'RGB').save(fname)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'checkpoint_path', help='Path to .pth checkpoint', required=True, metavar='PATH')
@click.option('--dpath', help='Path to input images (grayscale or RGB)', required=True, metavar='DIR')
@click.option('--epath', help='Path to input edges', required=True, metavar='DIR')
@click.option('--mpath', help='Path to input masks (optional)', required=False, metavar='DIR')
@click.option('--outdir', help='Output directory', required=True, metavar='DIR')
@click.option('--resolution', help='Process at this resolution', type=int, default=512, show_default=True)
@click.option('--generator', help='Generator class [default: from the run\'s training_options.json, else networks.new_mat.GeneratorSketch]', metavar='CLASS')
@click.option('--invert-edge/--no-invert-edge', help='Invert edge (Black-on-White → White-on-Black)', default=True)
@click.option('--cpu', help='Run on CPU', is_flag=True)
@click.option('--save-raw', help='Also save raw sigmoid output (not binarized)', is_flag=True)

def generate_edges(
    checkpoint_path: str,
    dpath: str,
    epath: str,
    mpath: str,
    outdir: str,
    resolution: int,
    generator: str,
    invert_edge: bool,
    cpu: bool,
    save_raw: bool,
):
    """Generate edge predictions using a trained GeneratorSketch.
    
    Expected Input:
    - Images (--dpath): Grayscale or RGB. Converted to grayscale, normalized to [-1, 1].
    - Edges (--epath): Grayscale.
        With --invert-edge (default): Black lines (0) on White bg (255) → 1.0=edge, 0.0=bg.
    - Masks (--mpath): Optional. White (255)=Hole, Black (0)=Valid.
        If not provided, random masks are generated.
    """
    device = torch.device('cuda') if torch.cuda.is_available() and not cpu else torch.device('cpu')
    print(f"Device: {device}")
    
    # Load model
    # Architecture options (swin_after, bottleneck, dec_norm, ...) come from the run's training_options.json
    cls, extra_kwargs, source = resolve_network(checkpoint_path, generator, 'networks.new_mat.GeneratorSketch')
    print(f"Network: {cls} ({source}) {extra_kwargs}")
    G, in_channels = load_generator(checkpoint_path, device, resolution, cls, extra_kwargs)
    missing, unexpected = G.load_report
    if missing or unexpected:
        raise click.ClickException(f'Checkpoint does not match {cls}: {len(missing)} missing, '
                                   f'{len(unexpected)} unexpected weights. Pass the right --generator.')
    
    # Create output dirs
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'highlights'), exist_ok=True)
    if save_raw:
        os.makedirs(os.path.join(outdir, 'raw'), exist_ok=True)
    
    # Gather files
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    img_files = sorted([f for ext in extensions for f in glob.glob(os.path.join(dpath, ext))])
    edge_files = sorted([f for ext in extensions for f in glob.glob(os.path.join(epath, ext))])
    
    mask_files = None
    if mpath is not None:
        mask_files = sorted([f for ext in extensions for f in glob.glob(os.path.join(mpath, ext))])
        print(f"Found {len(img_files)} images, {len(edge_files)} edges, {len(mask_files)} masks.")
    else:
        print(f"Found {len(img_files)} images, {len(edge_files)} edges. Using random masks.")
    
    # Determine how many to process
    counts = [len(img_files), len(edge_files)]
    if mask_files is not None:
        counts.append(len(mask_files))
    min_len = min(counts)
    if len(set(counts)) > 1:
        print(f"Warning: File counts don't match! Processing first {min_len}.")
    
    # Process
    for i in range(min_len):
        basename = os.path.splitext(os.path.basename(edge_files[i]))[0]
        print(f"[{i+1}/{min_len}] {basename}")
        
        # ── Load Image ──
        img = PIL.Image.open(img_files[i]).convert('L')
        img = img.resize((resolution, resolution), PIL.Image.BILINEAR)
        img_np = np.array(img)
        img_t = (torch.from_numpy(img_np).float().to(device) / 127.5) - 1.0
        img_t = img_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # ── Load Edge ──
        edge = PIL.Image.open(edge_files[i]).convert('L')
        edge = edge.resize((resolution, resolution), PIL.Image.NEAREST)
        edge_np = np.array(edge)
        if invert_edge:
            edge_np = 255 - edge_np
        edge_t = torch.from_numpy(edge_np).float().to(device).unsqueeze(0).unsqueeze(0) / 255.0
        
        # ── Load / Generate Mask ──
        if mask_files:
            mask = PIL.Image.open(mask_files[i]).convert('L')
            mask = mask.resize((resolution, resolution), PIL.Image.NEAREST)
            mask_np = np.array(mask)
            mask_t = torch.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0) / 255.0
        else:
            mask_np = RandomMask(resolution, hole_range=[0, 0.4])
            mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(device)
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            elif mask_t.ndim == 3:
                mask_t = mask_t.unsqueeze(0)
            mask_t = 1.0 - mask_t  # Convert to 1=Hole
        
        mask_t = (mask_t > 0.5).float()
        
        # ── Inference ──
        with torch.no_grad():
            logits = G(img_t, edge_t, mask_t)
            pred = torch.sigmoid(logits)
            pred_binary = (pred > 0.5).float()
            
            # Composite: keep real edge where valid, use prediction where hole
            output_composite = edge_t * (1 - mask_t) + pred_binary * mask_t
        
        # ── Save Results ──
        
        # 1. Composite result (Black-on-White for display)
        final_result = 1.0 - output_composite[0, 0].cpu().numpy()
        save_image(
            (final_result * 255).astype(np.uint8),
            os.path.join(outdir, 'results', f'{basename}.png')
        )
        
        # 2. Highlight overlay
        save_masked_with_transparent(
            fake_edges=pred_binary[0].cpu().numpy(),
            masks=mask_t[0].cpu().numpy(),
            fname=os.path.join(outdir, 'highlights', f'{basename}.png'),
            overlay_color=(100, 150, 255),
            alpha=0.3
        )
        
        # 3. Raw sigmoid (optional)
        if save_raw:
            raw = pred[0, 0].cpu().numpy()
            save_image(
                (raw * 255).astype(np.uint8),
                os.path.join(outdir, 'raw', f'{basename}.png')
            )
    
    print(f"\nDone! {min_len} images processed → {outdir}")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_edges()

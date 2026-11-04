"""Generate a complexity report for the TSR model (Generator).

This script calculates the number of parameters (total and trainable),
the estimated size in memory, and the computational complexity (FLOPs/MACs)
if a profiling library (like fvcore or thop) is installed.

Usage (from the repository root):
  python tools/report_complexity.py
  python tools/report_complexity.py --generator networks.new_mat.GeneratorSketch --resolution 512
"""
# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).

import os
import sys
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import click
import torch
import dnnlib

def get_model_size_mb(model):
    """Estimate the model size in MB based on parameter count and dtype."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

@click.command()
@click.option('--generator', default='networks.new_mat.GeneratorSketch', 
              help='Generator class path', show_default=True)
@click.option('--resolution', default=256, type=int, 
              help='Image resolution (width/height)', show_default=True)
@click.option('--batch', default=1, type=int, 
              help='Batch size for FLOP calculation', show_default=True)
@click.option('--device', default='cpu', 
              help='Device to run the model on (e.g., cpu, cuda:0)', show_default=True)
def main(generator, resolution, batch, device):
    print(f"==================================================")
    print(f" Model Complexity Report")
    print(f"==================================================")
    print(f"Generator Class : {generator}")
    print(f"Resolution      : {resolution}x{resolution}")
    print(f"Batch Size      : {batch}")
    print(f"Device          : {device}")
    print(f"--------------------------------------------------")

    # 1. Instantiate the model
    print("Constructing generator...")
    G_kwargs = dnnlib.EasyDict(
        class_name=generator,
        img_resolution=resolution,
    )
    try:
        model = dnnlib.util.construct_class_by_name(**G_kwargs).to(device)
        model.eval()
    except Exception as e:
        print(f"Error constructing model: {e}")
        return

    # 2. Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = get_model_size_mb(model)

    print(f"\n[Parameters]")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")
    print(f"  Estimated Size (MB)  : {model_size_mb:.2f} MB")

    # 3. Create dummy inputs for FLOPs calculation
    # The Generator takes (img, edge, mask)
    dummy_img = torch.zeros((batch, 1, resolution, resolution), device=device)
    dummy_edge = torch.zeros((batch, 1, resolution, resolution), device=device)
    dummy_mask = torch.zeros((batch, 1, resolution, resolution), device=device)
    inputs = (dummy_img, dummy_edge, dummy_mask)

    # 4. Attempt to calculate FLOPs/MACs
    print(f"\n[Computational Complexity]")
    
    flops_calculated = False

    # Try fvcore first (Facebook standard)
    if not flops_calculated:
        try:
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            flops = FlopCountAnalysis(model, inputs)
            # Suppress warnings
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            
            total_flops = flops.total()
            print(f"  MACs (fvcore)        : {total_flops / 1e9:.2f} G MACs (for batch size {batch})")
            print("\n  [Detailed fvcore Breakdown]")
            print(flop_count_table(flops))
            flops_calculated = True
        except ImportError:
            pass

    # Try thop if fvcore isn't available
    if not flops_calculated:
        try:
            from thop import profile
            macs, params = profile(model, inputs=inputs, verbose=False)
            print(f"  MACs (thop)          : {macs / 1e9:.2f} G MACs (for batch size {batch})")
            flops_calculated = True
        except ImportError:
            pass
            
    # Try ptflops
    if not flops_calculated:
        try:
            from ptflops import get_model_complexity_info
            
            # Wrapper to handle multiple inputs for ptflops
            class ModelWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    # We have to unpack x back into the 3 arguments for this specific generator
                    # ptflops usually passes a single tensor
                    c = x.shape[1] // 3
                    return self.m(x[:, 0:c], x[:, c:2*c], x[:, 2*c:3*c])
                    
            wrapper = ModelWrapper(model)
            # 3 channels (img, edge, mask) each with 1 channel
            macs, params = get_model_complexity_info(wrapper, (3, resolution, resolution), 
                                                     as_strings=True, print_per_layer_stat=False, verbose=False)
            print(f"  MACs (ptflops)       : {macs} (for batch size 1)")
            flops_calculated = True
        except ImportError:
            pass

    if not flops_calculated:
        print("  FLOPs / MACs         : N/A")
        print("\n  (Note: To view FLOPs/MACs, please install 'fvcore', 'thop', or 'ptflops')")
        print("  Example: pip install fvcore")
        
    print(f"\n==================================================")


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
"""Resize every image in the train/ val/ test/ sub-folders of a dataset to a square size,
keeping the folder structure.

Usage:
    python tools/resize_dataset.py --src <dataset_root> --dst <output_root> [--size 256]
"""
import os
import argparse
from PIL import Image

def get_resampling_filter():
    if hasattr(Image, 'Resampling'):
        return Image.Resampling.LANCZOS
    if hasattr(Image, 'LANCZOS'):
        return Image.LANCZOS
    if hasattr(Image, 'ANTIALIAS'):
        return Image.ANTIALIAS
    return Image.BICUBIC

def process_dataset(source_dir, dest_dir, size=256):
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return

    if not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir)
            print(f"Created destination directory: {dest_dir}")
        except OSError as e:
            print(f"Error creating destination directory: {e}")
            return
    else:
        print(f"Destination directory exists: {dest_dir}")

    resample_filter = get_resampling_filter()
    print(f"Using resampling filter: {resample_filter}")

    # Start processing
    parts = ['train', 'val', 'test'] 
    
    # Check what's actually there
    items = os.listdir(source_dir)
    print(f"Found in source root: {items}")
    
    total_processed = 0
    total_errors = 0
    
    for item in items:
        # Skip hidden files
        if item.startswith('.'):
            continue

        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            if item in parts:
                print(f"Processing part directory: {item}")
                
                # Walk through the directory structure
                for root, dirs, files in os.walk(item_path):
                    # compute relative path from source_dir to preserve structure
                    # e.g. "train/class_a" relative to "source_dir"
                    rel_path = os.path.relpath(root, source_dir) 
                    
                    target_root = os.path.join(dest_dir, rel_path)
                    if not os.path.exists(target_root):
                        try:
                            os.makedirs(target_root)
                        except OSError:
                            pass
                        
                    for file in files:
                        if file.startswith('.'):
                            continue
                            
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            src_file = os.path.join(root, file)
                            dst_file = os.path.join(target_root, file)
                            
                            try:
                                with Image.open(src_file) as img:
                                    img_resized = img.resize((size, size), resample_filter)
                                    img_resized.save(dst_file)
                                    total_processed += 1
                                    if total_processed % 100 == 0:
                                        print(f"Processed {total_processed} images...")
                            except Exception:
                                total_errors += 1
            else:
                print(f"Skipping non-part directory: {item}")
        else:
            # skipping files in root
            pass

    print(f"Done. Processed {total_processed} images successfully. Encountered {total_errors} errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='Dataset root containing train/ val/ test/')
    parser.add_argument('--dst', required=True, help='Output root (structure is preserved)')
    parser.add_argument('--size', type=int, default=256, help='Output width/height (default: 256)')
    args = parser.parse_args()
    process_dataset(args.src, args.dst, args.size)

# Copyright (c) 2026 Hassan Saad. Licensed under CC BY-NC 4.0 (see LICENSE).
import os
import cv2
import numpy as np
import argparse
import random
from tqdm import tqdm

def calculate_density(edge_crop, tau=0):
    """
    Calculate the density of edges in the crop.
    $$Density = \frac{\sum_{x=1}^{512} \sum_{y=1}^{512} \mathbb{I}(P_{x,y} > \tau)}{512 \times 512}$$
    """
    # Ensure it's single channel
    if len(edge_crop.shape) == 3:
        edge_crop = cv2.cvtColor(edge_crop, cv2.COLOR_BGR2GRAY)
    
    # Calculate density
    pixels_greater_than_tau = np.sum(edge_crop > tau)
    total_pixels = edge_crop.shape[0] * edge_crop.shape[1]
    
    if total_pixels == 0:
        return 0.0
        
    return pixels_greater_than_tau / total_pixels

def extract_crops(image, window_size=512, stride=256):
    """
    Extract sliding window crops from an image.
    Returns a list of crops and their coordinates (y, x).
    """
    h, w = image.shape[:2]
    crops = []
    coords = []
    
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            crop = image[y:y+window_size, x:x+window_size]
            crops.append(crop)
            coords.append((y, x))
            
    return crops, coords

def main():
    parser = argparse.ArgumentParser(description="Crop images based on edge density.")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing RGB images.")
    parser.add_argument("--edge_dir", type=str, required=True, help="Directory containing Edge images.")
    parser.add_argument("--out_rgb_dir", type=str, required=True, help="Output directory for RGB crops.")
    parser.add_argument("--out_edge_dir", type=str, required=True, help="Output directory for Edge crops.")
    parser.add_argument("--window_size", type=int, default=512, help="Size of the sliding window.")
    parser.add_argument("--stride", type=int, default=256, help="Stride of the sliding window.")
    parser.add_argument("--tau", type=int, default=0, help="Pixel intensity threshold for edge indicator function (tau).")
    parser.add_argument("--n_sample_images", type=int, default=10, help="Number of images to sample to calculate average density.")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.out_rgb_dir, exist_ok=True)
    os.makedirs(args.out_edge_dir, exist_ok=True)
    
    # Get all RGB files
    rgb_files = sorted([f for f in os.listdir(args.rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not rgb_files:
        print(f"No images found in {args.rgb_dir}.")
        return
        
    print(f"Found {len(rgb_files)} images.")
    
    # ==========================================
    # 1. Sample Phase to Calculate Average Density
    # ==========================================
    print(f"Sampling {args.n_sample_images} images to calculate average density...")
    sample_files = random.sample(rgb_files, min(args.n_sample_images, len(rgb_files)))
    
    sample_densities = []
    
    for f in tqdm(sample_files, desc="Sampling"):
        rgb_path = os.path.join(args.rgb_dir, f)
        edge_path = os.path.join(args.edge_dir, f)
        
        if not os.path.exists(edge_path):
            continue
            
        edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if edge_img is None:
            continue
            
        edge_crops, _ = extract_crops(edge_img, args.window_size, args.stride)
        for crop in edge_crops:
            density = calculate_density(crop, args.tau)
            sample_densities.append(density)
            
    if not sample_densities:
        print("Could not calculate densities on sample images. Please check your directories.")
        return
        
    avg_density = np.mean(sample_densities)
    max_density = np.max(sample_densities)
    min_density = np.min(sample_densities)
    
    print(f"\n--- Density Statistics ---")
    print(f"Calculated over {len(sample_densities)} sample crops.")
    print(f"Average Density: {avg_density:.4f}")
    print(f"Max Density:     {max_density:.4f}")
    print(f"Min Density:     {min_density:.4f}")
    print(f"--------------------------\n")
    
    # ==========================================
    # 2. Ask User for Threshold and Confirm
    # ==========================================
    while True:
        try:
            user_input = input("\nEnter the density threshold you would like to use (e.g., 0.05): ")
            density_threshold = float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue
            
        print("\nCalculating total number of valid crops (dry run)...")
        expected_crop_count = 0
        
        for f in tqdm(rgb_files, desc="Dry Run (Edges only)"):
            edge_path = os.path.join(args.edge_dir, f)
            if not os.path.exists(edge_path):
                continue
                
            edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            if edge_img is None:
                continue
                
            edge_crops, _ = extract_crops(edge_img, args.window_size, args.stride)
            for crop in edge_crops:
                density = calculate_density(crop, args.tau)
                if density >= density_threshold:
                    expected_crop_count += 1
                    
        print(f"\nWith threshold {density_threshold}, you will generate {expected_crop_count} crops.")
        confirm = input("Do you want to proceed and save these crops? (y to save / n to choose new threshold): ").strip().lower()
        
        if confirm == 'y':
            break
            
    # ==========================================
    # 3. Main Processing Loop
    # ==========================================
    print(f"\nProcessing and saving all images with density threshold: {density_threshold}")
    
    crop_count = 0
    for f in tqdm(rgb_files, desc="Cropping & Saving"):
        rgb_path = os.path.join(args.rgb_dir, f)
        edge_path = os.path.join(args.edge_dir, f)
        
        if not os.path.exists(edge_path):
            print(f"\nWarning: Edge image not found for {f}")
            continue
            
        rgb_img = cv2.imread(rgb_path)
        edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        
        if rgb_img is None or edge_img is None:
            print(f"\nWarning: Could not read {f}")
            continue
            
        # Ensure sizes match
        if rgb_img.shape[:2] != edge_img.shape[:2]:
            print(f"\nWarning: Size mismatch for {f}. Resizing edge to match rgb.")
            edge_img = cv2.resize(edge_img, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        rgb_crops, coords = extract_crops(rgb_img, args.window_size, args.stride)
        edge_crops, _ = extract_crops(edge_img, args.window_size, args.stride)
        
        base_name, ext = os.path.splitext(f)
        
        for (rgb_crop, edge_crop, (y, x)) in zip(rgb_crops, edge_crops, coords):
            density = calculate_density(edge_crop, args.tau)
            
            # Save if density condition is met
            if density >= density_threshold:
                # Add coordinates to filename to ensure uniqueness for multiple crops per image
                crop_name = f"{base_name}_{y}_{x}{ext}"
                
                out_rgb_path = os.path.join(args.out_rgb_dir, crop_name)
                out_edge_path = os.path.join(args.out_edge_dir, crop_name)
                
                cv2.imwrite(out_rgb_path, rgb_crop)
                cv2.imwrite(out_edge_path, edge_crop)
                crop_count += 1
                
    print(f"\nDone! Saved {crop_count} total crops.")

if __name__ == "__main__":
    main()

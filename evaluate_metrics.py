"""
Python implementation of PSNR and SSIM evaluation for image deraining
Compatible with Kaggle and can evaluate results against ground truth images
"""

import os
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from glob import glob
from tqdm import tqdm
import cv2


def rgb2ycbcr(img):
    """
    Convert RGB image to YCbCr and return Y channel
    Args:
        img: numpy array in RGB format (H, W, 3)
    Returns:
        Y channel as numpy array
    """
    if len(img.shape) == 2:
        return img
    
    # Convert to float
    img = img.astype(np.float32)
    
    # RGB to YCbCr conversion matrix
    transform = np.array([[65.481, 128.553, 24.966],
                          [-37.797, -74.203, 112.0],
                          [112.0, -93.786, -18.214]])
    
    offset = np.array([16, 128, 128])
    
    # Apply transformation
    ycbcr = np.dot(img, transform.T / 255.0) + offset
    
    # Return Y channel only
    return ycbcr[:, :, 0]


def compute_psnr(img1, img2, data_range=255):
    """
    Compute PSNR between two images on Y channel
    Args:
        img1: first image (H, W, C) or (H, W)
        img2: second image (H, W, C) or (H, W)
        data_range: data range of the images (default 255)
    Returns:
        PSNR value
    """
    # Convert to Y channel if RGB
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1)
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2)
    
    # Compute PSNR
    return psnr(img1, img2, data_range=data_range)


def compute_ssim(img1, img2, data_range=255):
    """
    Compute SSIM between two images on Y channel
    Args:
        img1: first image (H, W, C) or (H, W)
        img2: second image (H, W, C) or (H, W)
        data_range: data range of the images (default 255)
    Returns:
        SSIM value
    """
    # Convert to Y channel if RGB
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1)
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2)
    
    # Compute SSIM
    return ssim(img1, img2, data_range=data_range)


def evaluate_dataset(result_dir, gt_dir, dataset_name='Dataset'):
    """
    Evaluate PSNR and SSIM for a dataset
    Args:
        result_dir: directory containing derained results
        gt_dir: directory containing ground truth images
        dataset_name: name of the dataset for display
    Returns:
        tuple: (average_psnr, average_ssim)
    """
    # Get list of result images
    result_files = sorted(glob(os.path.join(result_dir, '*.png')) + 
                         glob(os.path.join(result_dir, '*.jpg')))
    
    # Get list of ground truth images
    gt_files = sorted(glob(os.path.join(gt_dir, '*.png')) + 
                     glob(os.path.join(gt_dir, '*.jpg')))
    
    if len(result_files) == 0:
        print(f"Error: No result images found in {result_dir}")
        return 0, 0
    
    if len(gt_files) == 0:
        print(f"Error: No ground truth images found in {gt_dir}")
        return 0, 0
    
    print(f"\nEvaluating {dataset_name}...")
    print(f"Result images: {len(result_files)}")
    print(f"Ground truth images: {len(gt_files)}")
    
    total_psnr = 0
    total_ssim = 0
    num_evaluated = 0
    
    for result_file in tqdm(result_files, desc=f"Evaluating {dataset_name}"):
        result_name = os.path.basename(result_file)
        
        # Try to find matching ground truth file
        gt_file = None
        for gt_path in gt_files:
            if os.path.basename(gt_path) == result_name:
                gt_file = gt_path
                break
        
        # If exact match not found, try without extension
        if gt_file is None:
            result_basename = os.path.splitext(result_name)[0]
            for gt_path in gt_files:
                gt_basename = os.path.splitext(os.path.basename(gt_path))[0]
                if gt_basename == result_basename:
                    gt_file = gt_path
                    break
        
        if gt_file is None:
            print(f"Warning: No matching ground truth found for {result_name}")
            continue
        
        try:
            # Load images
            result_img = np.array(Image.open(result_file))
            gt_img = np.array(Image.open(gt_file))
            
            # Check if images have the same size
            if result_img.shape != gt_img.shape:
                print(f"Warning: Size mismatch for {result_name}. "
                      f"Result: {result_img.shape}, GT: {gt_img.shape}")
                # Resize result to match ground truth
                result_img = cv2.resize(result_img, (gt_img.shape[1], gt_img.shape[0]))
            
            # Compute metrics
            psnr_val = compute_psnr(result_img, gt_img)
            ssim_val = compute_ssim(result_img, gt_img)
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            num_evaluated += 1
            
        except Exception as e:
            print(f"Error processing {result_name}: {str(e)}")
            continue
    
    if num_evaluated == 0:
        print(f"Error: No images were successfully evaluated for {dataset_name}")
        return 0, 0
    
    avg_psnr = total_psnr / num_evaluated
    avg_ssim = total_ssim / num_evaluated
    
    print(f"Results for {dataset_name}:")
    print(f"  Images evaluated: {num_evaluated}/{len(result_files)}")
    print(f"  Average PSNR: {avg_psnr:.4f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description='Evaluate PSNR and SSIM for image deraining')
    
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Directory containing derained result images')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth images')
    parser.add_argument('--dataset_name', type=str, default='Test Dataset',
                       help='Name of the dataset for display')
    parser.add_argument('--multiple_datasets', action='store_true',
                       help='Evaluate multiple datasets (result_dir and gt_dir should contain subdirectories)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PSNR/SSIM Evaluation for Image Deraining")
    print("="*80)
    
    if args.multiple_datasets:
        # Evaluate multiple datasets
        datasets = sorted([d for d in os.listdir(args.result_dir) 
                          if os.path.isdir(os.path.join(args.result_dir, d))])
        
        total_psnr = 0
        total_ssim = 0
        num_datasets = 0
        
        for dataset in datasets:
            result_subdir = os.path.join(args.result_dir, dataset)
            gt_subdir = os.path.join(args.gt_dir, dataset)
            
            if not os.path.exists(gt_subdir):
                print(f"Warning: Ground truth directory not found for {dataset}: {gt_subdir}")
                continue
            
            psnr_val, ssim_val = evaluate_dataset(result_subdir, gt_subdir, dataset)
            total_psnr += psnr_val
            total_ssim += ssim_val
            num_datasets += 1
        
        if num_datasets > 0:
            print("\n" + "="*80)
            print(f"Average across all {num_datasets} datasets:")
            print(f"  PSNR: {total_psnr/num_datasets:.4f} dB")
            print(f"  SSIM: {total_ssim/num_datasets:.4f}")
            print("="*80)
    else:
        # Evaluate single dataset
        evaluate_dataset(args.result_dir, args.gt_dir, args.dataset_name)
        print("="*80)


if __name__ == '__main__':
    main()

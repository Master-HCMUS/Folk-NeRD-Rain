"""
Enhanced data augmentation for night rain deraining
Includes color jittering, noise injection, and illumination adjustments
"""

import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF


class NightRainAugmentation:
    """
    Augmentation specifically designed for night rain scenarios
    """
    def __init__(self, 
                 brightness_range=(0.7, 1.3),
                 contrast_range=(0.95, 1.05),  # ⚠️ REDUCED from (0.8, 1.2) - high contrast sharpens rain
                 saturation_range=(0.9, 1.1),  # ⚠️ REDUCED from (0.8, 1.2) - gentler color augmentation
                 gamma_range=(0.8, 1.2),
                 noise_std=0.02,
                 apply_prob=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.apply_prob = apply_prob
    
    def __call__(self, img_pair):
        """
        Apply augmentation to (target, input) pair
        Args:
            img_pair: tuple of (target, input) tensors [C, H, W]
        Returns:
            augmented (target, input) pair
        """
        target, input_img = img_pair
        
        # Apply same geometric transformations to both
        if random.random() < self.apply_prob:
            # Random horizontal flip
            if random.random() < 0.5:
                target = TF.hflip(target)
                input_img = TF.hflip(input_img)
            
            # Random vertical flip
            if random.random() < 0.5:
                target = TF.vflip(target)
                input_img = TF.vflip(input_img)
            
            # Random rotation (0, 90, 180, 270)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                target = TF.rotate(target, angle)
                input_img = TF.rotate(input_img, angle)
        
        # Apply color jittering (helps with night illumination variations)
        if random.random() < self.apply_prob:
            brightness = random.uniform(*self.brightness_range)
            contrast = random.uniform(*self.contrast_range)
            saturation = random.uniform(*self.saturation_range)
            
            target = TF.adjust_brightness(target, brightness)
            target = TF.adjust_contrast(target, contrast)
            target = TF.adjust_saturation(target, saturation)
            
            input_img = TF.adjust_brightness(input_img, brightness)
            input_img = TF.adjust_contrast(input_img, contrast)
            input_img = TF.adjust_saturation(input_img, saturation)
        
        # Gamma correction (simulates different exposure levels)
        if random.random() < self.apply_prob:
            gamma = random.uniform(*self.gamma_range)
            target = TF.adjust_gamma(target, gamma)
            input_img = TF.adjust_gamma(input_img, gamma)
        
        # Add Gaussian noise to input only (simulates sensor noise in low light)
        if random.random() < self.apply_prob and self.noise_std > 0:
            noise = torch.randn_like(input_img) * self.noise_std
            input_img = torch.clamp(input_img + noise, 0, 1)
        
        return target, input_img


class MixupAugmentation:
    """
    Mixup augmentation for better generalization
    """
    def __init__(self, alpha=0.2, apply_prob=0.3):
        self.alpha = alpha
        self.apply_prob = apply_prob
    
    def __call__(self, batch_target, batch_input):
        """
        Apply mixup to a batch
        Args:
            batch_target: [B, C, H, W]
            batch_input: [B, C, H, W]
        Returns:
            mixed (target, input) batch
        """
        if random.random() > self.apply_prob or batch_target.size(0) < 2:
            return batch_target, batch_input
        
        batch_size = batch_target.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(batch_target.device)
        
        # Mix targets and inputs
        mixed_target = lam * batch_target + (1 - lam) * batch_target[index]
        mixed_input = lam * batch_input + (1 - lam) * batch_input[index]
        
        return mixed_target, mixed_input


def apply_retinex_decomposition(img, sigma=15):
    """
    Apply Retinex-based illumination normalization
    Helps with low-light enhancement
    Args:
        img: [B, C, H, W] or [C, H, W]
    Returns:
        illumination-normalized image
    """
    import torch.nn.functional as F
    
    # Gaussian blur to estimate illumination
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    x_cord = torch.arange(kernel_size, dtype=torch.float32)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    
    gaussian_kernel = torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape for convolution
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)
    
    # Estimate illumination
    padding = kernel_size // 2
    illumination = F.conv2d(img, gaussian_kernel.to(img.device), 
                           padding=padding, groups=3)
    
    # Reflectance = image / illumination
    reflectance = img / (illumination + 0.01)
    
    return reflectance

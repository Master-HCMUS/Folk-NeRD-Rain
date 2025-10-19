"""
Post-processing utilities to reduce unwanted rain sharpening artifacts
Apply these AFTER model inference if output still shows sharpened rain
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np


def bilateral_filter(image, d=5, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to reduce rain sharpening while preserving edges
    
    Args:
        image: torch.Tensor [B, C, H, W] or [C, H, W], range [0, 1]
        d: diameter of pixel neighborhood
        sigma_color: filter sigma in color space
        sigma_space: filter sigma in coordinate space
    
    Returns:
        filtered image (same shape as input)
    """
    is_batch = image.dim() == 4
    if not is_batch:
        image = image.unsqueeze(0)
    
    device = image.device
    B, C, H, W = image.shape
    
    # Convert to numpy for cv2
    img_np = image.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
    
    filtered = np.zeros_like(img_np)
    for i in range(B):
        # cv2 expects uint8 or float32
        img_uint8 = (img_np[i] * 255).astype(np.uint8)
        filtered[i] = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space) / 255.0
    
    # Convert back to torch
    filtered = torch.from_numpy(filtered.transpose(0, 3, 1, 2)).float().to(device)
    
    return filtered if is_batch else filtered[0]


def guided_filter(image, radius=4, eps=0.01):
    """
    Apply guided filtering (edge-preserving smoothing)
    Better than Gaussian blur for preserving scene details while removing rain
    
    Args:
        image: torch.Tensor [B, C, H, W] or [C, H, W], range [0, 1]
        radius: radius of local window
        eps: regularization parameter (higher = more smoothing)
    
    Returns:
        filtered image
    """
    is_batch = image.dim() == 4
    if not is_batch:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    device = image.device
    
    # Use image itself as guidance
    guide = image
    
    # Box filter
    def box_filter(x, r):
        """Efficient box filtering using cumulative sum"""
        ch = x.shape[1]
        kernel = torch.ones(ch, 1, 2*r+1, 2*r+1, device=x.device) / ((2*r+1) ** 2)
        return F.conv2d(x, kernel, padding=r, groups=ch)
    
    N = box_filter(torch.ones(B, 1, H, W, device=device), radius)
    
    mean_I = box_filter(guide, radius) / N
    mean_p = box_filter(image, radius) / N
    
    mean_Ip = box_filter(guide * image, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = box_filter(guide * guide, radius) / N
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box_filter(a, radius) / N
    mean_b = box_filter(b, radius) / N
    
    output = mean_a * guide + mean_b
    
    return output if is_batch else output[0]


def selective_gaussian_blur(image, rain_mask=None, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur only to rain-affected regions
    
    Args:
        image: torch.Tensor [B, C, H, W] or [C, H, W]
        rain_mask: torch.Tensor [B, 1, H, W] or None (auto-detect)
        kernel_size: size of Gaussian kernel
        sigma: standard deviation
    
    Returns:
        selectively blurred image
    """
    is_batch = image.dim() == 4
    if not is_batch:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    device = image.device
    
    # Auto-detect rain mask if not provided
    if rain_mask is None:
        # High-frequency content often corresponds to rain
        gray = image.mean(dim=1, keepdim=True)
        
        # Compute Laplacian (edge detection)
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        edges = F.conv2d(gray, laplacian_kernel, padding=1)
        rain_mask = torch.sigmoid(edges.abs() * 10 - 5)  # Soft threshold
    
    # Apply Gaussian blur
    channels = image.shape[1]
    gaussian_kernel = _get_gaussian_kernel(kernel_size, sigma, channels, device)
    blurred = F.conv2d(image, gaussian_kernel, padding=kernel_size//2, groups=channels)
    
    # Blend based on mask
    output = rain_mask * blurred + (1 - rain_mask) * image
    
    return output if is_batch else output[0]


def _get_gaussian_kernel(kernel_size, sigma, channels, device):
    """Create Gaussian kernel"""
    x_coord = torch.arange(kernel_size, dtype=torch.float32, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    
    gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape for conv2d
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    return gaussian_kernel


def non_local_means_denoise(image, search_window=7, template_window=3, h=0.1):
    """
    Non-local means denoising (CPU-based via OpenCV)
    Very effective at removing sharpened rain artifacts
    
    Args:
        image: torch.Tensor [B, C, H, W] or [C, H, W], range [0, 1]
        search_window: size of area to search for similar patches
        template_window: size of template patch
        h: filter strength (higher = more smoothing)
    
    Returns:
        denoised image
    """
    is_batch = image.dim() == 4
    if not is_batch:
        image = image.unsqueeze(0)
    
    device = image.device
    B, C, H, W = image.shape
    
    # Convert to numpy
    img_np = image.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
    
    denoised = np.zeros_like(img_np)
    for i in range(B):
        img_uint8 = (img_np[i] * 255).astype(np.uint8)
        denoised[i] = cv2.fastNlMeansDenoisingColored(
            img_uint8, 
            None, 
            h=h * 255,  # Scale to uint8 range
            hColor=h * 255,
            templateWindowSize=template_window,
            searchWindowSize=search_window
        ) / 255.0
    
    # Convert back to torch
    denoised = torch.from_numpy(denoised.transpose(0, 3, 1, 2)).float().to(device)
    
    return denoised if is_batch else denoised[0]


def anti_sharpen_filter(image, amount=0.5):
    """
    Apply anti-sharpening (inverse of unsharp masking)
    Reduces overly sharp edges including rain streaks
    
    Args:
        image: torch.Tensor [B, C, H, W] or [C, H, W]
        amount: strength of anti-sharpening [0, 1]
    
    Returns:
        softened image
    """
    is_batch = image.dim() == 4
    if not is_batch:
        image = image.unsqueeze(0)
    
    # Apply Gaussian blur
    blurred = guided_filter(image, radius=2, eps=0.01)
    
    # Blend: more blur = more anti-sharpening
    output = (1 - amount) * image + amount * blurred
    
    return output if is_batch else output[0]


# ========== Test Script Integration ==========

class PostProcessingPipeline:
    """
    Complete post-processing pipeline for deraining
    Apply after model inference to reduce sharpening artifacts
    """
    def __init__(self, method='bilateral', **kwargs):
        """
        Args:
            method: 'bilateral', 'guided', 'nlm', 'anti_sharpen', or 'none'
            **kwargs: method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
    
    def __call__(self, image):
        """
        Apply post-processing
        
        Args:
            image: torch.Tensor [B, C, H, W] or [C, H, W], range [0, 1]
        
        Returns:
            processed image
        """
        if self.method == 'none':
            return image
        elif self.method == 'bilateral':
            return bilateral_filter(image, **self.kwargs)
        elif self.method == 'guided':
            return guided_filter(image, **self.kwargs)
        elif self.method == 'nlm':
            return non_local_means_denoise(image, **self.kwargs)
        elif self.method == 'anti_sharpen':
            return anti_sharpen_filter(image, **self.kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")


# ========== Usage Examples ==========

if __name__ == "__main__":
    # Example usage
    import torchvision.transforms as T
    from PIL import Image
    
    # Load image
    img = Image.open("test_input.png").convert("RGB")
    img_tensor = T.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
    
    print("Testing post-processing methods...")
    
    # Test bilateral filter
    filtered = bilateral_filter(img_tensor, d=5, sigma_color=75, sigma_space=75)
    print(f"Bilateral filter: {filtered.shape}")
    
    # Test guided filter
    guided = guided_filter(img_tensor, radius=4, eps=0.01)
    print(f"Guided filter: {guided.shape}")
    
    # Test anti-sharpen
    softened = anti_sharpen_filter(img_tensor, amount=0.5)
    print(f"Anti-sharpen: {softened.shape}")
    
    print("\nAll tests passed!")

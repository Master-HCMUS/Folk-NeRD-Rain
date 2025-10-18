"""
Enhanced loss functions for night rain deraining
Includes perceptual loss, SSIM loss, and illumination-aware losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.models import vgg16
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False
    print("Warning: torchvision not available, perceptual loss disabled")
from math import exp


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    Critical for preserving semantic content in low-light conditions
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], 
                 weights=[1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        
        if not VGG_AVAILABLE:
            raise ImportError("VGG16 requires torchvision. Install with: pip install torchvision")
        
        self.layers = layers
        self.weights = weights
        self.device = None  # Will be set on first forward pass
        
        # Load pretrained VGG16
        try:
            vgg = vgg16(pretrained=True).features.eval()
        except:
            # Fallback for newer torchvision versions
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        
        # Extract specific layers
        self.slices = nn.ModuleList()
        layer_map = {
            'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16, 'relu4_3': 23
        }
        
        prev_idx = 0
        for layer_name in layers:
            idx = layer_map[layer_name]
            self.slices.append(vgg[prev_idx:idx+1])
            prev_idx = idx + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted image [B, 3, H, W]
            target: ground truth [B, 3, H, W]
        Returns:
            perceptual loss value
        """
        # Move VGG to same device as input (only once)
        if self.device != pred.device:
            self.device = pred.device
            self.slices = self.slices.to(pred.device)
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        loss = 0.0
        pred_features = pred_norm
        target_features = target_norm
        
        for i, slice_layer in enumerate(self.slices):
            pred_features = slice_layer(pred_features)
            target_features = slice_layer(target_features)
            loss += self.weights[i] * F.mse_loss(pred_features, target_features)
        
        return loss


class SSIMLoss(nn.Module):
    """
    SSIM loss for structural similarity
    More perceptually aligned than MSE, especially for low-light images
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                             for x in range(window_size)])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, 
                            groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, 
                            groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, 
                          groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)


class IlluminationAwareLoss(nn.Module):
    """
    Loss that adapts to different illumination levels
    Gives more weight to darker regions (common in night scenes)
    """
    def __init__(self, loss_type='l1'):
        super(IlluminationAwareLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted image [B, C, H, W]
            target: ground truth [B, C, H, W]
        Returns:
            illumination-weighted loss
        """
        # Compute illumination map (average across channels)
        illumination = target.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Weight darker regions more heavily
        # Use inverse illumination as weight
        weight = 1.0 / (illumination + 0.1)  # Add small epsilon
        weight = weight / weight.mean()  # Normalize
        
        # Compute weighted loss
        if self.loss_type == 'l1':
            loss = torch.abs(pred - target)
        elif self.loss_type == 'l2':
            loss = (pred - target) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        weighted_loss = (loss * weight).mean()
        return weighted_loss


class ColorConstancyLoss(nn.Module):
    """
    Ensures color consistency, important for night scenes with artificial lighting
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()
    
    def forward(self, pred):
        """
        Encourage gray-world assumption
        Args:
            pred: predicted image [B, C, H, W]
        Returns:
            color constancy loss
        """
        mean_rgb = torch.mean(pred, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Compute differences between channels
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        
        return torch.sqrt(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2))


class CombinedNightRainLoss(nn.Module):
    """
    Combined loss function optimized for night rain deraining
    """
    def __init__(self, 
                 use_perceptual=True,
                 use_ssim=True,
                 use_illumination=True,
                 use_color_constancy=True,
                 char_weight=1.0,
                 fft_weight=0.01,
                 edge_weight=0.05,
                 perceptual_weight=0.1,
                 ssim_weight=0.5,
                 illumination_weight=0.3,
                 color_weight=0.01):
        super(CombinedNightRainLoss, self).__init__()
        
        # Import existing losses
        from losses import CharbonnierLoss, EdgeLoss, fftLoss
        
        self.char_loss = CharbonnierLoss()
        self.fft_loss = fftLoss()
        self.edge_loss = EdgeLoss()
        
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        self.use_illumination = use_illumination
        self.use_color_constancy = use_color_constancy
        
        if use_perceptual:
            if not VGG_AVAILABLE:
                print("Warning: VGG16 not available, disabling perceptual loss")
                self.use_perceptual = False
            else:
                self.perceptual_loss = PerceptualLoss()
        if use_ssim:
            self.ssim_loss = SSIMLoss()
        if use_illumination:
            self.illumination_loss = IlluminationAwareLoss()
        if use_color_constancy:
            self.color_loss = ColorConstancyLoss()
        
        self.weights = {
            'char': char_weight,
            'fft': fft_weight,
            'edge': edge_weight,
            'perceptual': perceptual_weight,
            'ssim': ssim_weight,
            'illumination': illumination_weight,
            'color': color_weight
        }
    
    def forward(self, pred, target):
        """
        Compute combined loss
        Args:
            pred: predicted image or list of multi-scale predictions
            target: ground truth or list of multi-scale targets
        Returns:
            total loss and dict of individual losses
        """
        losses = {}
        
        # Handle multi-scale inputs
        if isinstance(pred, (list, tuple)):
            # Compute losses for each scale
            total_loss = 0
            for p, t in zip(pred[:3], target[:3]):  # First 3 scales
                losses['char'] = losses.get('char', 0) + self.char_loss(p, t)
                losses['fft'] = losses.get('fft', 0) + self.fft_loss(p, t)
                losses['edge'] = losses.get('edge', 0) + self.edge_loss(p, t)
            
            # Additional losses on finest scale only
            pred_finest = pred[0]
            target_finest = target[0]
        else:
            pred_finest = pred
            target_finest = target
            losses['char'] = self.char_loss(pred_finest, target_finest)
            losses['fft'] = self.fft_loss(pred_finest, target_finest)
            losses['edge'] = self.edge_loss(pred_finest, target_finest)
        
        # Enhanced losses for night rain
        if self.use_perceptual:
            losses['perceptual'] = self.perceptual_loss(pred_finest, target_finest)
        
        if self.use_ssim:
            losses['ssim'] = self.ssim_loss(pred_finest, target_finest)
        
        if self.use_illumination:
            losses['illumination'] = self.illumination_loss(pred_finest, target_finest)
        
        if self.use_color_constancy:
            losses['color'] = self.color_loss(pred_finest)
        
        # Compute weighted total
        total_loss = sum(losses[k] * self.weights[k] for k in losses.keys())
        
        return total_loss, losses

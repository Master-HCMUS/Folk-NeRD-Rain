"""
Enhanced architecture components for night rain deraining
Includes channel attention, spatial attention, and adaptive feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Helps model focus on informative channels (e.g., preserving color in night scenes)
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Helps model focus on rain-affected regions vs clean regions
    
    WARNING: Max pooling can focus on high-contrast rain streaks!
    Use with caution for deraining tasks.
    """
    def __init__(self, kernel_size=7, use_max=True):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        # Use only 1 channel if not using max (avg only)
        in_channels = 2 if use_max else 1
        self.conv = nn.Conv2d(in_channels, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.use_max = use_max
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        if self.use_max:
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
        else:
            # For deraining: use only avg to avoid focusing on high-contrast rain
            x_cat = avg_out
        
        attention = self.sigmoid(self.conv(x_cat))
        # Soften attention to avoid over-emphasis
        attention = 0.5 + 0.5 * attention  # Range [0.5, 1.0] instead of [0, 1]
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention
    
    For deraining: uses gentler spatial attention without max pooling
    to avoid focusing on high-contrast rain streaks.
    """
    def __init__(self, channels, reduction=16, kernel_size=7, use_max_spatial=False):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        # For deraining, disable max pooling in spatial attention
        self.spatial_attention = SpatialAttention(kernel_size, use_max=use_max_spatial)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion for multi-scale features
    Learns optimal fusion weights based on feature statistics
    """
    def __init__(self, channels):
        super(AdaptiveFeatureFusion, self).__init__()
        self.weight_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, feat1, feat2):
        """
        Adaptively fuse two feature maps
        Args:
            feat1, feat2: [B, C, H, W]
        Returns:
            fused features [B, C, H, W]
        """
        # Ensure same spatial size
        if feat1.shape[2:] != feat2.shape[2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], 
                                 mode='bilinear', align_corners=False)
        
        # Concatenate for weight prediction
        concat = torch.cat([feat1, feat2], dim=1)
        weights = self.weight_conv(concat)  # [B, 2, 1, 1]
        
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        
        fused = w1 * feat1 + w2 * feat2
        return fused


class EnhancedResBlock(nn.Module):
    """
    Enhanced Residual Block with CBAM attention
    Better feature extraction for night rain scenarios
    """
    def __init__(self, channels, use_attention=True):
        super(EnhancedResBlock, self).__init__()
        from layers import BasicConv
        
        self.conv1 = BasicConv(channels, channels, kernel_size=3, stride=1, 
                              relu=True, norm=False)
        self.conv2 = BasicConv(channels, channels, kernel_size=3, stride=1, 
                              relu=False, norm=False)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        return out + residual


class DynamicConv(nn.Module):
    """
    Dynamic Convolution - adapts kernel based on input
    Useful for handling diverse night lighting conditions
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 num_experts=4, reduction=4):
        super(DynamicConv, self).__init__()
        self.num_experts = num_experts
        
        # Multiple expert convolutions
        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     padding=kernel_size//2, bias=False)
            for _ in range(num_experts)
        ])
        
        # Attention for expert selection
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, num_experts, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Get attention weights for experts
        weights = self.attention(x)  # [B, num_experts, 1, 1]
        
        # Apply each expert
        expert_outs = [expert(x) for expert in self.experts]
        expert_outs = torch.stack(expert_outs, dim=1)  # [B, num_experts, C, H, W]
        
        # Weighted combination
        weights = weights.unsqueeze(2)  # [B, num_experts, 1, 1, 1]
        out = (expert_outs * weights).sum(dim=1)  # [B, C, H, W]
        
        return out


class LowLightEnhancementModule(nn.Module):
    """
    Dedicated module for low-light enhancement
    Can be inserted before the main deraining network
    
    WARNING: This module enhances brightness BEFORE deraining, which can
    inadvertently amplify rain streaks. For night rain deraining, consider:
    1. Using this module AFTER deraining (not before)
    2. Reducing enhancement strength (currently 0.3)
    3. Disabling this module entirely and relying on illumination-aware loss
    
    Recommended: Do NOT add this to the forward pass of deraining models.
    Use illumination_aware_loss instead during training.
    """
    def __init__(self, channels=32):
        super(LowLightEnhancementModule, self).__init__()
        
        # Illumination estimation branch
        self.illum_conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.illum_conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.illum_out = nn.Conv2d(channels, 3, 3, padding=1)
        
        # Enhancement branch
        self.enhance_conv1 = nn.Conv2d(6, channels, 3, padding=1)  # concat input + illum
        self.enhance_conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.enhance_out = nn.Conv2d(channels, 3, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Enhance low-light image
        Args:
            x: input image [B, 3, H, W]
        Returns:
            enhanced image [B, 3, H, W]
        """
        # Estimate illumination map
        illum = self.relu(self.illum_conv1(x))
        illum = self.relu(self.illum_conv2(illum))
        illum = self.sigmoid(self.illum_out(illum))
        
        # Enhancement
        concat = torch.cat([x, illum], dim=1)
        enhance = self.relu(self.enhance_conv1(concat))
        enhance = self.relu(self.enhance_conv2(enhance))
        enhance = self.sigmoid(self.enhance_out(enhance))
        
        # Residual connection
        return x + enhance * 0.3  # Gentle enhancement


class NightRainDecoderHead(nn.Module):
    """
    Specialized decoder head for night rain
    Uses progressive refinement with attention
    """
    def __init__(self, in_channels, out_channels=3):
        super(NightRainDecoderHead, self).__init__()
        
        self.refine1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(in_channels // 2)
        )
        
        self.refine2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(in_channels // 4)
        )
        
        self.out_conv = nn.Conv2d(in_channels // 4, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.refine1(x)
        x = self.refine2(x)
        x = self.sigmoid(self.out_conv(x))
        return x

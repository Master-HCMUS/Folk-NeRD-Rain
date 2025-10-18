"""
Enhanced training strategies for night rain deraining
Includes progressive training, curriculum learning, and adaptive learning rate
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing
    Better convergence for difficult night rain scenarios
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 min_lr=1e-6, max_lr=1e-4):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class ProgressiveTraining:
    """
    Progressive training strategy: train from easy to hard patches
    """
    def __init__(self, initial_patch_size=128, target_patch_size=256, 
                 transition_epoch=50):
        self.initial_size = initial_patch_size
        self.target_size = target_patch_size
        self.transition_epoch = transition_epoch
        self.current_size = initial_patch_size
    
    def get_patch_size(self, epoch):
        """Get current patch size based on epoch"""
        if epoch < self.transition_epoch:
            # Gradually increase patch size
            progress = epoch / self.transition_epoch
            size_diff = self.target_size - self.initial_size
            self.current_size = int(self.initial_size + size_diff * progress)
            # Ensure even number
            self.current_size = (self.current_size // 2) * 2
        else:
            self.current_size = self.target_size
        
        return self.current_size


class GradientAccumulator:
    """
    Efficient gradient accumulation for larger effective batch sizes
    """
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def step(self, loss):
        """
        Accumulate gradients and step when ready
        Args:
            loss: current loss value
        Returns:
            whether optimizer stepped
        """
        # Normalize loss for accumulation
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.current_step += 1
        
        if self.current_step >= self.accumulation_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.current_step = 0
            return True
        
        return False
    
    def reset(self):
        """Reset accumulation counter"""
        if self.current_step > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0


class ExponentialMovingAverage:
    """
    EMA of model weights for more stable evaluation
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + \
                             self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AdaptiveLossWeighting:
    """
    Dynamically adjust loss weights based on training progress
    """
    def __init__(self, initial_weights, adjust_every=10):
        self.weights = initial_weights.copy()
        self.loss_history = {k: [] for k in initial_weights.keys()}
        self.adjust_every = adjust_every
        self.step_count = 0
    
    def update(self, loss_dict):
        """
        Update loss history and adjust weights
        Args:
            loss_dict: dict of individual losses
        Returns:
            updated weights
        """
        for k, v in loss_dict.items():
            if k in self.loss_history:
                self.loss_history[k].append(v.item() if torch.is_tensor(v) else v)
        
        self.step_count += 1
        
        if self.step_count % self.adjust_every == 0:
            # Adjust weights based on loss magnitudes
            for k in self.weights.keys():
                if len(self.loss_history[k]) > 0:
                    # Get recent average
                    recent_loss = np.mean(self.loss_history[k][-10:])
                    
                    # If loss is very small, reduce its weight
                    if recent_loss < 1e-4:
                        self.weights[k] *= 0.9
                    # If loss is large, increase its weight slightly
                    elif recent_loss > 1.0:
                        self.weights[k] *= 1.05
                    
                    # Keep weights in reasonable range
                    self.weights[k] = np.clip(self.weights[k], 1e-4, 10.0)
        
        return self.weights


class CurriculumDataSampler:
    """
    Sample training data from easy to hard
    Sort by rain density/complexity
    """
    def __init__(self, dataset, num_epochs, warmup_epochs=20):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Precompute difficulty scores for each sample
        # (In practice, you'd analyze rain density, contrast, etc.)
        self.difficulties = torch.rand(len(dataset))  # Placeholder
    
    def get_sampler(self, epoch):
        """
        Get data sampler for current epoch
        Args:
            epoch: current training epoch
        Returns:
            sampler indices
        """
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Sample easier examples first
            threshold = (epoch / self.warmup_epochs)
            valid_indices = torch.where(self.difficulties < threshold)[0]
            
            if len(valid_indices) < 100:  # Minimum samples
                valid_indices = torch.argsort(self.difficulties)[:1000]
        else:
            # Use all samples
            valid_indices = torch.arange(len(self.dataset))
        
        return valid_indices.tolist()


def get_optimizer_with_layer_lr(model, base_lr=1e-4, 
                                encoder_lr_mult=0.1, decoder_lr_mult=1.0):
    """
    Create optimizer with different learning rates for different parts
    Lower LR for encoder (pretrained features), higher for decoder
    """
    encoder_params = []
    decoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'encoder' in name or 'downsample' in name:
            encoder_params.append(param)
        elif 'decoder' in name or 'upsample' in name or 'inr' in name.lower():
            decoder_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': base_lr * encoder_lr_mult},
        {'params': decoder_params, 'lr': base_lr * decoder_lr_mult},
        {'params': other_params, 'lr': base_lr}
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    
    return optimizer

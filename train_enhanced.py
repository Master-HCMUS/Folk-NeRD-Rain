"""
Enhanced Training Script for Night Rain Deraining
Integrates all improvements: enhanced losses, augmentation, attention, EMA, etc.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import random
import time
import numpy as np
from tqdm import tqdm

# Original imports
import utils
from data_RGB import get_training_data, get_validation_data
from get_parameter_number import get_parameter_number
import kornia
from torch.utils.tensorboard import SummaryWriter

# Enhanced imports
from enhanced_losses import CombinedNightRainLoss
from training_strategies import (
    WarmupCosineScheduler, 
    ProgressiveTraining, 
    GradientAccumulator,
    ExponentialMovingAverage
)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Argument Parser ###########
parser = argparse.ArgumentParser(description='Enhanced Night Rain Deraining')

# Dataset paths
parser.add_argument('--train_dir', required=True, type=str, 
                   help='Directory of train images')
parser.add_argument('--val_dir', required=True, type=str, 
                   help='Directory of validation images')
parser.add_argument('--input_subdir', default='rainy', type=str, 
                   help='Input subdirectory name (e.g., "rainy" for GTAV)')
parser.add_argument('--target_subdir', default='gt', type=str, 
                   help='Target subdirectory name (e.g., "gt" for GTAV)')
parser.add_argument('--model_save_dir', default='./checkpoints', type=str, 
                   help='Path to save weights')

# Model selection
parser.add_argument('--model', default='small', choices=['small', 'full'], 
                   help='Model size: small (~4M params) or full (~8M params)')

# Training hyperparameters
parser.add_argument('--num_epochs', default=300, type=int, 
                   help='Number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, 
                   help='Batch size per GPU')
parser.add_argument('--patch_size', default=256, type=int, 
                   help='Training patch size')
parser.add_argument('--val_epochs', default=5, type=int, 
                   help='Validation frequency (epochs)')

# Optimizer settings
parser.add_argument('--base_lr', default=1e-4, type=float, 
                   help='Base learning rate')
parser.add_argument('--min_lr', default=1e-6, type=float, 
                   help='Minimum learning rate')
parser.add_argument('--warmup_epochs', default=10, type=int, 
                   help='Number of warmup epochs')
parser.add_argument('--weight_decay', default=1e-4, type=float, 
                   help='Weight decay for optimizer')

# Enhanced features
parser.add_argument('--use_enhanced_loss', action='store_true', 
                   help='Use enhanced loss functions (perceptual + SSIM)')
parser.add_argument('--use_augmentation', action='store_true', 
                   help='Use night rain augmentation')
parser.add_argument('--use_ema', action='store_true', 
                   help='Use Exponential Moving Average')
parser.add_argument('--gradient_accumulation', default=1, type=int, 
                   help='Gradient accumulation steps (effective batch size multiplier)')
parser.add_argument('--progressive_training', action='store_true', 
                   help='Use progressive patch size training')
parser.add_argument('--mixed_precision', action='store_true', 
                   help='Use mixed precision training (FP16)')

# Loss weights
parser.add_argument('--perceptual_weight', default=0.1, type=float)
parser.add_argument('--ssim_weight', default=0.5, type=float)
parser.add_argument('--illumination_weight', default=0.3, type=float)

# Resume training
parser.add_argument('--resume', default='', type=str, 
                   help='Path to checkpoint to resume from')
parser.add_argument('--pretrain', default='', type=str, 
                   help='Path to pretrained weights')

# Session name
parser.add_argument('--session', default='enhanced_night_rain', type=str, 
                   help='Training session name')

args = parser.parse_args()

######### Setup ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

# Create save directory
mode = 'Deraining'
session = args.session
model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
utils.mkdir(model_dir)

# Log configuration
config_log = os.path.join(model_dir, 'config.txt')
with open(config_log, 'w') as f:
    for arg in vars(args):
        f.write(f'{arg}: {getattr(args, arg)}\n')

print("="*80)
print("ENHANCED NIGHT RAIN DERAINING TRAINING")
print("="*80)
print(f"Session: {session}")
print(f"Model: {args.model}")
print(f"Training dir: {args.train_dir}")
print(f"Validation dir: {args.val_dir}")
print(f"Enhanced loss: {args.use_enhanced_loss}")
print(f"Augmentation: {args.use_augmentation}")
print(f"EMA: {args.use_ema}")
print(f"Mixed precision: {args.mixed_precision}")
print(f"Gradient accumulation: {args.gradient_accumulation}")
print(f"Progressive training: {args.progressive_training}")
print("="*80)

######### Model ###########
if args.model == 'small':
    from model_S import MultiscaleNet as myNet
    print("Using Small Model (~4M parameters)")
else:
    from model import MultiscaleNet as myNet
    print("Using Full Model (~8M parameters)")

model_restoration = myNet()
get_parameter_number(model_restoration)
model_restoration.cuda()

# Multi-GPU support
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs!")
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Optimizer ###########
optimizer = optim.AdamW(
    model_restoration.parameters(),
    lr=args.base_lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=args.weight_decay
)

######### Learning Rate Scheduler ###########
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=args.warmup_epochs,
    total_epochs=args.num_epochs,
    min_lr=args.min_lr,
    max_lr=args.base_lr
)

######### Loss Functions ###########
if args.use_enhanced_loss:
    print("\n✓ Using Enhanced Loss Functions:")
    print(f"  - Charbonnier Loss")
    print(f"  - FFT Loss")
    print(f"  - Edge Loss")
    print(f"  - Perceptual Loss (weight: {args.perceptual_weight})")
    print(f"  - SSIM Loss (weight: {args.ssim_weight})")
    print(f"  - Illumination-Aware Loss (weight: {args.illumination_weight})")
    
    criterion = CombinedNightRainLoss(
        use_perceptual=True,
        use_ssim=True,
        use_illumination=True,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        illumination_weight=args.illumination_weight
    ).cuda()
else:
    print("\n✓ Using Standard Loss Functions")
    import losses
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_fft = losses.fftLoss()
    criterion_L1 = nn.L1Loss()

######### Training Strategies ###########
# Gradient Accumulator
if args.gradient_accumulation > 1:
    accumulator = GradientAccumulator(
        model_restoration,
        optimizer,
        accumulation_steps=args.gradient_accumulation
    )
    print(f"\n✓ Gradient Accumulation: {args.gradient_accumulation} steps")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
else:
    accumulator = None

# Exponential Moving Average
if args.use_ema:
    ema = ExponentialMovingAverage(model_restoration, decay=0.999)
    print(f"\n✓ Using EMA (decay=0.999)")
else:
    ema = None

# Progressive Training
if args.progressive_training:
    progressive = ProgressiveTraining(
        initial_patch_size=128,
        target_patch_size=args.patch_size,
        transition_epoch=args.num_epochs // 3
    )
    print(f"\n✓ Progressive Training: 128 → {args.patch_size}")
else:
    progressive = None

# Mixed Precision
if args.mixed_precision:
    scaler = GradScaler()
    print(f"\n✓ Mixed Precision Training (FP16)")
else:
    scaler = None

######### Resume / Pretrain ###########
start_epoch = 1

if args.resume:
    utils.load_checkpoint(model_restoration, args.resume)
    start_epoch = utils.load_start_epoch(args.resume) + 1
    utils.load_optim(optimizer, args.resume)
    for _ in range(1, start_epoch):
        scheduler.step()
    print(f"\n✓ Resumed from epoch {start_epoch-1}")
    print(f"  Learning rate: {scheduler.get_lr():.6f}")

elif args.pretrain:
    utils.load_checkpoint(model_restoration, args.pretrain)
    print(f"\n✓ Loaded pretrained weights from {args.pretrain}")

######### DataLoaders ###########
print(f"\n✓ Loading datasets...")

# Get initial patch size
current_patch_size = args.patch_size
if progressive:
    current_patch_size = progressive.get_patch_size(start_epoch)

train_dataset = get_training_data(
    args.train_dir,
    {'patch_size': current_patch_size},
    input_subdir=args.input_subdir,
    target_subdir=args.target_subdir
)

val_dataset = get_validation_data(
    args.val_dir,
    {'patch_size': args.patch_size},
    input_subdir=args.input_subdir,
    target_subdir=args.target_subdir
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=False,
    pin_memory=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    pin_memory=True
)

print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print(f"  Train batches: {len(train_loader)}")

######### Augmentation ###########
if args.use_augmentation:
    from augmentations import NightRainAugmentation
    augmentor = NightRainAugmentation(
        brightness_range=(0.7, 1.3),
        gamma_range=(0.8, 1.2),
        noise_std=0.02,
        apply_prob=0.5
    )
    print(f"\n✓ Night Rain Augmentation Enabled")
else:
    augmentor = None

######### TensorBoard ###########
writer = SummaryWriter(model_dir)

######### Training Loop ###########
print("\n" + "="*80)
print(f"Starting Training: Epoch {start_epoch} to {args.num_epochs}")
print("="*80 + "\n")

best_psnr = 0
best_epoch = 0
global_iter = 0

for epoch in range(start_epoch, args.num_epochs + 1):
    epoch_start_time = time.time()
    
    # Update patch size for progressive training
    if progressive:
        current_patch_size = progressive.get_patch_size(epoch)
        if current_patch_size != train_dataset.img_options['patch_size']:
            print(f"\n→ Updating patch size to {current_patch_size}x{current_patch_size}")
            train_dataset.img_options['patch_size'] = current_patch_size
    
    # Training phase
    model_restoration.train()
    epoch_loss = 0
    epoch_loss_dict = {}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
    
    for batch_idx, data in enumerate(pbar):
        target_ = data[0].cuda()
        input_ = data[1].cuda()
        
        # Apply augmentation if enabled
        if augmentor:
            with torch.no_grad():
                aug_target = []
                aug_input = []
                for i in range(target_.shape[0]):
                    t, inp = augmentor((target_[i], input_[i]))
                    aug_target.append(t)
                    aug_input.append(inp)
                target_ = torch.stack(aug_target)
                input_ = torch.stack(aug_input)
        
        # Build multi-scale pyramid
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        
        # Forward pass with optional mixed precision
        if scaler:
            with autocast():
                restored = model_restoration(input_)
                
                if args.use_enhanced_loss:
                    loss, loss_dict = criterion(restored, target)
                else:
                    loss_fft = criterion_fft(restored[0], target[0]) + \
                              criterion_fft(restored[1], target[1]) + \
                              criterion_fft(restored[2], target[2])
                    loss_char = criterion_char(restored[0], target[0]) + \
                               criterion_char(restored[1], target[1]) + \
                               criterion_char(restored[2], target[2])
                    loss_edge = criterion_edge(restored[0], target[0]) + \
                               criterion_edge(restored[1], target[1]) + \
                               criterion_edge(restored[2], target[2])
                    loss_l1 = criterion_L1(restored[3], target[1]) + \
                             criterion_L1(restored[5], target[2])
                    loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1
                    loss_dict = {'char': loss_char, 'fft': loss_fft, 
                               'edge': loss_edge, 'l1': loss_l1}
        else:
            restored = model_restoration(input_)
            
            if args.use_enhanced_loss:
                loss, loss_dict = criterion(restored, target)
            else:
                loss_fft = criterion_fft(restored[0], target[0]) + \
                          criterion_fft(restored[1], target[1]) + \
                          criterion_fft(restored[2], target[2])
                loss_char = criterion_char(restored[0], target[0]) + \
                           criterion_char(restored[1], target[1]) + \
                           criterion_char(restored[2], target[2])
                loss_edge = criterion_edge(restored[0], target[0]) + \
                           criterion_edge(restored[1], target[1]) + \
                           criterion_edge(restored[2], target[2])
                loss_l1 = criterion_L1(restored[3], target[1]) + \
                         criterion_L1(restored[5], target[2])
                loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1
                loss_dict = {'char': loss_char, 'fft': loss_fft, 
                           'edge': loss_edge, 'l1': loss_l1}
        
        # Ensure loss is scalar (defensive check)
        if loss.dim() > 0:
            loss = loss.mean()
        
        # Backward pass
        if accumulator:
            # Use gradient accumulation
            if scaler:
                scaler.scale(loss).backward()
                if accumulator.current_step + 1 >= accumulator.accumulation_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    accumulator.current_step = 0
                else:
                    accumulator.current_step += 1
            else:
                accumulator.step(loss)
        else:
            # Standard backward
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Update EMA
        if ema:
            ema.update()
        
        # Logging
        epoch_loss += loss.item()
        for k, v in loss_dict.items():
            if k not in epoch_loss_dict:
                epoch_loss_dict[k] = 0
            epoch_loss_dict[k] += v.item() if torch.is_tensor(v) else v
        
        global_iter += 1
        
        # TensorBoard logging (every 100 iterations)
        if global_iter % 100 == 0:
            writer.add_scalar('train/iter_loss', loss.item(), global_iter)
            for k, v in loss_dict.items():
                val = v.item() if torch.is_tensor(v) else v
                writer.add_scalar(f'train/{k}_loss', val, global_iter)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'lr': f'{scheduler.get_lr():.6f}'})
    
    # Epoch statistics
    avg_loss = epoch_loss / len(train_loader)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    for k, v in epoch_loss_dict.items():
        writer.add_scalar(f'train/epoch_{k}_loss', v / len(train_loader), epoch)
    
    # Learning rate step
    current_lr = scheduler.step()
    writer.add_scalar('train/learning_rate', current_lr, epoch)
    
    ######### Validation #########
    if epoch % args.val_epochs == 0:
        print(f"\n→ Running validation...")
        
        # Apply EMA weights for evaluation if enabled
        if ema:
            ema.apply_shadow()
        
        model_restoration.eval()
        psnr_val_rgb = []
        
        with torch.no_grad():
            for data_val in tqdm(val_loader, desc="Validating"):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                
                if scaler:
                    with autocast():
                        restored = model_restoration(input_)
                else:
                    restored = model_restoration(input_)
                
                for res, tar in zip(restored[0], target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
        
        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
        
        # Restore original weights if using EMA
        if ema:
            ema.restore()
        
        # Save best model
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            
            save_dict = {
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr
            }
            
            if ema:
                save_dict['ema_shadow'] = ema.shadow
            
            torch.save(save_dict, os.path.join(model_dir, "model_best.pth"))
            print(f"✓ New best model saved! PSNR: {best_psnr:.4f} dB")
        
        print(f"[Epoch {epoch}] PSNR: {psnr_val_rgb:.4f} dB | "
              f"Best: {best_psnr:.4f} dB (Epoch {best_epoch})")
        
        # Save periodic checkpoint
        if epoch % (args.val_epochs * 5) == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(save_dict, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
    
    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    print(f"\n{'='*80}")
    print(f"Epoch {epoch} Summary:")
    print(f"  Time: {epoch_time:.2f}s")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Learning Rate: {current_lr:.6f}")
    if progressive:
        print(f"  Patch Size: {current_patch_size}x{current_patch_size}")
    print(f"{'='*80}\n")
    
    # Save latest checkpoint
    save_dict = {
        'epoch': epoch,
        'state_dict': model_restoration.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_psnr': best_psnr
    }
    torch.save(save_dict, os.path.join(model_dir, "model_latest.pth"))

######### Training Complete ###########
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Best PSNR: {best_psnr:.4f} dB at Epoch {best_epoch}")
print(f"Models saved in: {model_dir}")
print("="*80)

writer.close()

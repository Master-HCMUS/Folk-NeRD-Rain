"""
Kaggle-optimized training script for GTAV-NightRain dataset
Handles:
- GTAV directory structure (rainy/gt instead of input/target)
- Kaggle GPU memory constraints
- Checkpoint saving to /kaggle/working
- Reduced batch size and validation frequency for faster iterations
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from model_S import MultiscaleNet as myNet  # Use smaller model for Kaggle
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from get_parameter_number import get_parameter_number
import kornia
from torch.utils.tensorboard import SummaryWriter
import argparse

from skimage import img_as_ubyte

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deraining - Kaggle GTAV-NightRain')

# Kaggle paths for GTAV-NightRain dataset
parser.add_argument('--train_dir', default='/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/', type=str, help='Directory of train images')
parser.add_argument('--val_dir', default='/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/', type=str, help='Directory of validation images')
parser.add_argument('--model_save_dir', default='/kaggle/working/checkpoints/', type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='', type=str, help='Path to pretrain-weights')

# Dataset-specific subdirectories
parser.add_argument('--input_subdir', default='rainy', type=str, help='Input subdirectory name')
parser.add_argument('--target_subdir', default='gt', type=str, help='Target subdirectory name')

# Training hyperparameters
parser.add_argument('--mode', default='Deraining', type=str)
parser.add_argument('--session', default='GTAV_NightRain', type=str, help='session')
parser.add_argument('--patch_size', default=256, type=int, help='patch size')
parser.add_argument('--num_epochs', default=300, type=int, help='num_epochs (reduced for Kaggle)')
parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
parser.add_argument('--val_epochs', default=5, type=int, help='validation frequency')
parser.add_argument('--save_epochs', default=10, type=int, help='checkpoint save frequency')

# Learning rate
parser.add_argument('--start_lr', default=2e-4, type=float, help='initial learning rate')
parser.add_argument('--end_lr', default=1e-6, type=float, help='final learning rate')

# Resume training
parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
parser.add_argument('--pretrain', action='store_true', help='use pretrained weights')

args = parser.parse_args()

mode = args.mode
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
utils.mkdir(model_dir)

train_dir = args.train_dir
val_dir = args.val_dir

num_epochs = args.num_epochs
batch_size = args.batch_size
val_epochs = args.val_epochs
save_epochs = args.save_epochs

start_lr = args.start_lr
end_lr = args.end_lr

print("\n" + "="*80)
print(f"KAGGLE TRAINING CONFIGURATION - GTAV-NightRain")
print("="*80)
print(f"Train directory: {train_dir}")
print(f"Val directory: {val_dir}")
print(f"Input subdir: {args.input_subdir}, Target subdir: {args.target_subdir}")
print(f"Model save directory: {model_dir}")
print(f"Patch size: {patch_size}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}")
print(f"Learning rate: {start_lr} → {end_lr}")
print(f"Validation frequency: every {val_epochs} epochs")
print(f"Checkpoint save frequency: every {save_epochs} epochs")
print("="*80 + "\n")

######### Model ###########
model_restoration = myNet()

# Print model parameters
print("Model: MultiscaleNet (Small)")
get_parameter_number(model_restoration)
print()

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!\n")

optimizer = optim.AdamW(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

######### Pretrain ###########
if args.pretrain and args.pretrain_weights:
    utils.load_checkpoint(model_restoration, args.pretrain_weights)
    print(f"==> Loaded pretrained weights from: {args.pretrain_weights}\n")

######### Resume ###########
if args.resume:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    if path_chk_rest:
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print(f"==> Resuming from epoch {start_epoch}, learning rate: {new_lr}\n")
    else:
        print("==> No checkpoint found, starting from scratch\n")

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss Functions ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()
criterion_L1 = nn.L1Loss()

######### DataLoaders ###########
print("Loading datasets...")
train_dataset = get_training_data(
    train_dir, 
    {'patch_size': patch_size},
    input_subdir=args.input_subdir,
    target_subdir=args.target_subdir
)
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2,  # Kaggle supports 2 workers
    drop_last=True,
    pin_memory=True
)

val_dataset = get_validation_data(
    val_dir, 
    {'patch_size': patch_size},
    input_subdir=args.input_subdir,
    target_subdir=args.target_subdir
)
val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2,
    drop_last=False,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Start Epoch: {start_epoch}, End Epoch: {num_epochs}\n")

best_psnr = 0
best_epoch = 0

# TensorBoard logging
writer = SummaryWriter(model_dir)
iter_count = 0

######### Training Loop ###########
for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    
    model_restoration.train()
    
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"), 0):
        # Zero gradients
        optimizer.zero_grad()

        target_ = data[0].cuda()
        input_ = data[1].cuda()
        
        # Create multi-scale targets
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        
        # Forward pass
        restored = model_restoration(input_)

        # Multi-scale losses
        loss_fft = sum([criterion_fft(restored[j], target[j]) for j in range(3)])
        loss_char = sum([criterion_char(restored[j], target[j]) for j in range(3)])
        loss_edge = sum([criterion_edge(restored[j], target[j]) for j in range(3)])
        loss_l1 = criterion_L1(restored[3], target[1]) + criterion_L1(restored[5], target[2])
        
        # Combined loss
        loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        iter_count += 1
        
        # Log to TensorBoard
        if iter_count % 10 == 0:
            writer.add_scalar('loss/fft_loss', loss_fft.item(), iter_count)
            writer.add_scalar('loss/char_loss', loss_char.item(), iter_count)
            writer.add_scalar('loss/edge_loss', loss_edge.item(), iter_count)
            writer.add_scalar('loss/l1_loss', loss_l1.item(), iter_count)
            writer.add_scalar('loss/iter_loss', loss.item(), iter_count)
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    writer.add_scalar('loss/epoch_loss', avg_epoch_loss, epoch)
    
    ######### Validation #########
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        
        with torch.no_grad():
            for ii, data_val in enumerate(tqdm(val_loader, desc="Validation"), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                
                restored = model_restoration(input_)
                
                for res, tar in zip(restored[0], target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
        
        # Save best model
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr
            }, os.path.join(model_dir, "model_best.pth"))
            print(f"✓ Saved best model at epoch {epoch}")

        print(f"\n[Epoch {epoch}] PSNR: {psnr_val_rgb:.4f} dB | Best: {best_psnr:.4f} dB (Epoch {best_epoch})\n")
    
    # Save periodic checkpoints
    if epoch % save_epochs == 0:
        torch.save({
            'epoch': epoch,
            'state_dict': model_restoration.state_dict(),
            'optimizer': optimizer.state_dict(),
            'psnr': psnr_val_rgb if epoch % val_epochs == 0 else None
        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    # Update learning rate
    scheduler.step()

    # Print epoch summary
    print("-" * 80)
    print(f"Epoch: {epoch}/{num_epochs} | Time: {time.time() - epoch_start_time:.2f}s | "
          f"Loss: {avg_epoch_loss:.6f} | LR: {scheduler.get_lr()[0]:.6e}")
    print("-" * 80 + "\n")

    # Save latest checkpoint (for resuming)
    torch.save({
        'epoch': epoch,
        'state_dict': model_restoration.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(model_dir, "model_latest.pth"))

writer.close()

print("\n" + "="*80)
print("TRAINING COMPLETED!")
print(f"Best PSNR: {best_psnr:.4f} dB at epoch {best_epoch}")
print(f"Models saved to: {model_dir}")
print("="*80)

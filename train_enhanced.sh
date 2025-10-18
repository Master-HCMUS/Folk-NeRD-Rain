#!/bin/bash
# Quick start training script for GTAV-NightRain with enhanced features

# =============================================================================
# CONFIGURATION - Modify these paths for your environment
# =============================================================================

# Dataset paths
TRAIN_DIR="/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train"
VAL_DIR="/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test"

# For Kaggle, use:
# TRAIN_DIR="/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train"
# VAL_DIR="/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test"

# For Colab, use:
# TRAIN_DIR="/content/GTAV-NightRain/train"
# VAL_DIR="/content/GTAV-NightRain/test"

# =============================================================================
# TRAINING CONFIGURATIONS
# =============================================================================

# Configuration 1: Quick Test (Fast, Lower Quality)
# Use this to verify everything works before full training
train_quick_test() {
    python Folk-NeRD-Rain/train_enhanced.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --input_subdir rainy \
        --target_subdir gt \
        --model small \
        --num_epochs 20 \
        --batch_size 1 \
        --patch_size 128 \
        --val_epochs 5 \
        --session quick_test \
        --use_enhanced_loss \
        --use_augmentation \
        --mixed_precision
}

# Configuration 2: Balanced (Good Quality, Reasonable Time)
# Recommended for most users
train_balanced() {
    python Folk-NeRD-Rain/train_enhanced.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --input_subdir rainy \
        --target_subdir gt \
        --model small \
        --num_epochs 100 \
        --batch_size 3 \
        --patch_size 256 \
        --val_epochs 10 \
        --gradient_accumulation 4 \
        --session balanced_v1 \
        --use_enhanced_loss \
        --use_augmentation \
        --use_ema \
        --mixed_precision \
        --warmup_epochs 10 \
        --base_lr 1e-4 \
        --min_lr 1e-6 \
        --perceptual_weight 0.1 \
        --ssim_weight 0.5 \
        --illumination_weight 0.3
}

# Configuration 3: Full Quality (Best Results, Longer Training)
# For final model or research
train_full_quality() {
    python Folk-NeRD-Rain/train_enhanced.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --input_subdir rainy \
        --target_subdir gt \
        --model full \
        --num_epochs 300 \
        --batch_size 1 \
        --patch_size 256 \
        --val_epochs 10 \
        --gradient_accumulation 4 \
        --session full_quality_v1 \
        --use_enhanced_loss \
        --use_augmentation \
        --use_ema \
        --progressive_training \
        --mixed_precision \
        --warmup_epochs 15 \
        --base_lr 1e-4 \
        --min_lr 1e-6 \
        --perceptual_weight 0.1 \
        --ssim_weight 0.5 \
        --illumination_weight 0.3
}

# Configuration 4: Resume Training
# Continue from a checkpoint
train_resume() {
    CHECKPOINT_PATH="./checkpoints/Deraining/models/balanced_v1/model_latest.pth"
    
    python Folk-NeRD-Rain/train_enhanced.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --input_subdir rainy \
        --target_subdir gt \
        --model small \
        --num_epochs 300 \
        --batch_size 1 \
        --patch_size 256 \
        --val_epochs 10 \
        --gradient_accumulation 4 \
        --session balanced_v1 \
        --resume "$CHECKPOINT_PATH" \
        --use_enhanced_loss \
        --use_augmentation \
        --use_ema \
        --mixed_precision
}

# Configuration 5: Minimal (Standard Loss, No Enhancements)
# Baseline for comparison
train_baseline() {
    python train_enhanced.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --input_subdir rainy \
        --target_subdir gt \
        --model small \
        --num_epochs 100 \
        --batch_size 1 \
        --patch_size 256 \
        --val_epochs 10 \
        --session baseline \
        --mixed_precision
}

# =============================================================================
# EXECUTION
# =============================================================================

# Uncomment the configuration you want to run:

# train_quick_test      # Fast test run
train_balanced        # Recommended
# train_full_quality    # Best quality
# train_resume          # Resume from checkpoint
# train_baseline        # Baseline comparison

echo ""
echo "Training started! Check TensorBoard for progress:"
echo "tensorboard --logdir=./checkpoints/Deraining/models/"

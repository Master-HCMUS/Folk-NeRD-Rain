@echo off
REM Quick start training script for GTAV-NightRain with enhanced features (Windows)

REM =============================================================================
REM CONFIGURATION - Modify these paths for your environment
REM =============================================================================

REM Dataset paths (modify these!)
set TRAIN_DIR=C:\path\to\GTAV-NightRain\train
set VAL_DIR=C:\path\to\GTAV-NightRain\test

REM For Kaggle, use:
REM set TRAIN_DIR=/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train
REM set VAL_DIR=/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test

REM =============================================================================
REM Choose your training configuration
REM =============================================================================

echo Select training configuration:
echo 1. Quick Test (20 epochs, fast validation)
echo 2. Balanced (200 epochs, recommended)
echo 3. Full Quality (300 epochs, best results)
echo 4. Resume Training (continue from checkpoint)
echo 5. Baseline (standard loss, no enhancements)
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto balanced
if "%choice%"=="3" goto full_quality
if "%choice%"=="4" goto resume
if "%choice%"=="5" goto baseline

echo Invalid choice!
exit /b 1

REM =============================================================================
REM Training Configurations
REM =============================================================================

:quick_test
echo Starting Quick Test training...
python train_enhanced.py ^
    --train_dir "%TRAIN_DIR%" ^
    --val_dir "%VAL_DIR%" ^
    --input_subdir rainy ^
    --target_subdir gt ^
    --model small ^
    --num_epochs 20 ^
    --batch_size 1 ^
    --patch_size 128 ^
    --val_epochs 5 ^
    --session quick_test ^
    --use_enhanced_loss ^
    --use_augmentation ^
    --mixed_precision
goto end

:balanced
echo Starting Balanced training (recommended)...
python train_enhanced.py ^
    --train_dir "%TRAIN_DIR%" ^
    --val_dir "%VAL_DIR%" ^
    --input_subdir rainy ^
    --target_subdir gt ^
    --model small ^
    --num_epochs 200 ^
    --batch_size 1 ^
    --patch_size 256 ^
    --val_epochs 10 ^
    --gradient_accumulation 4 ^
    --session balanced_v1 ^
    --use_enhanced_loss ^
    --use_augmentation ^
    --use_ema ^
    --mixed_precision ^
    --warmup_epochs 10 ^
    --base_lr 1e-4 ^
    --min_lr 1e-6 ^
    --perceptual_weight 0.1 ^
    --ssim_weight 0.5 ^
    --illumination_weight 0.3
goto end

:full_quality
echo Starting Full Quality training...
python train_enhanced.py ^
    --train_dir "%TRAIN_DIR%" ^
    --val_dir "%VAL_DIR%" ^
    --input_subdir rainy ^
    --target_subdir gt ^
    --model full ^
    --num_epochs 300 ^
    --batch_size 1 ^
    --patch_size 256 ^
    --val_epochs 10 ^
    --gradient_accumulation 4 ^
    --session full_quality_v1 ^
    --use_enhanced_loss ^
    --use_augmentation ^
    --use_ema ^
    --progressive_training ^
    --mixed_precision ^
    --warmup_epochs 15 ^
    --base_lr 1e-4 ^
    --min_lr 1e-6 ^
    --perceptual_weight 0.1 ^
    --ssim_weight 0.5 ^
    --illumination_weight 0.3
goto end

:resume
set /p CHECKPOINT_PATH="Enter checkpoint path (default: ./checkpoints/Deraining/models/balanced_v1/model_latest.pth): "
if "%CHECKPOINT_PATH%"=="" set CHECKPOINT_PATH=./checkpoints/Deraining/models/balanced_v1/model_latest.pth

echo Resuming training from %CHECKPOINT_PATH%...
python train_enhanced.py ^
    --train_dir "%TRAIN_DIR%" ^
    --val_dir "%VAL_DIR%" ^
    --input_subdir rainy ^
    --target_subdir gt ^
    --model small ^
    --num_epochs 300 ^
    --batch_size 1 ^
    --patch_size 256 ^
    --val_epochs 10 ^
    --gradient_accumulation 4 ^
    --session balanced_v1 ^
    --resume "%CHECKPOINT_PATH%" ^
    --use_enhanced_loss ^
    --use_augmentation ^
    --use_ema ^
    --mixed_precision
goto end

:baseline
echo Starting Baseline training (no enhancements)...
python train_enhanced.py ^
    --train_dir "%TRAIN_DIR%" ^
    --val_dir "%VAL_DIR%" ^
    --input_subdir rainy ^
    --target_subdir gt ^
    --model small ^
    --num_epochs 100 ^
    --batch_size 1 ^
    --patch_size 256 ^
    --val_epochs 10 ^
    --session baseline ^
    --mixed_precision
goto end

:end
echo.
echo Training started! Check TensorBoard for progress:
echo tensorboard --logdir=./checkpoints/Deraining/models/
pause

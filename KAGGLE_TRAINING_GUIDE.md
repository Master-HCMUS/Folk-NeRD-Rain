# Training NeRD-Rain on Kaggle with GTAV-NightRain Dataset

## Step 1: Create a New Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)
4. Set Internet to ON (for pip installs)

## Step 2: Add Dataset

1. Click "+ Add Data" in the right panel
2. Search for "gtav-nightrain-rerendered-version"
3. Add the dataset

## Step 3: Upload Code Files

Upload these files to your Kaggle notebook or create a dataset with them:
- `train_kaggle.py`
- `model.py` or `model_S.py`
- `data_RGB.py`
- `dataset_RGB.py`
- `layers.py`
- `mlp.py`
- `losses.py`
- `utils/` directory
- `get_parameter_number.py`
- `pytorch-gradual-warmup-lr/` directory

**Recommended**: Create a Kaggle dataset with all code files, then add it to your notebook.

## Step 4: Install Dependencies

```python
# Cell 1: Install dependencies
!pip install einops kornia timm warmup-scheduler -q

# Install warmup scheduler
!cd pytorch-gradual-warmup-lr && python setup.py install
```

## Step 5: Verify Dataset Structure

```python
# Cell 2: Check dataset paths
import os

train_rainy = '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/rainy'
train_gt = '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/gt'
test_rainy = '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/rainy'
test_gt = '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt'

print(f"Train rainy exists: {os.path.exists(train_rainy)}")
print(f"Train GT exists: {os.path.exists(train_gt)}")
print(f"Test rainy exists: {os.path.exists(test_rainy)}")
print(f"Test GT exists: {os.path.exists(test_gt)}")

if os.path.exists(train_rainy):
    train_count = len([f for f in os.listdir(train_rainy) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\nTraining samples: {train_count}")

if os.path.exists(test_rainy):
    test_count = len([f for f in os.listdir(test_rainy) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Test samples: {test_count}")
```

## Step 6: Start Training

### Option A: Quick Training (Recommended for Kaggle)
```python
# Cell 3: Train with small model and fewer epochs
!python train_kaggle.py \
    --train_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/ \
    --val_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/ \
    --model_save_dir /kaggle/working/checkpoints/ \
    --input_subdir rainy \
    --target_subdir gt \
    --num_epochs 100 \
    --batch_size 4 \
    --patch_size 256 \
    --val_epochs 5 \
    --save_epochs 10 \
    --session GTAV_NightRain_quick
```

### Option B: Full Training (For longer sessions)
```python
# Cell 3: Full training with more epochs
!python train_kaggle.py \
    --train_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/ \
    --val_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/ \
    --model_save_dir /kaggle/working/checkpoints/ \
    --input_subdir rainy \
    --target_subdir gt \
    --num_epochs 300 \
    --batch_size 4 \
    --patch_size 256 \
    --val_epochs 5 \
    --save_epochs 20 \
    --session GTAV_NightRain_full
```

### Option C: Resume Training
```python
# Cell 3: Resume from checkpoint
!python train_kaggle.py \
    --train_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/ \
    --val_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/ \
    --model_save_dir /kaggle/working/checkpoints/ \
    --input_subdir rainy \
    --target_subdir gt \
    --num_epochs 300 \
    --batch_size 4 \
    --resume \
    --session GTAV_NightRain_full
```

## Step 7: Monitor Training

```python
# Cell 4: Monitor training progress (run in separate cell while training)
import glob
checkpoint_dir = '/kaggle/working/checkpoints/Deraining/models/GTAV_NightRain_quick/'

# List all checkpoints
checkpoints = sorted(glob.glob(f'{checkpoint_dir}/*.pth'))
print(f"Found {len(checkpoints)} checkpoints:")
for ckpt in checkpoints:
    print(f"  - {os.path.basename(ckpt)}")
```

## Step 8: Download Results

```python
# Cell 5: Create zip file for download
import shutil

# Zip the checkpoint directory
output_zip = '/kaggle/working/checkpoints_gtav'
checkpoint_base = '/kaggle/working/checkpoints/Deraining/models/GTAV_NightRain_quick'

if os.path.exists(checkpoint_base):
    shutil.make_archive(output_zip, 'zip', checkpoint_base)
    print(f"✓ Created {output_zip}.zip")
    print(f"  Size: {os.path.getsize(output_zip + '.zip') / (1024*1024):.2f} MB")
else:
    print("No checkpoints found!")
```

## Important Kaggle Considerations

### Memory Management
- **Batch size**: Start with 4, reduce to 2 if OOM errors occur
- **Patch size**: 256 is optimal, 128 if memory issues persist
- **Model**: Use `model_S.py` (small) instead of `model.py` for Kaggle

### Session Time Limits
- Kaggle sessions timeout after ~12 hours
- Save checkpoints frequently (`--save_epochs 10`)
- Use `--resume` to continue training in new session

### GPU Optimization
```python
# Add to notebook to check GPU
!nvidia-smi
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Monitoring with TensorBoard
```python
# Load TensorBoard in Kaggle
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/checkpoints/Deraining/models/GTAV_NightRain_quick
```

## Troubleshooting

### Issue 1: Directory Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/.../rainy'
```
**Solution**: Verify the exact dataset path by listing directories:
```python
!ls -la /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/
```

### Issue 2: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce `--batch_size` from 4 to 2 or 1
- Reduce `--patch_size` from 256 to 128
- Use `model_S.py` instead of `model.py`

### Issue 3: Warmup Scheduler Import Error
```
ModuleNotFoundError: No module named 'warmup_scheduler'
```
**Solution**:
```bash
!cd pytorch-gradual-warmup-lr && python setup.py install && cd ..
```

### Issue 4: Checkpoint Not Saved
**Solution**: Ensure `/kaggle/working/` is writable:
```python
!mkdir -p /kaggle/working/checkpoints/Deraining/models/GTAV_NightRain_quick
!ls -la /kaggle/working/checkpoints/
```

## Performance Expectations

| Configuration | Training Time (100 epochs) | Expected PSNR |
|--------------|---------------------------|---------------|
| Batch=4, Patch=256 | ~8-10 hours | 28-30 dB |
| Batch=2, Patch=256 | ~12-14 hours | 28-30 dB |
| Batch=4, Patch=128 | ~4-6 hours | 26-28 dB |

**Note**: GTAV-NightRain is more challenging than Rain200L, expect slightly lower PSNR values.

## Sample Output Structure

After training, your `/kaggle/working/` directory will look like:
```
/kaggle/working/
└── checkpoints/
    └── Deraining/
        └── models/
            └── GTAV_NightRain_quick/
                ├── events.out.tfevents.xxx  (TensorBoard logs)
                ├── model_best.pth           (Best performing model)
                ├── model_latest.pth         (Latest checkpoint)
                ├── model_epoch_10.pth
                ├── model_epoch_20.pth
                └── ...
```

## Next Steps

After training, test the model:
```python
!python test.py \
    --input_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/rainy \
    --output_dir /kaggle/working/results/GTAV_test \
    --weights /kaggle/working/checkpoints/Deraining/models/GTAV_NightRain_quick/model_best.pth
```

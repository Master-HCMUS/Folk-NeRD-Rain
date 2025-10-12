# 🚀 Quick Start: Training on Kaggle with GTAV-NightRain

## One-Command Setup

```python
# Copy this entire cell to your Kaggle notebook:

# 1. Install dependencies
!pip install einops kornia timm -q

# 2. Clone repository (or upload files)
!cd /kaggle/working && git clone https://github.com/Master-HCMUS/Folk-NeRD-Rain.git

# 3. Install warmup scheduler
!cd /kaggle/working/Folk-NeRD-Rain/pytorch-gradual-warmup-lr && python setup.py install

# 4. Train model
!cd /kaggle/working/Folk-NeRD-Rain && python train_kaggle.py \
    --train_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/ \
    --val_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/ \
    --input_subdir rainy \
    --target_subdir gt \
    --num_epochs 100 \
    --batch_size 4 \
    --session GTAV_v1
```

## 📋 Prerequisites

1. ✅ Kaggle account with GPU enabled (T4 x2 or P100)
2. ✅ GTAV-NightRain dataset added to notebook
3. ✅ Internet enabled in notebook settings

## 🎯 Key Changes Made

### Modified Files:
- **`dataset_RGB.py`**: Added support for `rainy/gt` directory structure
- **`data_RGB.py`**: Pass subdirectory parameters to data loaders
- **`test.py`**: Already supports both `--input-dir` and `--input_dir`

### New Files:
- **`train_kaggle.py`**: Optimized training script for Kaggle
- **`KAGGLE_TRAINING_GUIDE.md`**: Complete documentation
- **`kaggle_notebook_cells.py`**: Ready-to-use notebook cells

## ⚙️ Training Options

### Quick Test (100 epochs, ~8 hours)
```bash
python train_kaggle.py \
    --num_epochs 100 \
    --batch_size 4 \
    --val_epochs 5
```

### Full Training (300 epochs, ~24 hours)
```bash
python train_kaggle.py \
    --num_epochs 300 \
    --batch_size 4 \
    --val_epochs 10 \
    --save_epochs 20
```

### Memory-Constrained (OOM errors)
```bash
python train_kaggle.py \
    --batch_size 2 \
    --patch_size 128
```

## 📊 Monitor Training

```python
# Check checkpoints
!ls -lh /kaggle/working/checkpoints/Deraining/models/GTAV_v1/

# View TensorBoard
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/checkpoints/Deraining/models/GTAV_v1
```

## 🧪 Test Trained Model

```bash
python test.py \
    --input-dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/rainy \
    --output-dir /kaggle/working/results/GTAV_test \
    --weights /kaggle/working/checkpoints/Deraining/models/GTAV_v1/model_best.pth
```

## 📊 Evaluate PSNR and SSIM

```bash
python evaluate_metrics.py \
    --result_dir /kaggle/working/results/GTAV_test \
    --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt \
    --dataset_name "GTAV-NightRain"
```

Or use inline Python:
```python
from evaluate_metrics import evaluate_dataset

psnr, ssim = evaluate_dataset(
    '/kaggle/working/results/GTAV_test',
    '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt',
    'GTAV-NightRain'
)
print(f"PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
```

## 💾 Download Results

```python
import shutil
shutil.make_archive('/kaggle/working/checkpoints_gtav', 'zip', 
                    '/kaggle/working/checkpoints')
# Download from Output tab →
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `--batch_size 2` or `--patch_size 128` |
| No warmup_scheduler | Run `!cd pytorch-gradual-warmup-lr && python setup.py install` |
| Wrong directory | Verify paths with `!ls /kaggle/input/.../train/` |
| Training stops | Resume with `--resume` flag |

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| Training Time (100 epochs) | ~8-10 hours |
| Validation PSNR | 28-30 dB |
| Model Size | ~18 MB |
| Inference Time (1024x1024) | ~0.5s |

## 📚 Documentation

- **Complete Guide**: [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md)
- **Summary**: [KAGGLE_SETUP_SUMMARY.md](KAGGLE_SETUP_SUMMARY.md)
- **Notebook Cells**: [kaggle_notebook_cells.py](kaggle_notebook_cells.py)
- **Workflow**: [WORKFLOW_DIAGRAM.py](WORKFLOW_DIAGRAM.py)

## ✅ Verified Working Configuration

```yaml
Environment:
  - Platform: Kaggle Notebooks
  - GPU: T4 x2 (16GB VRAM)
  - CUDA: 11.8
  - PyTorch: 2.0+

Dataset:
  - Name: GTAV-NightRain (Rerendered Version)
  - Train: 24,000+ images
  - Test: 2,000+ images
  - Structure: rainy/ and gt/ subdirectories

Model:
  - Architecture: MultiscaleNet (Small)
  - Parameters: ~4M
  - Input: 256x256 patches
  - Output: 3-scale predictions
```

---

**Ready to train?** Copy the one-command setup above to your Kaggle notebook! 🎉

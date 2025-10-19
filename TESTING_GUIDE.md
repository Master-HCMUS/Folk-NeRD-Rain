# Testing Guide - Folk-NeRD-Rain

## Quick Start

### 1. Identify Your Model Type

First, check which model was used for training:

**Small Model (~4M parameters):**
- Checkpoint saved from `model_S.py`
- Trained with `--model small`
- File usually named: `model_small_*.pth`

**Full Model (~8M parameters):**
- Checkpoint saved from `model.py`
- Trained with `--model full` or without `--model` argument
- File usually named: `model_best.pth`, `model_epoch_*.pth`

### 2. Run Test Script

**For Full Model:**
```bash
python test.py \
    --input_dir ./Datasets/Rain200L/test/input/ \
    --output_dir ./results/Rain200L \
    --weights ./checkpoints/model_best.pth \
    --model full
```

**For Small Model:**
```bash
python test.py \
    --input_dir ./Datasets/Rain200L/test/input/ \
    --output_dir ./results/Rain200L \
    --weights ./checkpoints/model_small_Rain200L.pth \
    --model small
```

**On Kaggle/Colab:**
```bash
python Folk-NeRD-Rain/test.py \
    --input_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/rainy/ \
    --output_dir /kaggle/working/results/ \
    --weights /kaggle/working/checkpoints/Deraining/models/balanced_v1/model_best.pth \
    --model full \
    --win_size 256
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | `./Datasets/Rain200L/test/input/` | Directory containing rainy test images |
| `--output_dir` | `./results/Rain200L` | Directory to save derained results |
| `--weights` | (required) | Path to checkpoint file (.pth) |
| `--model` | `full` | Model type: `full` or `small` |
| `--gpus` | `0` | GPU device IDs (e.g., `0`, `0,1`, `0,1,2,3`) |
| `--win_size` | `256` | Window size for processing (256 or 512) |

## Common Errors and Solutions

### Error 1: RuntimeError - Missing keys in state_dict

```
RuntimeError: Error(s) in loading state_dict for MultiscaleNet:
	Missing key(s) in state_dict: "patch_embed_small.proj.weight"
```

**Cause:** Model architecture mismatch between training and testing.

**Solution:**
1. Check your checkpoint file name:
   - If contains "small" → use `--model small`
   - Otherwise → use `--model full`

2. Or check training command:
   ```bash
   # If you trained with:
   python train.py --model small ...
   
   # Then test with:
   python test.py --model small ...
   ```

### Error 2: Weights only load failed

```
_pickle.UnpicklingError: Weights only load failed.
```

**Cause:** PyTorch 2.6+ changed default `weights_only` parameter.

**Solution:** This has been fixed in the latest `utils/model_utils.py`. Make sure you have the updated version with `safe_torch_load()` function.

### Error 3: CUDA out of memory

```
RuntimeError: CUDA out of memory.
```

**Solutions:**
1. Reduce window size:
   ```bash
   python test.py ... --win_size 128
   ```

2. Process images one at a time (batch_size=1 is already default)

3. Use CPU (slower):
   ```bash
   python test.py ... --gpus -1
   ```

### Error 4: No such file or directory

```
FileNotFoundError: [Errno 2] No such file or directory: './checkpoints/model_best.pth'
```

**Solution:** Provide correct path to checkpoint:
```bash
# Find your checkpoint
ls checkpoints/Deraining/models/*/model_best.pth

# Use full path
python test.py --weights checkpoints/Deraining/models/balanced_v1/model_best.pth
```

## Expected Output

After successful testing:

```
Using Full Model (model.py)
Total params: 8.03 M
===>Testing using weights: ./checkpoints/model_best.pth
100%|████████████████| 100/100 [00:45<00:00,  2.21it/s]
```

Results saved to `--output_dir`:
```
results/Rain200L/
├── 1.png
├── 2.png
├── 3.png
...
```

## Evaluate Results

After testing, evaluate PSNR and SSIM:

```bash
python evaluate_metrics.py \
    --result_dir ./results/Rain200L \
    --gt_dir ./Datasets/Rain200L/test/target \
    --filter_pattern "_00"
```

**Note:** Use `--filter_pattern "_00"` to filter out windowed outputs (e.g., `1_00.png`, `1_01.png` → only keep first window).

## Window Size Guidelines

| Image Size | Recommended `--win_size` | GPU Memory |
|------------|-------------------------|------------|
| 512×512 | 256 | 4-6 GB |
| 1024×1024 | 256 | 6-8 GB |
| 1024×1024 | 512 | 10-12 GB |
| 2048×2048 | 256 | 8-12 GB |
| 2048×2048 | 512 | 16+ GB |

**Tip:** Smaller window size = more windows = slower but uses less memory.

## Batch Processing Script

For processing multiple datasets:

```bash
#!/bin/bash

# Test on Rain200L
python test.py \
    --input_dir ./Datasets/Rain200L/test/input/ \
    --output_dir ./results/Rain200L \
    --weights ./checkpoints/model_best.pth \
    --model full

# Test on Rain200H
python test.py \
    --input_dir ./Datasets/Rain200H/test/input/ \
    --output_dir ./results/Rain200H \
    --weights ./checkpoints/model_best.pth \
    --model full

# Test on GTAV-NightRain
python test.py \
    --input_dir ./Datasets/GTAV-NightRain/test/rainy/ \
    --output_dir ./results/GTAV-NightRain \
    --weights ./checkpoints/model_best.pth \
    --model full

echo "Testing complete! Evaluating..."

# Evaluate all
python evaluate_metrics.py --result_dir ./results/Rain200L --gt_dir ./Datasets/Rain200L/test/target
python evaluate_metrics.py --result_dir ./results/Rain200H --gt_dir ./Datasets/Rain200H/test/target
python evaluate_metrics.py --result_dir ./results/GTAV-NightRain --gt_dir ./Datasets/GTAV-NightRain/test/gt --filter_pattern "_00"
```

## Troubleshooting Checklist

- [ ] Checkpoint file exists at specified path
- [ ] Model type (`--model`) matches checkpoint
- [ ] Input directory contains images
- [ ] Output directory has write permissions
- [ ] GPU is available (check `nvidia-smi`)
- [ ] CUDA version matches PyTorch version
- [ ] Enough GPU memory for window size
- [ ] PyTorch version ≥ 1.10

## Getting Help

If you encounter issues:

1. **Check error message carefully** - The improved `load_checkpoint` now provides detailed diagnostics

2. **Verify model and checkpoint match:**
   ```python
   # Quick test in Python
   import torch
   checkpoint = torch.load('checkpoints/model_best.pth', weights_only=False)
   print("Checkpoint keys (first 5):")
   for i, key in enumerate(list(checkpoint['state_dict'].keys())[:5]):
       print(f"  {key}")
   ```

3. **Check model parameters:**
   ```bash
   python get_parameter_number.py
   ```

4. **Open an issue** with:
   - Full error message
   - Command you ran
   - Checkpoint file name
   - Model type used

## Performance Tips

1. **Use appropriate window size:**
   - Larger = faster but more memory
   - Smaller = slower but less memory

2. **Multi-GPU testing:**
   ```bash
   python test.py --gpus 0,1,2,3 ...
   ```

3. **Batch processing:**
   - Process multiple images in parallel
   - See batch processing script above

4. **FP16 inference (experimental):**
   - Modify test.py to use mixed precision
   - 2× faster, uses less memory
   - May slightly reduce quality

---

**Last Updated:** October 19, 2025  
**Compatible with:** PyTorch 2.6+, CUDA 11.8+

# 📊 Evaluating PSNR and SSIM on Kaggle

## Quick Start: Single Command Evaluation

```bash
# After running test.py to generate results
python evaluate_metrics.py \
    --result_dir /kaggle/working/results/GTAV_test \
    --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt \
    --dataset_name "GTAV-NightRain Test"
```

## Method 1: Evaluate Single Dataset

### Step 1: Generate Results
First, run inference to generate derained images:

```bash
python test.py \
    --input-dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/rainy \
    --output-dir /kaggle/working/results/GTAV_test \
    --weights /kaggle/working/checkpoints/Deraining/models/GTAV_NightRain/model_best.pth
```

### Step 2: Evaluate Metrics
Run the evaluation script:

```bash
python evaluate_metrics.py \
    --result_dir /kaggle/working/results/GTAV_test \
    --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt \
    --dataset_name "GTAV-NightRain"
```

### Expected Output:
```
================================================================================
PSNR/SSIM Evaluation for Image Deraining
================================================================================

Evaluating GTAV-NightRain...
Result images: 2000
Ground truth images: 2000
Evaluating GTAV-NightRain: 100%|██████████| 2000/2000 [00:45<00:00, 44.12it/s]

Results for GTAV-NightRain:
  Images evaluated: 2000/2000
  Average PSNR: 28.5432 dB
  Average SSIM: 0.8765
================================================================================
```

## Method 2: Evaluate Multiple Datasets

If you have multiple datasets organized in subdirectories:

```bash
python evaluate_metrics.py \
    --result_dir /kaggle/working/results \
    --gt_dir /kaggle/input/datasets/ground_truth \
    --multiple_datasets
```

Directory structure:
```
/kaggle/working/results/
├── Rain200L/
│   ├── 1.png
│   └── 2.png
└── Rain200H/
    ├── 1.png
    └── 2.png

/kaggle/input/datasets/ground_truth/
├── Rain200L/
│   ├── 1.png
│   └── 2.png
└── Rain200H/
    ├── 1.png
    └── 2.png
```

## Method 3: Integrated Kaggle Notebook Cell

```python
# ==================== CELL: Evaluate Results ====================
import os
import sys
sys.path.append('/kaggle/working/Folk-NeRD-Rain')

# Import evaluation functions
from evaluate_metrics import evaluate_dataset

# Paths
result_dir = '/kaggle/working/results/GTAV_test'
gt_dir = '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt'

# Evaluate
print("Evaluating GTAV-NightRain test set...")
psnr, ssim = evaluate_dataset(result_dir, gt_dir, 'GTAV-NightRain')

print(f"\n{'='*60}")
print(f"Final Results:")
print(f"  PSNR: {psnr:.4f} dB")
print(f"  SSIM: {ssim:.4f}")
print(f"{'='*60}")
```

## Method 4: Inline Python Evaluation

For quick evaluation without running separate script:

```python
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from glob import glob
from tqdm import tqdm

def rgb2y(img):
    """Convert RGB to Y channel"""
    if len(img.shape) == 2:
        return img
    img = img.astype(np.float32)
    y = 16 + 65.481 * img[:,:,0]/255 + 128.553 * img[:,:,1]/255 + 24.966 * img[:,:,2]/255
    return y

def evaluate_quick(result_dir, gt_dir):
    result_files = sorted(glob(f"{result_dir}/*.png") + glob(f"{result_dir}/*.jpg"))
    gt_files = sorted(glob(f"{gt_dir}/*.png") + glob(f"{gt_dir}/*.jpg"))
    
    total_psnr = 0
    total_ssim = 0
    
    for res_file, gt_file in tqdm(zip(result_files, gt_files), total=len(result_files)):
        res_img = np.array(Image.open(res_file))
        gt_img = np.array(Image.open(gt_file))
        
        # Convert to Y channel
        if len(res_img.shape) == 3:
            res_img = rgb2y(res_img)
            gt_img = rgb2y(gt_img)
        
        total_psnr += psnr(res_img, gt_img, data_range=255)
        total_ssim += ssim(res_img, gt_img, data_range=255)
    
    n = len(result_files)
    print(f"Average PSNR: {total_psnr/n:.4f} dB")
    print(f"Average SSIM: {total_ssim/n:.4f}")
    return total_psnr/n, total_ssim/n

# Run evaluation
result_dir = '/kaggle/working/results/GTAV_test'
gt_dir = '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt'
evaluate_quick(result_dir, gt_dir)
```

## Method 5: Visualize Results with Metrics

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from evaluate_metrics import rgb2ycbcr, compute_psnr, compute_ssim

def visualize_with_metrics(input_path, result_path, gt_path, num_samples=5):
    """Visualize results with PSNR/SSIM overlays"""
    
    input_files = sorted(os.listdir(input_path))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    psnr_list = []
    ssim_list = []
    
    for idx, fname in enumerate(input_files):
        # Load images
        input_img = Image.open(os.path.join(input_path, fname))
        result_img = Image.open(os.path.join(result_path, fname))
        gt_img = Image.open(os.path.join(gt_path, fname))
        
        # Convert to numpy for metrics
        result_np = np.array(result_img)
        gt_np = np.array(gt_img)
        
        # Compute metrics
        psnr_val = compute_psnr(result_np, gt_np)
        ssim_val = compute_ssim(result_np, gt_np)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        # Display images
        axes[idx, 0].imshow(input_img)
        axes[idx, 0].set_title(f'Rainy Input\n{fname}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(result_img)
        axes[idx, 1].set_title(f'Derained\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(gt_img)
        axes[idx, 2].set_title('Ground Truth')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/evaluation_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Sample Statistics ({num_samples} images):")
    print(f"  Average PSNR: {np.mean(psnr_list):.4f} dB")
    print(f"  Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"  PSNR Range: [{np.min(psnr_list):.4f}, {np.max(psnr_list):.4f}]")
    print(f"  SSIM Range: [{np.min(ssim_list):.4f}, {np.max(ssim_list):.4f}]")
    print(f"{'='*60}")

# Run visualization
visualize_with_metrics(
    input_path='/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/rainy',
    result_path='/kaggle/working/results/GTAV_test',
    gt_path='/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt',
    num_samples=5
)
```

## Complete Workflow: Train → Test → Evaluate

```python
# ==================== CELL 1: Train Model ====================
!cd /kaggle/working/Folk-NeRD-Rain && python train_kaggle.py \
    --train_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/train/ \
    --val_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/ \
    --input_subdir rainy \
    --target_subdir gt \
    --num_epochs 100

# ==================== CELL 2: Test Model ====================
!cd /kaggle/working/Folk-NeRD-Rain && python test.py \
    --input-dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/rainy \
    --output-dir /kaggle/working/results/GTAV_test \
    --weights /kaggle/working/checkpoints/Deraining/models/GTAV_NightRain/model_best.pth

# ==================== CELL 3: Evaluate Metrics ====================
!cd /kaggle/working/Folk-NeRD-Rain && python evaluate_metrics.py \
    --result_dir /kaggle/working/results/GTAV_test \
    --gt_dir /kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt \
    --dataset_name "GTAV-NightRain"

# ==================== CELL 4: Visualize Best/Worst Cases ====================
import os
import numpy as np
from PIL import Image
from glob import glob
from evaluate_metrics import compute_psnr, compute_ssim
import matplotlib.pyplot as plt

# Compute metrics for all images
result_files = sorted(glob('/kaggle/working/results/GTAV_test/*.png'))
gt_dir = '/kaggle/input/gtav-nightrain-rerendered-version/GTAV-NightRain/test/gt'

metrics = []
for res_file in result_files:
    fname = os.path.basename(res_file)
    gt_file = os.path.join(gt_dir, fname)
    
    if os.path.exists(gt_file):
        res_img = np.array(Image.open(res_file))
        gt_img = np.array(Image.open(gt_file))
        
        psnr_val = compute_psnr(res_img, gt_img)
        ssim_val = compute_ssim(res_img, gt_img)
        
        metrics.append({
            'file': fname,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'combined': psnr_val + 100*ssim_val  # Combined score
        })

# Sort by combined score
metrics_sorted = sorted(metrics, key=lambda x: x['combined'])

# Show best and worst cases
print("Top 5 Best Results:")
for m in metrics_sorted[-5:][::-1]:
    print(f"  {m['file']}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}")

print("\nTop 5 Worst Results:")
for m in metrics_sorted[:5]:
    print(f"  {m['file']}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}")
```

## Dependencies

The evaluation script requires:
- `numpy`
- `pillow` (PIL)
- `scikit-image` (for SSIM/PSNR computation)
- `opencv-python` (for image resizing)
- `tqdm` (for progress bars)

All are pre-installed on Kaggle! 🎉

## Notes

1. **Y Channel Evaluation**: Following the MATLAB code, evaluation is performed on the Y channel (luminance) of YCbCr color space for RGB images.

2. **Image Matching**: The script automatically matches result images with ground truth images by filename.

3. **Size Mismatch**: If result and GT images have different sizes, the result is automatically resized to match GT dimensions.

4. **Performance**: Evaluation takes ~30-60 seconds for 2000 images on Kaggle.

## Troubleshooting

### Issue: "No result images found"
**Solution**: Make sure you've run `test.py` first to generate results

### Issue: "No matching ground truth found"
**Solution**: Check that result and GT filenames match:
```python
!ls /kaggle/working/results/GTAV_test/ | head -5
!ls /kaggle/input/.../test/gt/ | head -5
```

### Issue: Different PSNR/SSIM than expected
**Solution**: Verify you're using Y channel evaluation (matches MATLAB implementation)

## Expected Performance

| Dataset | PSNR (dB) | SSIM |
|---------|-----------|------|
| Rain200L (Paper) | 39.00 | 0.9860 |
| Rain200H (Paper) | 31.20 | 0.9080 |
| GTAV-NightRain (Expected) | 28-30 | 0.85-0.90 |

**Note**: GTAV-NightRain is significantly more challenging due to night conditions and complex rain patterns.

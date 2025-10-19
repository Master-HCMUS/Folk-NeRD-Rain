# Loss Weights Guide for Night Rain Deraining

## ⚠️ Critical Issue: Edge Enhancement Sharpens Rain Streaks

### The Problem

When training deraining models, **certain loss functions can inadvertently enhance rain streaks** instead of removing them:

1. **EdgeLoss** - Penalizes differences in high-frequency edges
   - Rain streaks ARE edges
   - High edge_weight → model preserves rain edges
   - **Symptom**: Output has sharper, more visible rain

2. **SSIM Loss** - Preserves structural similarity
   - Can preserve rain structure in dark regions
   - Too high weight → maintains rain patterns
   - **Symptom**: Rain patterns preserved with higher contrast

3. **Low-Light Enhancement** - Brightens dark regions
   - Amplifies ALL content including rain
   - If applied BEFORE deraining → disaster
   - **Symptom**: Bright, visible rain streaks in enhanced areas

---

## ✅ Recommended Loss Weights

### For Enhanced Loss (with Perceptual + SSIM)

```python
CombinedNightRainLoss(
    char_weight=1.0,           # Main reconstruction loss - KEEP HIGH
    fft_weight=0.01,           # Frequency domain - gentle constraint
    edge_weight=0.01,          # ⚠️ REDUCED from 0.05 - avoid sharpening rain
    perceptual_weight=0.1,     # VGG16 features - semantic preservation
    ssim_weight=0.3,           # ⚠️ REDUCED from 0.5 - avoid preserving rain structure
    illumination_weight=0.2,   # ⚠️ REDUCED from 0.3 - gentler brightening
    color_weight=0.01          # Color consistency - gentle
)
```

### For Standard Loss (no Perceptual/SSIM)

```python
loss = loss_char + 0.01 * loss_fft + 0.01 * loss_edge + 0.1 * loss_l1
#                                     ^^^ REDUCED from 0.05
```

---

## 📊 Impact Comparison

| Edge Weight | Effect on Rain | Effect on Scene Details |
|-------------|----------------|-------------------------|
| **0.05** (OLD) | ❌ Sharpens rain streaks | ✅ Sharp scene edges |
| **0.01** (NEW) | ✅ Removes rain smoothly | ✅ Still preserves important edges |
| **0.00** | ✅✅ Best rain removal | ⚠️ May blur scene details slightly |

| SSIM Weight | Effect on Rain | Effect on Structure |
|-------------|----------------|---------------------|
| **0.5** (OLD) | ❌ Preserves rain patterns | ✅ Strong structural preservation |
| **0.3** (NEW) | ✅ Better rain removal | ✅ Sufficient structure preservation |
| **0.0** | ✅✅ Best rain removal | ⚠️ May lose some structural details |

---

## 🔧 How to Fix Existing Training

### Option 1: Restart Training with Corrected Weights

**Best for**: Early training (< 20 epochs)

```bash
# Stop current training
# Update enhanced_losses.py with new default weights
python train_enhanced.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --use_enhanced_loss \
    --use_ema \
    --mixed_precision \
    --progressive_training \
    --gradient_accumulation 4 \
    --use_augmentation \
    --num_epochs 50 \
    --session night_rain_fixed_losses
```

### Option 2: Continue Training with Manual Weights

**Best for**: Late training (> 20 epochs, don't want to restart)

Add explicit weight arguments:

```bash
python train_enhanced.py \
    --resume /path/to/checkpoint.pth \
    --ssim_weight 0.3 \
    --illumination_weight 0.2 \
    # ... other args
```

**Note**: This requires adding CLI arguments to `train_enhanced.py`:

```python
# Add to argument parser
parser.add_argument('--edge_weight', default=0.01, type=float)
parser.add_argument('--ssim_weight', default=0.3, type=float)
parser.add_argument('--illumination_weight', default=0.2, type=float)

# Update criterion creation
criterion = CombinedNightRainLoss(
    edge_weight=args.edge_weight,
    ssim_weight=args.ssim_weight,
    illumination_weight=args.illumination_weight,
    # ...
)
```

### Option 3: Test-Time Fixes (if model already trained)

If your model already sharpens rain, you can apply post-processing:

```python
# In test.py, after restoration
import cv2

def soften_rain_artifacts(image, kernel_size=3):
    """Gentle bilateral filtering to remove sharpened rain"""
    # Convert to numpy
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    # Bilateral filter (preserves edges but smooths rain streaks)
    filtered = cv2.bilateralFilter(img_np, kernel_size, 75, 75)
    return torch.from_numpy(filtered.transpose(2, 0, 1))

# Apply after restoration
restored = model(input_)
restored = soften_rain_artifacts(restored[0])
```

---

## 🧪 Debugging: Is EdgeLoss Sharpening Rain?

### Visual Test

1. **Train for 10 epochs** with old weights (edge_weight=0.05)
2. **Evaluate on validation set**
3. **Compare with ground truth**:
   - If output has MORE visible rain than input → EdgeLoss too high
   - If output has sharper rain streaks → EdgeLoss sharpening

### Loss Monitoring

Check TensorBoard:

```bash
tensorboard --logdir checkpoints/Deraining/models/
```

Look for:
- `train/edge_loss` - Should DECREASE over time
- `val/psnr` - Should INCREASE over time
- If edge_loss decreases but PSNR doesn't improve → edge weight too high

### Quantitative Test

```python
# Compute edge magnitude before/after
def edge_magnitude(img):
    sobel_x = torch.nn.functional.conv2d(img, sobel_x_kernel)
    sobel_y = torch.nn.functional.conv2d(img, sobel_y_kernel)
    return torch.sqrt(sobel_x**2 + sobel_y**2).mean()

input_edges = edge_magnitude(input_img)
output_edges = edge_magnitude(restored_img)
target_edges = edge_magnitude(gt_img)

print(f"Input edge magnitude: {input_edges:.4f}")
print(f"Output edge magnitude: {output_edges:.4f}")  # Should be closer to target
print(f"Target edge magnitude: {target_edges:.4f}")

# If output_edges > input_edges → Model is sharpening!
```

---

## 📖 Loss Function Best Practices for Deraining

### Core Principle

> **Rain removal is OPPOSITE of edge preservation**
> Rain streaks are high-frequency edges that should be REMOVED, not preserved

### Recommended Priority

1. **Charbonnier Loss (weight=1.0)** - Main pixel-wise reconstruction
2. **Perceptual Loss (weight=0.1)** - Semantic content preservation (VGG features are robust to rain)
3. **FFT Loss (weight=0.01)** - Frequency consistency (gentle)
4. **SSIM Loss (weight=0.3)** - Structural similarity (moderate to avoid preserving rain structure)
5. **Edge Loss (weight=0.01)** - Edge preservation (LOW to avoid preserving rain edges)
6. **Illumination Loss (weight=0.2)** - Night-specific brightening (gentle)

### What NOT to Do

❌ **High EdgeLoss weight** (> 0.05) - Preserves rain edges
❌ **Pre-deraining brightness enhancement** - Amplifies rain
❌ **High SSIM weight** (> 0.5) - Can preserve rain patterns
❌ **Using LowLightEnhancementModule before deraining** - Sharpens rain
❌ **Gradient clipping too aggressive** (< 0.5) - Can cause edge artifacts

### What TO Do

✅ **Pixel-wise losses dominant** (Charbonnier, L1)
✅ **Perceptual loss moderate** - Semantic understanding
✅ **Edge/SSIM losses gentle** - Preserve important structures only
✅ **Illumination-aware loss** - Handles dark regions correctly
✅ **Post-deraining enhancement** (if needed)

---

## 🎯 Expected Results

### With Corrected Weights

**Training behavior**:
- PSNR increases steadily (25-28 dB by epoch 50)
- Edge loss decreases but doesn't dominate
- Perceptual loss provides semantic guidance

**Output quality**:
- ✅ Clean removal of rain streaks
- ✅ No sharpening artifacts
- ✅ Preserved scene details (buildings, signs, lights)
- ✅ Natural-looking night scenes

### With Old Weights (edge_weight=0.05)

**Training behavior**:
- PSNR plateaus or fluctuates
- Edge loss dominates training
- Model focuses on preserving edges (including rain)

**Output quality**:
- ❌ Sharp, visible rain streaks
- ❌ Enhanced rain contrast
- ✅ Sharp scene edges (but also rain edges!)
- ⚠️ Unnatural over-sharpening

---

## 🔬 Advanced: Adaptive Loss Weighting

For even better results, consider **adaptive loss weights** based on training progress:

```python
class AdaptiveLossWeights:
    def __init__(self):
        self.epoch = 0
    
    def get_weights(self, epoch):
        """Gradually increase perceptual/SSIM, decrease edge"""
        self.epoch = epoch
        
        # Start with strong pixel-wise losses
        if epoch < 10:
            return {
                'char': 1.0,
                'edge': 0.005,  # Very low initially
                'perceptual': 0.05,
                'ssim': 0.2
            }
        # Gradually shift to perceptual
        elif epoch < 30:
            return {
                'char': 1.0,
                'edge': 0.01,
                'perceptual': 0.1,
                'ssim': 0.3
            }
        # Final stable weights
        else:
            return {
                'char': 1.0,
                'edge': 0.01,
                'perceptual': 0.15,  # Slightly higher
                'ssim': 0.3
            }
```

---

## Summary

**The fix applied**:
1. ✅ Reduced `edge_weight` from **0.05 → 0.01** in both standard and enhanced losses
2. ✅ Reduced `ssim_weight` from **0.5 → 0.3** to avoid preserving rain structure
3. ✅ Reduced `illumination_weight` from **0.3 → 0.2** for gentler brightening
4. ✅ Added warning to `LowLightEnhancementModule` about pre-deraining usage

**Expected impact**:
- Rain streaks will be removed more smoothly
- No artificial sharpening of rain
- Scene details still well-preserved
- Better PSNR on validation set

**If you're mid-training**: Consider restarting with new weights if < 20 epochs. Otherwise continue and expect gradual improvement.

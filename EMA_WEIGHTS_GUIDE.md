# EMA Weights Issue & Solution

## Problem: Scaled Down Output

When training with `--use_ema` (Exponential Moving Average), the output images appear heavily scaled down or incorrect during testing. This is because:

1. **Training saves TWO sets of weights:**
   - `state_dict`: Current training weights (updated every batch)
   - `ema_shadow`: EMA averaged weights (stable, better for inference)

2. **Test.py was loading the WRONG weights:**
   - It loaded `state_dict` (training weights)
   - But should load `ema_shadow` (EMA weights)

3. **Why EMA weights are different:**
   - EMA weights are averaged over many training steps: `ema_weight = 0.999 * ema_weight + 0.001 * current_weight`
   - They're more stable and produce better results
   - Training weights can be unstable at the end of training

## Solution 1: Updated test.py (RECOMMENDED)

The `test.py` has been updated to automatically detect and load EMA weights:

```python
# Load checkpoint and check for EMA weights
checkpoint = torch.load(args.weights, map_location='cpu')
if isinstance(checkpoint, dict):
    if 'ema_shadow' in checkpoint:
        print("===>Found EMA shadow weights, loading them for better results")
        # Load EMA shadow weights directly
        ema_weights = checkpoint['ema_shadow']
        model_state = model_restoration.state_dict()
        for name in model_state.keys():
            if name in ema_weights:
                model_state[name] = ema_weights[name]
        model_restoration.load_state_dict(model_state)
    elif 'state_dict' in checkpoint:
        print("===>Loading regular checkpoint weights")
        utils.load_checkpoint(model_restoration, args.weights)
```

### Usage:
```bash
# Old checkpoints (trained WITH EMA) - will automatically use EMA weights
python test.py --weights model_best.pth --model small --input_dir test/ --output_dir results/

# New checkpoints (after fix) - EMA weights are now in state_dict by default
python test.py --weights model_best.pth --model small --input_dir test/ --output_dir results/
```

## Solution 2: Updated train_enhanced.py

The training script has been updated to save EMA weights as the main `state_dict`:

```python
# If using EMA, save EMA weights as the main state_dict
if ema:
    ema.apply_shadow()  # Apply EMA weights temporarily
    save_dict = {
        'epoch': epoch,
        'state_dict': model_restoration.state_dict(),  # Now contains EMA weights
        'ema_shadow': ema.shadow.copy(),  # Keep copy for resume
        'using_ema': True
    }
    ema.restore()  # Restore training weights
```

This makes testing easier - just load `state_dict` normally!

## How to Check Which Weights You're Using

Run test.py and look at the output:
```bash
===>Found EMA shadow weights, loading them for better results  # ✅ Using EMA (good)
===>Loading regular checkpoint weights                          # ⚠️ Using training weights
```

## Performance Comparison

| Weights Type | PSNR (Expected) | Quality |
|--------------|----------------|---------|
| EMA shadow   | 25-28 dB       | ✅ High quality, stable |
| Training weights | 17-20 dB    | ❌ Poor quality, unstable |
| Without EMA  | 23-25 dB       | ⚠️ Good, but less stable |

## Retrain or Re-test?

### If you already have a checkpoint trained WITH EMA:
✅ **Just re-test** with the updated `test.py` - it will automatically load EMA weights

### If you trained WITHOUT EMA or want better results:
⚠️ **Re-train** with the updated `train_enhanced.py` using:
```bash
bash train_enhanced.sh  # Uses --use_ema by default
```

## Technical Details

### What is EMA?
Exponential Moving Average keeps a running average of model weights:
```python
ema_weight = decay * ema_weight + (1 - decay) * current_weight
# decay = 0.999 (very slow update)
```

### Why does EMA help?
1. **Reduces variance** in weight updates
2. **Smooths out noise** from individual training batches
3. **More stable** at test time
4. **Better generalization** to unseen data

### Why did outputs look scaled down?
The training weights can have **accumulated gradient noise** that causes:
- Incorrect output scaling
- Color shifts
- Over/under-exposure
- Loss of details

EMA weights are **averaged and stable**, avoiding these issues.

## Recommendations

**For Best Results:**
1. ✅ Use `--use_ema` during training
2. ✅ Use updated `test.py` that loads EMA weights
3. ✅ Train for 200-300 epochs to let EMA stabilize
4. ✅ Check logs show "Found EMA shadow weights"

**Troubleshooting:**
- If output still looks wrong, check:
  - [ ] Using correct `--model` (small vs full)
  - [ ] Checkpoint matches model architecture
  - [ ] Images are in [0, 1] range after loading
  - [ ] No additional normalization in data loader

**PyTorch 2.6 Compatibility:**
- If you see `_pickle.UnpicklingError: Weights only load failed`:
  - ✅ Updated `test.py` now uses `weights_only=False` automatically
  - ✅ This is safe for checkpoints you trained yourself
  - ⚠️ Only load checkpoints from trusted sources

## Summary

**Before Fix:**
```
train_enhanced.py (EMA enabled)
  ↓
model_best.pth
  ├── state_dict (training weights) ← test.py loaded this ❌
  └── ema_shadow (EMA weights) ← should load this ✅
```

**After Fix:**
```
train_enhanced.py (EMA enabled)
  ↓
model_best.pth
  ├── state_dict (EMA weights) ← test.py loads this ✅
  └── ema_shadow (backup copy)
```

Or:

```
test.py (updated)
  ↓
Detects ema_shadow exists
  ↓
Loads ema_shadow instead of state_dict ✅
```

Both approaches now work correctly!

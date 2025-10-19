"""
Checkpoint Inspector - Check what's inside your checkpoint file
Helps diagnose EMA weights and model architecture issues
"""

import torch
import argparse

parser = argparse.ArgumentParser(description='Inspect checkpoint file')
parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint file')
args = parser.parse_args()

print("="*80)
print("CHECKPOINT INSPECTOR")
print("="*80)

# Handle PyTorch 2.6+ compatibility
try:
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
except TypeError:
    # Fallback for older PyTorch versions
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        exit(1)
except Exception as e:
    print(f"\n❌ Error loading checkpoint: {e}")
    exit(1)

print(f"\n📁 File: {args.checkpoint}")
print(f"📦 Type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"\n🔑 Keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Check for EMA
    if 'ema_shadow' in checkpoint:
        print(f"\n✅ EMA WEIGHTS FOUND!")
        print(f"   EMA shadow contains {len(checkpoint['ema_shadow'])} parameters")
        print(f"   📊 This checkpoint was trained with --use_ema")
    else:
        print(f"\n⚠️  NO EMA WEIGHTS")
        print(f"   This checkpoint was trained WITHOUT --use_ema")
    
    # Check state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"\n📊 Model State Dict:")
        print(f"   Total parameters: {len(state_dict)}")
        
        # Check for model type based on key names
        sample_keys = list(state_dict.keys())[:5]
        print(f"\n   Sample parameter names:")
        for key in sample_keys:
            print(f"     - {key}: {state_dict[key].shape}")
        
        # Detect model type
        if any('patch_embed_small' in key for key in state_dict.keys()):
            print(f"\n   🏷️  Model Type: SMALL (model_S.py)")
            print(f"      Use: --model small")
        else:
            print(f"\n   🏷️  Model Type: FULL (model.py)")
            print(f"      Use: --model full")
    
    # Check training info
    if 'epoch' in checkpoint:
        print(f"\n📈 Training Info:")
        print(f"   Epoch: {checkpoint['epoch']}")
    
    if 'best_psnr' in checkpoint:
        print(f"   Best PSNR: {checkpoint['best_psnr']:.4f} dB")
    
    if 'using_ema' in checkpoint:
        print(f"   Using EMA: {checkpoint['using_ema']}")
        if checkpoint['using_ema']:
            print(f"   ✅ State dict contains EMA weights (ready for test.py)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if 'ema_shadow' in checkpoint and 'using_ema' not in checkpoint:
        print(f"   ⚠️  This is an OLD checkpoint with EMA weights")
        print(f"   📝 Updated test.py will automatically load EMA weights")
        print(f"   ✅ Just run: python test.py --weights {args.checkpoint} --model [small/full]")
    elif 'ema_shadow' in checkpoint and checkpoint.get('using_ema', False):
        print(f"   ✅ This is a NEW checkpoint with EMA in state_dict")
        print(f"   ✅ Run: python test.py --weights {args.checkpoint} --model [small/full]")
    else:
        print(f"   ℹ️  This checkpoint was trained WITHOUT EMA")
        print(f"   ✅ Run: python test.py --weights {args.checkpoint} --model [small/full]")
        print(f"   💡 For better results, retrain with --use_ema")

else:
    print(f"\n⚠️  Checkpoint is not a dictionary (old format)")
    print(f"   Trying to count parameters...")
    try:
        print(f"   Total parameters: {len(checkpoint)}")
    except:
        print(f"   Could not determine parameter count")

print(f"\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)

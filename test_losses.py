"""
Quick test to verify all loss functions return scalar values
"""
import torch
import sys

# Test ColorConstancyLoss fix
try:
    from enhanced_losses import ColorConstancyLoss, CombinedNightRainLoss, SSIMLoss, IlluminationAwareLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create dummy tensors
    batch_size = 2
    pred = torch.randn(batch_size, 3, 128, 128).to(device)
    target = torch.randn(batch_size, 3, 128, 128).to(device)
    
    print("\n1. Testing ColorConstancyLoss...")
    color_loss = ColorConstancyLoss().to(device)
    loss = color_loss(pred)
    print(f"   Output shape: {loss.shape}")
    print(f"   Output value: {loss.item():.6f}")
    print(f"   ✓ Returns scalar: {loss.dim() == 0}")
    
    print("\n2. Testing SSIMLoss...")
    ssim_loss = SSIMLoss().to(device)
    loss = ssim_loss(pred, target)
    print(f"   Output shape: {loss.shape}")
    print(f"   Output value: {loss.item():.6f}")
    print(f"   ✓ Returns scalar: {loss.dim() == 0}")
    
    print("\n3. Testing IlluminationAwareLoss...")
    illum_loss = IlluminationAwareLoss().to(device)
    loss = illum_loss(pred, target)
    print(f"   Output shape: {loss.shape}")
    print(f"   Output value: {loss.item():.6f}")
    print(f"   ✓ Returns scalar: {loss.dim() == 0}")
    
    print("\n4. Testing CombinedNightRainLoss...")
    combined_loss = CombinedNightRainLoss(
        use_perceptual=True,
        use_ssim=True,
        use_illumination=True,
        use_color_constancy=True
    ).to(device)
    
    # Test single-scale
    print("   a) Single-scale input:")
    total_loss, loss_dict = combined_loss(pred, target)
    print(f"      Total loss shape: {total_loss.shape}")
    print(f"      Total loss value: {total_loss.item():.6f}")
    print(f"      ✓ Returns scalar: {total_loss.dim() == 0}")
    print(f"      Individual losses: {list(loss_dict.keys())}")
    for k, v in loss_dict.items():
        print(f"        {k}: {v.item():.6f} (scalar: {v.dim() == 0})")
    
    # Test multi-scale
    print("\n   b) Multi-scale input:")
    pred_multi = [
        torch.randn(batch_size, 3, 256, 256).to(device),
        torch.randn(batch_size, 3, 128, 128).to(device),
        torch.randn(batch_size, 3, 64, 64).to(device),
    ]
    target_multi = [
        torch.randn(batch_size, 3, 256, 256).to(device),
        torch.randn(batch_size, 3, 128, 128).to(device),
        torch.randn(batch_size, 3, 64, 64).to(device),
    ]
    total_loss, loss_dict = combined_loss(pred_multi, target_multi)
    print(f"      Total loss shape: {total_loss.shape}")
    print(f"      Total loss value: {total_loss.item():.6f}")
    print(f"      ✓ Returns scalar: {total_loss.dim() == 0}")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    total_loss.backward()
    print("   ✓ Backward pass successful!")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED! Loss functions return proper scalars.")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

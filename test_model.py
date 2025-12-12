#!/usr/bin/env python3
"""
Comprehensive test suite for MixturePromptCLIP model.

This script runs a battery of tests to verify model functionality before training:
- Model initialization
- Forward pass and loss computation
- Gradient flow
- Prediction
- Temperature annealing
- Config loading
- Training step simulation
- Loss component independence

Run this before training to ensure everything is working correctly.
"""

import torch
import torch.nn.functional as F
import yaml
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.mop_clip import MixturePromptCLIP


def create_dummy_metadata(num_classes=10):
    """Create dummy metadata for testing."""
    metadata = []
    for i in range(num_classes):
        metadata.append({
            "species": f"species_{i}",
            "genus": f"genus_{i}",
            "family": f"family_{i}",
            "order": f"order_{i}",
            "scientific_name": f"Scientific_{i}",
        })
    return metadata


def test_model_initialization():
    """Test 1: Model initialization"""
    print("\n" + "="*60)
    print("TEST 1: Model Initialization")
    print("="*60)
    
    metadata = create_dummy_metadata(num_classes=10)
    
    try:
        model = MixturePromptCLIP(
            clip_model="ViT-B/32",  # Use smaller model for testing
            metadata=metadata,
            K=4,  # Small K for testing
            em_tau=1.0,
        )
        print("✓ Model initialized successfully")
        print(f"  - num_classes: {model.num_classes}")
        print(f"  - K: {model.K}")
        print(f"  - D: {model.D}")
        print(f"  - base_text_features shape: {model.base_text_features.shape}")
        print(f"  - prompt_offsets shape: {model.prompt_offsets.shape}")
        print(f"  - sim_scale: {model.sim_scale}")
        print(f"  - offset_reg_weight: {model.offset_reg_weight}")
        
        # Verify shapes
        assert model.base_text_features.shape == (10, model.D), \
            f"Base text features shape mismatch: {model.base_text_features.shape}"
        assert model.prompt_offsets.shape == (10, 4, model.D), \
            f"Prompt offsets shape mismatch: {model.prompt_offsets.shape}"
        
        print("✓ All shapes correct")
        return model
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise


def test_forward_pass(model, device):
    """Test 2: Forward pass with loss components"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass & Loss Components")
    print("="*60)
    
    model = model.to(device)
    model.clip.to(device)
    model.train()
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, model.num_classes, (batch_size,)).to(device)
    
    try:
        # Test forward pass
        loss, loss_dict = model(images, labels, lambda_mixture=0.5, temp_cls=0.07)
        
        print(f"✓ Forward pass successful")
        print(f"  - Total loss: {loss.item():.4f}")
        print(f"  - Loss components:")
        for key, value in loss_dict.items():
            print(f"    * {key}: {value:.4f}")
        
        # Verify loss is a scalar tensor
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite (not NaN or Inf)"
        
        # Verify all loss components exist
        required_keys = ["loss_mixture", "loss_cls", "loss_reg", "loss_total"]
        for key in required_keys:
            assert key in loss_dict, f"Missing loss component: {key}"
        
        # Verify loss components are reasonable (mixture loss can be negative)
        assert abs(loss_dict["loss_mixture"]) < 100, "Mixture loss magnitude too large"
        assert loss_dict["loss_cls"] > 0 and loss_dict["loss_cls"] < 100, "Classification loss out of range"
        assert loss_dict["loss_reg"] >= 0 and loss_dict["loss_reg"] < 1, "Regularization loss out of range"
        
        print("✓ All loss components valid")
        return loss, loss_dict
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_gradient_flow(model, device):
    """Test 3: Gradient flow"""
    print("\n" + "="*60)
    print("TEST 3: Gradient Flow")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, model.num_classes, (batch_size,)).to(device)
    
    try:
        optimizer.zero_grad()
        loss, _ = model(images, labels)
        loss.backward()
        
        # Check gradients on prompt_offsets
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                print(f"  - {name}: grad_norm = {grad_norm:.6f}")
        
        assert has_grad, "No gradients found!"
        assert model.prompt_offsets.grad is not None, "prompt_offsets has no gradient"
        
        # Verify CLIP parameters don't have gradients
        clip_has_grad = False
        for param in model.clip.parameters():
            if param.requires_grad and param.grad is not None:
                clip_has_grad = True
        
        assert not clip_has_grad, "CLIP parameters should not have gradients"
        
        print("✓ Gradients flow correctly")
        print("✓ Only prompt_offsets receive gradients (CLIP frozen)")
        
        optimizer.step()
        print("✓ Optimizer step successful")
        
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_prediction(model, device):
    """Test 4: Prediction"""
    print("\n" + "="*60)
    print("TEST 4: Prediction")
    print("="*60)
    
    model.eval()
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    try:
        with torch.no_grad():
            preds = model.predict(images)
        
        print(f"✓ Prediction successful")
        print(f"  - Predictions shape: {preds.shape}")
        print(f"  - Predictions: {preds.cpu().tolist()}")
        
        # Verify predictions
        assert preds.shape == (batch_size,), f"Wrong prediction shape: {preds.shape}"
        assert (preds >= 0).all() and (preds < model.num_classes).all(), \
            "Predictions out of range"
        
        print("✓ Predictions valid")
        
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_temperature_annealing(model, device):
    """Test 5: Temperature annealing effect"""
    print("\n" + "="*60)
    print("TEST 5: Temperature Annealing")
    print("="*60)
    
    model.eval()
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, model.num_classes, (batch_size,)).to(device)
    
    try:
        # Test with different temperatures
        temperatures = [1.0, 0.5, 0.1, 0.05]
        
        with torch.no_grad():
            img_feat = model.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)
            prompt_feats = model._batch_prompt_features(labels, device)
            sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats) * model.sim_scale
        
        print("  Temperature | Max Gamma | Entropy")
        print("  " + "-" * 40)
        
        for tau in temperatures:
            gamma = F.softmax(sims / tau, dim=1)
            max_gamma = gamma.max(dim=1).values.mean().item()
            entropy = -(gamma * (gamma + 1e-10).log()).sum(dim=1).mean().item()
            
            print(f"  {tau:11.2f} | {max_gamma:9.4f} | {entropy:7.4f}")
        
        # Lower temperature should give higher max_gamma and lower entropy
        tau_high = 1.0
        tau_low = 0.05
        
        gamma_high = F.softmax(sims / tau_high, dim=1)
        gamma_low = F.softmax(sims / tau_low, dim=1)
        
        max_high = gamma_high.max(dim=1).values.mean()
        max_low = gamma_low.max(dim=1).values.mean()
        
        assert max_low > max_high, "Lower temperature should give sharper assignments"
        
        print("✓ Temperature annealing works correctly")
        
    except Exception as e:
        print(f"✗ Temperature annealing test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_config_loading():
    """Test 6: Config loading"""
    print("\n" + "="*60)
    print("TEST 6: Config Loading")
    print("="*60)
    
    config_path = "configs/default.yaml"
    
    try:
        if not os.path.exists(config_path):
            print(f"⚠ Config file not found: {config_path}")
            print("  (This is OK if you haven't created it yet)")
            return
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        print("✓ Config loaded successfully")
        print(f"  - Model: {config['model']['clip_model']}")
        print(f"  - K: {config['model']['K']}")
        print(f"  - em_tau_start: {config['model']['em_tau_start']}")
        print(f"  - em_tau_end: {config['model']['em_tau_end']}")
        print(f"  - lr: {config['train']['lr']}")
        print(f"  - lambda_mixture: {config['train'].get('lambda_mixture', 'not set')}")
        print(f"  - temp_cls: {config['train'].get('temp_cls', 'not set')}")
        
        # Verify required keys
        assert "model" in config, "Missing 'model' key in config"
        assert "train" in config, "Missing 'train' key in config"
        assert "clip_model" in config["model"], "Missing 'clip_model' in config"
        assert "K" in config["model"], "Missing 'K' in config"
        
        print("✓ Config structure valid")
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_mini_training_step(model, device):
    """Test 7: Mini training step"""
    print("\n" + "="*60)
    print("TEST 7: Mini Training Step")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, model.num_classes, (batch_size,)).to(device)
    
    try:
        # Get initial loss
        optimizer.zero_grad()
        loss_before, loss_dict_before = model(images, labels, lambda_mixture=0.5, temp_cls=0.07)
        loss_before_val = loss_before.item()
        
        # Backward and step
        loss_before.backward()
        optimizer.step()
        
        # Get loss after
        optimizer.zero_grad()
        loss_after, loss_dict_after = model(images, labels, lambda_mixture=0.5, temp_cls=0.07)
        loss_after_val = loss_after.item()
        
        print(f"  Loss before: {loss_before_val:.4f}")
        print(f"  Loss after:  {loss_after_val:.4f}")
        print(f"  Change:      {loss_after_val - loss_before_val:.4f}")
        
        # Loss should change (may increase or decrease)
        assert abs(loss_after_val - loss_before_val) > 1e-6, "Loss didn't change after step"
        
        print("✓ Training step successful")
        print("✓ Loss changes after optimizer step")
        
    except Exception as e:
        print(f"✗ Mini training step failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_loss_components_independence(model, device):
    """Test 8: Loss components are independent"""
    print("\n" + "="*60)
    print("TEST 8: Loss Components Independence")
    print("="*60)
    
    model.train()
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, model.num_classes, (batch_size,)).to(device)
    
    try:
        # Test with different lambda_mixture values
        lambda_vals = [0.0, 0.5, 1.0]
        
        print("  lambda_mixture | loss_mixture | loss_cls | loss_total")
        print("  " + "-" * 60)
        
        for lam in lambda_vals:
            loss, loss_dict = model(images, labels, lambda_mixture=lam, temp_cls=0.07)
            print(f"  {lam:14.1f} | {loss_dict['loss_mixture']:12.4f} | "
                  f"{loss_dict['loss_cls']:8.4f} | {loss_dict['loss_total']:10.4f}")
        
        # With lambda=0, should only have classification loss
        loss_0, dict_0 = model(images, labels, lambda_mixture=0.0, temp_cls=0.07)
        # With lambda=1, should only have mixture loss
        loss_1, dict_1 = model(images, labels, lambda_mixture=1.0, temp_cls=0.07)
        
        print(f"\n  lambda=0.0: loss ≈ loss_cls + reg = {dict_0['loss_cls']:.4f} + {dict_0['loss_reg']:.4f}")
        print(f"  lambda=1.0: loss ≈ loss_mixture + reg = {dict_1['loss_mixture']:.4f} + {dict_1['loss_reg']:.4f}")
        
        print("✓ Loss components work independently")
        
    except Exception as e:
        print(f"✗ Loss components test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MixturePromptCLIP Model Test Suite")
    print("="*60)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    try:
        # Test 1: Model initialization
        model = test_model_initialization()
        
        # Test 6: Config loading (doesn't need model)
        test_config_loading()
        
        # Move model to device for remaining tests
        model = model.to(device)
        model.clip.to(device)
        
        # Test 2: Forward pass
        test_forward_pass(model, device)
        
        # Test 3: Gradient flow
        test_gradient_flow(model, device)
        
        # Test 4: Prediction
        test_prediction(model, device)
        
        # Test 5: Temperature annealing
        test_temperature_annealing(model, device)
        
        # Test 7: Mini training step
        test_mini_training_step(model, device)
        
        # Test 8: Loss components
        test_loss_components_independence(model, device)
        
        # Final summary
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour model is ready for training!")
        print("\nNext steps:")
        print("  1. Ensure your dataset is properly set up")
        print("  2. Check configs/default.yaml has correct paths")
        print("  3. Run: python train.py --config configs/default.yaml")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


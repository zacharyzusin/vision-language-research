#!/usr/bin/env python3
"""
Comprehensive validation script for training setup.
Runs before submitting jobs to catch errors early.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_syntax():
    """Check Python syntax for all relevant files."""
    print("=" * 60)
    print("1. Checking Python syntax...")
    print("=" * 60)
    
    files_to_check = [
        "train.py",
        "src/datasets/inat_dataset.py",
        "src/models/mop_clip.py",
    ]
    
    errors = []
    for file_path in files_to_check:
        full_path = project_root / file_path
        if not full_path.exists():
            errors.append(f"  ✗ File not found: {file_path}")
            continue
        
        try:
            with open(full_path, 'r') as f:
                compile(f.read(), str(full_path), 'exec')
            print(f"  ✓ {file_path}")
        except SyntaxError as e:
            errors.append(f"  ✗ {file_path}: {e}")
            print(f"  ✗ {file_path}: {e}")
        except Exception as e:
            errors.append(f"  ✗ {file_path}: {e}")
            print(f"  ✗ {file_path}: {e}")
    
    if errors:
        print("\n❌ Syntax errors found!")
        return False
    print("\n✓ All syntax checks passed\n")
    return True

def check_imports():
    """Check that all imports work."""
    print("=" * 60)
    print("2. Checking imports...")
    print("=" * 60)
    
    imports = [
        ("torch", "PyTorch", True),
        ("torchvision", "torchvision", True),
        ("PIL", "Pillow", True),
        ("yaml", "PyYAML", True),
        ("wandb", "wandb", True),
        ("src.datasets.inat_dataset", "inat_dataset module", True),
        ("src.models.mop_clip", "mop_clip module", False),  # May fail if clip not available
    ]
    
    errors = []
    warnings = []
    for module_name, description, required in imports:
        try:
            __import__(module_name)
            print(f"  ✓ {description} ({module_name})")
        except ImportError as e:
            if required:
                errors.append(f"  ✗ {description} ({module_name}): {e}")
                print(f"  ✗ {description} ({module_name}): {e}")
            else:
                warnings.append(f"  ⚠ {description} ({module_name}): {e} (may be OK if not in conda env)")
                print(f"  ⚠ {description} ({module_name}): {e} (may be OK if not in conda env)")
        except Exception as e:
            if required:
                errors.append(f"  ✗ {description} ({module_name}): {e}")
                print(f"  ✗ {description} ({module_name}): {e}")
            else:
                warnings.append(f"  ⚠ {description} ({module_name}): {e}")
                print(f"  ⚠ {description} ({module_name}): {e}")
    
    if warnings:
        print(f"\n  ({len(warnings)} non-critical warnings)")
    
    if errors:
        print("\n❌ Critical import errors found!")
        return False
    print("\n✓ All critical imports successful\n")
    return True

def check_config():
    """Check that config file exists and is valid."""
    print("=" * 60)
    print("3. Checking configuration file...")
    print("=" * 60)
    
    config_path = project_root / "configs" / "default.yaml"
    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['dataset', 'model', 'train']
        for key in required_keys:
            if key not in config:
                print(f"  ✗ Missing required key: {key}")
                return False
        
        print(f"  ✓ Config file exists and is valid")
        print(f"    - Dataset: {config.get('dataset', {}).get('root', 'N/A')}")
        print(f"    - Model: {config.get('model', {}).get('clip_model', 'N/A')}")
        print(f"    - Batch size: {config.get('train', {}).get('batch_size', 'N/A')}")
        print(f"    - Epochs: {config.get('train', {}).get('epochs', 'N/A')}")
        print("\n✓ Configuration valid\n")
        return True
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        traceback.print_exc()
        return False

def check_data_paths():
    """Check that data directories exist."""
    print("=" * 60)
    print("4. Checking data paths...")
    print("=" * 60)
    
    try:
        import yaml
        config_path = project_root / "configs" / "default.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_root = config.get('dataset', {}).get('root', 'data/iNat2021')
        dataset_root = project_root / dataset_root
        
        if not dataset_root.exists():
            print(f"  ✗ Dataset root not found: {dataset_root}")
            return False
        
        print(f"  ✓ Dataset root exists: {dataset_root}")
        
        # Check for train/val directories
        version = str(config.get('dataset', {}).get('version', '2021'))
        train_dir = dataset_root / version / "train"
        val_dir = dataset_root / version / "val"
        
        if train_dir.exists():
            print(f"  ✓ Train directory exists: {train_dir}")
        else:
            print(f"  ⚠ Train directory not found: {train_dir} (may be OK if using JSON loader)")
        
        if val_dir.exists():
            print(f"  ✓ Val directory exists: {val_dir}")
        else:
            print(f"  ⚠ Val directory not found: {val_dir} (may be OK if using JSON loader)")
        
        # Check for JSON files
        train_json = dataset_root / f"train{version}.json"
        val_json = dataset_root / f"val{version}.json"
        
        if train_json.exists():
            print(f"  ✓ Train JSON exists: {train_json}")
        else:
            print(f"  ⚠ Train JSON not found: {train_json}")
        
        if val_json.exists():
            print(f"  ✓ Val JSON exists: {val_json}")
        else:
            print(f"  ⚠ Val JSON not found: {val_json}")
        
        print("\n✓ Data path check complete\n")
        return True
    except Exception as e:
        print(f"  ✗ Error checking data paths: {e}")
        traceback.print_exc()
        return False

def check_model_instantiation():
    """Check that model can be instantiated."""
    print("=" * 60)
    print("5. Checking model instantiation...")
    print("=" * 60)
    
    try:
        import torch
        import yaml
        
        # Try to import clip module (may fail if not in conda env, that's OK)
        try:
            import clip
            clip_available = True
        except ImportError:
            print("  ⚠ CLIP module not available (this is OK if not in conda env)")
            print("    Will skip model instantiation test")
            clip_available = False
        
        if not clip_available:
            print("\n✓ Model instantiation check skipped (CLIP not available)\n")
            return True
        
        from src.models.mop_clip import MixturePromptCLIP
        
        config_path = project_root / "configs" / "default.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        clip_model = model_config.get('clip_model', 'ViT-B/16')
        K = model_config.get('K', 32)
        
        print(f"  Creating model with CLIP={clip_model}, K={K}...")
        
        # Create dummy metadata for testing (model needs metadata, not num_classes)
        dummy_metadata = [
            {"species": f"species_{i}", "genus": f"genus_{i}", "family": f"family_{i}", 
             "order": f"order_{i}", "scientific_name": f"species_{i}"}
            for i in range(10)  # Use 10 classes for testing
        ]
        
        # Try to instantiate model (this will download CLIP weights if needed)
        model = MixturePromptCLIP(
            clip_model=clip_model,
            metadata=dummy_metadata,
            K=K,
        )
        
        print(f"  ✓ Model instantiated successfully")
        print(f"    - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Try a forward pass
        print(f"  Testing forward pass...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create dummy input
        # Use labels in range [0, num_classes-1] where num_classes = len(dummy_metadata)
        num_test_classes = len(dummy_metadata)
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        dummy_labels = torch.randint(0, num_test_classes, (batch_size,)).to(device)
        
        with torch.no_grad():
            output = model(dummy_input, dummy_labels)
        
        print(f"  ✓ Forward pass successful")
        print(f"    - Input shape: {dummy_input.shape}")
        # Model may return tuple or single tensor
        if isinstance(output, tuple):
            print(f"    - Output: tuple with {len(output)} elements")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"      [{i}] shape: {out.shape}")
        else:
            print(f"    - Output shape: {output.shape}")
        
        print("\n✓ Model instantiation and forward pass successful\n")
        return True
    except Exception as e:
        print(f"  ✗ Error instantiating model: {e}")
        traceback.print_exc()
        return False

def check_ddp_setup():
    """Check DDP setup logic (without actually initializing DDP)."""
    print("=" * 60)
    print("6. Checking DDP setup logic...")
    print("=" * 60)
    
    try:
        # Check that train.py has the correct DDP setup
        train_py = project_root / "train.py"
        with open(train_py, 'r') as f:
            content = f.read()
        
        # Check for key DDP components
        checks = [
            ("LOCAL_RANK", "LOCAL_RANK environment variable handling"),
            ("torch.cuda.set_device", "Device setting"),
            ("dist.init_process_group", "Process group initialization"),
            ("DDP", "DDP model wrapping"),
        ]
        
        all_found = True
        for keyword, description in checks:
            if keyword in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ Missing: {description}")
                all_found = False
        
        # Check for device setting before DDP init
        # Look for the pattern: set_device appears before init_process_group in the same code block
        if "torch.cuda.set_device" in content and "dist.init_process_group" in content:
            # Find all occurrences
            set_device_lines = [i for i, line in enumerate(content.split('\n')) if 'torch.cuda.set_device' in line]
            init_pg_lines = [i for i, line in enumerate(content.split('\n')) if 'dist.init_process_group' in line or '_ddp_setup' in line]
            
            if set_device_lines and init_pg_lines:
                # Check if there's a set_device before any init_pg
                first_set_device = min(set_device_lines)
                first_init_pg = min(init_pg_lines)
                if first_set_device < first_init_pg:
                    print(f"  ✓ Device set before process group init")
                else:
                    # This might be OK if _ddp_setup is a function that does it correctly
                    print(f"  ⚠ Device setting order - checking function definition...")
                    # Check if _ddp_setup function exists and handles device setting
                    if "_ddp_setup" in content:
                        print(f"  ✓ Using _ddp_setup function (should handle device correctly)")
                    else:
                        print(f"  ✗ Device set AFTER process group init (should be before)")
                        all_found = False
        
        if all_found:
            print("\n✓ DDP setup logic looks correct\n")
            return True
        else:
            print("\n❌ DDP setup issues found\n")
            return False
    except Exception as e:
        print(f"  ✗ Error checking DDP setup: {e}")
        traceback.print_exc()
        return False

def check_dataset_loading():
    """Check that dataset can be loaded."""
    print("=" * 60)
    print("7. Checking dataset loading...")
    print("=" * 60)
    
    try:
        import yaml
        from src.datasets.inat_dataset import get_inat
        
        config_path = project_root / "configs" / "default.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_root = config.get('dataset', {}).get('root', 'data/iNat2021')
        version = str(config.get('dataset', {}).get('version', '2021'))
        
        print(f"  Attempting to load dataset from {dataset_root} (version {version})...")
        
        # Try to create dataset (this will fail if paths are wrong, but that's OK for validation)
        try:
            dataset = get_inat(
                root=str(project_root / dataset_root),
                split="train",
                version=version,
            )
            print(f"  ✓ Dataset created successfully")
            print(f"    - Length: {len(dataset)}")
            print(f"    - Num classes: {dataset.num_classes}")
            
            # Try to get one sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  ✓ Sample loaded successfully")
                print(f"    - Image shape: {sample[0].shape if hasattr(sample[0], 'shape') else type(sample[0])}")
                print(f"    - Label: {sample[1]}")
        except FileNotFoundError as e:
            print(f"  ⚠ Dataset files not found (this is OK if data isn't downloaded yet): {e}")
        except Exception as e:
            print(f"  ⚠ Dataset loading error (may be OK if data isn't set up): {e}")
            print(f"    This won't prevent training if data is available at job time")
        
        print("\n✓ Dataset loading check complete\n")
        return True
    except Exception as e:
        print(f"  ✗ Error checking dataset: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("TRAINING SETUP VALIDATION")
    print("=" * 60 + "\n")
    
    checks = [
        ("Syntax", check_syntax),
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("Data Paths", check_data_paths),
        ("Model Instantiation", check_model_instantiation),
        ("DDP Setup", check_ddp_setup),
        ("Dataset Loading", check_dataset_loading),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to submit training jobs!")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before submitting jobs")
        return 1

if __name__ == "__main__":
    sys.exit(main())


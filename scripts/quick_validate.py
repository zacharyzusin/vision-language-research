#!/usr/bin/env python3
"""
Quick validation script - checks only critical issues that would cause immediate failures.
Runs in seconds instead of minutes.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_syntax():
    """Check Python syntax for all relevant files."""
    print("1. Checking syntax...")
    files_to_check = [
        "train.py",
        "src/datasets/inat_dataset.py",
        "src/models/mop_clip.py",
    ]
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"  ✗ File not found: {file_path}")
            return False
        try:
            with open(full_path, 'r') as f:
                compile(f.read(), str(full_path), 'exec')
            print(f"  ✓ {file_path}")
        except SyntaxError as e:
            print(f"  ✗ {file_path}: {e}")
            return False
    print("  ✓ All syntax OK\n")
    return True

def check_critical_imports():
    """Check only critical imports that would cause immediate failure."""
    print("2. Checking critical imports...")
    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        from src.datasets.inat_dataset import get_inat, extract_hierarchical_metadata
        print("  ✓ inat_dataset")
    except Exception as e:
        print(f"  ✗ inat_dataset: {e}")
        return False
    
    try:
        # Just check if file can be imported, don't instantiate model
        import src.models.mop_clip
        print("  ✓ mop_clip (import only)")
    except Exception as e:
        print(f"  ✗ mop_clip: {e}")
        return False
    
    print("  ✓ All critical imports OK\n")
    return True

def check_config():
    """Check config file exists and has required keys."""
    print("3. Checking config...")
    config_path = project_root / "configs" / "default.yaml"
    if not config_path.exists():
        print(f"  ✗ Config not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required = ['dataset', 'model', 'train']
        for key in required:
            if key not in config:
                print(f"  ✗ Missing key: {key}")
                return False
        
        print(f"  ✓ Config valid (batch_size={config['train'].get('batch_size')}, epochs={config['train'].get('epochs')})\n")
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False

def check_ddp_setup():
    """Quick check of DDP setup in train.py."""
    print("4. Checking DDP setup...")
    train_py = project_root / "train.py"
    with open(train_py, 'r') as f:
        content = f.read()
    
    # Check critical DDP components exist
    if "LOCAL_RANK" not in content:
        print("  ✗ Missing LOCAL_RANK handling")
        return False
    if "torch.cuda.set_device" not in content:
        print("  ✗ Missing device setting")
        return False
    if "dist.init_process_group" not in content and "_ddp_setup" not in content:
        print("  ✗ Missing DDP initialization")
        return False
    
    # Check device is set before DDP init
    lines = content.split('\n')
    set_device_line = None
    init_pg_line = None
    for i, line in enumerate(lines):
        if 'torch.cuda.set_device' in line:
            set_device_line = i
        if 'dist.init_process_group' in line or '_ddp_setup' in line:
            if init_pg_line is None:
                init_pg_line = i
    
    if set_device_line is not None and init_pg_line is not None:
        if set_device_line > init_pg_line:
            print("  ⚠ Device set after DDP init (check if _ddp_setup handles it)")
        else:
            print("  ✓ Device set before DDP init")
    
    print("  ✓ DDP setup looks OK\n")
    return True

def main():
    """Run quick validation checks."""
    print("=" * 60)
    print("QUICK VALIDATION (Fast Checks Only)")
    print("=" * 60 + "\n")
    
    checks = [
        check_syntax,
        check_critical_imports,
        check_config,
        check_ddp_setup,
    ]
    
    all_passed = True
    for check in checks:
        try:
            if not check():
                all_passed = False
        except Exception as e:
            print(f"  ✗ Check failed: {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ QUICK VALIDATION PASSED")
        print("  Ready to submit jobs (full validation can catch more issues)")
        return 0
    else:
        print("✗ QUICK VALIDATION FAILED")
        print("  Fix issues before submitting jobs")
        return 1

if __name__ == "__main__":
    sys.exit(main())


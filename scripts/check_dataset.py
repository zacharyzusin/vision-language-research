#!/usr/bin/env python3
"""
Check if iNaturalist dataset is properly set up.
"""

import os
import sys
import argparse


def check_dataset(root: str, version: str):
    """Check if dataset files exist and are properly structured."""
    print(f"\nChecking iNaturalist {version} dataset at: {root}\n")
    
    if not os.path.exists(root):
        print(f"❌ Directory does not exist: {root}")
        return False
    
    # Required files
    required_files = {
        "train": os.path.join(root, f"train{version}.json"),
        "val": os.path.join(root, f"val{version}.json"),
        "categories": os.path.join(root, "categories.json"),
    }
    
    # Image directory
    image_dir = os.path.join(root, version)
    
    all_good = True
    
    # Check JSON files
    for name, path in required_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {name}.json found ({size:.1f} MB)")
        else:
            print(f"❌ {name}.json NOT FOUND: {path}")
            all_good = False
    
    # Check image directory
    if os.path.exists(image_dir):
        num_subdirs = len([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        print(f"✓ Image directory found: {image_dir} ({num_subdirs} subdirectories)")
    else:
        print(f"❌ Image directory NOT FOUND: {image_dir}")
        all_good = False
    
    if all_good:
        print("\n✅ Dataset appears to be properly set up!")
    else:
        print("\n❌ Dataset is incomplete. Please download and set up the dataset.")
        print("\nFor iNaturalist 2021, you need:")
        print("  1. Download train2021.json, val2021.json, and categories.json")
        print("  2. Download and extract the image archive")
        print("  3. Organize images in the structure shown above")
    
    return all_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check iNaturalist dataset setup")
    parser.add_argument("--root", type=str, default="data/iNat2021", help="Dataset root directory")
    parser.add_argument("--version", type=str, default="2021", choices=["2018", "2021"], help="Dataset version")
    args = parser.parse_args()
    
    success = check_dataset(args.root, args.version)
    sys.exit(0 if success else 1)

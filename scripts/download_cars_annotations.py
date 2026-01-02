#!/usr/bin/env python3
"""Quick script to download Stanford Cars annotation files."""

import os
import urllib.request
from pathlib import Path

def download_file(url, dest_path):
    """Download a file from URL to destination."""
    try:
        print(f"Downloading {dest_path.name}...")
        urllib.request.urlretrieve(url, dest_path)
        if dest_path.exists() and dest_path.stat().st_size > 0:
            size_kb = dest_path.stat().st_size / 1024
            print(f"  ✓ Success! ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  ✗ File is empty")
            if dest_path.exists():
                dest_path.unlink()
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    """Download annotation files."""
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_dir = project_root / "data" / "stanford_cars"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Stanford Cars Annotation Files Downloader")
    print("=" * 60)
    print(f"Target directory: {dataset_dir}")
    print()
    
    # Change to dataset directory for downloads
    original_dir = os.getcwd()
    os.chdir(dataset_dir)
    
    try:
        # Try multiple sources for cars_annos.mat
        annos_sources = [
            "https://github.com/ducha-aiki/manuallyAnnotatedImagesEvaluation/raw/master/cars_annos.mat",
        ]
        
        annos_path = Path("cars_annos.mat")
        if annos_path.exists():
            size_kb = annos_path.stat().st_size / 1024
            print(f"✓ cars_annos.mat already exists ({size_kb:.1f} KB)")
            annos_downloaded = True
        else:
            annos_downloaded = False
            for url in annos_sources:
                if download_file(url, annos_path):
                    annos_downloaded = True
                    break
            
            if not annos_downloaded:
                print()
                print("⚠ Could not download cars_annos.mat automatically")
                print("Please download manually from:")
                print("  https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
        
        # Try to download cars_meta.mat (optional)
        meta_path = Path("cars_meta.mat")
        if meta_path.exists():
            size_kb = meta_path.stat().st_size / 1024
            print(f"✓ cars_meta.mat already exists ({size_kb:.1f} KB)")
        else:
            meta_sources = [
                "https://github.com/ducha-aiki/manuallyAnnotatedImagesEvaluation/raw/master/cars_meta.mat",
            ]
            
            meta_downloaded = False
            for url in meta_sources:
                if download_file(url, meta_path):
                    meta_downloaded = True
                    break
            
            if not meta_downloaded:
                print("⚠ cars_meta.mat not found (optional - dataset will work without it)")
        
        print()
        print("=" * 60)
        
        # Verify
        if annos_path.exists():
            print("✓ cars_annos.mat found - dataset should work!")
        else:
            print("✗ cars_annos.mat missing - dataset will not work")
        
        if meta_path.exists():
            print("✓ cars_meta.mat found")
        else:
            print("⚠ cars_meta.mat missing (optional)")
        
        print("=" * 60)
        
        return 0 if annos_path.exists() else 1
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    import sys
    sys.exit(main())


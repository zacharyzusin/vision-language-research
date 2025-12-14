#!/usr/bin/env python3
"""
Download script for Stanford Cars dataset.
Tries multiple alternative sources since the original URL is broken.
"""

import os
import sys
import requests
import tarfile
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.NC}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.NC}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.NC}")

def download_file(url, output_path, description="file"):
    """Download a file with progress bar."""
    try:
        print_info(f"Downloading {description} from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print_success(f"Downloaded {description}: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            return True
        else:
            print_error(f"Downloaded file is empty or missing")
            return False
    except Exception as e:
        print_error(f"Failed to download {description}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def extract_tar(file_path, extract_to="."):
    """Extract tar/tgz file."""
    try:
        print_info(f"Extracting {file_path}...")
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(path=extract_to)
        print_success(f"Extracted {file_path}")
        return True
    except Exception as e:
        print_error(f"Failed to extract {file_path}: {e}")
        return False

def extract_zip(file_path, extract_to="."):
    """Extract zip file."""
    try:
        print_info(f"Extracting {file_path}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_to)
        print_success(f"Extracted {file_path}")
        return True
    except Exception as e:
        print_error(f"Failed to extract {file_path}: {e}")
        return False

def find_file_in_dir(directory, filename):
    """Recursively find a file in directory."""
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def setup_from_github():
    """Try to download from GitHub repository."""
    print_info("Attempting download from GitHub...")
    
    github_urls = [
        "https://github.com/cyizhuo/Stanford_Cars_dataset/archive/refs/heads/main.zip",
        "https://github.com/cyizhuo/Stanford_Cars_dataset/archive/refs/heads/master.zip",
    ]
    
    for url in github_urls:
        try:
            zip_path = "stanford_cars_github.zip"
            if download_file(url, zip_path, "GitHub repository"):
                if extract_zip(zip_path):
                    # Look for the dataset files
                    repo_dir = None
                    for item in os.listdir("."):
                        if item.startswith("Stanford_Cars_dataset") and os.path.isdir(item):
                            repo_dir = item
                            break
                    
                    if repo_dir:
                        # Check for different possible structures
                        found_files = {}
                        
                        # Look for cars_annos.mat
                        annos_path = find_file_in_dir(repo_dir, "cars_annos.mat")
                        if annos_path:
                            shutil.copy2(annos_path, "cars_annos.mat")
                            found_files['annos'] = True
                        
                        # Look for cars_meta.mat
                        meta_path = find_file_in_dir(repo_dir, "cars_meta.mat")
                        if meta_path:
                            shutil.copy2(meta_path, "cars_meta.mat")
                            found_files['meta'] = True
                        
                        # Look for images
                        for root, dirs, files in os.walk(repo_dir):
                            if "car_ims" in root or any(f.endswith(('.jpg', '.png')) for f in files[:10]):
                                # Found image directory
                                if not os.path.exists("car_ims"):
                                    if "car_ims" in root:
                                        shutil.copytree(root, "car_ims", dirs_exist_ok=True)
                                    else:
                                        # Create car_ims and copy images
                                        os.makedirs("car_ims", exist_ok=True)
                                        for f in files:
                                            if f.endswith(('.jpg', '.png')):
                                                shutil.copy2(os.path.join(root, f), "car_ims")
                                        # Copy from subdirectories
                                        for d in dirs:
                                            src_dir = os.path.join(root, d)
                                            if any(f.endswith(('.jpg', '.png')) for f in os.listdir(src_dir)[:5]):
                                                for img_file in os.listdir(src_dir):
                                                    if img_file.endswith(('.jpg', '.png')):
                                                        shutil.copy2(
                                                            os.path.join(src_dir, img_file),
                                                            os.path.join("car_ims", img_file)
                                                        )
                                found_files['images'] = True
                                break
                        
                        # Cleanup
                        shutil.rmtree(repo_dir, ignore_errors=True)
                        os.remove(zip_path)
                        
                        if found_files:
                            return found_files
        except Exception as e:
            print_warning(f"GitHub download failed: {e}")
            continue
    
    return None

def setup_from_huggingface():
    """Try to download from Hugging Face."""
    print_info("Attempting download from Hugging Face...")
    print_warning("Hugging Face download requires 'datasets' library")
    print_info("Install with: pip install datasets")
    
    try:
        from datasets import load_dataset
        
        print_info("Loading dataset from Hugging Face...")
        dataset = load_dataset("HuggingFaceM4/Stanford-Cars", trust_remote_code=True)
        
        # Create car_ims directory
        os.makedirs("car_ims", exist_ok=True)
        
        # Download images
        print_info("Downloading images from Hugging Face...")
        for split_name in ['train', 'test']:
            if split_name in dataset:
                split_data = dataset[split_name]
                for i, example in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    # Save image
                    if 'image' in example:
                        img = example['image']
                        img_path = f"car_ims/{split_name}_{i:05d}.jpg"
                        img.save(img_path)
        
        # Create annotation file (simplified)
        print_warning("Note: Hugging Face format may require annotation conversion")
        return {'images': True, 'annos': False}
        
    except ImportError:
        print_warning("'datasets' library not installed. Skipping Hugging Face.")
        return None
    except Exception as e:
        print_warning(f"Hugging Face download failed: {e}")
        return None

def setup_from_kaggle():
    """Try to download from Kaggle (requires kaggle API)."""
    print_info("Kaggle download requires manual setup:")
    print_info("1. Install: pip install kaggle")
    print_info("2. Set up Kaggle API credentials")
    print_info("3. Download from: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
    return None

def verify_dataset():
    """Verify that the dataset is properly set up."""
    print_info("Verifying dataset structure...")
    
    verified = True
    
    # Check car_ims directory
    if os.path.exists("car_ims") and os.path.isdir("car_ims"):
        img_files = [f for f in os.listdir("car_ims") if f.endswith(('.jpg', '.png', '.jpeg'))]
        if img_files:
            print_success(f"Found {len(img_files)} images in car_ims/")
        else:
            print_error("car_ims/ directory exists but contains no images")
            verified = False
    else:
        print_error("car_ims/ directory not found")
        verified = False
    
    # Check cars_annos.mat
    if os.path.exists("cars_annos.mat"):
        print_success("cars_annos.mat found")
    else:
        print_warning("cars_annos.mat not found (required for training)")
        verified = False
    
    # Check cars_meta.mat (optional)
    if os.path.exists("cars_meta.mat"):
        print_success("cars_meta.mat found")
    else:
        print_warning("cars_meta.mat not found (optional)")
    
    return verified

def main():
    """Main download function."""
    # Get target directory
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "data/stanford_cars"
    
    # Create directory
    os.makedirs(data_root, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(data_root)
    
    print(f"{Colors.GREEN}{'='*50}")
    print("Stanford Cars Dataset Download Script")
    print(f"{'='*50}{Colors.NC}")
    print(f"Target directory: {os.path.abspath(data_root)}")
    print()
    
    # Try different sources
    sources_tried = []
    
    # Try GitHub first
    result = setup_from_github()
    if result:
        sources_tried.append("GitHub")
        if verify_dataset():
            print()
            print_success("Dataset downloaded and verified successfully!")
            os.chdir(original_dir)
            return 0
    
    # Try Hugging Face
    result = setup_from_huggingface()
    if result:
        sources_tried.append("Hugging Face")
        if verify_dataset():
            print()
            print_success("Dataset downloaded and verified successfully!")
            os.chdir(original_dir)
            return 0
    
    # If we get here, automatic download failed
    print()
    print_error("Automatic download from all sources failed")
    print()
    print_info("Please download manually from one of these sources:")
    print("  1. GitHub: https://github.com/cyizhuo/Stanford_Cars_dataset")
    print("  2. Hugging Face: https://huggingface.co/datasets/HuggingFaceM4/Stanford-Cars")
    print("  3. Kaggle: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
    print("  4. Original (may be broken): https://ai.stanford.edu/~jkrause/cars/car_dataset.html")
    print()
    print_info("After downloading, extract files to:")
    print(f"  {os.path.abspath(data_root)}/")
    print()
    print_info("Required structure:")
    print("  car_ims/          # Directory with all images")
    print("  cars_annos.mat    # Annotations file")
    print("  cars_meta.mat     # Metadata file (optional)")
    
    os.chdir(original_dir)
    return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)


"""
Dataset loader for Caltech101 dataset.

The Caltech101 dataset contains 101 object categories plus 1 background category (102 total classes).
Each class contains approximately 40 to 800 images.

Dataset structure (torchvision format):
    caltech101/
        101_ObjectCategories/
            accordion/
                image_0001.jpg
                ...
            airplanes/
                ...
            ...
        annotations/
            ...
"""

import os
import json
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's Caltech101 dataset
try:
    from torchvision.datasets import Caltech101
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Use CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _make_transform():
    """
    CLIP-style preprocessing:
      - ensure RGB
      - Resize to 224 (short side) with bicubic interpolation
      - Center crop 224
      - Normalize with CLIP mean/std
    """
    def _safe_load(img):
        try:
            return img.convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    return T.Compose([
        _safe_load,
        T.Resize(224, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class Caltech101Dataset(Dataset):
    """
    Custom dataset loader for Caltech101 dataset.
    
    Supports both torchvision's Caltech101 format and custom directory structures.
    
    Note: Caltech101 doesn't have a standard train/test split.
    By convention, we use the first 30 images per class for training and the rest for test/val.
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "val"],
        transform=None,
        use_torchvision: bool = True,
        download: bool = False,
        train_ratio: float = 0.7,
    ):
        """
        Args:
            root: Root directory containing the dataset
            split: Dataset split ("train", "test", or "val" - val maps to test)
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's Caltech101 loader
            download: If True, download the dataset if not present
            train_ratio: Ratio of images to use for training (default 0.7)
        """
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else _make_transform()
        self.use_torchvision = use_torchvision and TORCHVISION_AVAILABLE
        self.train_ratio = train_ratio
        
        # Map "val" to "test" since Caltech101 doesn't have a separate validation set
        if split == "val":
            self.split = "test"
        
        # Try torchvision loader first
        if self.use_torchvision:
            try:
                # Caltech101 doesn't have a built-in split, so we load all data and split manually
                self.base_dataset = Caltech101(
                    root=root,
                    transform=None,  # We'll apply transform ourselves
                    download=download,
                )
                
                # Get all classes and images
                self.classes = self.base_dataset.categories
                self.num_classes = len(self.classes)
                
                # Caltech101 doesn't have a standard split, so we need to create one
                # Group images by category and split per category
                self.samples = []
                
                # Get all indices grouped by category
                category_to_indices = {}
                for idx in range(len(self.base_dataset)):
                    _, category = self.base_dataset[idx]
                    if category not in category_to_indices:
                        category_to_indices[category] = []
                    category_to_indices[category].append(idx)
                
                # Split each category
                for category, indices in category_to_indices.items():
                    indices.sort()  # Ensure deterministic ordering
                    n_train = int(len(indices) * train_ratio)
                    
                    if self.split == "train":
                        split_indices = indices[:n_train]
                    else:  # test or val
                        split_indices = indices[n_train:]
                    
                    for idx in split_indices:
                        # category is already 0-indexed from Caltech101
                        self.samples.append((idx, category))
                
                print(f"Caltech101 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (using torchvision)")
                return
            except Exception as e:
                print(f"Warning: torchvision Caltech101 loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader: try to find class directories
        categories_dir = os.path.join(root, "101_ObjectCategories")
        if not os.path.exists(categories_dir):
            raise FileNotFoundError(
                f"Could not find Caltech101 dataset at {categories_dir}\n"
                f"Expected structure: {root}/101_ObjectCategories/category_name/image_*.jpg"
            )
        
        # Load from directory structure
        self._load_from_directory(categories_dir)
    
    def _load_from_directory(self, categories_dir: str):
        """Load dataset from directory structure."""
        # Get all category directories (sorted for consistent ordering)
        category_dirs = sorted([d for d in os.listdir(categories_dir) 
                               if os.path.isdir(os.path.join(categories_dir, d)) 
                               and not d.startswith('.')])
        
        self.classes = category_dirs
        self.num_classes = len(self.classes)
        category_to_label = {cat: idx for idx, cat in enumerate(self.classes)}
        
        # Collect all images grouped by category
        category_to_images = {}
        for category in self.classes:
            cat_dir = os.path.join(categories_dir, category)
            images = sorted([f for f in os.listdir(cat_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                           and not f.startswith('.')])
            category_to_images[category] = [os.path.join(cat_dir, img) for img in images]
        
        # Split each category
        self.samples = []
        for category, image_paths in category_to_images.items():
            image_paths.sort()  # Ensure deterministic ordering
            n_train = int(len(image_paths) * self.train_ratio)
            
            if self.split == "train":
                split_paths = image_paths[:n_train]
            else:  # test or val
                split_paths = image_paths[n_train:]
            
            label = category_to_label[category]
            for img_path in split_paths:
                self.samples.append((img_path, label))
        
        print(f"Caltech101 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (custom loader)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_torchvision:
            # samples are (base_idx, category)
            base_idx, label = self.samples[idx]
            img, _ = self.base_dataset[base_idx]
        else:
            # samples are (path, label)
            img_path, label = self.samples[idx]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


def get_caltech101(
    root: str,
    split: Literal["train", "test", "val"],
    transform=None,
    use_torchvision: bool = True,
    download: bool = False,
    train_ratio: float = 0.7,
) -> Dataset:
    """
    Get Caltech101 dataset.
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "test", or "val")
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's loader
        download: If True, download the dataset if not present
        train_ratio: Ratio of images to use for training (default 0.7)
    
    Returns:
        Dataset instance
    """
    return Caltech101Dataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
        download=download,
        train_ratio=train_ratio,
    )


def extract_caltech101_metadata(root: str) -> list:
    """
    Extract metadata for Caltech101 dataset.
    
    Returns a list of metadata dicts, one per class.
    """
    try:
        # Try to load using torchvision to get class names
        if TORCHVISION_AVAILABLE:
            try:
                base_dataset = Caltech101(root=root, download=False)
                class_names = base_dataset.categories
            except:
                # Fallback: try to read from directory
                categories_dir = os.path.join(root, "101_ObjectCategories")
                if os.path.exists(categories_dir):
                    class_names = sorted([d for d in os.listdir(categories_dir) 
                                        if os.path.isdir(os.path.join(categories_dir, d)) 
                                        and not d.startswith('.')])
                else:
                    # Final fallback: use generic names
                    class_names = [f"object_{i+1}" for i in range(102)]
        else:
            categories_dir = os.path.join(root, "101_ObjectCategories")
            if os.path.exists(categories_dir):
                class_names = sorted([d for d in os.listdir(categories_dir) 
                                    if os.path.isdir(os.path.join(categories_dir, d)) 
                                    and not d.startswith('.')])
            else:
                class_names = [f"object_{i+1}" for i in range(102)]
    except:
        # Fallback to generic names
        class_names = [f"object_{i+1}" for i in range(102)]
    
    metadata = []
    for class_name in class_names:
        metadata.append({
            "species": class_name.lower(),
            "genus": "object",
            "family": "caltech101",
            "order": "caltech101",
            "scientific_name": class_name.lower(),
            "full_name": class_name,
        })
    
    return metadata


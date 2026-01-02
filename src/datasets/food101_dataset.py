"""
Dataset loader for Food-101 dataset.

The Food-101 dataset contains 101 food categories.
Each class contains approximately 750 training images and 250 test images.

Dataset structure (torchvision format):
    food101/
        images/
            class_name/
                image_0001.jpg
                ...
        meta/
            train.txt
            test.txt
            classes.txt
"""

import os
import json
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's Food101 dataset
try:
    from torchvision.datasets import Food101
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


class Food101Dataset(Dataset):
    """
    Custom dataset loader for Food-101 dataset.
    
    Supports both torchvision's Food101 format and custom directory structures.
    
    Note: Food101 uses 'train' and 'test' splits.
    For validation, we split the training set (70/30).
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "val"],
        transform=None,
        use_torchvision: bool = True,
        download: bool = False,
    ):
        """
        Args:
            root: Root directory containing the dataset
            split: Dataset split ("train", "test", or "val")
                   - "train" -> uses 70% of training data
                   - "val" -> uses 30% of training data (split from train)
                   - "test" -> uses test split from torchvision
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's Food101 loader
            download: If True, download the dataset if not present
        """
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else _make_transform()
        self.use_torchvision = use_torchvision and TORCHVISION_AVAILABLE
        self.download = download
        
        # Try torchvision loader first
        if self.use_torchvision:
            try:
                if split == "val":
                    # Load train split and split it 70/30 for train/val
                    base_dataset = Food101(
                        root=root,
                        split="train",
                        transform=None,  # We'll apply transform ourselves
                        download=download,
                    )
                    self.classes = base_dataset.classes
                    self.num_classes = len(self.classes)
                    
                    # Split train into train/val (70/30)
                    from torch.utils.data import random_split
                    total_len = len(base_dataset)
                    val_len = int(total_len * 0.3)
                    train_len = total_len - val_len
                    _, val_subset = random_split(base_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
                    
                    # Store subset indices and create samples
                    self.subset_indices = val_subset.indices
                    self.base_dataset = base_dataset
                    self.samples = [(idx, base_dataset[idx][1]) for idx in self.subset_indices]
                elif split == "train":
                    # Load train split and use 70% of it
                    base_dataset = Food101(
                        root=root,
                        split="train",
                        transform=None,  # We'll apply transform ourselves
                        download=download,
                    )
                    self.classes = base_dataset.classes
                    self.num_classes = len(self.classes)
                    
                    # Split train into train/val (70/30)
                    from torch.utils.data import random_split
                    total_len = len(base_dataset)
                    val_len = int(total_len * 0.3)
                    train_len = total_len - val_len
                    train_subset, _ = random_split(base_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
                    
                    # Store subset indices and create samples
                    self.subset_indices = train_subset.indices
                    self.base_dataset = base_dataset
                    self.samples = [(idx, base_dataset[idx][1]) for idx in self.subset_indices]
                else:  # test
                    self.base_dataset = Food101(
                        root=root,
                        split="test",
                        transform=None,  # We'll apply transform ourselves
                        download=download,
                    )
                    self.classes = self.base_dataset.classes
                    self.num_classes = len(self.classes)
                    self.samples = [(idx, label) for idx, (_, label) in enumerate(self.base_dataset)]
                
                print(f"Food101 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (using torchvision)")
                return
            except Exception as e:
                print(f"Warning: torchvision Food101 loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader: try to find image directories
        images_dir = os.path.join(root, "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Could not find Food101 dataset at {images_dir}\n"
                f"Expected structure: {root}/images/class_name/*.jpg"
            )
        
        # Load from directory structure
        self._load_from_directory(images_dir, split)
    
    def _load_from_directory(self, images_dir: str, split: str):
        """Load dataset from directory structure."""
        # Food101 structure: images/class_name/image_XXXX.jpg
        class_dirs = sorted([d for d in os.listdir(images_dir) 
                           if os.path.isdir(os.path.join(images_dir, d)) 
                           and not d.startswith('.')])
        
        self.classes = class_dirs
        self.num_classes = len(self.classes)
        class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load split files if available
        meta_dir = os.path.join(self.root, "meta")
        split_file = None
        if os.path.exists(meta_dir):
            if split == "train":
                split_file = os.path.join(meta_dir, "train.txt")
            elif split == "test":
                split_file = os.path.join(meta_dir, "test.txt")
            elif split == "val":
                # For val, we'll use train.txt and split it
                split_file = os.path.join(meta_dir, "train.txt")
        
        if split_file and os.path.exists(split_file):
            # Load from split file
            with open(split_file, 'r') as f:
                image_list = [line.strip() for line in f if line.strip()]
            
            # Format: "class_name/image_XXXX.jpg"
            all_samples = []
            for img_path in image_list:
                class_name = img_path.split('/')[0]
                if class_name in class_to_label:
                    full_path = os.path.join(images_dir, img_path)
                    if os.path.exists(full_path):
                        label = class_to_label[class_name]
                        all_samples.append((full_path, label))
            
            # For val, split the training samples
            if split == "val":
                all_samples.sort()  # Sort for deterministic ordering
                n_train = int(len(all_samples) * 0.7)
                self.samples = all_samples[n_train:]
            elif split == "train":
                all_samples.sort()
                n_train = int(len(all_samples) * 0.7)
                self.samples = all_samples[:n_train]
            else:  # test
                self.samples = all_samples
        else:
            # Fallback: load all images and split manually
            all_samples = []
            for class_name in self.classes:
                class_dir = os.path.join(images_dir, class_name)
                images = sorted([f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                               and not f.startswith('.')])
                label = class_to_label[class_name]
                for img_name in images:
                    img_path = os.path.join(class_dir, img_name)
                    all_samples.append((img_path, label))
            
            # Split samples (simple: use first 70% for train, last 30% for val/test)
            all_samples.sort()
            n_train = int(len(all_samples) * 0.7)
            
            if split == "train":
                self.samples = all_samples[:n_train]
            elif split == "val":
                self.samples = all_samples[n_train:]
            else:  # test
                self.samples = all_samples[n_train:]
        
        print(f"Food101 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (custom loader)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_torchvision:
            if hasattr(self, 'subset_indices'):
                # For train/val splits using subset
                base_idx = self.subset_indices[idx]
                img, label = self.base_dataset[base_idx]
            else:
                # For test split
                base_idx, label = self.samples[idx]
                img, _ = self.base_dataset[base_idx]
        else:
            # Custom loader: samples are (path, label)
            img_path, label = self.samples[idx]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


def get_food101(
    root: str,
    split: Literal["train", "test", "val"],
    transform=None,
    use_torchvision: bool = True,
    download: bool = False,
) -> Dataset:
    """
    Get Food-101 dataset.
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "test", or "val")
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's loader
        download: If True, download the dataset if not present
    
    Returns:
        Dataset instance
    """
    return Food101Dataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
        download=download,
    )


def extract_food101_metadata(root: str) -> list:
    """
    Extract metadata for Food-101 dataset.
    
    Returns a list of metadata dicts, one per class.
    """
    try:
        # Try to load using torchvision to get class names
        if TORCHVISION_AVAILABLE:
            try:
                base_dataset = Food101(root=root, split="train", download=False)
                class_names = base_dataset.classes
            except:
                # Fallback: try to read from directory or classes.txt
                classes_file = os.path.join(root, "meta", "classes.txt")
                if os.path.exists(classes_file):
                    with open(classes_file, 'r') as f:
                        class_names = [line.strip() for line in f if line.strip()]
                else:
                    images_dir = os.path.join(root, "images")
                    if os.path.exists(images_dir):
                        class_names = sorted([d for d in os.listdir(images_dir) 
                                            if os.path.isdir(os.path.join(images_dir, d)) 
                                            and not d.startswith('.')])
                    else:
                        # Final fallback: use generic names
                        class_names = [f"food_class_{i+1}" for i in range(101)]
        else:
            classes_file = os.path.join(root, "meta", "classes.txt")
            if os.path.exists(classes_file):
                with open(classes_file, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
            else:
                images_dir = os.path.join(root, "images")
                if os.path.exists(images_dir):
                    class_names = sorted([d for d in os.listdir(images_dir) 
                                        if os.path.isdir(os.path.join(images_dir, d)) 
                                        and not d.startswith('.')])
                else:
                    class_names = [f"food_class_{i+1}" for i in range(101)]
    except:
        # Fallback to generic names
        class_names = [f"food_class_{i+1}" for i in range(101)]
    
    metadata = []
    for class_name in class_names:
        metadata.append({
            "species": class_name.lower(),
            "genus": "food",
            "family": "food101",
            "order": "food101",
            "scientific_name": class_name.lower(),
            "full_name": class_name,
        })
    
    return metadata


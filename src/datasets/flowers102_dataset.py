"""
Dataset loader for Flowers102 dataset.

The Flowers102 dataset contains 102 classes of flowers with approximately
8,189 images. Each class represents a different flower species.

Dataset structure (torchvision format):
    flowers102/
        flowers102/
            jpg/
                image_00001.jpg
                ...
            labels.txt (optional)
"""

import os
import json
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's Flowers102 dataset
try:
    from torchvision.datasets import Flowers102
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


class Flowers102Dataset(Dataset):
    """
    Custom dataset loader for Flowers102 dataset.
    
    Supports both torchvision's Flowers102 format and custom directory structures.
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "val"],
        transform=None,
        use_torchvision: bool = True,
    ):
        """
        Args:
            root: Root directory containing the dataset
            split: Dataset split ("train", "test", or "val")
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's Flowers102 loader
        """
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else _make_transform()
        self.use_torchvision = use_torchvision and TORCHVISION_AVAILABLE
        
        if self.use_torchvision:
            try:
                self.dataset = Flowers102(
                    root=root,
                    split=split,
                    transform=None,  # We'll apply transform ourselves
                    download=False,  # Don't auto-download, we'll handle it separately
                )
                self.classes = self.dataset.classes
                self.num_classes = len(self.classes)
                self.samples = [(img_idx, self.dataset._labels[img_idx]) for img_idx in range(len(self.dataset))]
                print(f"Using torchvision Flowers102 loader for {self.split} split.")
                return
            except Exception as e:
                print(f"Warning: torchvision Flowers102 loader failed: {e}. Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader fallback
        self._load_from_directory()
        
        # Ensure classes are sorted for consistent indexing
        self.classes = sorted(list(set(self.classes)))
        self.num_classes = len(self.classes)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Re-map samples to sorted class indices
        remapped_samples = []
        for img_path, label_name in self.samples:
            remapped_samples.append((img_path, class_to_idx[label_name]))
        self.samples = remapped_samples
        
        if not self.samples:
            raise RuntimeError(f"No images found for {self.split} split")
    
    def _load_from_directory(self):
        """Load dataset from directory structure."""
        # Flowers102 structure: root/flowers102/jpg/*.jpg
        # Labels are typically in the filename or a separate file
        jpg_dir = os.path.join(self.root, "flowers102", "jpg")
        if not os.path.exists(jpg_dir):
            # Try alternative structure
            jpg_dir = os.path.join(self.root, "jpg")
        
        if not os.path.exists(jpg_dir):
            raise RuntimeError(f"Image directory not found. Tried: {jpg_dir}")
        
        # Load labels file if available
        labels_file = os.path.join(self.root, "flowers102", "labels.txt")
        if not os.path.exists(labels_file):
            labels_file = os.path.join(self.root, "labels.txt")
        
        label_map = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            label_map[parts[0]] = " ".join(parts[1:])
        
        # Load split files if available
        split_files = {
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
        }
        
        split_file = os.path.join(self.root, "flowers102", split_files.get(self.split, "train.txt"))
        if not os.path.exists(split_file):
            split_file = os.path.join(self.root, split_files.get(self.split, "train.txt"))
        
        self.samples = []
        self.classes = []
        
        if os.path.exists(split_file):
            # Load from split file
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    img_name = line
                    img_path = os.path.join(jpg_dir, img_name)
                    
                    if os.path.exists(img_path):
                        # Extract label from filename or label_map
                        label_name = label_map.get(img_name, f"class_{hash(img_name) % 102}")
                        if label_name not in self.classes:
                            self.classes.append(label_name)
                        self.samples.append((img_path, label_name))
        else:
            # Fallback: load all images from jpg_dir
            print(f"Warning: Split file not found at {split_file}, loading all images...")
            for img_name in sorted(os.listdir(jpg_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(jpg_dir, img_name)
                    label_name = label_map.get(img_name, f"class_{hash(img_name) % 102}")
                    if label_name not in self.classes:
                        self.classes.append(label_name)
                    self.samples.append((img_path, label_name))
        
        if not self.samples:
            raise RuntimeError(f"No images found for {self.split} split")
    
    def __len__(self):
        if self.use_torchvision:
            return len(self.dataset)
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_torchvision:
            img, label = self.dataset[idx]
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
        
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new("RGB", (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


def get_flowers102(
    root: str,
    split: Literal["train", "test", "val"],
    transform=None,
    use_torchvision: bool = True,
) -> Dataset:
    """
    Get Flowers102 dataset for specified split.
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "test", or "val")
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's Flowers102 loader
    
    Returns:
        Flowers102Dataset instance
    """
    return Flowers102Dataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
    )


def extract_flowers102_metadata(root: str) -> list:
    """
    Extract metadata for Flowers102 dataset.
    
    Creates metadata compatible with the hierarchical metadata format used by MoP-CLIP.
    
    Args:
        root: Root directory containing the dataset
    
    Returns:
        List of metadata dictionaries (one for each class)
    """
    try:
        train_ds = Flowers102Dataset(root=root, split="train", use_torchvision=True)
        class_names = train_ds.classes
    except:
        try:
            train_ds = Flowers102Dataset(root=root, split="train", use_torchvision=False)
            class_names = train_ds.classes
        except:
            # Fallback: use default class names
            class_names = [f"flower species {i+1}" for i in range(102)]
    
    metadata = []
    for class_name in class_names:
        # Clean up class name
        flower_name = class_name.strip()
        
        # Try to parse genus and species if possible
        parts = flower_name.split()
        genus = parts[0].lower() if len(parts) > 0 else "flower"
        species = " ".join(parts[1:]).lower() if len(parts) > 1 else flower_name.lower()
        
        metadata.append({
            "species": species,
            "genus": genus,
            "family": "flower",
            "order": "angiosperm",
            "scientific_name": flower_name.lower(),
            "full_name": flower_name,
        })
    
    return metadata


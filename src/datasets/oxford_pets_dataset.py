"""
Dataset loader for Oxford-IIIT Pet dataset.

The Oxford-IIIT Pet dataset contains 37 categories of pet images.
Each class contains approximately 200 images.

Dataset structure (torchvision format):
    oxford-pets/
        images/
            Abyssinian_001.jpg
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

# Try to import torchvision's OxfordIIITPet dataset
try:
    from torchvision.datasets import OxfordIIITPet
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


class OxfordPetsDataset(Dataset):
    """
    Custom dataset loader for Oxford-IIIT Pet dataset.
    
    Supports both torchvision's OxfordIIITPet format and custom directory structures.
    
    Note: OxfordIIITPet uses 'trainval' and 'test' splits.
    We map 'train' and 'val' to appropriate splits.
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
                   - "train" -> "trainval" split from torchvision
                   - "val" -> splits trainval into train/val (70/30)
                   - "test" -> "test" split from torchvision
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's OxfordIIITPet loader
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
                # OxfordIIITPet uses 'trainval' and 'test' splits
                # We need to split trainval into train/val (70/30)
                if split in ["train", "val"]:
                    # Load trainval and split it
                    base_dataset = OxfordIIITPet(
                        root=root,
                        split="trainval",
                        target_types="category",
                        transform=None,  # We'll apply transform ourselves
                        download=download,
                    )
                    self.classes = base_dataset.classes
                    self.num_classes = len(self.classes)
                    
                    # Split trainval into train/val (70/30)
                    from torch.utils.data import random_split
                    total_len = len(base_dataset)
                    val_len = int(total_len * 0.3)
                    train_len = total_len - val_len
                    train_subset, val_subset = random_split(base_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
                    
                    # Store subset indices based on split
                    if split == "train":
                        self.subset_indices = train_subset.indices
                    else:  # val
                        self.subset_indices = val_subset.indices
                    
                    self.base_dataset = base_dataset
                    self.samples = [(idx, base_dataset[idx][1]) for idx in self.subset_indices]
                else:  # test
                    self.base_dataset = OxfordIIITPet(
                        root=root,
                        split="test",
                        target_types="category",
                        transform=None,  # We'll apply transform ourselves
                        download=download,
                    )
                    self.classes = self.base_dataset.classes
                    self.num_classes = len(self.classes)
                    self.samples = [(idx, label) for idx, (_, label) in enumerate(self.base_dataset)]
                
                print(f"OxfordPets ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (using torchvision)")
                return
            except Exception as e:
                print(f"Warning: torchvision OxfordIIITPet loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader: try to find image directories
        images_dir = os.path.join(root, "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Could not find OxfordPets dataset at {images_dir}\n"
                f"Expected structure: {root}/images/*.jpg"
            )
        
        # Load from directory structure
        self._load_from_directory(images_dir, split)
    
    def _load_from_directory(self, images_dir: str, split: str):
        """Load dataset from directory structure."""
        # OxfordPets images are named like: "Abyssinian_001.jpg"
        # Class name is the first part before the underscore
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                             and not f.startswith('.')])
        
        # Extract class names from filenames
        class_names = set()
        filename_to_class = {}
        for img_file in image_files:
            # Format: "classname_XXX.jpg"
            class_name = img_file.split('_')[0]
            class_names.add(class_name)
            filename_to_class[img_file] = class_name
        
        self.classes = sorted(list(class_names))
        self.num_classes = len(self.classes)
        class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create samples
        all_samples = []
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            class_name = filename_to_class[img_file]
            label = class_to_label[class_name]
            all_samples.append((img_path, label))
        
        # Split samples (simple: use first 70% for train, last 30% for val/test)
        # This is a simple split - real OxfordPets has predefined splits
        all_samples.sort()  # Sort by path for deterministic ordering
        n_train = int(len(all_samples) * 0.7)
        
        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            # Use last 30% as validation
            self.samples = all_samples[n_train:]
        else:  # test
            self.samples = all_samples[n_train:]
        
        print(f"OxfordPets ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (custom loader)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_torchvision:
            if hasattr(self, 'subset_indices'):
                # For val split using subset
                base_idx = self.subset_indices[idx]
                img, label = self.base_dataset[base_idx]
            else:
                # For train/test splits
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


def get_oxford_pets(
    root: str,
    split: Literal["train", "test", "val"],
    transform=None,
    use_torchvision: bool = True,
    download: bool = False,
) -> Dataset:
    """
    Get Oxford-IIIT Pet dataset.
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "test", or "val")
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's loader
        download: If True, download the dataset if not present
    
    Returns:
        Dataset instance
    """
    return OxfordPetsDataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
        download=download,
    )


def extract_oxford_pets_metadata(root: str) -> list:
    """
    Extract metadata for Oxford-IIIT Pet dataset.
    
    Returns a list of metadata dicts, one per class.
    """
    try:
        # Try to load using torchvision to get class names
        if TORCHVISION_AVAILABLE:
            try:
                base_dataset = OxfordIIITPet(root=root, split="trainval", target_types="category", download=False)
                class_names = base_dataset.classes
            except:
                # Fallback: try to read from directory
                images_dir = os.path.join(root, "images")
                if os.path.exists(images_dir):
                    image_files = sorted([f for f in os.listdir(images_dir) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                                         and not f.startswith('.')])
                    class_names = sorted(list(set([f.split('_')[0] for f in image_files])))
                else:
                    # Final fallback: use generic names
                    class_names = [f"pet_class_{i+1}" for i in range(37)]
        else:
            images_dir = os.path.join(root, "images")
            if os.path.exists(images_dir):
                image_files = sorted([f for f in os.listdir(images_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                                     and not f.startswith('.')])
                class_names = sorted(list(set([f.split('_')[0] for f in image_files])))
            else:
                class_names = [f"pet_class_{i+1}" for i in range(37)]
    except:
        # Fallback to generic names
        class_names = [f"pet_class_{i+1}" for i in range(37)]
    
    metadata = []
    for class_name in class_names:
        metadata.append({
            "species": class_name.lower(),
            "genus": "pet",
            "family": "oxford_pets",
            "order": "oxford_pets",
            "scientific_name": class_name.lower(),
            "full_name": class_name,
        })
    
    return metadata


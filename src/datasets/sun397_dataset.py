"""
Dataset loader for SUN397 (Scene UNderstanding dataset).

The SUN397 dataset contains 397 scene categories.
Each class contains approximately 100-500 images.
Total dataset: ~108,754 images across 397 scene categories.
"""

import os
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's SUN397 dataset
try:
    from torchvision.datasets import SUN397
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


class SUN397Dataset(Dataset):
    """
    Custom dataset loader for SUN397 (Scene UNderstanding dataset).
    
    Supports both torchvision's SUN397 format and custom directory structures.
    
    SUN397 doesn't have official train/val/test splits, so we create our own
    by splitting the data (70% train, 15% val, 15% test) deterministically.
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        transform=None,
        use_torchvision: bool = True,
        download: bool = False,
        train_ratio: float = 0.7,
    ):
        """
        Args:
            root: Root directory containing the dataset
            split: Dataset split ("train", "val", or "test")
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's SUN397 loader
            download: If True, download the dataset if not present
            train_ratio: Ratio of data to use for training (rest split equally between val and test)
        """
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else _make_transform()
        self.use_torchvision = use_torchvision and TORCHVISION_AVAILABLE
        self.download = download
        self.train_ratio = train_ratio
        
        # Try torchvision loader first
        if self.use_torchvision:
            try:
                # SUN397 doesn't have official splits, so we load all data and split it ourselves
                base_dataset = SUN397(
                    root=root,
                    transform=None,  # We'll apply transform ourselves
                    download=download,
                )
                self.classes = base_dataset.classes
                self.num_classes = len(self.classes)
                
                # Create deterministic split
                from torch.utils.data import random_split
                total_len = len(base_dataset)
                train_len = int(total_len * train_ratio)
                val_len = int((total_len - train_len) / 2)
                test_len = total_len - train_len - val_len
                
                train_subset, val_subset, test_subset = random_split(
                    base_dataset, 
                    [train_len, val_len, test_len], 
                    generator=torch.Generator().manual_seed(42)
                )
                
                if split == "train":
                    self.base_dataset = train_subset
                elif split == "val":
                    self.base_dataset = val_subset
                else:  # test
                    self.base_dataset = test_subset
                
                # Create samples list from subset
                self.samples = [(idx, self.base_dataset.dataset[self.base_dataset.indices[idx]][1]) 
                               for idx in range(len(self.base_dataset))]
                
                print(f"SUN397 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (using torchvision)")
                return
            except Exception as e:
                print(f"Warning: torchvision SUN397 loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader fallback
        # SUN397 structure: images/class_name/image_XXXX.jpg
        images_dir = os.path.join(root, "SUN397", "images")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(root, "images")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Could not find SUN397 dataset at {images_dir}\n"
                f"Expected structure: {root}/SUN397/images/class_name/*.jpg"
            )
        
        self._load_from_directory(images_dir, split)
    
    def _load_from_directory(self, images_dir: str, split: str):
        """Load dataset from directory structure."""
        # SUN397 structure: images/class_name/image_XXXX.jpg
        class_dirs = sorted([d for d in os.listdir(images_dir) 
                           if os.path.isdir(os.path.join(images_dir, d)) 
                           and not d.startswith('.')])
        
        self.classes = class_dirs
        self.num_classes = len(self.classes)
        class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all samples
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
        
        # Deterministic split (same seed as torchvision split)
        import random
        random.seed(42)
        random.shuffle(all_samples)
        
        n_total = len(all_samples)
        n_train = int(n_total * self.train_ratio)
        n_val = int((n_total - n_train) / 2)
        n_test = n_total - n_train - n_val
        
        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        else:  # test
            self.samples = all_samples[n_train + n_val:]
        
        print(f"SUN397 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (custom loader)")
    
    def __len__(self):
        if self.use_torchvision:
            return len(self.base_dataset)
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_torchvision:
            img, label = self.base_dataset[idx]
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


def get_sun397(
    root: str,
    split: Literal["train", "val", "test"],
    transform=None,
    use_torchvision: bool = True,
    download: bool = False,
    train_ratio: float = 0.7,
) -> Dataset:
    """
    Get SUN397 (Scene UNderstanding dataset).
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "val", or "test")
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's loader
        download: If True, download the dataset if not present
        train_ratio: Ratio of data to use for training
    
    Returns:
        Dataset instance
    """
    return SUN397Dataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
        download=download,
        train_ratio=train_ratio,
    )


def extract_sun397_metadata(root: str) -> list:
    """
    Extract metadata for SUN397 dataset.
    
    Returns a list of metadata dicts, one per class.
    """
    try:
        # Try to load using torchvision to get class names
        if TORCHVISION_AVAILABLE:
            try:
                base_dataset = SUN397(root=root, download=False)
                class_names = base_dataset.classes
            except:
                # Fallback: try to read from directory
                images_dir = os.path.join(root, "SUN397", "images")
                if not os.path.exists(images_dir):
                    images_dir = os.path.join(root, "images")
                
                if os.path.exists(images_dir):
                    class_names = sorted([d for d in os.listdir(images_dir) 
                                        if os.path.isdir(os.path.join(images_dir, d)) 
                                        and not d.startswith('.')])
                else:
                    # Final fallback: use generic names
                    class_names = [f"scene_{i+1}" for i in range(397)]
        else:
            images_dir = os.path.join(root, "SUN397", "images")
            if not os.path.exists(images_dir):
                images_dir = os.path.join(root, "images")
            
            if os.path.exists(images_dir):
                class_names = sorted([d for d in os.listdir(images_dir) 
                                    if os.path.isdir(os.path.join(images_dir, d)) 
                                    and not d.startswith('.')])
            else:
                class_names = [f"scene_{i+1}" for i in range(397)]
    except:
        # Fallback to generic names
        class_names = [f"scene_{i+1}" for i in range(397)]
    
    metadata = []
    for class_name in class_names:
        scene_name = class_name.replace('_', ' ').replace('/', ' ').strip()
        
        metadata.append({
            "species": scene_name.lower(),
            "genus": "scene",
            "family": "sun397",
            "order": "scene",
            "scientific_name": scene_name.lower(),
            "full_name": scene_name,
        })
    
    return metadata


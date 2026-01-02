"""
Dataset loader for DTD (Describable Textures Dataset).

The DTD dataset contains 47 texture categories.
Each class contains approximately 120 images divided into train/val/test splits.
"""

import os
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's DTD dataset
try:
    from torchvision.datasets import DTD
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


class DTDDataset(Dataset):
    """
    Custom dataset loader for DTD (Describable Textures Dataset).
    
    Supports both torchvision's DTD format and custom directory structures.
    
    DTD has official 'train', 'val', and 'test' splits.
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        transform=None,
        use_torchvision: bool = True,
        download: bool = False,
    ):
        """
        Args:
            root: Root directory containing the dataset
            split: Dataset split ("train", "val", or "test")
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's DTD loader
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
                self.base_dataset = DTD(
                    root=root,
                    split=split,
                    transform=None,  # We'll apply transform ourselves
                    download=download,
                )
                self.classes = self.base_dataset.classes
                self.num_classes = len(self.classes)
                self.samples = [(idx, label) for idx, (_, label) in enumerate(self.base_dataset)]
                
                print(f"DTD ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (using torchvision)")
                return
            except Exception as e:
                print(f"Warning: torchvision DTD loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader fallback
        # DTD structure: images/texture_name/image_XXXX.jpg
        images_dir = os.path.join(root, "dtd", "images")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(root, "images")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"Could not find DTD dataset at {images_dir}\n"
                f"Expected structure: {root}/dtd/images/texture_name/*.jpg"
            )
        
        self._load_from_directory(images_dir, split)
    
    def _load_from_directory(self, images_dir: str, split: str):
        """Load dataset from directory structure."""
        # DTD structure: images/texture_name/image_XXXX.jpg
        class_dirs = sorted([d for d in os.listdir(images_dir) 
                           if os.path.isdir(os.path.join(images_dir, d)) 
                           and not d.startswith('.')])
        
        self.classes = class_dirs
        self.num_classes = len(self.classes)
        class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load split files if available
        split_files_dir = os.path.join(self.root, "dtd", "labels")
        if not os.path.exists(split_files_dir):
            split_files_dir = os.path.join(self.root, "labels")
        
        split_file = None
        if os.path.exists(split_files_dir):
            split_file = os.path.join(split_files_dir, f"{split}1.txt")
            if not os.path.exists(split_file):
                # Try without the '1' suffix
                split_file = os.path.join(split_files_dir, f"{split}.txt")
        
        if split_file and os.path.exists(split_file):
            # Load from split file
            # Format: "texture_name/image_XXXX.jpg"
            with open(split_file, 'r') as f:
                image_list = [line.strip() for line in f if line.strip()]
            
            self.samples = []
            for img_path in image_list:
                class_name = img_path.split('/')[0]
                if class_name in class_to_label:
                    full_path = os.path.join(images_dir, img_path)
                    if os.path.exists(full_path):
                        label = class_to_label[class_name]
                        self.samples.append((full_path, label))
        else:
            # Fallback: load all images (not ideal, but works)
            print(f"Warning: Split file not found, loading all images for {split} split")
            self.samples = []
            for class_name in self.classes:
                class_dir = os.path.join(images_dir, class_name)
                images = sorted([f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                               and not f.startswith('.')])
                label = class_to_label[class_name]
                for img_name in images:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, label))
            
            # Simple split if we couldn't find split files
            if split == "val":
                # Use last 20% for val
                n_val = int(len(self.samples) * 0.2)
                self.samples = self.samples[-n_val:]
            elif split == "test":
                # Use last 20% for test
                n_test = int(len(self.samples) * 0.2)
                self.samples = self.samples[-n_test:]
            else:  # train
                # Use first 60% for train
                n_train = int(len(self.samples) * 0.6)
                self.samples = self.samples[:n_train]
        
        print(f"DTD ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (custom loader)")
    
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


def get_dtd(
    root: str,
    split: Literal["train", "val", "test"],
    transform=None,
    use_torchvision: bool = True,
    download: bool = False,
) -> Dataset:
    """
    Get DTD (Describable Textures Dataset).
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "val", or "test")
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's loader
        download: If True, download the dataset if not present
    
    Returns:
        Dataset instance
    """
    return DTDDataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
        download=download,
    )


def extract_dtd_metadata(root: str) -> list:
    """
    Extract metadata for DTD dataset.
    
    Returns a list of metadata dicts, one per class.
    """
    try:
        # Try to load using torchvision to get class names
        if TORCHVISION_AVAILABLE:
            try:
                base_dataset = DTD(root=root, split="train", download=False)
                class_names = base_dataset.classes
            except:
                # Fallback: try to read from directory
                images_dir = os.path.join(root, "dtd", "images")
                if not os.path.exists(images_dir):
                    images_dir = os.path.join(root, "images")
                
                if os.path.exists(images_dir):
                    class_names = sorted([d for d in os.listdir(images_dir) 
                                        if os.path.isdir(os.path.join(images_dir, d)) 
                                        and not d.startswith('.')])
                else:
                    # Final fallback: use generic names
                    class_names = [f"texture_{i+1}" for i in range(47)]
        else:
            images_dir = os.path.join(root, "dtd", "images")
            if not os.path.exists(images_dir):
                images_dir = os.path.join(root, "images")
            
            if os.path.exists(images_dir):
                class_names = sorted([d for d in os.listdir(images_dir) 
                                    if os.path.isdir(os.path.join(images_dir, d)) 
                                    and not d.startswith('.')])
            else:
                class_names = [f"texture_{i+1}" for i in range(47)]
    except:
        # Fallback to generic names
        class_names = [f"texture_{i+1}" for i in range(47)]
    
    metadata = []
    for class_name in class_names:
        texture_name = class_name.replace('_', ' ').strip()
        
        metadata.append({
            "species": texture_name.lower(),
            "genus": "texture",
            "family": "dtd",
            "order": "texture",
            "scientific_name": texture_name.lower(),
            "full_name": texture_name,
        })
    
    return metadata


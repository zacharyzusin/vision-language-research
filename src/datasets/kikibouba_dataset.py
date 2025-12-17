"""
Dataset loader for KikiBouba dataset.

The KikiBouba dataset is a binary classification dataset with two classes:
- kiki (sharp/angular shapes)
- bouba (round/curved shapes)

Dataset structure:
    kiki_bouba_v2_split/
        train/
            bouba/  (images)
            kiki/   (images) - may be named differently like 'galaga', 'kepike'
        val/ or test/
            bouba/
            kiki/
"""

import os
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _make_transform():
    """Create standard image preprocessing transform for CLIP."""
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class KikiBoubaDataset(Dataset):
    """
    Dataset loader for KikiBouba dataset.
    
    Supports binary classification: kiki vs bouba.
    Automatically detects class directories and maps them to labels.
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        transform=None,
        version: Optional[str] = None,
    ):
        """
        Args:
            root: Root directory containing kiki_bouba_v2_split/
            split: Dataset split ("train", "val", or "test")
            transform: Optional image transform (defaults to CLIP preprocessing)
        """
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else _make_transform()

        # ----------------------------------------
        # Locate dataset directory (v2 or v1)
        # ----------------------------------------
        #
        # We support both:
        #   - kiki_bouba_v2_split/ (current default)
        #   - kiki_bouba_v1_split/ (legacy v1)
        # as well as the case where the user unzips directly into `root/`
        # and `root` itself contains `train/`, `val/` or `test/`.
        #
        # If `version` is provided, we prioritize / require that version;
        # otherwise we will auto-detect whichever split folder exists.
        #
        if version is not None:
            version = version.lower()
            if version not in {"v1", "v2"}:
                raise ValueError(f"Unsupported KikiBouba version: {version}. Expected 'v1' or 'v2'.")

        if version == "v2":
            candidate_roots = [
                os.path.join(root, "kiki_bouba_v2_split"),
                root,
            ]
        elif version == "v1":
            candidate_roots = [
                os.path.join(root, "kiki_bouba_v1_split"),
                root,
            ]
        else:
            candidate_roots = [
                os.path.join(root, "kiki_bouba_v2_split"),
                os.path.join(root, "kiki_bouba_v1_split"),
                root,
            ]

        dataset_dir = None
        for cand in candidate_roots:
            if os.path.exists(os.path.join(cand, split)):
                dataset_dir = cand
                break

        if dataset_dir is None:
            # As a last resort, fall back to the original error message for clarity
            base_msg = "Could not find KikiBouba dataset directory.\n"
            if version == "v2":
                expected = [
                    os.path.join(root, "kiki_bouba_v2_split", split),
                    os.path.join(root, split),
                ]
            elif version == "v1":
                expected = [
                    os.path.join(root, "kiki_bouba_v1_split", split),
                    os.path.join(root, split),
                ]
            else:
                expected = [
                    os.path.join(root, "kiki_bouba_v2_split", split),
                    os.path.join(root, "kiki_bouba_v1_split", split),
                    os.path.join(root, split),
                ]

            tried_str = "\n".join([f"    - {p}" for p in expected])
            raise FileNotFoundError(
                base_msg
                + "  Tried:\n"
                + tried_str
                + "\nPlease ensure you unzipped the dataset so that one of these "
                  "paths exists, or set dataset.version correctly in the config."
            )

        # Find split directory
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            # Try 'test' if 'val' doesn't exist
            if split == "val" and os.path.exists(os.path.join(dataset_dir, "test")):
                split_dir = os.path.join(dataset_dir, "test")
            else:
                raise FileNotFoundError(
                    f"Could not find {split} split directory: {split_dir}"
                )
        
        # Find class directories
        # KikiBouba has multiple classes: bouba, galaga, kepike, kiki, maluma
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        class_dirs.sort()
        
        if len(class_dirs) < 1:
            raise ValueError(
                f"Expected at least 1 class directory in {split_dir}, found: {class_dirs}"
            )
        
        # Map class directories to labels - multiclass classification
        # Each directory becomes its own class
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Sort class directories for consistent ordering
        class_dirs.sort()
        
        # Map each directory to a unique label
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir] = idx
            self.idx_to_class[idx] = class_dir
        
        self.num_classes = len(class_dirs)
        
        # Build list of (image_path, label) tuples
        self.samples = []
        for class_dir in class_dirs:  # Use all class_dirs, not just class_to_idx keys
            label = self.class_to_idx[class_dir]
            class_path = os.path.join(split_dir, class_dir)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, filename)
                    self.samples.append((img_path, label))
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                f"Found class directories: {class_dirs}"
            )
        
        print(f"KikiBouba {split}: {len(self.samples)} samples, {self.num_classes} classes")
        print(f"  Classes: {', '.join([f'{self.idx_to_class[i]} (label {i})' for i in range(self.num_classes)])}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to blank image if loading fails
            print(f"Warning: Failed to load {img_path}: {e}")
            img = Image.new("RGB", (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_kikibouba(
    root: str,
    split: Literal["train", "val", "test"],
    transform=None,
    version: Optional[str] = None,
):
    """
    Get KikiBouba dataset for specified split.
    
    Args:
        root: Root directory containing kiki_bouba_v2_split/
        split: Dataset split ("train", "val", or "test")
        transform: Optional image transform
    
    Returns:
        KikiBoubaDataset instance
    """
    return KikiBoubaDataset(root=root, split=split, transform=transform, version=version)


def extract_kikibouba_metadata(root: str) -> list:
    """
    Extract metadata for KikiBouba dataset.
    
    Creates metadata for multiclass classification, with each class directory
    as a separate class. Compatible with the hierarchical metadata format used by MoP-CLIP.
    
    Args:
        root: Root directory containing the dataset
    
    Returns:
        List of metadata dictionaries (one for each class)
    """
    # Load dataset to get class names
    try:
        train_ds = KikiBoubaDataset(root=root, split="train")
        num_classes = train_ds.num_classes
        class_names = [train_ds.idx_to_class[i] for i in range(num_classes)]
    except:
        # Fallback to default names if dataset can't be loaded
        class_names = ["bouba", "galaga", "kepike", "kiki", "maluma"]
    
    metadata = []
    for class_name in class_names:
        metadata.append({
            "species": class_name.lower(),
            "genus": class_name.lower(),
            "family": "kikibouba",
            "order": "kikibouba",
            "scientific_name": class_name.lower(),
        })
    
    return metadata


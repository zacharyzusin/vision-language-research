"""
iNaturalist Dataset Loading and Preprocessing.

This module provides utilities for loading the iNaturalist dataset (2018 or 2021)
with proper train/val splits and hierarchical metadata extraction.
"""

import os
import json
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import INaturalist
from torchvision.io import read_image
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _make_transform(use_torchvision_io=False):
    """
    CLIP-style preprocessing:
      - ensure RGB
      - Resize to 224 (short side) with bicubic interpolation
      - Center crop 224
      - Normalize with CLIP mean/std
    
    Args:
        use_torchvision_io: If True, expects tensor input (CHW, uint8) from torchvision.io.read_image
                           If False, expects PIL Image input
    """
    if use_torchvision_io:
        # Transform for torchvision.io.read_image output (tensor CHW, uint8 [0-255])
        return T.Compose([
            # Convert uint8 [0-255] to float [0-1]
            T.Lambda(lambda x: x.float() / 255.0),
            # Resize and crop (works on tensors)
            T.Resize(224, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            # Normalize with CLIP mean/std
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        # Original PIL-based transform
        def _safe_load(img):
            try:
                return img.convert("RGB")
            except Exception:
                import PIL.Image as Image
                return Image.new("RGB", (224, 224))

        return T.Compose([
            _safe_load,
            T.Resize(224, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])


def _load_split_paths(root: str, version: str = "2021"):
    """
    Load train/val split paths from JSON files.
    
    Args:
        root: Root directory containing JSON files
        version: Dataset version ("2018" or "2021")
    
    Returns:
      train_rel (set of 'Aves/2761/xxx.jpg')
      val_rel (set of 'Aves/2761/xxx.jpg')
    """
    train_json = os.path.join(root, f"train{version}.json")
    val_json = os.path.join(root, f"val{version}.json")

    with open(train_json, "r") as f:
        train_data = json.load(f)
    with open(val_json, "r") as f:
        val_data = json.load(f)

    def to_rel_set(data):
        rels = set()
        for img in data["images"]:
            # file_name like "train_val2018/Aves/2761/xxx.jpg"
            fname = img["file_name"]
            parts = fname.split("/", 1)
            rel = parts[1] if len(parts) == 2 else fname  # "Aves/2761/xxx.jpg"
            rels.add(rel)
        return rels

    return to_rel_set(train_data), to_rel_set(val_data)


class SubsetDataset(Dataset):
    """
    Wrapper dataset that filters to specific category IDs and remaps labels to [0, num_subset_classes-1].
    """
    
    def __init__(self, base_dataset: Dataset, category_ids: list):
        """
        Args:
            base_dataset: Base dataset (e.g., INatJSONDataset)
            category_ids: List of category IDs to include in subset
        """
        self.base_dataset = base_dataset
        self.category_ids = sorted(category_ids)
        
        # Create mapping from original category_id to subset index [0, len(category_ids)-1]
        self.category_to_subset_idx = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}
        
        # Filter samples to only include those with category_ids in our subset
        # The base_dataset (INatJSONDataset) stores (path, category_id) before remapping
        # We need to access the original category_ids before they're remapped to indices
        self.samples = []
        
        # Access the original samples (before label remapping) from base_dataset
        if hasattr(base_dataset, 'original_samples'):
            # base_dataset.original_samples contains (path, category_id) tuples before remapping
            for path, cat_id in base_dataset.original_samples:
                if cat_id in self.category_to_subset_idx:
                    subset_label = self.category_to_subset_idx[cat_id]
                    self.samples.append((path, subset_label))
        elif hasattr(base_dataset, 'samples'):
            # Fallback: try to reverse map from remapped samples
            # This requires the category_to_idx mapping
            if hasattr(base_dataset, 'category_to_idx'):
                idx_to_category = {idx: cat_id for cat_id, idx in base_dataset.category_to_idx.items()}
                # We can't easily reverse map without storing original category_ids
                # So we'll use the slower method below
                pass
        else:
            # Fallback: iterate through dataset and check labels
            # This is slower but works if we can't access internal samples
            idx_to_category = {}
            if hasattr(base_dataset, 'category_to_idx'):
                idx_to_category = {idx: cat_id for cat_id, idx in base_dataset.category_to_idx.items()}
            
            for idx in range(len(base_dataset)):
                sample = base_dataset[idx]
                if len(sample) == 2:
                    img, label_idx = sample
                    # Map label_idx back to original category_id
                    original_cat_id = idx_to_category.get(label_idx)
                    if original_cat_id in self.category_to_subset_idx:
                        self.samples.append((idx, self.category_to_subset_idx[original_cat_id]))
        
        self.num_classes = len(self.category_ids)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if isinstance(self.samples[0][0], str):
            # samples are (path, label) - need to load image
            path, subset_label = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            if self.base_dataset.transform:
                img = self.base_dataset.transform(img)
            return img, subset_label
        else:
            # samples are (base_idx, label) - delegate to base dataset
            base_idx, subset_label = self.samples[idx]
            img, _ = self.base_dataset[base_idx]
            return img, subset_label


class INatJSONDataset(Dataset):
    """
    Custom dataset loader that reads directly from JSON files.
    This bypasses torchvision's structure requirements and works with raw iNaturalist data.
    """

    def __init__(self, root: str, split: Literal["train", "val"], version: str, transform=None, only_class_id: Optional[int] = None, category_ids: Optional[list] = None):
        self.root = root
        self.split = split
        self.version = version
        self.transform = transform
        self.only_class_id = only_class_id
        self.category_ids = category_ids

        # Load categories.json to get consistent category ordering
        cat_path = os.path.join(root, "categories.json")
        if os.path.exists(cat_path):
            with open(cat_path, "r") as f:
                categories = json.load(f)
            categories.sort(key=lambda c: c["id"])
            # Create mapping from category_id to index [0, C-1]
            self.category_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
            self.num_classes = len(categories)
        else:
            # Fallback: build mapping from data
            self.category_to_idx = None
            self.num_classes = None

        # Load JSON files
        # Try different possible file name formats
        possible_names = [
            os.path.join(root, f"{split}{version}.json"),  # train2021.json, val2021.json
            os.path.join(root, f"{split}_{version}.json"),  # train_2021.json, val_2021.json
            os.path.join(root, f"{split}.json"),  # train.json, val.json
        ]
        
        json_file = None
        for name in possible_names:
            if os.path.exists(name):
                json_file = name
                break
        
        if json_file is None:
            # List what files actually exist
            existing_files = []
            if os.path.exists(root):
                existing_files = [f for f in os.listdir(root) if f.endswith('.json')]
            
            error_msg = (
                f"Could not find JSON file for {split} split. Tried:\n" +
                "\n".join(f"  - {name}" for name in possible_names) +
                f"\n\nDirectory: {root}\n"
            )
            if existing_files:
                error_msg += f"Found JSON files: {', '.join(existing_files)}\n"
            else:
                error_msg += f"No JSON files found in directory.\n"
            error_msg += "\nPlease ensure the dataset is properly set up."
            raise FileNotFoundError(error_msg)
        
        with open(json_file, "r") as f:
            data = json.load(f)

        # Build image_id -> category_id mapping
        ann_dict = {ann["image_id"]: ann["category_id"] for ann in data["annotations"]}

        # Build list of (image_path, category_id) tuples
        self.samples = []
        
        # Try different possible image directory locations
        # For 2021, images might be in 2021/train/ or 2021/val/ or just 2021/
        # The base directory should contain both train/ and val/ subdirectories
        possible_image_dirs = [
            os.path.join(root, version),  # e.g., data/iNat2021/2021 (contains train/ and val/)
            os.path.join(root, version, "train"),  # e.g., data/iNat2021/2021/train (for train split)
            os.path.join(root, version, "val"),  # e.g., data/iNat2021/2021/val (for val split)
            os.path.join(root, "images"),  # e.g., data/iNat2021/images
            root,  # e.g., data/iNat2021 (images directly in root)
        ]
        
        # For 2021, the base directory (2021/) contains train/ and val/ subdirectories
        # So we should use the base directory and let file paths handle train/val/
        image_dir = None
        base_dir = os.path.join(root, version)  # e.g., data/iNat2021/2021
        if os.path.exists(base_dir):
            image_dir = base_dir  # Use base directory, file paths will include train/ or val/
        else:
            # Fallback to other locations
            for img_dir in possible_image_dirs[1:]:  # Skip base_dir, already checked
                if os.path.exists(img_dir):
                    image_dir = img_dir
                    break
        
        if image_dir is None:
            raise FileNotFoundError(
                f"Could not find image directory. Tried:\n" +
                "\n".join(f"  - {d}" for d in possible_image_dirs) +
                f"\n\nPlease ensure images are in one of these locations under: {root}"
            )

        missing_files = 0
        for img_info in data["images"]:
            image_id = img_info["id"]
            category_id = ann_dict[image_id]

            # Skip if filtering by class_id (use original category_id, not mapped index)
            if only_class_id is not None and category_id != only_class_id:
                continue
            
            # Skip if filtering by category_ids list
            if category_ids is not None and category_id not in category_ids:
                continue

            # Extract relative path from file_name
            # For 2021: file_name is like "train/02912_.../file.jpg" or "val/03938_.../file.jpg"
            # For 2018: file_name might be like "train_val2018/Aves/2761/xxx.jpg" or just "Aves/2761/xxx.jpg"
            fname = img_info["file_name"]
            
            # Optimized path resolution: try most likely paths first to minimize os.path.exists() calls
            # Most common case: image_dir + file_name (e.g., "2021/train/02912_.../file.jpg")
            full_path = os.path.join(image_dir, fname)
            if not os.path.exists(full_path):
                # Second most common: if file_name has train/val prefix, try without it
                if fname.startswith("train/") or fname.startswith("val/"):
                    rel_path = fname.split("/", 1)[1]
                    full_path = os.path.join(image_dir, rel_path)
                    if not os.path.exists(full_path):
                        # Try direct from root
                        full_path = os.path.join(root, fname)
                        if not os.path.exists(full_path):
                            # Last resort: try root + relative path
                            full_path = os.path.join(root, rel_path)
                            if not os.path.exists(full_path):
                                full_path = None
                else:
                    # No train/val prefix, try root
                    full_path = os.path.join(root, fname)
                    if not os.path.exists(full_path):
                        full_path = None
            
            if full_path:
                self.samples.append((full_path, category_id))
            else:
                missing_files += 1
        
        if missing_files > 0:
            print(f"Warning: {missing_files} image files not found (out of {len(data['images'])} total)")
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image files found! Check that:\n"
                f"  1. JSON file exists: {json_file}\n"
                f"  2. Image directory exists: {image_dir}\n"
                f"  3. File paths in JSON match actual file locations"
            )

        # If we don't have categories.json, build mapping from data
        if self.category_to_idx is None:
            unique_categories = sorted(set(cat_id for _, cat_id in self.samples))
            self.category_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
            self.num_classes = len(unique_categories)
        
        # Store original category_ids before remapping (needed for SubsetDataset)
        # Store ALL samples with original category_ids, even if filtered
        self.original_samples = [(path, cat_id) for path, cat_id in self.samples]
        
        # Remap category IDs to indices [0, C-1]
        # Only include samples whose category_id exists in our mapping
        self.samples = [(path, self.category_to_idx[cat_id]) 
                       for path, cat_id in self.samples 
                       if cat_id in self.category_to_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Use torchvision.io.read_image for faster I/O (uses libjpeg-turbo/libpng)
            # Returns tensor in CHW format, uint8 [0-255]
            img = read_image(img_path)
            # Ensure RGB (handle grayscale/alpha channels)
            if img.shape[0] == 1:
                # Grayscale: repeat to 3 channels
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                # RGBA: take first 3 channels
                img = img[:3]
            elif img.shape[0] != 3:
                # Unexpected: create blank RGB image
                img = torch.zeros(3, 224, 224, dtype=torch.uint8)
        except Exception as e:
            # Fallback to blank image if loading fails
            img = torch.zeros(3, 224, 224, dtype=torch.uint8)

        if self.transform:
            img = self.transform(img)

        return img, label


class INatSplit(Dataset):
    """
    Thin wrapper around torchvision.datasets.INaturalist that:
      - restricts to a given list of indices
      - returns (image, species_label) with integer labels [0..C-1]
    """

    def __init__(self, base_ds: INaturalist, indices):
        self.base_ds = base_ds
        self.indices = list(indices)

        # Convenience: number of species-level classes
        self.num_classes = len(self.base_ds.all_categories)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, target = self.base_ds[base_idx]

        # If target is a dict (target_type="full"), extract species/category id.
        if isinstance(target, dict):
            # 'category_id' is species-level ID aligned with all_categories
            label = target.get("category_id", None)
            if label is None:
                # fallback: some versions use 'species' key
                label = target.get("species", 0)
        else:
            # If we used target_type="species", target is already an int
            label = target

        label = int(label)
        return img, label




def get_inat(root: str, split: Literal["train", "val"], version: str = "2021", only_class_id: int = None, category_ids: list = None):
    """
    Returns a Dataset corresponding to the official train or val split.
    
    This is a version-agnostic wrapper that supports both 2018 and 2021.
    For 2021, uses a custom JSON-based loader. For 2018, uses torchvision's loader.

    Args:
        root: Root directory containing the dataset
        split: "train" or "val"
        version: Dataset version ("2018" or "2021")
        only_class_id: If set, restricts to only that class ID (int in [0, num_classes))

    Returns:
        Dataset with samples (image_tensor, label_int), where labels are species IDs in [0, num_classes).
    """
    # Use faster torchvision.io-based transform for 2021 (custom loader)
    # For 2018, keep PIL-based transform (torchvision's INaturalist uses PIL internally)
    use_torchvision_io = (version == "2021")
    transform = _make_transform(use_torchvision_io=use_torchvision_io)

    # For 2021, use custom JSON-based loader (more robust, doesn't require torchvision structure)
    if version == "2021":
        dataset = INatJSONDataset(
            root=root,
            split=split,
            version=version,
            transform=transform,
            only_class_id=only_class_id,
            category_ids=category_ids
        )
        
        # If category_ids provided, wrap in SubsetDataset to remap labels
        if category_ids is not None:
            return SubsetDataset(dataset, category_ids)
        return dataset

    # For 2018 and earlier, use torchvision's loader
    # Map version to torchvision's expected format
    tv_version = version

    try:
        base_ds = INaturalist(
            root=root,
            version=tv_version,
            target_type="full",
            transform=transform,
            download=False,
        )

        train_rel, val_rel = _load_split_paths(root, version)
        target_set = train_rel if split == "train" else val_rel

        indices = []
        for idx, (cat_id, fname) in enumerate(base_ds.index):
            rel = os.path.join(base_ds.all_categories[cat_id], fname)
            if rel not in target_set:
                continue

            if only_class_id is not None and cat_id != only_class_id:
                continue  # skip samples not matching the desired class

            indices.append(idx)

        return INatSplit(base_ds, indices)
    except RuntimeError:
        # Fallback to JSON loader if torchvision loader fails
        print(f"Warning: torchvision INaturalist loader failed, falling back to JSON loader")
        return INatJSONDataset(
            root=root,
            split=split,
            version=version,
            transform=transform,
            only_class_id=only_class_id
        )


def get_inat2018(root: str, split: Literal["train", "val"], only_class_id: int = None):
    """
    Returns a Dataset corresponding to the official 2018 train or val split.
    
    DEPRECATED: Use get_inat() with version="2018" instead.
    Kept for backward compatibility.

    If only_class_id is set, restricts to only that class ID (int in [0, num_classes)).

    Output samples are (image_tensor, label_int), where labels are species IDs in [0, num_classes).
    """
    return get_inat(root, split, version="2018", only_class_id=only_class_id)


def extract_hierarchical_metadata(root: str, category_ids: Optional[list] = None) -> list:
    """
    Extract hierarchical metadata from iNaturalist categories.json.

    Args:
        root: Root directory containing categories.json
        category_ids: Optional list of category IDs to filter to (for subset training)

    Returns:
        List of dictionaries, each containing:
            - species: Species name (lowercase)
            - genus: Genus name (lowercase)
            - family: Family name (lowercase)
            - order: Order name (lowercase)
            - scientific_name: Scientific name (lowercase)
        
        The list is sorted by category ID to ensure consistent ordering.
        If category_ids is provided, only returns metadata for those categories.
    """

    cat_path = os.path.join(root, "categories.json")
    with open(cat_path, "r") as f:
        categories = json.load(f)

    # ensure sorted by numeric ID (species id)
    categories.sort(key=lambda c: c["id"])
    
    # Filter to subset if category_ids provided
    if category_ids is not None:
        category_ids_set = set(category_ids)
        categories = [c for c in categories if c["id"] in category_ids_set]
        # Sort by the order in category_ids to maintain consistent label mapping
        id_to_order = {cat_id: idx for idx, cat_id in enumerate(sorted(category_ids))}
        categories.sort(key=lambda c: id_to_order.get(c["id"], 999999))

    metadata = []
    for c in categories:
        metadata.append(
            dict(
                species=c["name"].lower(),
                genus=c["genus"].lower(),
                family=c["family"].lower(),
                order=c["order"].lower(),
                scientific_name=c["name"].lower(),
            )
        )

    return metadata

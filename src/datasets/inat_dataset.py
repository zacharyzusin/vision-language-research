"""
iNaturalist 2018 Dataset Loading and Preprocessing.

This module provides utilities for loading the iNaturalist 2018 dataset
with proper train/val splits and hierarchical metadata extraction.
"""

import os
import json
from typing import Literal

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import INaturalist


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


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
            import PIL.Image as Image
            return Image.new("RGB", (224, 224))

    return T.Compose([
        _safe_load,
        T.Resize(224, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def _load_split_paths(root: str):
    """
    Returns:
      train_rel (set of 'Aves/2761/xxx.jpg')
      val_rel (set of 'Aves/2761/xxx.jpg')
    """
    train_json = os.path.join(root, "train2018.json")
    val_json = os.path.join(root, "val2018.json")

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


class INat2018Split(Dataset):
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


def get_inat2018(root: str, split: Literal["train", "val"], only_class_id: int = None):
    """
    Returns a Dataset corresponding to the official 2018 train or val split.

    If only_class_id is set, restricts to only that class ID (int in [0, num_classes)).

    Output samples are (image_tensor, label_int), where labels are species IDs in [0, num_classes).
    """
    transform = _make_transform()

    base_ds = INaturalist(
        root=root,
        version="2018",
        target_type="full",
        transform=transform,
        download=False,
    )

    train_rel, val_rel = _load_split_paths(root)
    target_set = train_rel if split == "train" else val_rel

    indices = []
    for idx, (cat_id, fname) in enumerate(base_ds.index):
        rel = os.path.join(base_ds.all_categories[cat_id], fname)
        if rel not in target_set:
            continue

        if only_class_id is not None and cat_id != only_class_id:
            continue  # skip samples not matching the desired class

        indices.append(idx)

    return INat2018Split(base_ds, indices)

def extract_hierarchical_metadata(root: str) -> list:
    """
    Extract hierarchical metadata from iNaturalist 2018 categories.json.

    Args:
        root: Root directory containing categories.json

    Returns:
        List of dictionaries, each containing:
            - species: Species name (lowercase)
            - genus: Genus name (lowercase)
            - family: Family name (lowercase)
            - order: Order name (lowercase)
            - scientific_name: Scientific name (lowercase)
        
        The list is sorted by category ID to ensure consistent ordering.
    """

    cat_path = os.path.join(root, "categories.json")
    with open(cat_path, "r") as f:
        categories = json.load(f)

    # ensure sorted by numeric ID (species id)
    categories.sort(key=lambda c: c["id"])

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

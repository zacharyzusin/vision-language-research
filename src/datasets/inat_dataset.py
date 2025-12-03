# src/datasets/inat_dataset.py

import os
import json
from typing import Literal

import torch
from torch.utils.data import Subset
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.datasets import INaturalist


def _make_transform():
    return T.Compose([
        lambda img: img.convert("RGB"), 
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
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


def get_inat2018(root: str, split: Literal["train", "val"]):
    """
    Returns a Subset of torchvision.datasets.INaturalist corresponding
    to the official 2018 train or val split.
    """

    transform = _make_transform()

    # This loads ALL 2018 images from:
    #   root/2018/<supercategory>/<species_id>/*.jpg
    base_ds = INaturalist(
        root=root,
        version="2018",
        target_type="full",
        transform=transform,
        download=False,  # you already have the data
    )

    train_rel, val_rel = _load_split_paths(root)

    target_set = train_rel if split == "train" else val_rel

    indices = []
    for idx, (cat_id, fname) in enumerate(base_ds.index):
        # base_ds.all_categories[cat_id] like "Aves/2761"
        rel = os.path.join(base_ds.all_categories[cat_id], fname)  # "Aves/2761/xxx.jpg"
        if rel in target_set:
            indices.append(idx)

    subset = Subset(base_ds, indices)

    # Attach a num_classes attribute for convenience if you want
    subset.num_classes = len(base_ds.all_categories)

    return subset


def extract_hierarchical_metadata(root: str):
    """
    Read categories.json (unobfuscated taxonomy) and return list of
    dicts with species/genus/family/order/scientific_name per class id.
    """

    cat_path = os.path.join(root, "categories.json")
    with open(cat_path, "r") as f:
        categories = json.load(f)

    # ensure sorted by numeric ID
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

"""
Convert iNaturalist 2018 dataset to torchvision-compatible format.

This script reorganizes the iNaturalist 2018 dataset directory structure
to be compatible with torchvision.datasets.INaturalist.
"""

import os
import json
import shutil
from tqdm import tqdm

BASE = "data/iNat2018"

SRC_IMG_ROOT = os.path.join(BASE, "2018")    # your extracted images folder
TRAIN_JSON = os.path.join(BASE, "train2018.json")
VAL_JSON = os.path.join(BASE, "val2018.json")
CAT_JSON = os.path.join(BASE, "categories.json")

OUT_ROOT = os.path.join(BASE, "torchvision_format")
DEST_2018 = os.path.join(OUT_ROOT, "2018")

os.makedirs(DEST_2018, exist_ok=True)

# ----------------------------------------------------------------------
# Load JSON files
# ----------------------------------------------------------------------
print("Loading annotations...")

with open(TRAIN_JSON) as f:
    train_data = json.load(f)

with open(VAL_JSON) as f:
    val_data = json.load(f)

with open(CAT_JSON) as f:
    categories = json.load(f)

categories.sort(key=lambda x: x["id"])
cat_by_id = {c["id"]: c for c in categories}

# Map species ID → class name
species_to_class = {c["id"]: c["class"] for c in categories}

def link_or_copy(src: str, dst: str):
    """
    Create symbolic link, falling back to copy if symlink fails.

    Args:
        src: Source file path
        dst: Destination file path
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy(src, dst)


def convert_split(split_data: dict):
    """
    Convert a dataset split (train or val) to torchvision format.

    Args:
        split_data: Dictionary containing 'images' and 'annotations' keys
    """
    print("Converting split...")

    ann = {a["image_id"]: a["category_id"] for a in split_data["annotations"]}

    for img in tqdm(split_data["images"]):
        img_id = img["id"]
        category_id = ann[img_id]
        classname = species_to_class[category_id].replace(" ", "_")

        # input path: 2018/Aves/2761/file.jpg
        parts = img["file_name"].split("/", 1)
        rel_path = parts[1] if len(parts) == 2 else img["file_name"]
        src = os.path.join(SRC_IMG_ROOT, rel_path)

        dst = os.path.join(
            DEST_2018,
            classname,
            str(category_id),
            os.path.basename(src)
        )

        link_or_copy(src, dst)

# ----------------------------------------------------------------------
# Run structure creation
# ----------------------------------------------------------------------
convert_split(train_data)
convert_split(val_data)

# ----------------------------------------------------------------------
# Write metadata.json
# ----------------------------------------------------------------------
metadata_path = os.path.join(OUT_ROOT, "metadata.json")

metadata = {
    "categories": {
        "full":    [c["name"]    for c in categories],
        "kingdom": [c["kingdom"] for c in categories],
        "phylum":  [c["phylum"]  for c in categories],
        "class":   [c["class"]   for c in categories],
        "order":   [c["order"]   for c in categories],
        "family":  [c["family"]  for c in categories],
        "genus":   [c["genus"]   for c in categories],
    },
    "class_to_idx": {
        "full": list(range(len(categories)))
    }
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✔ Conversion complete!")
print(f"Torchvision-ready dataset written to:\n  {OUT_ROOT}")

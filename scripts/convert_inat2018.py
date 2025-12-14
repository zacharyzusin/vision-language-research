"""
Convert iNaturalist dataset to torchvision-compatible format.

This script reorganizes the iNaturalist dataset directory structure
to be compatible with torchvision.datasets.INaturalist.
Supports both 2018 and 2021 versions.
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert iNaturalist dataset to torchvision format")
    parser.add_argument("--data_root", type=str, default="data/iNat2021", 
                        help="Root directory containing the dataset")
    parser.add_argument("--version", type=str, default="2021", choices=["2018", "2021"],
                        help="Dataset version")
    args = parser.parse_args()
    
    BASE = args.data_root
    VERSION = args.version

    SRC_IMG_ROOT = os.path.join(BASE, VERSION)    # your extracted images folder
    TRAIN_JSON = os.path.join(BASE, f"train{VERSION}.json")
    VAL_JSON = os.path.join(BASE, f"val{VERSION}.json")
    CAT_JSON = os.path.join(BASE, "categories.json")

    OUT_ROOT = os.path.join(BASE, "torchvision_format")
    DEST_DIR = os.path.join(OUT_ROOT, VERSION)

    os.makedirs(DEST_DIR, exist_ok=True)

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


    def convert_split(split_data: dict, split_name: str):
        """
        Convert a dataset split (train or val) to torchvision format.

        Args:
            split_data: Dictionary containing 'images' and 'annotations' keys
            split_name: Name of the split ("train" or "val")
        """
        print(f"Converting {split_name} split...")

        ann = {a["image_id"]: a["category_id"] for a in split_data["annotations"]}

        for img in tqdm(split_data["images"], desc=f"Processing {split_name}"):
            img_id = img["id"]
            category_id = ann[img_id]
            classname = species_to_class[category_id].replace(" ", "_")

            # input path: 2021/Aves/2761/file.jpg or 2018/Aves/2761/file.jpg
            parts = img["file_name"].split("/", 1)
            rel_path = parts[1] if len(parts) == 2 else img["file_name"]
            src = os.path.join(SRC_IMG_ROOT, rel_path)

            dst = os.path.join(
                DEST_DIR,
                classname,
                str(category_id),
                os.path.basename(src)
            )

            link_or_copy(src, dst)

    # ----------------------------------------------------------------------
    # Run structure creation
    # ----------------------------------------------------------------------
    convert_split(train_data, "train")
    convert_split(val_data, "val")

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


if __name__ == "__main__":
    main()

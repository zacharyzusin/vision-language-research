import os
import json
import shutil
from tqdm import tqdm

ROOT = "data/iNat2018"

# IMPORTANT: your JSON file paths already include "train_val2018/"
# Therefore the real image source directory is:
SRC = os.path.join(ROOT, "train_val2018")  # outer folder

TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR = os.path.join(ROOT, "val")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

def move_from_json(json_path, out_dir):
    print(f"\nReading metadata: {json_path}")
    with open(json_path, "r") as f:
        meta = json.load(f)

    print(f"Copying images into: {out_dir}")
    for ann in tqdm(meta["annotations"]):
        img_id = ann["image_id"]
        rel_path = meta["images"][img_id-1]["file_name"]   # file_name includes "train_val2018/"

        # Build full path
        src_path = os.path.join(ROOT, rel_path)

        if not os.path.exists(src_path):
            print("\n❌ Missing:", src_path)
            raise FileNotFoundError(src_path)

        dst_path = os.path.join(out_dir, "/".join(rel_path.split("/")[1:]))  
        # removes the leading "train_val2018/"

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

print("Starting reorganization...")

move_from_json(os.path.join(ROOT, "train2018.json"), TRAIN_DIR)
move_from_json(os.path.join(ROOT, "val2018.json"), VAL_DIR)

print("\n✔ Done!")

"""
Visualization script for Mixture-of-Prompts CLIP model.

This script visualizes which images are assigned to each sub-prompt for a given class.
It generates grid images showing the top-N images per sub-prompt based on gamma scores.
"""

import os
import argparse
import math
import sys

# Add repository root to PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.datasets import INaturalist

from train import load_config
from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
from src.models.mop_clip import MixturePromptCLIP


def find_class_index_by_name(metadata, name: str):
    """
    Find class index whose 'species' field matches the given name (case-insensitive).
    """
    name = name.lower()
    matches = []
    for idx, md in enumerate(metadata):
        if md["species"].lower() == name:
            matches.append(idx)
    if not matches:
        raise ValueError(f"No species named '{name}' found in metadata.")
    if len(matches) > 1:
        print(f"Warning: multiple matches for '{name}', using first index {matches[0]}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Target class index (0-based).")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Alternatively, species name to search for.")
    parser.add_argument("--top_n", type=int, default=16,
                        help="Top-N images per sub-prompt to visualize.")
    parser.add_argument("--out_dir", type=str, default="subprompt_viz",
                        help="Where to save visualization grids.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------
    # Load config & metadata
    # ---------------------------
    config = load_config(args.config)
    root = config["dataset"]["root"]

    metadata = extract_hierarchical_metadata(root)
    num_classes = len(metadata)
    print(f"Num classes: {num_classes}")

    # Determine target class
    if args.class_idx is not None:
        class_idx = int(args.class_idx)
        if not (0 <= class_idx < num_classes):
            raise ValueError(f"class_idx {class_idx} out of range [0, {num_classes})")
        cls_name = metadata[class_idx]["species"]
    elif args.class_name is not None:
        class_idx = find_class_index_by_name(metadata, args.class_name)
        cls_name = metadata[class_idx]["species"]
    else:
        raise ValueError("You must specify either --class_idx or --class_name.")

    print(f"Target class index: {class_idx} (species='{cls_name}')")

    # ---------------------------
    # Load dataset split
    # ---------------------------
    subset = get_inat2018(root, args.split)
    print(f"Loaded split '{args.split}': {len(subset)} samples")

    # subset is a torch.utils.data.Subset over a base INaturalist dataset
    base_ds = subset.dataset
    indices = subset.indices  # list mapping subset index -> base index

    # Make a second "raw" dataset (no CLIP normalization, just resized tensor) for visualization
    raw_transform = T.Compose([
        lambda img: img.convert("RGB"),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),  # [0,1]
    ])
    raw_base = INaturalist(
        root=root,
        version="2018",
        target_type="full",
        transform=raw_transform,
        download=False,
    )

    # ---------------------------
    # Load model & checkpoint
    # ---------------------------
    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=int(config["model"]["K"]),
        em_tau=float(config["model"].get("em_tau", 1.0)),
    ).to(device)
    model.clip.to(device)

    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    K = model.K
    print(f"Model has K = {K} sub-prompts per class.")

    # ---------------------------
    # Collect top-N images per sub-prompt
    # ---------------------------
    # For each k in [0..K-1], store list of (gamma, subset_idx)
    per_k = {k: [] for k in range(K)}

    print("Scanning dataset for target class samples...")
    for sub_idx in range(len(subset)):
        img, label = subset[sub_idx]
        if int(label) != class_idx:
            continue

        img_batch = img.unsqueeze(0).to(device)
        label_batch = torch.tensor([class_idx], device=device)

        # Compute gamma (soft assignment) using model's internal methods
        with torch.no_grad():
            img_feat = model.clip.encode_image(img_batch).float()
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            prompt_feats = model._batch_prompt_features(label_batch, device)  # (1, K, D)
            sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats) * model.sim_scale
            gamma = torch.nn.functional.softmax(sims / model.em_tau, dim=1)
            gamma = gamma[0].cpu()  # (K,)

        for k in range(K):
            score = float(gamma[k].item())
            per_k[k].append((score, sub_idx))

    # Sort & keep top-N per sub-prompt
    for k in range(K):
        per_k[k].sort(key=lambda x: x[0], reverse=True)
        per_k[k] = per_k[k][: args.top_n]

    # ---------------------------
    # Build and save grids
    # ---------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    for k in range(K):
        entries = per_k[k]
        if not entries:
            print(f"No samples for sub-prompt k={k}, skipping.")
            continue

        imgs = []
        scores = []
        for score, sub_idx in entries:
            base_idx = indices[sub_idx]
            img_raw, _ = raw_base[base_idx]  # tensor [3,H,W]
            imgs.append(img_raw)
            scores.append(score)

        # Build grid
        n = len(imgs)
        nrow = int(math.sqrt(n)) if n >= 4 else n
        grid = vutils.make_grid(imgs, nrow=nrow, padding=2)

        out_path = os.path.join(
            args.out_dir,
            f"class{class_idx}_{cls_name.replace(' ', '_')}_k{k}.png",
        )
        vutils.save_image(grid, out_path)
        print(f"Saved grid for sub-prompt k={k} with {n} images to {out_path}")
        print("  gamma scores:", [f"{s:.3f}" for s in scores])


if __name__ == "__main__":
    main()

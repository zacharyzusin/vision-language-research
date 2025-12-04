#!/usr/bin/env python3
# eval_clip_zeroshot.py

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata


@torch.no_grad()
def build_text_features(model, metadata, device, batch_size=256):
    """
    Build zero-shot text embeddings using multiple templates.
    """
    print("\nBuilding zero-shot text features...")

    templates = [
        "a photo of a {}.",
        "a wildlife photo of a {}.",
        "a close-up photo of a {}.",
        "a natural habitat photo of a {}.",
        "a high quality image of a {}.",
        "an organism known as {}.",
    ]

    prompts = []
    for md in metadata:
        species = md["species"]
        for tmpl in templates:
            prompts.append(tmpl.format(species))

    print(f"Total prompts = {len(prompts)} "
          f"= {len(metadata)} classes × {len(templates)} templates")

    tokenized = clip.tokenize(prompts).to(device)

    all_feats = []
    for i in tqdm(range(0, len(tokenized), batch_size), desc="Encoding text"):
        batch = tokenized[i : i + batch_size]
        feat = model.encode_text(batch)
        feat = F.normalize(feat.float(), dim=-1)
        all_feats.append(feat)

    all_feats = torch.cat(all_feats, dim=0)  # (C*T, D)

    C = len(metadata)
    T = len(templates)
    text_features = all_feats.view(C, T, -1).mean(dim=1)
    text_features = F.normalize(text_features, dim=-1)

    print("Text features:", text_features.shape)
    return text_features


@torch.no_grad()
def evaluate_zero_shot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ------------------------------------------
    # Load CLIP
    # ------------------------------------------
    print("Loading CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # ------------------------------------------
    # Load dataset splits
    # ------------------------------------------
    root = os.path.join("data", "iNat2018")

    print("\nLoading iNat2018 splits...")
    train_ds = get_inat2018(root, split="train")  # only to ensure class ordering
    val_ds = get_inat2018(root, split="val")
    val_ds.transform = preprocess

    # ------------------------------------------
    # Load metadata (defines class ordering)
    # ------------------------------------------
    metadata = extract_hierarchical_metadata(root)
    num_classes = len(metadata)

    print(f"Validation samples: {len(val_ds)}, classes: {num_classes}")

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------
    # Build text embeddings
    # ------------------------------------------
    text_features = build_text_features(model, metadata, device)

    # ------------------------------------------
    # Evaluate
    # ------------------------------------------
    print("\nEvaluating zero-shot CLIP...")

    correct = 0
    total = 0

    for imgs, labels in tqdm(val_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        img_feat = model.encode_image(imgs).float()
        img_feat = F.normalize(img_feat, dim=-1)

        # similarity (B, C)
        sims = img_feat @ text_features.t()
        preds = sims.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += len(labels)

    acc = correct / total
    print("\n======================================")
    print(f"Zero-Shot CLIP (ViT-B/32) Accuracy: {acc:.4f}")
    print("======================================\n")

    return acc


if __name__ == "__main__":
    evaluate_zero_shot()

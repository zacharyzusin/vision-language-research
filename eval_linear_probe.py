#!/usr/bin/env python3
# eval_linear_probe.py

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata


def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr


def train_linear_probe(
    backbone="ViT-B/32",
    root="data/iNat2018",
    batch_size=256,
    epochs=10,
    lr=1e-3,
    warmup_steps=1000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # Load CLIP
    # -----------------------------
    print(f"Loading CLIP {backbone}...")
    clip_model, preprocess = clip.load(backbone, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # -----------------------------
    # Dataset
    # -----------------------------
    print("\nLoading iNat2018 train/val...")
    train_ds = get_inat2018(root, "train")
    val_ds = get_inat2018(root, "val")

    # override transforms to use CLIP preprocess
    train_ds.dataset.transform = preprocess
    val_ds.dataset.transform = preprocess

    metadata = extract_hierarchical_metadata(root)
    num_classes = len(metadata)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    feat_dim = clip_model.visual.output_dim
    print("Feature dim:", feat_dim)

    # -----------------------------
    # Linear classifier
    # -----------------------------
    clf = nn.Linear(feat_dim, num_classes, bias=True).to(device)

    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    max_steps = len(train_loader) * epochs
    step = 0
    best_val = 0.0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        clf.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[LinearProbe] Epoch {epoch}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            cosine_lr(optimizer, lr, step, max_steps, warmup_steps)
            step += 1

            with torch.no_grad():
                feats = clip_model.encode_image(imgs).float()
                feats = F.normalize(feats, dim=-1)

            logits = clf(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch}: Train Loss = {running / len(train_loader):.4f}")

        # -------------------------
        # Validation
        # -------------------------
        clf.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                feats = clip_model.encode_image(imgs).float()
                feats = F.normalize(feats, dim=-1)
                logits = clf(feats)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        acc = correct / total
        print(f"Epoch {epoch}: Val Acc = {acc:.4f}")

        best_val = max(best_val, acc)

    print("\n[LinearProbe] Training complete. Best Val Acc =", best_val)
    return best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/iNat2018")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    args = parser.parse_args()

    train_linear_probe(
        backbone=args.backbone,
        root=args.root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

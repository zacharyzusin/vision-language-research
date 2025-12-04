import os
import argparse
import yaml
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
from src.models.mop_clip import MixturePromptCLIP


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def validate(model, dataloader, device):
    """
    Fast validation using cached prompt features (Option C).
    """
    model.eval()

    total = 0
    correct = 0

    start_time = time.time()

    for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model.predict(imgs)
        correct += (preds == labels).sum().item()
        total += len(labels)

    duration = time.time() - start_time
    print(f"[Validation] time = {duration:.2f}s")

    return correct / total if total else 0.0


def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr


def train(config, resume=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root = config["dataset"]["root"]

    print("\n=== Loading INaturalist 2018 dataset ===")
    train_ds = get_inat2018(root, "train")
    val_ds = get_inat2018(root, "val")

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    metadata = extract_hierarchical_metadata(root)
    num_classes = len(metadata)
    print(f"Num classes (species): {num_classes}")

    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        ctx_len=config["model"]["ctx_len"],
    ).to(device)

    print("\n================= RUN CONFIG SUMMARY =================")
    print(f"Model: {config['model']['clip_model']}")
    print(f"Dataset Root: {root}")
    print(f"Train Samples: {len(train_ds)}")
    print(f"Val Samples:   {len(val_ds)}")
    print(f"Num Classes:   {num_classes}")
    print("=====================================================\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    scaler = GradScaler("cuda" if device == "cuda" else "cpu")

    # Load checkpoint if needed
    start_epoch = 1
    best_val = 0.0
    step = 0
    max_steps = len(train_loader) * config["train"]["epochs"]

    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val_acc"]
        step = ckpt.get("step", 0)
        print(f"Resuming from epoch {start_epoch}")

    os.makedirs("checkpoints", exist_ok=True)

    # -----------------------------------------------------------------
    # BUILD INFERENCE CACHE ONCE BEFORE TRAINING
    # -----------------------------------------------------------------
    print("\n=== Building inference prompt cache BEFORE training ===")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.predict(dummy)  # triggers cache build

    # -----------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------
    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # LR schedule
            cosine_lr(
                optimizer,
                config["train"]["lr"],
                step,
                max_steps,
                warmup_steps=config["train"].get("warmup_steps", 1000),
            )
            step += 1

            optimizer.zero_grad(set_to_none=True)

            with autocast(device):
                loss = model(imgs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch}: Train Loss = {running_loss / len(train_loader):.4f}")

        # -------------------------------------------------------------
        # VALIDATE — fast because of prebuilt cache
        # -------------------------------------------------------------
        acc = validate(model, val_loader, device)
        print(f"Epoch {epoch}: Val Acc = {acc:.4f}")

        if acc > best_val:
            best_val = acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "step": step,
                },
                f"checkpoints/best_epoch{epoch}.pt",
            )
            print("Saved new best checkpoint!")

    print("Training complete. Best Val Acc =", best_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["train"]["lr"] = float(config["train"]["lr"])
    train(config, resume=args.resume)

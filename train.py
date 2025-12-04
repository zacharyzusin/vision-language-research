# train.py

import os
import argparse
import yaml
import math
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
from src.models.mop_clip import MixturePromptCLIP


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model.predict(images)

        # preds returned on SAME DEVICE as images
        assert preds.device == labels.device, \
            "BUG: preds and labels must live on same device!"

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr


def train(config, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = config["dataset"]["root"]

    print("\n=== Loading INaturalist 2018 dataset ===")
    train_ds = get_inat2018(root, "train")
    val_ds   = get_inat2018(root, "val")

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
    print(f"Num classes: {len(metadata)}")

    # MODEL INIT
    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        ctx_len=config["model"]["ctx_len"],
        em_tau=config["model"]["em_tau"],
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    scaler = GradScaler(device.type)

    start_epoch = 1
    best_val = 0.0
    step = 0
    max_steps = len(train_loader) * config["train"]["epochs"]

    # RESUME CHECKPOINT
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val_acc"]
        step = ckpt["step"]
        print(f"Resumed training from epoch {start_epoch}")

    os.makedirs("checkpoints", exist_ok=True)

    # TRAIN LOOP
    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cosine_lr(optimizer, config["train"]["lr"], step, max_steps,
                      warmup_steps=config["train"]["warmup_steps"])
            step += 1

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                loss = model(images, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            ckpt_path = f"checkpoints/best_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val,
                "step": step,
            }, ckpt_path)
            print(f"Saved new best checkpoint to {ckpt_path}")

    print("\nTraining complete. Best Val Acc =", best_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["train"]["lr"] = float(config["train"]["lr"])
    train(config, resume=args.resume)

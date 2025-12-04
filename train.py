# train.py

import os
import argparse
import math
import yaml

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
from src.models.mop_clip import MixturePromptCLIP


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0

    for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model.predict(imgs)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return correct / total if total else 0.0


def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    if step < warmup_steps:
        lr = base_lr * step / max_steps if warmup_steps == 0 else base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
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

    # DataLoaders
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

    # Hierarchical metadata
    metadata = extract_hierarchical_metadata(root)
    num_classes = len(metadata)
    print(f"Num classes (species): {num_classes}")

    # Model
    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        ctx_len=config["model"]["ctx_len"],
        em_tau=float(config["model"].get("em_tau", 0.5)),
        cache_dir="text_cache",
    ).to(device)

    # -------------------------------------------
    # Print summary
    # -------------------------------------------
    print("\n================= RUN CONFIG SUMMARY =================")
    print(f"Model:          {config['model']['clip_model']}")
    print(f"Dataset Root:   {root}")
    print(f"Train Samples:  {len(train_ds)}")
    print(f"Val Samples:    {len(val_ds)}")
    print(f"Num Classes:    {num_classes}")
    print(f"K (prompts):    {config['model']['K']}")
    print(f"ctx_len:        {config['model']['ctx_len']}")
    print(f"em_tau:         {config['model'].get('em_tau', 0.5)}")
    print("=====================================================\n")

    base_lr = float(config["train"]["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scaler = GradScaler("cuda" if device == "cuda" else "cpu")

    epochs = config["train"]["epochs"]
    warmup_steps = config["train"].get("warmup_steps", 1000)
    max_steps = len(train_loader) * epochs

    start_epoch = 1
    best_val = 0.0
    step = 0

    # Resume logic
    if resume is not None and os.path.exists(resume):
        print(f"Loading checkpoint from {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val_acc", 0.0)
        step = ckpt.get("step", 0)
        print(
            f"Resuming from epoch {start_epoch} | best val acc so far = {best_val:.4f} | step={step}"
        )

    os.makedirs("checkpoints", exist_ok=True)

    # -----------------
    # Training loop
    # -----------------
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # LR schedule
            cosine_lr(
                optimizer,
                base_lr,
                step,
                max_steps,
                warmup_steps=warmup_steps,
            )
            step += 1

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if device == "cuda" else "cpu"):
                loss = model(imgs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

        # Validation
        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

        # Save "last" checkpoint every epoch
        last_ckpt_path = "checkpoints/last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val,
                "step": step,
            },
            last_ckpt_path,
        )

        # Save best checkpoint by validation accuracy
        if val_acc > best_val:
            best_val = val_acc
            best_path = f"checkpoints/best_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "step": step,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to {best_path} (Val Acc = {best_val:.4f})")

    print("Training complete. Best Val Acc =", best_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume=args.resume)

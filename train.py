# train.py

import os
import argparse
import yaml
import math
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import wandb

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
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(1, total)


def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    """Cosine LR with linear warmup."""
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    for g in optimizer.param_groups:
        g["lr"] = lr


def train(config, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = config["dataset"]["root"]

    # ----------------------------------
    # Dataset
    # ----------------------------------
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

    # ----------------------------------
    # W&B init (minimal)
    # ----------------------------------
    wandb.init(project="inat-mop-clip", config=config)

    # ----------------------------------
    # Model
    # ----------------------------------
    em_tau_start = float(
        config["model"].get("em_tau_start", config["model"].get("em_tau", 1.0))
    )
    em_tau_end = float(config["model"].get("em_tau_end", 0.3))

    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        em_tau=em_tau_start,
    )

    model = model.to(device)
    model.clip.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    scaler = GradScaler(device.type)

    start_epoch = 1
    best_val = 0.0
    step = 0
    max_steps = len(train_loader) * config["train"]["epochs"]
    tau_anneal_steps = max_steps

    # ----------------------------------
    # Optional checkpoint resume
    # ----------------------------------
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val_acc"]
        step = ckpt["step"]
        print(f"Resumed training from epoch {start_epoch}")

    os.makedirs("checkpoints", exist_ok=True)

    # ----------------------------------
    # Training Loop
    # ----------------------------------
    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # LR scheduler
            cosine_lr(
                optimizer,
                config["train"]["lr"],
                step,
                max_steps,
                warmup_steps=config["train"]["warmup_steps"],
            )

            # EM temperature annealing
            progress = min(1.0, step / max(1, tau_anneal_steps))
            current_tau = em_tau_end + (em_tau_start - em_tau_end) * (1.0 - progress)
            model.em_tau = float(current_tau)

            step += 1

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                loss = model(images, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # WandB logging
            wandb.log({
                "loss": loss.item(),
                "tau": current_tau,
                "lr": optimizer.param_groups[0]["lr"],
                "step": step,
            })

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "tau": f"{current_tau:.3f}"})

        # Epoch summary
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        wandb.log({"train_loss_epoch": avg_loss})

        # Validation
        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f} | Best = {best_val:.4f}")
        wandb.log({"val_acc": val_acc})

        # Best checkpoint
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

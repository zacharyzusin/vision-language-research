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
        images = images.to(device)
        labels = labels.to(device)

        preds = model.predict(images)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(1, total)


def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    """Cosine LR with linear warmup."""
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    for g in optimizer.param_groups:
        g["lr"] = lr


def train(config, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =====================
    # Dataset
    # =====================
    root = config["dataset"]["root"]

    train_ds = get_inat2018(root, "train")
    val_ds   = get_inat2018(root, "val")

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

    # =====================
    # W&B
    # =====================
    wandb.init(project="inat-mop-clip", config=config)

    # =====================
    # Model Setup
    # =====================
    em_tau_start = float(config["model"].get("em_tau_start", 1.0))
    em_tau_end   = float(config["model"].get("em_tau_end", 0.3))

    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        em_tau=em_tau_start,
    )
    model = model.to(device)
    model.clip.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    scaler = GradScaler(device.type)

    # =====================
    # Resume Variables
    # =====================
    start_epoch = 1
    step = 0
    best_val = 0.0
    current_tau = em_tau_start

    total_epochs = config["train"]["epochs"]
    max_steps = len(train_loader) * total_epochs
    tau_anneal_steps = max_steps

    # =====================
    # LOAD CHECKPOINT
    # =====================
    if resume and os.path.exists(resume):
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"] + 1
        step = ckpt["step"]
        best_val = ckpt["best_val_acc"]

        # Restore τ precisely
        current_tau = ckpt.get("tau", current_tau)
        model.em_tau = current_tau

        print(f"→ Resumed epoch {start_epoch}")
        print(f"→ Resumed global step {step}")
        print(f"→ Restored τ = {current_tau}")

    os.makedirs("checkpoints", exist_ok=True)

    # =====================
    # TRAIN LOOP
    # =====================
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # -----------------------
            # LR Scheduling
            # -----------------------
            cosine_lr(
                optimizer,
                config["train"]["lr"],
                step,
                max_steps,
                warmup_steps=config["train"]["warmup_steps"],
            )

            # -----------------------
            # τ-Annealing (RESUMABLE)
            # -----------------------
            progress = min(1.0, step / tau_anneal_steps)
            current_tau = em_tau_end + (em_tau_start - em_tau_end) * (1.0 - progress)
            model.em_tau = float(current_tau)

            step += 1

            optimizer.zero_grad(set_to_none=True)

            # -----------------------
            # Loss
            # -----------------------
            with autocast(device_type=device.type):
                result = model(
                    images, 
                    labels,
                    lambda_mixture=config["train"].get("lambda_mixture", 0.5),
                    temp_cls=config["train"].get("temp_cls", 0.07),
                )
                
                if isinstance(result, tuple):
                    loss, loss_dict = result
                else:
                    loss = result
                    loss_dict = {}

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            log_dict = {
                "loss": loss.item(),
                "tau": current_tau,
                "lr": optimizer.param_groups[0]["lr"],
                "step": step,
                "epoch": epoch,
            }
            log_dict.update(loss_dict)
            wandb.log(log_dict)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "tau": f"{current_tau:.3f}",
                "mix": f"{loss_dict.get('loss_mixture', 0):.3f}",
                "cls": f"{loss_dict.get('loss_cls', 0):.3f}",
            })

        # =====================
        # Validation
        # =====================
        val_acc = validate(model, val_loader, device)

        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc

        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f} (best={best_val:.4f})")
        wandb.log({"val_acc": val_acc, "best_val_acc": best_val})

        # =====================
        # Save Checkpoint
        # =====================
        if is_best:
            ckpt_path = f"checkpoints/best_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val,
                "tau": current_tau,
            }, ckpt_path)

            print(f"Saved new BEST checkpoint: {ckpt_path}")

    print("\nTraining Complete! Best Val Acc:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["train"]["lr"] = float(config["train"]["lr"])
    train(config, resume=args.resume)

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


# -------------------------------------------------------------------
#                      CONFIG LOADING
# -------------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------
#                      CHECKPOINTING
# -------------------------------------------------------------------
def save_checkpoint(path, epoch, model, optimizer, scaler, best_val, step):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_acc": best_val,
        "step": step,
    }
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    print(f"[Checkpoint] Loading from: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    else:
        print("[Checkpoint] WARNING: No scaler state in checkpoint!")

    epoch = ckpt.get("epoch", 1)
    step = ckpt.get("step", 0)
    best_val = ckpt.get("best_val_acc", 0.0)

    print(
        f"[Checkpoint] Restored epoch={epoch}, step={step}, best_val={best_val:.4f}"
    )

    return epoch, step, best_val


# -------------------------------------------------------------------
#                      VALIDATION LOOP
# -------------------------------------------------------------------
@torch.no_grad()
def validate(model, dataloader, device):
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
    print(f"[Validation] Time = {duration:.2f}s")

    return correct / total if total else 0.0


# -------------------------------------------------------------------
#                      LR SCHEDULER
# -------------------------------------------------------------------
def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    if step < warmup_steps:
        lr = base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    for g in optimizer.param_groups:
        g["lr"] = lr


# -------------------------------------------------------------------
#                      TRAINING LOOP
# -------------------------------------------------------------------
def train(config, resume=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Load metadata + model
    # ---------------------------------------------------------------
    metadata = extract_hierarchical_metadata(root)
    num_classes = len(metadata)
    print(f"Num classes: {num_classes}")

    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        ctx_len=config["model"]["ctx_len"],
    ).to(device)

    # Print run summary
    print("\n================= RUN CONFIG SUMMARY =================")
    print(f"Model: {config['model']['clip_model']}")
    print(f"Dataset Root: {root}")
    print(f"Train Samples: {len(train_ds)}")
    print(f"Val Samples:   {len(val_ds)}")
    print(f"Num Classes:   {num_classes}")
    print("=====================================================\n")

    # ---------------------------------------------------------------
    # Optimizer + AMP
    # ---------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    scaler = GradScaler("cuda" if device == "cuda" else "cpu")

    start_epoch = 1
    best_val = 0.0
    step = 0
    max_steps = len(train_loader) * config["train"]["epochs"]

    # ---------------------------------------------------------------
    # Resume training if checkpoint provided
    # ---------------------------------------------------------------
    if resume and os.path.exists(resume):
        start_epoch, step, best_val = load_checkpoint(
            resume, model, optimizer, scaler, device
        )
        start_epoch += 1

    # ---------------------------------------------------------------
    # Build inference cache before training (fast validation)
    # ---------------------------------------------------------------
    print("\n=== Building inference cache BEFORE training ===")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.predict(dummy)  # triggers cache build
    print("=== Inference cache ready ===\n")

    # ---------------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------------
    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # LR scheduling
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

        # -------------------
        # VALIDATION
        # -------------------
        acc = validate(model, val_loader, device)
        print(f"Epoch {epoch}: Val Acc = {acc:.4f}")

        # -------------------
        # SAVE CHECKPOINT
        # -------------------
        if acc > best_val:
            best_val = acc
            ckpt_path = f"checkpoints/best_epoch{epoch}.pt"
            save_checkpoint(
                ckpt_path, epoch, model, optimizer, scaler, best_val, step
            )

    print("Training complete. Best Val Acc =", best_val)


# -------------------------------------------------------------------
#                      ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["train"]["lr"] = float(config["train"]["lr"])

    train(config, resume=args.resume)

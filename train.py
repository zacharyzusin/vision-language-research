"""
Training script for Mixture-of-Prompts CLIP model.

This script handles:
- Dataset loading and preprocessing
- Model initialization and checkpoint resuming
- Training loop with cosine LR scheduling and tau annealing
- Validation and checkpoint saving
- Weights & Biases logging
"""

import os
import argparse
import yaml
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import wandb

from src.datasets.inat_dataset import get_inat, extract_hierarchical_metadata
from src.datasets.kikibouba_dataset import get_kikibouba, extract_kikibouba_metadata
from src.datasets.stanford_cars_dataset import get_stanford_cars, extract_stanford_cars_metadata
from src.models.mop_clip import MixturePromptCLIP


def load_config(path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def validate(model, dataloader, device):
    """
    Validate model on a dataset split.

    Args:
        model: MixturePromptCLIP model instance
        dataloader: DataLoader for validation set
        device: Device to run validation on

    Returns:
        Validation accuracy (float between 0 and 1)
    """
    model.eval()
    total = 0
    correct = 0

    for images, labels in tqdm(dataloader, desc="Validating", leave=False, disable=not _is_main_process()):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # If wrapped in DDP, expose underlying module.
        if hasattr(model, "module"):
            preds = model.module.predict(images)
        else:
            preds = model.predict(images)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Aggregate across ranks if distributed.
    if _is_distributed():
        t = torch.tensor([correct, total], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct = int(t[0].item())
        total = int(t[1].item())

    return correct / max(1, total)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _ddp_setup(backend: str = "nccl"):
    """
    Initialize torch.distributed using environment variables set by torchrun or Slurm.
    """
    if _is_distributed():
        return
    dist.init_process_group(backend=backend, init_method="env://")


def cosine_lr(optimizer, base_lr, step, max_steps, warmup_steps=1000):
    """
    Apply cosine learning rate schedule with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        base_lr: Base learning rate
        step: Current training step
        max_steps: Total number of training steps
        warmup_steps: Number of warmup steps with linear LR increase
    """
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    for g in optimizer.param_groups:
        g["lr"] = lr


def _build_experiment_name(config: dict, dataset_type: str) -> str:
    """
    Build a descriptive experiment name for organizing checkpoints.

    Format examples:
      - inat_v2021_K1_ViT-B-16
      - inat_subset_wrasse_K32_ViT-B-16
      - kikibouba_v1_K32_ViT-B-16
      - stanford_cars_K32_ViT-B-16
    """
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    clip_model = str(model_cfg.get("clip_model", "unknown")).replace("/", "-")
    K = model_cfg.get("K", "K")

    if dataset_type == "inat":
        version = str(dataset_cfg.get("version", "unknown"))
        subset_name = dataset_cfg.get("subset_name", None)
        if subset_name is None and dataset_cfg.get("category_ids") is not None:
            subset_name = "subset"
        if subset_name:
            ds_part = f"inat_{subset_name}_v{version}"
        else:
            ds_part = f"inat_v{version}"
    elif dataset_type == "kikibouba":
        kb_version = dataset_cfg.get("version", None)
        if kb_version:
            ds_part = f"kikibouba_{kb_version}"
        else:
            ds_part = "kikibouba"
    elif dataset_type == "stanford_cars":
        ds_part = "stanford_cars"
    else:
        ds_part = dataset_type

    return f"{ds_part}_K{K}_{clip_model}"


def train(config: dict, resume: str = None):
    """
    Main training function.

    Args:
        config: Configuration dictionary from YAML file
        resume: Optional path to checkpoint to resume from
    """
    # DDP: torchrun sets LOCAL_RANK/WORLD_SIZE/RANK.
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        _ddp_setup(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _is_main_process():
        print(f"Using device: {device} | distributed={use_ddp} | world_size={_get_world_size()}")

    # =====================
    # Dataset
    # =====================
    root = config["dataset"]["root"]
    dataset_type = config["dataset"].get("type", "inat")  # "inat", "kikibouba", or "stanford_cars"
    
    if dataset_type == "kikibouba":
        # KikiBouba dataset (multiclass classification)
        kb_version = config["dataset"].get("version", None)
        if kb_version is not None:
            print(f"Using KikiBouba dataset version: {kb_version}")

        train_ds = get_kikibouba(root, "train", version=kb_version)
        val_ds = get_kikibouba(root, "val", version=kb_version)
        metadata = extract_kikibouba_metadata(root)
        category_ids = None  # Not used for KikiBouba
    elif dataset_type == "stanford_cars":
        # Stanford Cars dataset
        train_ds = get_stanford_cars(root, "train")
        val_ds = get_stanford_cars(root, "val")  # Maps to test split
        metadata = extract_stanford_cars_metadata(root)
        category_ids = None  # Not used for Stanford Cars
    else:
        # iNaturalist dataset
        version = str(config["dataset"]["version"])
        
        # Support subset training via category_ids
        category_ids = config["dataset"].get("category_ids", None)
        if category_ids is not None:
            print(f"Training on subset with {len(category_ids)} classes: {category_ids}")
        
        train_ds = get_inat(root, "train", version=version, category_ids=category_ids)
        val_ds   = get_inat(root, "val", version=version, category_ids=category_ids)
        metadata = extract_hierarchical_metadata(root, category_ids=category_ids)

    # Build experiment name for organizing checkpoints
    experiment_name = _build_experiment_name(config, dataset_type)
    print(f"Experiment name: {experiment_name}")

    # Use more workers for large datasets (2021 has 2.7M samples)
    # With 48 CPU cores, we can use more workers for better I/O parallelism
    # For KikiBouba and Stanford Cars, use fewer workers since they're smaller datasets
    default_workers = 8 if dataset_type in ["kikibouba", "stanford_cars"] else 16
    num_workers = config["train"].get("num_workers", default_workers)
    prefetch_factor = config["train"].get("prefetch_factor", 4)  # Prefetch more batches
    
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None
    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    # Metadata already loaded above based on dataset type

    if _is_main_process():
        print(f"[DEBUG] Created DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches")

    # =====================
    # W&B
    # =====================
    if _is_main_process():
        print("[DEBUG] Initializing W&B...")
        wandb.init(project="inat-mop-clip", config=config)
        print("[DEBUG] W&B initialized")

    # =====================
    # Model Setup
    # =====================
    if _is_main_process():
        print("[DEBUG] Creating model...")
    em_tau_start = float(config["model"].get("em_tau_start", 1.0))
    em_tau_end   = float(config["model"].get("em_tau_end", 0.3))

    # Ablation parameters
    use_semantic_init = config["model"].get("use_semantic_init", True)
    offset_reg_weight = config["model"].get("offset_reg_weight", 0.001)
    use_hard_assignment = config["model"].get("use_hard_assignment", False)
    
    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        em_tau=em_tau_start,
        use_semantic_init=use_semantic_init,
        offset_reg_weight=offset_reg_weight,
        use_hard_assignment=use_hard_assignment,
    )
    model = model.to(device)
    model.clip.to(device)

    if use_ddp:
        if _is_main_process():
            print("[DEBUG] Wrapping model with DDP...")
        # Wrap after moving to device.
        # NOTE: MixturePromptCLIP keeps `base_text_features` on CPU intentionally (see `MixturePromptCLIP._apply`).
        # DDP's default `broadcast_buffers=True` would try to broadcast that CPU buffer with NCCL and crash.
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        if _is_main_process():
            print("[DEBUG] DDP wrapper created")

    # Optional: Use torch.compile() for faster training (PyTorch 2.0+)
    # This can provide 20-30% speedup on compatible models
    use_compile = config.get("train", {}).get("use_compile", False)
    if use_compile and hasattr(torch, "compile"):
        if _is_main_process():
            print("[DEBUG] Compiling model with torch.compile()...")
        # Compile the underlying model (not DDP wrapper)
        target_model = model.module if hasattr(model, "module") else model
        target_model = torch.compile(target_model, mode="reduce-overhead")
        if hasattr(model, "module"):
            model.module = target_model
        else:
            model = target_model
        if _is_main_process():
            print("[DEBUG] Model compiled")

    if _is_main_process():
        print("[DEBUG] Creating optimizer and scaler...")
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
    if _is_main_process():
        print(f"[DEBUG] Computing max_steps from train_loader length...")
    max_steps = len(train_loader) * total_epochs
    tau_anneal_steps = max_steps
    if _is_main_process():
        print(f"[DEBUG] max_steps={max_steps}, starting training loop...")

    # =====================
    # LOAD CHECKPOINT
    # =====================
    if resume and os.path.exists(resume):
        if _is_main_process():
            print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)

        # Handle both DDP-wrapped and non-DDP checkpoints
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"] + 1
        step = ckpt["step"]
        best_val = ckpt["best_val_acc"]

        # Restore τ precisely
        current_tau = ckpt.get("tau", current_tau)
        model.em_tau = current_tau

        if _is_main_process():
            print(f"→ Resumed epoch {start_epoch}")
            print(f"→ Resumed global step {step}")
            print(f"→ Restored τ = {current_tau}")

    ckpt_dir = os.path.join("checkpoints", experiment_name)
    if _is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

    # =====================
    # TRAIN LOOP
    # =====================
    if _is_main_process():
        print(f"[DEBUG] Entering training loop: epochs {start_epoch} to {total_epochs}")
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running_loss = 0.0

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not _is_main_process())

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
            use_tau_annealing = config["model"].get("use_tau_annealing", True)
            if use_tau_annealing:
                progress = min(1.0, step / tau_anneal_steps)
                current_tau = em_tau_end + (em_tau_start - em_tau_end) * (1.0 - progress)
                # DDP-safe attribute update
                target_model = model.module if hasattr(model, "module") else model
                target_model.em_tau = float(current_tau)
            else:
                # Fixed temperature
                target_model = model.module if hasattr(model, "module") else model
                target_model.em_tau = float(em_tau_start)

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
            
            # Invalidate cached all_prompts after optimizer step (prompt_offsets changed)
            target_model = model.module if hasattr(model, "module") else model
            if hasattr(target_model, "_invalidate_all_prompts_cache"):
                target_model._invalidate_all_prompts_cache()

            running_loss += loss.item()

            if _is_main_process():
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
        # Validation (with configurable frequency)
        # =====================
        val_freq = config["train"].get("val_freq", 1)  # Default: every epoch
        should_validate = (epoch % val_freq == 0) or (epoch == total_epochs)
        
        if should_validate:
            val_acc = validate(model, val_loader, device)

            is_best = val_acc > best_val
            if is_best:
                best_val = val_acc

            if _is_main_process():
                print(f"Epoch {epoch}: Val Acc = {val_acc:.4f} (best={best_val:.4f})")
                wandb.log({"val_acc": val_acc, "best_val_acc": best_val})
        else:
            # Still log best_val_acc even if not validating
            if _is_main_process():
                print(f"Epoch {epoch}: (skipping validation, next at epoch {((epoch // val_freq) + 1) * val_freq})")
                wandb.log({"best_val_acc": best_val})

        # =====================
        # Save Checkpoint
        # =====================
        if is_best and _is_main_process():
            ckpt_filename = f"{experiment_name}_best_epoch{epoch}.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
            save_model = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "step": step,
                "model": save_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val,
                "tau": current_tau,
            }, ckpt_path)

            print(f"Saved new BEST checkpoint: {ckpt_path}")

    if _is_main_process():
        print("\nTraining Complete! Best Val Acc:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["train"]["lr"] = float(config["train"]["lr"])
    train(config, resume=args.resume)

    # Cleanly tear down distributed group (prevents hang on some clusters)
    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()

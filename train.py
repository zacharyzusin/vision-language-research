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
import sys
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
from src.datasets.fgvc_aircraft_dataset import get_fgvc_aircraft, extract_fgvc_aircraft_metadata
from src.datasets.flowers102_dataset import get_flowers102, extract_flowers102_metadata
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
    
    rank = int(os.environ.get("RANK", "0"))
    print(f"[DEBUG] Rank {rank}: Calling dist.init_process_group with backend={backend}", flush=True)
    
    try:
        dist.init_process_group(
            backend=backend, 
            init_method="env://"
        )
        print(f"[DEBUG] Rank {rank}: dist.init_process_group completed successfully", flush=True)
    except Exception as e:
        print(f"[ERROR] Rank {rank}: dist.init_process_group failed: {e}", flush=True)
        raise
    
    # Note: No barrier needed after init_process_group - it already synchronizes
    print(f"[DEBUG] Rank {rank}: DDP setup complete, is_initialized={dist.is_initialized()}", flush=True)


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
    # DDP: torchrun or Slurm (with one GPU per task) set LOCAL_RANK/WORLD_SIZE/RANK.
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    print(f"[DEBUG] Rank {os.environ.get('RANK', '0')}: Starting train(), use_ddp={use_ddp}", flush=True)
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        # CRITICAL: Set device BEFORE initializing process group!
        # NCCL needs the correct device to be set before init_process_group()
        if torch.cuda.is_available():
            # When launched via Slurm with --gpus-per-task=1, each process sees a single
            # visible GPU (CUDA_VISIBLE_DEVICES is set to one id), so device_count == 1
            # and the correct choice is always "cuda:0" regardless of LOCAL_RANK.
            #
            # When launched via torchrun on a single node, all physical GPUs are visible
            # and LOCAL_RANK indexes into them (device_count == num_gpus_per_node).
            if torch.cuda.device_count() == 1:
                device = torch.device("cuda:0")
                torch.cuda.set_device(0)
            else:
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(local_rank)
        else:
            device = torch.device("cpu")
        
        print(f"[DEBUG] Rank {os.environ.get('RANK', '0')}: Device set to {device}, initializing DDP...", flush=True)
        # Now initialize process group with device already set
        _ddp_setup(backend="nccl" if torch.cuda.is_available() else "gloo")
        print(f"[DEBUG] Rank {_get_rank()}: DDP initialized", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _is_main_process():
        print(f"Using device: {device} | distributed={use_ddp} | world_size={_get_world_size()}", flush=True)

    # =====================
    # Dataset
    # =====================
    if _is_main_process() or not use_ddp:
        print(f"[DEBUG] Loading dataset (type={config['dataset'].get('type', 'inat')})...", flush=True)
    root = config["dataset"]["root"]
    dataset_type = config["dataset"].get("type", "inat")  # "inat", "kikibouba", "stanford_cars", or "fgvc_aircraft"
    
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
    elif dataset_type == "fgvc_aircraft":
        # FGVC Aircraft dataset
        train_ds = get_fgvc_aircraft(root, "train")
        val_ds = get_fgvc_aircraft(root, "val")  # Maps to test split
        metadata = extract_fgvc_aircraft_metadata(root)
        category_ids = None  # Not used for FGVC Aircraft
    elif dataset_type == "flowers102":
        # Flowers102 dataset
        train_ds = get_flowers102(root, "train")
        val_ds = get_flowers102(root, "val")
        metadata = extract_flowers102_metadata(root)
        category_ids = None  # Not used for Flowers102
    elif dataset_type == "caltech101":
        # Caltech101 dataset
        from src.datasets.caltech101_dataset import get_caltech101, extract_caltech101_metadata
        train_ds = get_caltech101(root, "train")
        val_ds = get_caltech101(root, "val")
        metadata = extract_caltech101_metadata(root)
        category_ids = None  # Not used for Caltech101
    elif dataset_type == "oxford_pets":
        # Oxford-IIIT Pet dataset
        from src.datasets.oxford_pets_dataset import get_oxford_pets, extract_oxford_pets_metadata
        train_ds = get_oxford_pets(root, "train")
        val_ds = get_oxford_pets(root, "val")
        metadata = extract_oxford_pets_metadata(root)
        category_ids = None  # Not used for OxfordPets
    elif dataset_type == "food101":
        # Food-101 dataset
        from src.datasets.food101_dataset import get_food101, extract_food101_metadata
        train_ds = get_food101(root, "train")
        val_ds = get_food101(root, "val")
        metadata = extract_food101_metadata(root)
        category_ids = None  # Not used for Food101
    elif dataset_type == "dtd":
        # DTD (Describable Textures Dataset)
        from src.datasets.dtd_dataset import get_dtd, extract_dtd_metadata
        train_ds = get_dtd(root, "train")
        val_ds = get_dtd(root, "val")
        metadata = extract_dtd_metadata(root)
        category_ids = None  # Not used for DTD
    elif dataset_type == "sun397":
        # SUN397 (Scene UNderstanding dataset)
        from src.datasets.sun397_dataset import get_sun397, extract_sun397_metadata
        train_ds = get_sun397(root, "train")
        val_ds = get_sun397(root, "val")
        metadata = extract_sun397_metadata(root)
        category_ids = None  # Not used for SUN397
    elif dataset_type == "eurosat":
        # EuroSAT dataset
        from src.datasets.eurosat_dataset import get_eurosat, extract_eurosat_metadata
        train_ds = get_eurosat(root, "train", download=True)
        val_ds = get_eurosat(root, "val", download=True)
        metadata = extract_eurosat_metadata(root)
        category_ids = None  # Not used for EuroSAT
    elif dataset_type == "ucf101":
        # UCF101 dataset
        # Note: UCF101 doesn't support automatic download - dataset must be manually downloaded
        from src.datasets.ucf101_dataset import get_ucf101, extract_ucf101_metadata
        train_ds = get_ucf101(root, "train", download=False)
        val_ds = get_ucf101(root, "val", download=False)
        metadata = extract_ucf101_metadata(root)
        category_ids = None  # Not used for UCF101
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

    # Experiment name will be built after config merge (to reflect actual K value)
    experiment_name = None

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
    # Note: All ranks read from the same config file, so K should be the same
    # We'll verify this after W&B init (in case sweep changes it)
    # =====================
    
    # =====================
    # W&B and Config Merge (after broadcast to avoid deadlock)
    # =====================
    if _is_main_process():
        print("[DEBUG] Initializing W&B...")
        
        # Determine wandb project name based on dataset type
        dataset_type_for_project = config.get("dataset", {}).get("type", "inat")
        if dataset_type_for_project == "stanford_cars":
            wandb_project = "stanford-cars-mop-clip"
        elif dataset_type_for_project == "kikibouba":
            wandb_project = "kikibouba-mop-clip"
        elif dataset_type_for_project == "fgvc_aircraft":
            wandb_project = "fgvc-aircraft-mop-clip"
        elif dataset_type_for_project == "flowers102":
            wandb_project = "flowers102-mop-clip"
        elif dataset_type_for_project == "caltech101":
            wandb_project = "caltech101-mop-clip"
        elif dataset_type_for_project == "oxford_pets":
            wandb_project = "oxford-pets-mop-clip"
        elif dataset_type_for_project == "food101":
            wandb_project = "food101-mop-clip"
        elif dataset_type_for_project == "dtd":
            wandb_project = "dtd-mop-clip"
        elif dataset_type_for_project == "sun397":
            wandb_project = "sun397-mop-clip"
        elif dataset_type_for_project == "eurosat":
            wandb_project = "eurosat-mop-clip"
        elif dataset_type_for_project == "ucf101":
            wandb_project = "ucf101-mop-clip"
        else:
            # For iNaturalist, check if it's a subset
            if config.get("dataset", {}).get("category_ids") is not None:
                subset_name = config.get("dataset", {}).get("subset_name", "subset")
                wandb_project = f"inat-{subset_name}-mop-clip"
            else:
                version = config.get("dataset", {}).get("version", "2021")
                wandb_project = f"inat-{version}-mop-clip"
        
        # Initialize W&B - wandb.config will be populated if this is a sweep run
        wandb.init(project=wandb_project, config=config)
        
        # Merge wandb.config (sweep parameters) into config to override YAML values
        def deep_update(base_dict, update_dict):
            """Recursively update nested dictionaries."""
            for key, value in update_dict.items():
                # Handle dot-separated keys like "model.K" -> nested dict structure
                if '.' in key:
                    parts = key.split('.')
                    current = base_dict
                    for i, part in enumerate(parts[:-1]):
                        if part not in current:
                            current[part] = {}
                        elif not isinstance(current[part], dict):
                            # Convert existing value to dict if needed
                            current[part] = {f"_{part}": current[part]}
                        current = current[part]
                    current[parts[-1]] = value
                elif key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        # Convert wandb.config to dict and merge into config
        try:
            if wandb.run is not None:
                sweep_config = {}
                try:
                    # Direct iteration approach - wandb.config is iterable and supports dict-like access
                    for key in wandb.config:
                        sweep_config[str(key)] = wandb.config[key]
                except (TypeError, AttributeError) as e:
                    try:
                        if hasattr(wandb.config, '_items'):
                            sweep_config = {str(k): v for k, v in wandb.config._items()}
                        elif hasattr(wandb.config, 'items'):
                            sweep_config = {str(k): v for k, v in wandb.config.items()}
                    except Exception as e2:
                        print(f"[DEBUG] Error converting wandb.config (fallback): {e2}")
                        sweep_config = {}
                
                if sweep_config:
                    deep_update(config, sweep_config)
                    print(f"[DEBUG] Merged wandb.config into config: {sweep_config}")
        except (TypeError, AttributeError) as e:
            print(f"[DEBUG] Could not merge wandb.config (not in sweep?): {e}")
    
        # Build experiment name AFTER config merge
        if experiment_name is None:
            experiment_name = _build_experiment_name(config, dataset_type)
            print(f"Experiment name: {experiment_name}")
        
        print("[DEBUG] W&B initialized")
        
        # After W&B init and config merge, broadcast K value if it changed
        final_k = config.get("model", {}).get("K", 32)
        if use_ddp:
            # Broadcast the final K value to all ranks
            if dist.is_initialized():
                backend = dist.get_backend()
            else:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            
            if backend == "nccl" and torch.cuda.is_available():
                k_tensor = torch.tensor(final_k, dtype=torch.int32, device=device)
            else:
                k_tensor = torch.tensor(final_k, dtype=torch.int32, device="cpu")
            
            # Wait for non-main processes to be ready (they wait at barrier after W&B disabled)
            print(f"[DEBUG] Rank 0: Waiting for other ranks to be ready for broadcast...", flush=True)
            dist.barrier()  # Wait for rank 1 to finish setup and be ready for broadcast
            print(f"[DEBUG] Rank 0: All ranks ready, broadcasting final K={final_k}...", flush=True)
            dist.broadcast(k_tensor, src=0)
            print(f"[DEBUG] Rank 0: Broadcast complete", flush=True)
    else:
        # Non-main processes: disable wandb
        os.environ["WANDB_MODE"] = "disabled"
        print(f"[DEBUG] Rank {_get_rank()}: W&B disabled (non-main process)")
        
        if use_ddp:
            # Wait for main process to finish W&B init before broadcast
            rank = _get_rank()
            print(f"[DEBUG] Rank {rank}: Waiting for main process to finish W&B init...", flush=True)
            dist.barrier()  # Wait for rank 0 to finish W&B init
            print(f"[DEBUG] Rank {rank}: Main process finished W&B, preparing for broadcast...", flush=True)
            
            # Receive K value from main process
            if dist.is_initialized():
                backend = dist.get_backend()
            else:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            
            if backend == "nccl" and torch.cuda.is_available():
                k_tensor = torch.tensor(0, dtype=torch.int32, device=device)
            else:
                k_tensor = torch.tensor(0, dtype=torch.int32, device="cpu")
            
            print(f"[DEBUG] Rank {rank}: Waiting for K value broadcast...", flush=True)
            dist.broadcast(k_tensor, src=0)
            received_k = int(k_tensor.cpu().item() if k_tensor.is_cuda else k_tensor.item())
            print(f"[DEBUG] Rank {rank}: Received K={received_k}", flush=True)
            
            # Update config with received K value
            if "model" not in config:
                config["model"] = {}
            config["model"]["K"] = received_k
    
    # Synchronize all ranks after W&B init and K broadcast
    if use_ddp:
        rank = _get_rank()
        print(f"[DEBUG] Rank {rank}: Synchronizing after W&B init...", flush=True)
        dist.barrier()
        print(f"[DEBUG] Rank {rank}: Synchronization after W&B complete", flush=True)
    
    # Build experiment name on non-main processes (if not already set)
    if experiment_name is None:
        experiment_name = _build_experiment_name(config, dataset_type)

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
    
    # Diversity and entropy regularization (to prevent mode collapse)
    diversity_loss_weight = config["model"].get("diversity_loss_weight", 0.1)
    entropy_loss_weight = config["model"].get("entropy_loss_weight", 0.01)
    min_usage_loss_weight = config["model"].get("min_usage_loss_weight", 0.0)
    min_usage_threshold = config["model"].get("min_usage_threshold", 0.05)
    
    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=config["model"]["K"],
        em_tau=em_tau_start,
        use_semantic_init=use_semantic_init,
        offset_reg_weight=offset_reg_weight,
        use_hard_assignment=use_hard_assignment,
        diversity_loss_weight=diversity_loss_weight,
        entropy_loss_weight=entropy_loss_weight,
        min_usage_loss_weight=min_usage_loss_weight,
        min_usage_threshold=min_usage_threshold,
    )
    model = model.to(device)
    model.clip.to(device)

    if use_ddp:
        rank = _get_rank()
        print(f"[DEBUG] Rank {rank}: About to wrap model with DDP...", flush=True)
        # Wrap after moving to device.
        # NOTE: MixturePromptCLIP keeps `base_text_features` on CPU intentionally (see `MixturePromptCLIP._apply`).
        # DDP's default `broadcast_buffers=True` would try to broadcast that CPU buffer with NCCL and crash.
        try:
            model = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            print(f"[DEBUG] Rank {rank}: DDP wrapper created successfully", flush=True)
        except Exception as e:
            print(f"[ERROR] Rank {rank}: DDP wrapping failed: {e}", flush=True)
            raise

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
                    "best_val_acc": best_val,  # Include best_val_acc in training step logs
                }
                log_dict.update(loss_dict)
                if wandb.run is not None:
                    wandb.log(log_dict, commit=True)  # commit=True ensures immediate sync

            postfix = {
                "loss": f"{loss.item():.4f}",
                "tau": f"{current_tau:.3f}",
                "mix": f"{loss_dict.get('loss_mixture', 0):.3f}",
                "cls": f"{loss_dict.get('loss_cls', 0):.3f}",
            }
            # Add diversity and entropy if present
            if 'loss_diversity' in loss_dict:
                postfix["div"] = f"{loss_dict['loss_diversity']:.3f}"
            if 'loss_entropy' in loss_dict:
                postfix["ent"] = f"{loss_dict['loss_entropy']:.3f}"
            if 'loss_min_usage' in loss_dict:
                postfix["min_use"] = f"{loss_dict['loss_min_usage']:.3f}"
            pbar.set_postfix(postfix)

        # =====================
        # Validation (with configurable frequency)
        # =====================
        val_freq = config["train"].get("val_freq", 1)  # Default: every epoch
        should_validate = (epoch % val_freq == 0) or (epoch == total_epochs)
        
        # Initialize is_best and val_acc
        is_best = False
        val_acc = None
        
        if should_validate:
            val_acc = validate(model, val_loader, device)

            is_best = val_acc > best_val
            if is_best:
                best_val = val_acc

            if _is_main_process():
                print(f"Epoch {epoch}: Val Acc = {val_acc:.4f} (best={best_val:.4f})")
                # Log both current val_acc and best_val_acc for each epoch
                # Use the current training step, not epoch number, to avoid step ordering conflicts
                if wandb.run is not None:
                    wandb.log({"val_acc": val_acc, "best_val_acc": best_val}, step=step)
        else:
            # Still log best_val_acc even if not validating
            if _is_main_process():
                print(f"Epoch {epoch}: (skipping validation, next at epoch {((epoch // val_freq) + 1) * val_freq})")
                if wandb.run is not None:
                    wandb.log({"best_val_acc": best_val}, step=epoch)

        # =====================
        # Save Checkpoint
        # =====================
        if _is_main_process():
            save_model = model.module if hasattr(model, "module") else model
            
            # Save best checkpoint if validation improved
            if is_best:
                ckpt_filename = f"{experiment_name}_best_epoch{epoch}.pt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model": save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "tau": current_tau,
                }, ckpt_path)
                print(f"Saved new BEST checkpoint: {ckpt_path}")
            
            # Also save final epoch checkpoint (in addition to best, if different)
            if epoch == total_epochs:
                ckpt_filename = f"{experiment_name}_final_epoch{epoch}.pt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                final_val_acc = val_acc if should_validate else None
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "model": save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "val_acc": final_val_acc,
                    "tau": current_tau,
                }, ckpt_path)
                print(f"Saved FINAL epoch checkpoint: {ckpt_path} (val_acc={final_val_acc:.4f if final_val_acc is not None else 'N/A'})")

    if _is_main_process():
        print("\nTraining Complete! Best Val Acc:", best_val)
        # Log final metrics to wandb summary for sweep comparison
        # Ensure both val_acc and best_val_acc are in summary
        # Also log model.K so it appears in sweep comparisons
        if wandb.run is not None:
            # Get the actual K value used (may have been overridden by sweep)
            k_value = config.get("model", {}).get("K", None)
            
            # Method 1: Update summary dictionary directly
            wandb.run.summary["best_val_acc"] = best_val
            wandb.run.summary["val_acc"] = best_val
            wandb.run.summary["final_epoch"] = total_epochs
            if k_value is not None:
                wandb.run.summary["model.K"] = k_value
                wandb.run.summary["config.model.K"] = k_value  # Also log with dot notation for compatibility
            
            # Method 2: Also use summary.update() 
            summary_dict = {
                "best_val_acc": best_val,
                "val_acc": best_val,
                "final_epoch": total_epochs
            }
            if k_value is not None:
                summary_dict["model.K"] = k_value
                summary_dict["config.model.K"] = k_value
            wandb.summary.update(summary_dict)
            
            # Method 3: Log as a final step to ensure it's recorded
            log_dict = {
                "best_val_acc": best_val,
                "val_acc": best_val,
                "final_epoch": total_epochs
            }
            if k_value is not None:
                log_dict["model.K"] = k_value
            wandb.log(log_dict, step=total_epochs)
        
        wandb.finish()

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

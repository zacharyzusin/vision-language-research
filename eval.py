"""
Evaluation script for Mixture-of-Prompts CLIP model.

This script loads a trained checkpoint and evaluates it on the validation set.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from typing import Dict, Tuple

from src.datasets.inat_dataset import get_inat, extract_hierarchical_metadata
from src.datasets.kikibouba_dataset import get_kikibouba, extract_kikibouba_metadata
from src.datasets.stanford_cars_dataset import get_stanford_cars, extract_stanford_cars_metadata
from src.models.mop_clip import MixturePromptCLIP
from src.metrics.evaluation_metrics import compute_class_shots, compute_metrics, print_metrics


@torch.no_grad()
def get_model_logits(model, images, device):
    """
    Get logits/scores from model for given images.
    
    This replicates the logic from model.predict() but returns scores instead of argmax.
    
    Args:
        model: MixturePromptCLIP model instance
        images: Image tensor (B, 3, H, W)
        device: Device
    
    Returns:
        scores: (B, C) similarity scores for each class
    """
    import torch.nn.functional as F
    
    img_feat = model.clip.encode_image(images).float()
    img_feat = F.normalize(img_feat, dim=-1)
    
    all_prompts = model._all_prompt_features(device)  # (C, K, D)
    C = all_prompts.size(0)
    
    chunk_size = 512
    scores_accum = []
    
    for chunk in torch.split(torch.arange(C, device=device), chunk_size):
        pf = all_prompts[chunk]                     # (chunk, K, D)
        sims = torch.einsum("bd,ckd->bck", img_feat, pf)  # (B, chunk, K)
        scores = sims.max(dim=2).values             # (B, chunk)
        scores_accum.append(scores)
    
    final_scores = torch.cat(scores_accum, dim=1)  # (B, C)
    return final_scores


@torch.no_grad()
def evaluate_with_metrics(
    model, 
    dataloader, 
    device,
    class_shots: Dict[int, int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and compute comprehensive metrics.

    Args:
        model: MixturePromptCLIP model instance
        dataloader: DataLoader for evaluation set
        device: Device to run evaluation on
        class_shots: Dictionary mapping class_id -> number of training examples

    Returns:
        Dictionary of metric names -> values
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        # Get logits/scores from model
        logits = get_model_logits(model, images, device)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    # Concatenate all predictions
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(logits, labels, class_shots, device)
    
    return metrics




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/iNat2021")
    parser.add_argument("--version", type=str, default="2021", help="Dataset version: 2018, 2021, kikibouba, or stanford_cars")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--category_ids", type=int, nargs="+", default=None,
                        help="Category IDs for subset evaluation (e.g., --category_ids 5439 5440 5441)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file to load category_ids from")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine dataset type
    if args.version == "kikibouba":
        dataset_type = "kikibouba"
    elif args.version == "stanford_cars":
        dataset_type = "stanford_cars"
    else:
        dataset_type = "inat"
    
    # Load category_ids from config if provided
    category_ids = args.category_ids
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            dataset_type = config.get("dataset", {}).get("type", dataset_type)
            category_ids = config.get("dataset", {}).get("category_ids", None)
            if category_ids:
                print(f"Loaded category_ids from config: {category_ids}")

    print("Loading dataset...")
    if dataset_type == "kikibouba":
        train_ds = get_kikibouba(args.data_root, "train")
        val_ds = get_kikibouba(args.data_root, "val")
        metadata = extract_kikibouba_metadata(args.data_root)
    elif dataset_type == "stanford_cars":
        train_ds = get_stanford_cars(args.data_root, "train")
        val_ds = get_stanford_cars(args.data_root, "val")  # Maps to test split
        metadata = extract_stanford_cars_metadata(args.data_root)
    else:
        train_ds = get_inat(args.data_root, "train", version=args.version, category_ids=category_ids)
        val_ds = get_inat(args.data_root, "val", version=args.version, category_ids=category_ids)
        metadata = extract_hierarchical_metadata(args.data_root, category_ids=category_ids)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Compute class shots from training set (optimized - no image loading)
    print("Computing class shot statistics from training set...")
    class_shots = compute_class_shots(train_ds, train_ds.num_classes)
    
    # Count classes by shot category
    many_shot = sum(1 for count in class_shots.values() if count > 100)
    medium_shot = sum(1 for count in class_shots.values() if 20 <= count <= 100)
    few_shot = sum(1 for count in class_shots.values() if count < 20)
    print(f"  Class distribution: {many_shot} many-shot (>100), {medium_shot} medium-shot (20-100), {few_shot} few-shot (<20)")

    # Metadata already loaded above based on dataset type

    print("Loading model checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Extract model configuration from checkpoint or use defaults
    clip_model = ckpt.get("clip_model", "ViT-B/16")
    K = ckpt.get("K", 32)  # Try to get K from checkpoint, fallback to 32
    
    # If K is not in checkpoint, try to infer from model state dict
    if K == 32 and "model" in ckpt:
        # Try to infer K from prompt_offsets shape: (C, K, D)
        try:
            prompt_offsets_shape = ckpt["model"]["prompt_offsets"].shape
            if len(prompt_offsets_shape) == 3:
                K = prompt_offsets_shape[1]
                print(f"Inferred K={K} from checkpoint")
        except (KeyError, AttributeError):
            pass

    model = MixturePromptCLIP(
        clip_model=clip_model,
        metadata=metadata,
        K=K,
        em_tau=0.3,   # Not used for prediction, but required for initialization
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.clip.to(device)

    # Evaluate with comprehensive metrics
    metrics = evaluate_with_metrics(model, val_loader, device, class_shots=class_shots)
    
    # Print results
    print_metrics(metrics, title="Mixture-of-Prompts CLIP Evaluation")


if __name__ == "__main__":
    main()

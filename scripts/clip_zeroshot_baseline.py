"""
Zero-shot CLIP baseline evaluation with hand-crafted prompts.

This script evaluates a standard CLIP model (without fine-tuning) on various datasets
to establish a baseline for comparison with the Mixture-of-Prompts approach.

Supports:
- iNaturalist: with both common names and scientific names
- Stanford Cars: with car make/model names
- KikiBouba: with class names
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import os
import json
import argparse
from typing import List, Dict, Literal

# Import dataset loaders
from src.datasets.inat_dataset import get_inat, extract_hierarchical_metadata
from src.datasets.stanford_cars_dataset import get_stanford_cars, extract_stanford_cars_metadata
from src.datasets.kikibouba_dataset import get_kikibouba, extract_kikibouba_metadata
from src.metrics.evaluation_metrics import compute_class_shots, compute_metrics, print_metrics

def get_inat_templates(name_type: Literal["common", "scientific", "both"] = "both") -> List[str]:
    """
    Get prompt templates for iNaturalist dataset.
    
    Args:
        name_type: "common" for common names, "scientific" for scientific names, "both" for both
    
    Returns:
        List of template strings
    """
    if name_type == "common":
        return [
            "a photo of a {}",
            "a picture of a {}",
            "a wildlife photo of a {}",
            "an image of a {}",
        ]
    elif name_type == "scientific":
        return [
            "a photo of {}",
            "a picture of {}",
            "a wildlife photo of {}",
            "an image of {}",
        ]
    else:  # both
        return [
            "a photo of a {}",
            "a picture of a {}",
            "a wildlife photo of a {}",
            "an image of a {}",
            "a photo of {}",
            "a picture of {}",
        ]


def get_cars_templates() -> List[str]:
    """Get prompt templates for Stanford Cars dataset."""
    return [
        "a photo of a {}",
        "a picture of a {}",
        "a {} car",
        "an image of a {}",
        "a {} vehicle",
    ]


def get_kikibouba_templates() -> List[str]:
    """Get prompt templates for KikiBouba dataset."""
    return [
        "a photo of {}",
        "a picture of {}",
        "an image of {}",
        "a {} shape",
    ]


@torch.no_grad()
def build_zeroshot_classifier(
    model, 
    class_names: List[str], 
    device, 
    templates: List[str] = None
):
    """
    Build zero-shot text classifier weights from class names.

    Args:
        model: CLIP model instance
        class_names: List of class name strings
        device: Device to run computation on
        templates: List of prompt templates (default: ["a photo of a {}"])

    Returns:
        Normalized text embeddings of shape (num_classes, embedding_dim)
    """
    if templates is None:
        templates = ["a photo of a {}"]

    texts = []
    for name in class_names:
        # Clean up name (replace underscores with spaces)
        clean_name = name.replace("_", " ")
        texts += [template.format(clean_name) for template in templates]

    tokenized = clip.tokenize(texts).to(device)
    text_embeddings = model.encode_text(tokenized).float()
    text_embeddings = text_embeddings.view(len(class_names), len(templates), -1)
    text_embeddings = text_embeddings.mean(dim=1)
    return F.normalize(text_embeddings, dim=-1).float()


@torch.no_grad()
def build_inat_zeroshot_classifier(
    model,
    categories: List[Dict],
    device,
    name_type: Literal["common", "scientific", "both"] = "both",
    templates: List[str] = None
):
    """
    Build zero-shot classifier for iNaturalist with common/scientific names.
    
    Args:
        model: CLIP model instance
        categories: List of category dictionaries from categories.json
        device: Device to run computation on
        name_type: "common" for common names, "scientific" for scientific names, "both" for both
        templates: Optional custom templates (if None, uses default for name_type)
    
    Returns:
        Normalized text embeddings of shape (num_classes, embedding_dim)
    """
    if templates is None:
        templates = get_inat_templates(name_type)
    
    texts = []
    for cat in categories:
        if name_type == "common":
            # Use common name if available, fallback to scientific
            name = cat.get("common_name", cat.get("name", ""))
            if not name:
                name = cat.get("name", "")
            clean_name = name.replace("_", " ")
            texts += [template.format(clean_name) for template in templates]
            
        elif name_type == "scientific":
            # Use scientific name
            name = cat.get("name", "")
            clean_name = name.replace("_", " ")
            texts += [template.format(clean_name) for template in templates]
            
        else:  # both
            # Use both common and scientific names
            common_name = cat.get("common_name", "")
            scientific_name = cat.get("name", "")
            
            # Add prompts for common name (if available)
            if common_name:
                clean_common = common_name.replace("_", " ")
                texts += [template.format(clean_common) for template in templates]
            
            # Add prompts for scientific name
            if scientific_name:
                clean_scientific = scientific_name.replace("_", " ")
                texts += [template.format(clean_scientific) for template in templates]
            
            # If neither available, skip (shouldn't happen)
            if not common_name and not scientific_name:
                # Fallback: use empty string
                texts += [template.format("") for template in templates]
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    all_embeddings = []
    
    print(f"  Processing {len(texts)} prompts in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding prompts", leave=False):
        batch_texts = texts[i:i+batch_size]
        tokenized = clip.tokenize(batch_texts).to(device)
        batch_embeddings = model.encode_text(tokenized).float()
        all_embeddings.append(batch_embeddings)
    
    text_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Calculate number of templates per class
    if name_type == "both":
        # Count how many names we actually have per class
        num_names_per_class = []
        for cat in categories:
            common_name = cat.get("common_name", "")
            scientific_name = cat.get("name", "")
            count = (1 if common_name else 0) + (1 if scientific_name else 0)
            if count == 0:
                count = 1  # Fallback
            num_names_per_class.append(count)
        
        # Group embeddings by class and average
        class_embeddings = []
        idx = 0
        for num_names in num_names_per_class:
            num_prompts = num_names * len(templates)
            class_emb = text_embeddings[idx:idx+num_prompts].mean(dim=0)
            class_embeddings.append(class_emb)
            idx += num_prompts
        
        text_embeddings = torch.stack(class_embeddings, dim=0)
    else:
        text_embeddings = text_embeddings.view(len(categories), len(templates), -1)
        text_embeddings = text_embeddings.mean(dim=1)
    
    return F.normalize(text_embeddings, dim=-1).float()

@torch.no_grad()
def evaluate_with_metrics(model, dataloader, zeroshot_weights, device):
    """
    Evaluate zero-shot CLIP model on dataset and return logits for metrics.

    Args:
        model: CLIP model instance
        dataloader: DataLoader for evaluation set
        zeroshot_weights: Text embeddings of shape (num_classes, embedding_dim)
        device: Device to run evaluation on

    Returns:
        Tuple of (all_logits, all_labels) tensors
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        image_features = model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)
        logits = 100.0 * image_features @ zeroshot_weights.T
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot CLIP baseline evaluation with hand-crafted prompts"
    )
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to dataset root directory")
    parser.add_argument("--dataset", type=str, default="inat",
                       choices=["inat", "stanford_cars", "kikibouba"],
                       help="Dataset type")
    parser.add_argument("--version", type=str, default="2021",
                       help="Dataset version (for iNaturalist: 2018 or 2021)")
    parser.add_argument("--name_type", type=str, default="both",
                       choices=["common", "scientific", "both"],
                       help="For iNaturalist: use common names, scientific names, or both")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       help="CLIP model name (e.g., ViT-B/16, ViT-B/32)")
    parser.add_argument("--split", type=str, default="val",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == "inat":
        print(f"  Version: {args.version}")
        print(f"  Name type: {args.name_type}")

    # Load CLIP model
    print(f"\nLoading CLIP model: {args.clip_model}")
    model, _ = clip.load(args.clip_model, device=device)
    model.eval()

    # Load datasets (both train and val for shot statistics)
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "inat":
        train_ds = get_inat(args.data_root, "train", version=args.version)
        val_ds = get_inat(args.data_root, args.split, version=args.version)
        val_loader = DataLoader(
            val_ds, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=8,
            pin_memory=True
        )
        
        # Load categories for building classifier
        cat_path = os.path.join(args.data_root, "categories.json")
        with open(cat_path, "r") as f:
            categories = json.load(f)
        categories.sort(key=lambda x: x["id"])
        
        print(f"  Classes: {len(categories)}")
        print(f"  Training samples: {len(train_ds)}")
        print(f"  Validation samples: {len(val_ds)}")
        
        # Build zero-shot classifier
        print(f"\nBuilding zero-shot classifier with {args.name_type} names...")
        zeroshot_weights = build_inat_zeroshot_classifier(
            model, categories, device, name_type=args.name_type
        )
        
    elif args.dataset == "stanford_cars":
        train_ds = get_stanford_cars(args.data_root, "train")
        val_ds = get_stanford_cars(args.data_root, args.split)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        # Get class names
        class_names = val_ds.classes
        print(f"  Classes: {len(class_names)}")
        print(f"  Training samples: {len(train_ds)}")
        print(f"  Validation samples: {len(val_ds)}")
        
        # Build zero-shot classifier
        print("\nBuilding zero-shot classifier...")
        templates = get_cars_templates()
        zeroshot_weights = build_zeroshot_classifier(
            model, class_names, device, templates=templates
        )
        
    elif args.dataset == "kikibouba":
        train_ds = get_kikibouba(args.data_root, "train")
        val_ds = get_kikibouba(args.data_root, args.split)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Get class names
        class_names = [val_ds.idx_to_class[i] for i in range(val_ds.num_classes)]
        print(f"  Classes: {len(class_names)}")
        print(f"  Training samples: {len(train_ds)}")
        print(f"  Validation samples: {len(val_ds)}")
        
        # Build zero-shot classifier
        print("\nBuilding zero-shot classifier...")
        templates = get_kikibouba_templates()
        zeroshot_weights = build_zeroshot_classifier(
            model, class_names, device, templates=templates
        )

    # Compute class shots from training set (optimized - no image loading)
    print("\nComputing class shot statistics from training set...")
    class_shots = compute_class_shots(train_ds, train_ds.num_classes)
    
    # Count classes by shot category
    many_shot = sum(1 for count in class_shots.values() if count > 100)
    medium_shot = sum(1 for count in class_shots.values() if 20 <= count <= 100)
    few_shot = sum(1 for count in class_shots.values() if count < 20)
    print(f"  Class distribution: {many_shot} many-shot (>100), {medium_shot} medium-shot (20-100), {few_shot} few-shot (<20)")

    # Evaluate
    print(f"\nEvaluating zero-shot CLIP on {args.split} split...")
    logits, labels = evaluate_with_metrics(model, val_loader, zeroshot_weights, device)
    
    # Compute comprehensive metrics
    print("\nComputing comprehensive metrics...")
    metrics = compute_metrics(logits, labels, class_shots=class_shots, device=device)
    
    # Print results
    title = f"Zero-shot CLIP Evaluation"
    if args.dataset == "inat":
        title += f" ({args.name_type} names)"
    print_metrics(metrics, title=title)

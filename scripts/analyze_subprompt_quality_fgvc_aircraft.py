"""
Quantitative analysis of sub-prompt cluster quality for FGVC Aircraft dataset.

This script analyzes:
1. Assignment entropy (how spread out are assignments)
2. Sub-prompt diversity (how different are the sub-prompts)
3. Usage statistics (how many images use each sub-prompt)
4. Cluster coherence (within vs between cluster similarity)
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# Add repository root to PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from train import load_config
from src.datasets.fgvc_aircraft_dataset import get_fgvc_aircraft, extract_fgvc_aircraft_metadata
from src.models.mop_clip import MixturePromptCLIP


def compute_assignment_entropy(gamma, eps=1e-8):
    """Compute entropy of assignment distribution."""
    # gamma: [batch_size, K]
    # Entropy: -sum(p * log(p))
    gamma_clamped = torch.clamp(gamma, min=eps)
    entropy = -(gamma_clamped * torch.log(gamma_clamped)).sum(dim=1)  # [batch_size]
    return entropy


def compute_subprompt_diversity(model, device):
    """Compute cosine similarity between all pairs of sub-prompts for each class."""
    num_classes = len(model.metadata)
    K = model.K
    
    # Get prompt features for all classes
    # Shape: [num_classes, K, embedding_dim]
    all_prompt_features = []
    for class_idx in range(num_classes):
        label_tensor = torch.tensor([class_idx], device=device)
        prompt_feats = model._batch_prompt_features(label_tensor, device)  # [1, K, D]
        all_prompt_features.append(prompt_feats[0])  # [K, D]
    
    all_prompt_features = torch.stack(all_prompt_features)  # [num_classes, K, D]
    
    # Normalize
    all_prompt_features = F.normalize(all_prompt_features, dim=-1)
    
    # Compute pairwise similarities within each class
    similarities = []
    for class_idx in range(num_classes):
        class_prompts = all_prompt_features[class_idx]  # [K, D]
        # Compute all pairwise similarities
        sim_matrix = torch.mm(class_prompts, class_prompts.t())  # [K, K]
        # Get upper triangle (excluding diagonal)
        triu_indices = torch.triu_indices(K, K, offset=1)
        sims = sim_matrix[triu_indices[0], triu_indices[1]]
        similarities.append(sims.detach().cpu().numpy())
    
    return np.array(similarities)  # [num_classes, K*(K-1)/2]


def analyze_cluster_quality(model, dataloader, device, temperature=5.0):
    """Analyze assignment quality and cluster statistics."""
    model.eval()
    
    all_gammas = []
    all_labels = []
    all_img_features = []
    
    print("Computing assignments for all images...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get image features
            img_feat = model.clip.encode_image(images).float()
            img_feat = F.normalize(img_feat, dim=-1)
            all_img_features.append(img_feat.cpu())
            
            # Get prompt features for all labels in batch
            batch_size = images.size(0)
            K = model.K
            
            prompt_feats_list = []
            for label in labels:
                label_tensor = label.unsqueeze(0)
                prompt_feats = model._batch_prompt_features(label_tensor, device)  # [1, K, D]
                prompt_feats_list.append(prompt_feats[0])  # [K, D]
            
            prompt_feats = torch.stack(prompt_feats_list)  # [batch_size, K, D]
            
            # Compute similarities
            sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats) * model.sim_scale
            
            # Compute gamma with visualization temperature
            gamma = F.softmax(sims / temperature, dim=1)  # [batch_size, K]
            
            all_gammas.append(gamma.cpu())
            all_labels.append(labels.cpu())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    all_gammas = torch.cat(all_gammas, dim=0)  # [total_samples, K]
    all_labels = torch.cat(all_labels, dim=0)  # [total_samples]
    all_img_features = torch.cat(all_img_features, dim=0)  # [total_samples, D]
    
    print(f"Total samples: {len(all_gammas)}")
    
    # ===================
    # 1. Assignment Entropy
    # ===================
    entropies = compute_assignment_entropy(all_gammas)
    print(f"\n{'='*60}")
    print("1. ASSIGNMENT ENTROPY (Higher = More Spread Out)")
    print(f"{'='*60}")
    print(f"  Mean entropy: {entropies.mean():.4f} (max possible: {np.log(model.K):.4f})")
    print(f"  Std entropy: {entropies.std():.4f}")
    print(f"  Min entropy: {entropies.min():.4f}")
    print(f"  Max entropy: {entropies.max():.4f}")
    print(f"  Entropy ratio (mean/max): {entropies.mean() / np.log(model.K):.4f}")
    
    # ===================
    # 2. Hard Assignment Statistics
    # ===================
    hard_assignments = all_gammas.argmax(dim=1)  # [total_samples]
    usage_per_k = torch.bincount(hard_assignments, minlength=model.K).float()
    usage_per_k = usage_per_k / usage_per_k.sum()  # Normalize to percentages
    
    print(f"\n{'='*60}")
    print("2. SUB-PROMPT USAGE (Hard Assignments)")
    print(f"{'='*60}")
    for k in range(model.K):
        count = (hard_assignments == k).sum().item()
        pct = usage_per_k[k].item() * 100
        print(f"  Sub-prompt k={k}: {count:5d} images ({pct:5.2f}%)")
    
    print(f"\n  Usage balance (std of percentages): {usage_per_k.std():.4f} (lower = more balanced)")
    print(f"  Min usage: {usage_per_k.min()*100:.2f}%, Max usage: {usage_per_k.max()*100:.2f}%")
    
    # ===================
    # 3. Average Gamma Values
    # ===================
    avg_gammas = all_gammas.mean(dim=0)  # [K]
    print(f"\n{'='*60}")
    print("3. AVERAGE GAMMA VALUES (Soft Assignments)")
    print(f"{'='*60}")
    for k in range(model.K):
        print(f"  Sub-prompt k={k}: avg_gamma={avg_gammas[k]:.4f} (expected: {1.0/model.K:.4f})")
    
    print(f"\n  Expected uniform: {1.0/model.K:.4f} per sub-prompt")
    print(f"  Gamma balance (std): {avg_gammas.std():.4f} (lower = more balanced)")
    
    # ===================
    # 4. Per-Class Statistics
    # ===================
    num_classes = len(model.metadata)
    per_class_entropy = []
    per_class_usage = []
    
    for class_idx in range(num_classes):
        class_mask = (all_labels == class_idx)
        if class_mask.sum() == 0:
            continue
        
        class_gammas = all_gammas[class_mask]
        class_entropy = compute_assignment_entropy(class_gammas).mean().item()
        per_class_entropy.append(class_entropy)
        
        class_hard = class_gammas.argmax(dim=1)
        class_usage = torch.bincount(class_hard, minlength=model.K).float()
        class_usage = class_usage / class_usage.sum()
        per_class_usage.append(class_usage.numpy())
    
    per_class_usage = np.array(per_class_usage)  # [num_classes, K]
    
    print(f"\n{'='*60}")
    print("4. PER-CLASS STATISTICS")
    print(f"{'='*60}")
    print(f"  Mean entropy per class: {np.mean(per_class_entropy):.4f}")
    print(f"  Classes with entropy > {np.log(model.K) * 0.8:.4f}: {(np.array(per_class_entropy) > np.log(model.K) * 0.8).sum()}/{num_classes}")
    print(f"  Classes with entropy < {np.log(model.K) * 0.5:.4f}: {(np.array(per_class_entropy) < np.log(model.K) * 0.5).sum()}/{num_classes}")
    
    # Classes with most balanced usage
    class_usage_balance = per_class_usage.std(axis=1)  # Lower = more balanced
    best_classes = np.argsort(class_usage_balance)[:5]
    worst_classes = np.argsort(class_usage_balance)[-5:][::-1]
    
    print(f"\n  Top 5 most balanced classes (by usage std):")
    for idx in best_classes:
        cls_name = model.metadata[idx].get("variant", f"class_{idx}")
        print(f"    Class {idx} ({cls_name}): std={class_usage_balance[idx]:.4f}")
    
    print(f"\n  Top 5 least balanced classes (by usage std):")
    for idx in worst_classes:
        cls_name = model.metadata[idx].get("variant", f"class_{idx}")
        print(f"    Class {idx} ({cls_name}): std={class_usage_balance[idx]:.4f}")
    
    # ===================
    # 5. Sub-prompt Diversity
    # ===================
    print(f"\n{'='*60}")
    print("5. SUB-PROMPT DIVERSITY (Within-Class Similarity)")
    print(f"{'='*60}")
    similarities = compute_subprompt_diversity(model, device)  # [num_classes, K*(K-1)/2]
    
    mean_sim_per_class = similarities.mean(axis=1)  # [num_classes]
    overall_mean_sim = similarities.mean()
    overall_std_sim = similarities.std()
    
    print(f"  Mean pairwise similarity (all classes): {overall_mean_sim:.4f}")
    print(f"  Std pairwise similarity: {overall_std_sim:.4f}")
    print(f"  Max similarity (worst diversity): {similarities.max():.4f}")
    print(f"  Min similarity (best diversity): {similarities.min():.4f}")
    
    # Classes with most/least diverse sub-prompts
    best_diverse = np.argsort(mean_sim_per_class)[:5]
    worst_diverse = np.argsort(mean_sim_per_class)[-5:][::-1]
    
    print(f"\n  Top 5 most diverse classes (lowest similarity):")
    for idx in best_diverse:
        cls_name = model.metadata[idx].get("variant", f"class_{idx}")
        print(f"    Class {idx} ({cls_name}): mean_sim={mean_sim_per_class[idx]:.4f}")
    
    print(f"\n  Top 5 least diverse classes (highest similarity):")
    for idx in worst_diverse:
        cls_name = model.metadata[idx].get("variant", f"class_{idx}")
        print(f"    Class {idx} ({cls_name}): mean_sim={mean_sim_per_class[idx]:.4f}")
    
    # ===================
    # 6. Cluster Coherence (Within vs Between Cluster)
    # ===================
    print(f"\n{'='*60}")
    print("6. CLUSTER COHERENCE")
    print(f"{'='*60}")
    
    # For each class, compute:
    # - Within-cluster similarity: avg similarity of images to their assigned sub-prompt
    # - Between-cluster similarity: avg similarity of images to other sub-prompts
    within_cluster_sims = []
    between_cluster_sims = []
    
    with torch.no_grad():
        # Get prompt features for each class
        prompt_feats_all = []
        for class_idx in range(num_classes):
            label_tensor = torch.tensor([class_idx], device=device)
            prompt_feats = model._batch_prompt_features(label_tensor, device)[0]  # [K, D]
            prompt_feats_all.append(prompt_feats)
        
        # Analyze per class
        for class_idx in range(num_classes):
            class_mask = (all_labels == class_idx)
            if class_mask.sum() == 0:
                continue
            
            class_img_feat = all_img_features[class_mask].to(device)  # [N, D]
            class_gammas = all_gammas[class_mask].to(device)  # [N, K]
            class_hard = class_gammas.argmax(dim=1)  # [N]
            
            prompt_feats = prompt_feats_all[class_idx]  # [K, D]
            
            # Compute similarities to all sub-prompts
            sims = torch.mm(class_img_feat, prompt_feats.t()) * model.sim_scale  # [N, K]
            
            # Within-cluster: similarity to assigned sub-prompt
            within_sims = sims[torch.arange(len(class_img_feat)), class_hard]
            within_cluster_sims.append(within_sims.cpu().numpy())
            
            # Between-cluster: similarity to non-assigned sub-prompts
            for k in range(model.K):
                mask = (class_hard != k)
                if mask.sum() > 0:
                    between_sims = sims[mask, k]
                    between_cluster_sims.append(between_sims.cpu().numpy())
    
    within_sims_all = np.concatenate(within_cluster_sims)
    between_sims_all = np.concatenate(between_cluster_sims)
    
    print(f"  Within-cluster similarity: {within_sims_all.mean():.4f} ± {within_sims_all.std():.4f}")
    print(f"  Between-cluster similarity: {between_sims_all.mean():.4f} ± {between_sims_all.std():.4f}")
    print(f"  Separation (within - between): {within_sims_all.mean() - between_sims_all.mean():.4f}")
    print(f"  Separation ratio: {(within_sims_all.mean() - between_sims_all.mean()) / (within_sims_all.std() + 1e-8):.4f}")
    
    # ===================
    # Summary
    # ===================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Assignment entropy: {entropies.mean():.4f} / {np.log(model.K):.4f} ({entropies.mean() / np.log(model.K) * 100:.1f}% of max)")
    print(f"✓ Usage balance: std={usage_per_k.std():.4f} (lower is better)")
    print(f"✓ Gamma balance: std={avg_gammas.std():.4f} (lower is better)")
    print(f"✓ Sub-prompt diversity: mean similarity={overall_mean_sim:.4f} (lower is better)")
    print(f"✓ Cluster coherence: separation={within_sims_all.mean() - between_sims_all.mean():.4f} (higher is better)")
    
    if entropies.mean() / np.log(model.K) > 0.7:
        print("\n✅ Good: Assignments are well spread out")
    else:
        print("\n⚠️  Warning: Assignments may be too concentrated")
    
    if usage_per_k.std() < 0.05:
        print("✅ Good: All sub-prompts are being used fairly equally")
    else:
        print("⚠️  Warning: Some sub-prompts are underused")
    
    if overall_mean_sim < 0.3:
        print("✅ Good: Sub-prompts are diverse")
    elif overall_mean_sim > 0.7:
        print("⚠️  Warning: Sub-prompts may be too similar (mode collapse)")
    else:
        print("⚠️  Moderate: Sub-prompt diversity could be improved")
    
    if (within_sims_all.mean() - between_sims_all.mean()) > 0.1:
        print("✅ Good: Clear cluster separation")
    else:
        print("⚠️  Warning: Clusters may not be well separated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fgvc_aircraft.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Temperature for soft assignments")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    root = config["dataset"]["root"]
    metadata = extract_fgvc_aircraft_metadata(root)
    
    # Load dataset (combined train + val)
    from torch.utils.data import ConcatDataset
    train_dataset = get_fgvc_aircraft(root, "train", use_torchvision=False)
    val_dataset = get_fgvc_aircraft(root, "val", use_torchvision=False)
    dataset = ConcatDataset([train_dataset, val_dataset])
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print(f"\nLoading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt
    
    # Remove 'module.' prefix if present (from DDP)
    if any(k.startswith('module.') for k in model_state.keys()):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items() if k.startswith('module.')}
    
    # Determine K from checkpoint
    if "prompt_offsets" in model_state:
        K_from_ckpt = model_state["prompt_offsets"].shape[1]
    else:
        K_from_ckpt = int(config["model"]["K"])
    
    print(f"Detected K={K_from_ckpt}")
    
    # Initialize model
    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=K_from_ckpt,
        em_tau=float(config["model"].get("em_tau", 1.0)),
    ).to(device)
    model.clip.to(device)
    
    model.load_state_dict(model_state, strict=False)
    model.eval()
    
    print(f"Model has K={model.K} sub-prompts per class")
    print(f"Using temperature={args.temperature} for analysis")
    
    # Run analysis
    analyze_cluster_quality(model, dataloader, device, temperature=args.temperature)


if __name__ == "__main__":
    main()


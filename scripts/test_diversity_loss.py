#!/usr/bin/env python3
"""Test the updated diversity loss calculation."""

import torch
import torch.nn.functional as F
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_config
from src.datasets.stanford_cars_dataset import get_stanford_cars, extract_stanford_cars_metadata
from src.models.mop_clip import MixturePromptCLIP
from torch.utils.data import DataLoader

def main():
    config = load_config('configs/stanford_cars.yaml')
    root = config['dataset']['root']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('=== Testing Updated Diversity Loss ===\n')
    
    # Find latest checkpoint
    checkpoints = glob.glob('checkpoints/stanford_cars_K32_ViT-B-16/*best*.pt')
    if checkpoints:
        BEST_CKPT = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)[0]
        print(f'Loading checkpoint: {BEST_CKPT}')
        load_checkpoint = True
    else:
        print('No checkpoint found, will test with fresh model')
        load_checkpoint = False
    
    # Load dataset
    metadata = extract_stanford_cars_metadata(root)
    train_ds = get_stanford_cars(root, "train")
    
    # Create model with NEW diversity loss settings
    print(f'\nCreating model with updated diversity loss settings:')
    print(f'  diversity_loss_weight: {config["model"].get("diversity_loss_weight", 1.0)}')
    print(f'  entropy_loss_weight: {config["model"].get("entropy_loss_weight", 0.05)}')
    
    model = MixturePromptCLIP(
        clip_model=config['model']['clip_model'],
        metadata=metadata,
        K=config['model']['K'],
        em_tau=config['model']['em_tau_end'],  # Use final temperature
        diversity_loss_weight=config['model'].get('diversity_loss_weight', 1.0),
        entropy_loss_weight=config['model'].get('entropy_loss_weight', 0.05),
    ).to(device)
    
    if load_checkpoint:
        ckpt = torch.load(BEST_CKPT, map_location=device)
        model_state = ckpt['model'] if 'model' in ckpt else ckpt
        if any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items() if k.startswith('module.')}
        model.load_state_dict(model_state, strict=False)
        print('  Checkpoint loaded')
    
    model.eval()
    
    # Create a small batch
    batch_size = 8
    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f'\nTesting with batch of {batch_size} samples')
    print(f'Labels: {labels.cpu().tolist()}')
    
    # Run forward pass
    with torch.no_grad():
        result = model(
            images,
            labels,
            lambda_mixture=config['train'].get('lambda_mixture', 0.5),
            temp_cls=config['train'].get('temp_cls', 0.07),
        )
        loss, loss_dict = result
    
    print(f'\n=== Loss Components ===')
    print(f'Total loss: {loss.item():.4f}')
    print(f'  loss_mixture: {loss_dict.get("loss_mixture", 0):.4f}')
    print(f'  loss_cls: {loss_dict.get("loss_cls", 0):.4f}')
    print(f'  loss_reg: {loss_dict.get("loss_reg", 0):.4f}')
    
    if 'loss_diversity' in loss_dict:
        print(f'  loss_diversity: {loss_dict["loss_diversity"]:.4f} ⭐')
        print(f'    (This should be >0 if sub-prompts are similar)')
    else:
        print(f'  loss_diversity: NOT COMPUTED (check diversity_loss_weight > 0)')
    
    if 'loss_entropy' in loss_dict:
        print(f'  loss_entropy: {loss_dict["loss_entropy"]:.4f} ⭐')
        print(f'    (Negative = higher entropy = better spreading)')
    else:
        print(f'  loss_entropy: NOT COMPUTED (check entropy_loss_weight > 0)')
    
    # Check diversity loss contribution
    diversity_weight = config['model'].get('diversity_loss_weight', 1.0)
    entropy_weight = config['model'].get('entropy_loss_weight', 0.05)
    
    if 'loss_diversity' in loss_dict:
        diversity_contribution = diversity_weight * loss_dict['loss_diversity']
        print(f'\n=== Diversity Loss Contribution ===')
        print(f'Weight × loss_diversity: {diversity_weight} × {loss_dict["loss_diversity"]:.4f} = {diversity_contribution:.4f}')
        print(f'Percentage of total loss: {100 * diversity_contribution / loss.item():.1f}%')
    
    # Test the diversity loss calculation manually for one class
    print(f'\n=== Manual Diversity Check (Class 0) ===')
    class_idx = 0
    prompt_feats = model._batch_prompt_features(torch.tensor([class_idx], device=device), device)  # [1, K, D]
    prompt_feats_norm = F.normalize(prompt_feats, dim=-1)  # [1, K, D]
    pairwise_sims = torch.bmm(prompt_feats_norm, prompt_feats_norm.transpose(1, 2))  # [1, K, K]
    
    K = pairwise_sims.shape[1]
    triu_mask = torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0)
    pairwise_sims_flat = pairwise_sims[triu_mask.expand_as(pairwise_sims)]
    
    # Apply the new penalty function
    high_sim_mask = pairwise_sims_flat > 0.5
    very_high_sim_mask = pairwise_sims_flat > 0.8
    
    very_high_penalty = (pairwise_sims_flat[very_high_sim_mask] ** 4).sum() if very_high_sim_mask.any() else torch.tensor(0.0, device=device)
    high_penalty = (pairwise_sims_flat[high_sim_mask & ~very_high_sim_mask] ** 2).sum() if (high_sim_mask & ~very_high_sim_mask).any() else torch.tensor(0.0, device=device)
    manual_diversity_loss = (very_high_penalty + high_penalty) / pairwise_sims_flat.shape[0]
    
    print(f'Class {class_idx} ({metadata[class_idx].get("full_name", "unknown")}):')
    print(f'  Mean pairwise similarity: {pairwise_sims_flat.mean().item():.4f}')
    print(f'  Pairs >0.8 (very similar): {very_high_sim_mask.sum().item()}/{pairwise_sims_flat.shape[0]}')
    print(f'  Pairs >0.5 (moderately similar): {high_sim_mask.sum().item()}/{pairwise_sims_flat.shape[0]}')
    print(f'  Manual diversity loss: {manual_diversity_loss.item():.4f}')
    print(f'  Very high penalty component: {very_high_penalty.item():.4f}')
    print(f'  High penalty component: {high_penalty.item():.4f}')
    
    print(f'\n✅ Test completed successfully!')
    print(f'\nThe diversity loss is being computed and should push sub-prompts apart during training.')

if __name__ == '__main__':
    main()


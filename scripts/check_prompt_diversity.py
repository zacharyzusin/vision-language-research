#!/usr/bin/env python3
"""Check prompt feature diversity in a trained model."""

import torch
import torch.nn.functional as F
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_config
from src.datasets.stanford_cars_dataset import extract_stanford_cars_metadata
from src.models.mop_clip import MixturePromptCLIP

def main():
    config = load_config('configs/stanford_cars.yaml')
    root = config['dataset']['root']
    metadata = extract_stanford_cars_metadata(root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find latest checkpoint
    checkpoints = glob.glob('checkpoints/stanford_cars_K32_ViT-B-16/*best*.pt')
    if not checkpoints:
        print('No checkpoint found!')
        return
    BEST_CKPT = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)[0]

    print('=== Checking Prompt Feature Diversity ===')
    print(f'Loading checkpoint: {BEST_CKPT}')

    ckpt = torch.load(BEST_CKPT, map_location=device)
    model_state = ckpt['model'] if 'model' in ckpt else ckpt
    if any(k.startswith('module.') for k in model_state.keys()):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items() if k.startswith('module.')}
    K_from_ckpt = model_state['prompt_offsets'].shape[1]

    model = MixturePromptCLIP(
        clip_model=config['model']['clip_model'],
        metadata=metadata,
        K=K_from_ckpt,
        em_tau=2.0,
    ).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # Check multiple classes
    for class_idx in [0, 1, 5, 10, 15]:
        prompt_feats = model._batch_prompt_features(torch.tensor([class_idx], device=device), device)
        prompt_feats = prompt_feats[0]  # [K, D]

        print(f'\nClass {class_idx} ({metadata[class_idx].get("full_name", "unknown")})')
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(K_from_ckpt):
            for j in range(i+1, K_from_ckpt):
                sim = F.cosine_similarity(prompt_feats[i:i+1], prompt_feats[j:j+1])
                similarities.append(sim.item())

        print(f'  Mean similarity: {sum(similarities)/len(similarities):.4f}')
        print(f'  Min: {min(similarities):.4f}, Max: {max(similarities):.4f}')
        very_similar = sum(1 for s in similarities if s > 0.9)
        print(f'  Pairs >0.9: {very_similar}/{len(similarities)} ({100*very_similar/len(similarities):.1f}%)')

if __name__ == '__main__':
    main()


"""
Evaluation script for Mixture-of-Prompts CLIP model.

This script loads a trained checkpoint and evaluates it on the validation set.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
from src.models.mop_clip import MixturePromptCLIP


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on a dataset.

    Args:
        model: MixturePromptCLIP model instance
        dataloader: DataLoader for evaluation set
        device: Device to run evaluation on

    Returns:
        Top-1 accuracy (float between 0 and 1)
    """
    model.eval()
    total = 0
    correct = 0

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        preds = model.predict(images)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/iNat2018")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    val_ds = get_inat2018(args.data_root, "val")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    metadata = extract_hierarchical_metadata(args.data_root)

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

    acc = evaluate(model, val_loader, device)
    print("\nFinal Validation Accuracy:", acc)


if __name__ == "__main__":
    main()

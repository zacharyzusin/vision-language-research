# eval.py

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
from src.models.mop_clip import MixturePromptCLIP


@torch.no_grad()
def evaluate(model, dataloader, device):
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

    model = MixturePromptCLIP(
        clip_model=ckpt.get("clip_model", "ViT-B/16"),
        metadata=metadata,
        K=32,
        em_tau=0.3,   # not used for prediction
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.clip.to(device)

    acc = evaluate(model, val_loader, device)
    print("\nFinal Validation Accuracy:", acc)


if __name__ == "__main__":
    main()

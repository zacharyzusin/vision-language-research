# visualize_subprompts.py

import os
import argparse
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
from src.models.mop_clip import MixturePromptCLIP


@torch.no_grad()
def collect_assignments(model, dataloader, target_class, device, max_per_k=16):
    """
    For a given class c, compute gamma assignments for all its images.
    Return the top-N images for each sub-prompt k.
    """
    model.eval()

    buckets = {k: [] for k in range(model.K)}

    for images, labels in tqdm(dataloader, desc="Collecting γ"):
        images = images.to(device)
        labels = labels.to(device)

        mask = (labels == target_class)
        if mask.sum() == 0:
            continue

        imgs = images[mask]
        lbls = labels[mask]

        # Forward: get sims and gamma
        with torch.no_grad():
            img_feat = model.clip.encode_image(imgs).float()
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)

            prompt_feats = model._batch_prompt_features(lbls, device)
            sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats)
            gamma = torch.softmax(sims / model.em_tau, dim=1)

        for i in range(imgs.size(0)):
            for k in range(model.K):
                buckets[k].append((gamma[i, k].item(), imgs[i].cpu()))

    # Sort each bucket by gamma highest
    for k in buckets:
        buckets[k] = sorted(buckets[k], key=lambda x: -x[0])
        buckets[k] = [img for _, img in buckets[k][:max_per_k]]

    return buckets


def save_grids(buckets, out_dir="subprompt_vis", prefix="class"):
    os.makedirs(out_dir, exist_ok=True)

    for k, imgs in buckets.items():
        if len(imgs) == 0:
            continue
        grid = make_grid(imgs, nrow=4, normalize=True, padding=2)
        path = os.path.join(out_dir, f"{prefix}_k{k}.png")
        save_image(grid, path)
        print("Saved:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/iNat2018")
    parser.add_argument("--class_id", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_per_k", type=int, default=16)
    parser.add_argument("--out_dir", default="subprompt_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = get_inat2018(args.data_root, "val")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    metadata = extract_hierarchical_metadata(args.data_root)

    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = MixturePromptCLIP(
        clip_model=ckpt.get("clip_model", "ViT-B/16"),
        metadata=metadata,
        K=32,
        em_tau=0.3,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.clip.to(device)

    print("Collecting assignments for class:", args.class_id)
    buckets = collect_assignments(
        model, val_loader, args.class_id, device, max_per_k=args.max_per_k
    )

    save_grids(buckets, args.out_dir, prefix=f"class_{args.class_id}")


if __name__ == "__main__":
    main()

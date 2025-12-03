import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip

from src.datasets.inat_dataset import get_inat2018
from train import extract_hierarchical_metadata   # <-- use SAME function


@torch.no_grad()
def run_zero_shot():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Load CLIP
    # ----------------------------
    print("Loading CLIP model: ViT-B/32")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # ----------------------------
    # Load BOTH train + val sets
    # ----------------------------
    root = os.path.join(os.path.dirname(__file__), "data/iNat2018")

    train_ds = get_inat2018(root, split="train")     # <-- ensures class order matches training
    val_ds   = get_inat2018(root, split="val")
    val_ds.transform = preprocess

    print(f"Loaded iNat2018 split: val, samples={len(val_ds)}, classes={len(train_ds.categories)}")

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ----------------------------
    # Extract SAME metadata as training
    # ----------------------------
    metadata = extract_hierarchical_metadata(train_ds)  
    num_classes = len(metadata)

    print(f"Loaded iNat2018 val set: {len(val_ds)} samples, {num_classes} classes")

    # ----------------------------
    # Build prompts
    # ----------------------------
    templates = [
        "a photo of a {species}",
        "a wildlife photo of the species {scientific_name}",
        "a photograph of an organism in genus {genus}",
        "an organism belonging to family {family}",
        "a close-up photo of a {species}",
    ]

    prompts = []
    for md in metadata:
        for t in templates:
            prompts.append(
                t.format(
                    species=md["species"],
                    scientific_name=md["scientific_name"],
                    genus=md["genus"],
                    family=md["family"],
                )
            )

    # ----------------------------
    # Encode prompts in batches
    # ----------------------------
    print("Encoding text prompts...")
    tokenized = clip.tokenize(prompts).to(device)
    BATCH = 128

    all_feats = []
    for i in tqdm(range(0, len(tokenized), BATCH), desc="Encoding text"):
        batch = tokenized[i : i + BATCH]
        emb = model.encode_text(batch).float()
        emb = F.normalize(emb, dim=-1)
        all_feats.append(emb.cpu())

    all_feats = torch.cat(all_feats, dim=0)

    T = len(templates)
    C = num_classes
    D = all_feats.size(-1)

    text_features = all_feats.view(C, T, D).mean(dim=1)
    text_features = F.normalize(text_features, dim=-1).to(device)

    # ----------------------------
    # Evaluate
    # ----------------------------
    print("Running zero-shot inference...")

    correct = 0
    total = 0

    for imgs, labels in tqdm(val_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        img_feat = model.encode_image(imgs).float()
        img_feat = F.normalize(img_feat, dim=-1)

        sims = img_feat @ text_features.T
        preds = sims.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += len(labels)

    acc = correct / total
    print(f"\nZero-shot CLIP ViT-B/32 Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    run_zero_shot()

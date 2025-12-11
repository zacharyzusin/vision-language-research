import torch
import torch.nn.functional as F
from torchvision.datasets import INaturalist
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import os
import json
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# CLIP normalization stats
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Basic CLIP-style preprocessing
def preprocess():
    def convert_rgb(img):
        return img.convert("RGB") if isinstance(img, Image.Image) else img

    return transforms.Compose([
        convert_rgb,
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

# Load iNaturalist metadata for zeroshot text prompts
def load_inat_metadata(cat_path):
    with open(cat_path, "r") as f:
        categories = json.load(f)
    categories.sort(key=lambda x: x["id"])  # ensure consistent order
    return categories

# Build zeroshot prompt embeddings
@torch.no_grad()
def build_zeroshot_classifier(model, categories, device, templates=None):
    if templates is None:
        templates = ["a photo of a {}."]

    texts = []
    for cat in categories:
        name = cat["name"].replace("_", " ")
        texts += [template.format(name) for template in templates]

    tokenized = clip.tokenize(texts).to(device)
    text_embeddings = model.encode_text(tokenized).float()
    text_embeddings = text_embeddings.view(len(categories), len(templates), -1)
    text_embeddings = text_embeddings.mean(dim=1)
    return F.normalize(text_embeddings, dim=-1).float()

@torch.no_grad()
def evaluate(model, dataloader, zeroshot_weights, device):
    model.eval()
    total = 0
    correct = 0
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        image_features = model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)
        logits = 100.0 * image_features @ zeroshot_weights.T
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to iNat2018 root")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, preprocess_clip = clip.load(args.clip_model, device=device)
    model.eval()

    # Dataset
    val_ds = INaturalist(
        root=args.data_root,
        version="2018",
        target_type="full",
        transform=preprocess(),
        download=False
    )

    # Restrict to validation split
    with open(os.path.join(args.data_root, "val2018.json"), "r") as f:
        val_json = json.load(f)
    val_paths = set(img["file_name"].split("/", 1)[-1] for img in val_json["images"])
    val_indices = []
    for idx, (cat_id, fname) in enumerate(val_ds.index):
        rel = os.path.join(val_ds.all_categories[cat_id], fname)
        if rel in val_paths:
            val_indices.append(idx)

    class ValSubset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices):
            self.base = base_ds
            self.indices = indices
        def __getitem__(self, idx):
            return self.base[self.indices[idx]]
        def __len__(self):
            return len(self.indices)

    val_subset = ValSubset(val_ds, val_indices)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build zeroshot classifier
    categories = load_inat_metadata(os.path.join(args.data_root, "categories.json"))
    zeroshot_weights = build_zeroshot_classifier(model, categories, device)

    # Evaluate
    acc = evaluate(model, val_loader, zeroshot_weights, device)
    print(f"Zero-shot CLIP Top-1 Accuracy: {acc * 100:.2f}%")

import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def unnormalize_clip(tensor):
    mean = torch.tensor(CLIP_MEAN).view(-1, 1, 1)
    std = torch.tensor(CLIP_STD).view(-1, 1, 1)
    return (tensor * std) + mean

def get_class_index(metadata, species_name):
    species_name = species_name.lower()
    for i, meta in enumerate(metadata):
        if meta["species"] == species_name:
            return i
    raise ValueError(f"Species '{species_name}' not found.")

@torch.no_grad()
def extract_topn_per_prompt(model, dataloader, class_index, device, top_n=8):
    model.eval()
    K = model.K
    subprompt_scores = [[] for _ in range(K)]
    subprompt_images = [[] for _ in range(K)]

    for images, labels in tqdm(dataloader, desc="Gathering γ"):
        images = images.to(device)

        img_feat = model.clip.encode_image(images).float()
        img_feat = F.normalize(img_feat, dim=-1)

        labels_c = torch.full((images.size(0),), class_index, dtype=torch.long, device=device)
        prompt_feats = model._batch_prompt_features(labels_c, device)

        sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats) * model.sim_scale
        gamma = F.softmax(sims / model.em_tau, dim=1)

        for i in range(images.size(0)):
            for k in range(K):
                score = gamma[i, k].item()
                img = images[i].cpu()
                subprompt_scores[k].append(score)
                subprompt_images[k].append(img)

    topn = []
    for k in range(K):
        pairs = list(zip(subprompt_scores[k], subprompt_images[k]))
        pairs.sort(reverse=True, key=lambda x: x[0])
        topn.append(pairs[:top_n])
    return topn

def draw_gamma_grid(topn_data, species_name, output_dir="viz_outputs", image_size=224, nrow=8):
    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.load_default()
    padding = 2
    rows = []

    for k, row_data in enumerate(topn_data):
        row_images = []
        for gamma, img_tensor in row_data:
            img = unnormalize_clip(img_tensor).clamp(0, 1)
            img = to_pil_image(img)
            draw = ImageDraw.Draw(img)
            label = f"γ={gamma:.2f}"
            draw.rectangle([0, 0, 60, 12], fill="black")
            draw.text((2, 0), label, font=font, fill="white")
            row_images.append(img)
        if len(row_images) < nrow:
            blanks = [Image.new("RGB", (image_size, image_size), (255, 255, 255))] * (nrow - len(row_images))
            row_images.extend(blanks)
        row_concat = Image.new("RGB", (nrow * image_size, image_size))
        for i, img in enumerate(row_images):
            row_concat.paste(img, (i * image_size, 0))
        draw = ImageDraw.Draw(row_concat)
        draw.text((5, 5), f"Prompt {k}", font=font, fill="yellow")
        rows.append(row_concat)

    full_height = len(rows) * image_size
    full_image = Image.new("RGB", (nrow * image_size, full_height), (255, 255, 255))
    for i, row in enumerate(rows):
        full_image.paste(row, (0, i * image_size))

    save_path = os.path.join(output_dir, f"{species_name.replace(' ', '_')}_subprompt_grid_gamma.png")
    full_image.save(save_path)
    print(f"Saved grid with γ labels to {save_path}")

# === Main Entry ===
if __name__ == "__main__":
    from src.datasets.inat_dataset import get_inat2018, extract_hierarchical_metadata
    from src.models.mop_clip import MixturePromptCLIP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "data/iNat2018"
    model_path = "checkpoints/best_epoch15.pt"
    species = "danaus plexippus"  # change as needed
    top_n = 8
    batch_size = 16

    metadata = extract_hierarchical_metadata(root)
    class_index = get_class_index(metadata, species)

    dataset = get_inat2018(root, split="train", only_class_id=class_index)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MixturePromptCLIP(clip_model="ViT-B/16", metadata=metadata, K=32)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    topn = extract_topn_per_prompt(model, dataloader, class_index, device, top_n=top_n)
    draw_gamma_grid(topn, species_name=species)

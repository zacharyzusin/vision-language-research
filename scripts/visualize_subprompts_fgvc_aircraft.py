"""
Visualization script for Mixture-of-Prompts CLIP model - FGVC Aircraft dataset.

This script visualizes which images are assigned to each sub-prompt for a given class.
It generates grid images showing the top-N images per sub-prompt based on gamma scores.
"""

import os
import argparse
import math
import sys

# Add repository root to PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from train import load_config
from src.datasets.fgvc_aircraft_dataset import get_fgvc_aircraft, extract_fgvc_aircraft_metadata
from src.models.mop_clip import MixturePromptCLIP


def find_class_index_by_name(metadata, name: str):
    """
    Find class index whose 'full_name' or 'variant' field matches the given name (case-insensitive).
    """
    name = name.lower()
    matches = []
    for idx, md in enumerate(metadata):
        full_name = md.get("full_name", md.get("variant", "")).lower()
        if name in full_name or full_name == name:
            matches.append(idx)
    if not matches:
        raise ValueError(f"No class named '{name}' found in metadata.")
    if len(matches) > 1:
        print(f"Warning: multiple matches for '{name}', using first index {matches[0]}")
    return matches[0]


def add_gamma_text(img_tensor, gamma_value, font_size=20):
    """
    Add gamma value text overlay to image tensor.
    
    Args:
        img_tensor: torch.Tensor of shape [3, H, W] with values in [0, 1]
        gamma_value: float, the gamma score to display
        font_size: int, font size for text
    
    Returns:
        torch.Tensor: modified image with text overlay
    """
    # Convert tensor to PIL Image
    # img_tensor is [3, H, W] in [0, 1]
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    
    # Create drawing context
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Format gamma value with more precision for small values
    # Special case: if gamma_value < 0, display "EMPTY" (for blank placeholder images)
    if gamma_value < 0:
        text = "EMPTY"
    elif gamma_value < 0.001:
        text = f"{gamma_value:.4f}"
    else:
        text = f"{gamma_value:.3f}"
    
    # Draw semi-transparent background rectangle for text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position in top-left corner with padding
    padding = 5
    x, y = padding, padding
    
    # Draw semi-transparent rectangle
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [x - 2, y - 2, x + text_width + 2, y + text_height + 2],
        fill=(0, 0, 0, 180)  # Semi-transparent black
    )
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    
    # Draw white text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    # Convert back to tensor
    img_np = np.array(pil_img).astype(np.float32) / 255.0  # [H, W, 3] in [0, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]
    
    return img_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fgvc_aircraft.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Target class index (0-based).")
    parser.add_argument("--class_name", type=str, default=None,
                        help="Alternatively, aircraft variant name to search for.")
    parser.add_argument("--top_n", type=int, default=8,
                        help="Top-N images per sub-prompt to visualize.")
    parser.add_argument("--out_dir", type=str, default="subprompt_viz_fgvc_aircraft",
                        help="Where to save visualization grids.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for soft assignments (higher = softer). Default: use em_tau_start from config (1.0).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------
    # Load config & metadata
    # ---------------------------
    config = load_config(args.config)
    root = config["dataset"]["root"]
    dataset_type = config["dataset"].get("type", "fgvc_aircraft")

    if dataset_type != "fgvc_aircraft":
        raise ValueError(f"Expected dataset type 'fgvc_aircraft', got '{dataset_type}'")

    metadata = extract_fgvc_aircraft_metadata(root)
    num_classes = len(metadata)
    print(f"Num classes: {num_classes}")

    # Determine target class
    if args.class_idx is not None:
        class_idx = int(args.class_idx)
        if not (0 <= class_idx < num_classes):
            raise ValueError(f"class_idx {class_idx} out of range [0, {num_classes})")
        cls_name = metadata[class_idx].get("full_name", metadata[class_idx].get("variant", f"class_{class_idx}"))
    elif args.class_name is not None:
        class_idx = find_class_index_by_name(metadata, args.class_name)
        cls_name = metadata[class_idx].get("full_name", metadata[class_idx].get("variant", f"class_{class_idx}"))
    else:
        raise ValueError("You must specify either --class_idx or --class_name.")

    print(f"Target class index: {class_idx} (variant='{cls_name}')")

    # ---------------------------
    # Load dataset split(s)
    # ---------------------------
    # Combine both train and val splits to visualize all available images
    from torch.utils.data import ConcatDataset
    
    train_dataset = get_fgvc_aircraft(root, "train", use_torchvision=False)
    val_dataset = get_fgvc_aircraft(root, "val", use_torchvision=False)
    dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"Loaded TRAIN split: {len(train_dataset)} samples")
    print(f"Loaded VAL split: {len(val_dataset)} samples")
    print(f"Combined dataset: {len(dataset)} total samples")

    # Make a "raw" dataset (no CLIP normalization, just resized tensor) for visualization
    raw_transform = T.Compose([
        lambda img: img.convert("RGB") if hasattr(img, 'convert') else img,
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),  # [0,1]
    ])
    
    # Create raw datasets for visualization (combined)
    raw_train_dataset = get_fgvc_aircraft(root, "train", transform=raw_transform, use_torchvision=False)
    raw_val_dataset = get_fgvc_aircraft(root, "val", transform=raw_transform, use_torchvision=False)
    raw_dataset = ConcatDataset([raw_train_dataset, raw_val_dataset])

    # ---------------------------
    # Load model & checkpoint
    # ---------------------------
    # First, load checkpoint to determine K value (checkpoint may have different K than config)
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Handle checkpoint structure
    if "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt
    
    # Remove 'module.' prefix if present (from DDP)
    if any(k.startswith('module.') for k in model_state.keys()):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items() if k.startswith('module.')}
    
    # Determine K from checkpoint shape (more reliable than config)
    if "prompt_offsets" in model_state:
        # Shape is [num_classes, K, embedding_dim]
        K_from_ckpt = model_state["prompt_offsets"].shape[1]
        print(f"Detected K={K_from_ckpt} from checkpoint (config has K={config['model']['K']})")
    else:
        # Fall back to config
        K_from_ckpt = int(config["model"]["K"])
        print(f"Using K={K_from_ckpt} from config")
    
    # Initialize model with correct K
    model = MixturePromptCLIP(
        clip_model=config["model"]["clip_model"],
        metadata=metadata,
        K=K_from_ckpt,
        em_tau=float(config["model"].get("em_tau", 1.0)),
    ).to(device)
    model.clip.to(device)
    
    model.load_state_dict(model_state, strict=False)
    model.eval()

    K = model.K
    print(f"Model has K = {K} sub-prompts per class.")
    
    # Determine visualization temperature
    # Training uses em_tau_end (0.05) which causes hard assignments
    # For visualization, use a higher temperature to see soft assignments
    if args.temperature is not None:
        viz_temperature = float(args.temperature)
        print(f"Using user-specified visualization temperature: {viz_temperature}")
    else:
        # Default to 5.0 for better visualization of soft assignments
        # Higher temperature shows the underlying soft assignments more clearly
        viz_temperature = 5.0
        em_tau_start = float(config["model"].get("em_tau_start", 1.0))
        em_tau_end = float(config["model"].get("em_tau_end", 0.05))
        print(f"Using default visualization temperature: {viz_temperature}")
        print(f"  (Config has em_tau_start={em_tau_start}, em_tau_end={em_tau_end})")
    
    training_tau_end = float(config["model"].get("em_tau_end", 0.05))
    print(f"Note: Model was trained with temperature annealing down to em_tau_end={training_tau_end}")
    print(f"      Using temperature={viz_temperature} for visualization to reveal soft assignments")

    # ---------------------------
    # Collect top-N images per sub-prompt
    # ---------------------------
    # For each sub-prompt k, show the top-N images ranked by gamma[k] (assignment to that sub-prompt)
    # This means the same image can appear in multiple rows if it has high gamma for multiple sub-prompts
    # This is the correct way to visualize: "which images are best matched to each sub-prompt?"
    all_images_per_k = {k: [] for k in range(K)}  # Store all (gamma[k], dataset_idx) for each k

    print(f"Scanning dataset for target class samples (class {class_idx})...")
    class_count = 0
    for dataset_idx in range(len(dataset)):
        img, label = dataset[dataset_idx]
        if int(label) != class_idx:
            continue
        
        class_count += 1
        img_batch = img.unsqueeze(0).to(device)
        label_batch = torch.tensor([class_idx], device=device)

        # Compute gamma (soft assignment) using model's internal methods
        # Use visualization temperature instead of model.em_tau for softer assignments
        with torch.no_grad():
            img_feat = model.clip.encode_image(img_batch).float()
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            prompt_feats = model._batch_prompt_features(label_batch, device)  # (1, K, D)
            sims = torch.einsum("bd,bkd->bk", img_feat, prompt_feats) * model.sim_scale
            gamma = torch.nn.functional.softmax(sims / viz_temperature, dim=1)
            gamma = gamma[0].cpu()  # (K,)

        # For each sub-prompt k, store this image with its gamma[k] value
        # This allows the same image to appear in multiple rows if it has high gamma for multiple k
        for k in range(K):
            gamma_k = gamma[k].item()
            all_images_per_k[k].append((gamma_k, dataset_idx))

    print(f"Found {class_count} samples for class {class_idx} ({cls_name})")

    # Sort & keep top-N per sub-prompt (ranked by gamma[k] for that specific k)
    per_k = {}
    for k in range(K):
        all_images_per_k[k].sort(key=lambda x: x[0], reverse=True)  # Sort by gamma[k], descending
        per_k[k] = all_images_per_k[k][: args.top_n]
        if per_k[k]:
            print(f"Sub-prompt k={k}: {len(per_k[k])} images (top gamma[k]={per_k[k][0][0]:.4f})")
        else:
            print(f"Sub-prompt k={k}: 0 images (no samples for this class)")

    # ---------------------------
    # Build and save single combined grid
    # ---------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect all images per sub-prompt (top-N per k)
    # IMPORTANT: Always show ALL K rows, even if empty (so user sees full picture)
    all_row_imgs = []
    all_row_scores = []
    
    for k in range(K):
        entries = per_k[k][:args.top_n]  # Top-N per sub-prompt (top 8)

        imgs = []
        scores = []
        for score, dataset_idx in entries:
            img_raw, _ = raw_dataset[dataset_idx]  # tensor [3,H,W]
            # Add gamma value text overlay
            img_with_text = add_gamma_text(img_raw, score)
            imgs.append(img_with_text)
            scores.append(score)
        
        all_row_imgs.append((k, imgs))
        all_row_scores.append((k, scores))
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            avg_score = sum(scores) / len(scores)
            print(f"Sub-prompt k={k}: {len(imgs)} images, top={max_score:.4f}, avg={avg_score:.4f}, min={min_score:.4f}")
        else:
            print(f"Sub-prompt k={k}: 0 images (empty row)")
    
    if not all_row_imgs:
        print("Error: No images to visualize!")
        return

    # Use exactly top_n images per row (8 by default)
    max_images = args.top_n
    
    # Combine all rows: each row is one sub-prompt k
    # Flatten into a single list: [k0_img0, k0_img1, ..., k0_img7, k1_img0, k1_img1, ...]
    combined_imgs = []
    for k, imgs in all_row_imgs:
        # Add exactly top_n images for this sub-prompt
        combined_imgs.extend(imgs[:max_images])
        
        # Pad row if needed (with blank images) for consistent grid
        # Use gray background with "EMPTY" label so empty rows are visible
        num_missing = max_images - len(imgs)
        for _ in range(num_missing):
            # Create gray background instead of black (so it's visible)
            blank = torch.ones(3, 224, 224) * 0.3  # Gray (30% white)
            # Add "EMPTY" text to make it clear this is a blank placeholder
            blank = add_gamma_text(blank, -1.0, font_size=30)  # Use -1 as sentinel for "EMPTY"
            combined_imgs.append(blank)
    
    # Create grid: nrow determines how many images per row
    # We want: K rows (one per sub-prompt), each with exactly top_n (8) images
    # So we set nrow = max_images (images per row)
    grid = vutils.make_grid(
        combined_imgs, 
        nrow=max_images,  # Number of images per row (8 images per row)
        padding=2, 
        normalize=False
    )

    # Sanitize class name for filename
    safe_cls_name = cls_name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_')
    out_path = os.path.join(
        args.out_dir,
        f"class{class_idx}_{safe_cls_name}_all_k.png",
    )
    vutils.save_image(grid, out_path)
    
    print(f"\n✅ Saved combined visualization to {out_path}")
    print(f"   Layout: {K} rows (one per sub-prompt k=0..{K-1}), {max_images} images per row")
    print(f"   Each image shows gamma value in top-left corner")
    print(f"   Total images: {len([img for _, imgs in all_row_imgs for img in imgs])} (excluding padding/blanks)")
    
    # Print summary of scores
    print(f"\n   Top scores per sub-prompt:")
    for k, scores in all_row_scores:
        if scores:
            print(f"     k={k}: {scores[0]:.3f}")


if __name__ == "__main__":
    main()


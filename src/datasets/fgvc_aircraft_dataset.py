"""
Dataset loader for FGVC Aircraft dataset.

The FGVC Aircraft dataset contains 102 classes of aircraft variants with approximately
10,000 images. Each class represents a specific aircraft model variant.

Dataset structure (torchvision format):
    fgvc-aircraft-2013b/
        data/
            images/
                *.jpg files
        data/
            images_variant_train.txt
            images_variant_test.txt
            images_variant_val.txt
            variants.txt (class names)
"""

import os
import json
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's FGVCAircraft dataset
try:
    from torchvision.datasets import FGVCAircraft
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Use CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _make_transform():
    """
    CLIP-style preprocessing:
      - ensure RGB
      - Resize to 224 (short side) with bicubic interpolation
      - Center crop 224
      - Normalize with CLIP mean/std
    """
    def _safe_load(img):
        try:
            return img.convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    return T.Compose([
        _safe_load,
        T.Resize(224, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class FGVCAircraftDataset(Dataset):
    """
    Custom dataset loader for FGVC Aircraft dataset.
    
    Supports both torchvision's FGVCAircraft format and custom directory structures.
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "val"],
        transform=None,
        use_torchvision: bool = True,
    ):
        """
        Args:
            root: Root directory containing the dataset
            split: Dataset split ("train", "test", or "val")
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's FGVCAircraft loader
        """
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else _make_transform()
        self.use_torchvision = use_torchvision and TORCHVISION_AVAILABLE
        
        if self.use_torchvision:
            try:
                # Map "val" to "test" for torchvision compatibility (FGVC uses train/test splits)
                tv_split = "test" if split == "val" else split
                self.dataset = FGVCAircraft(
                    root=root,
                    split=tv_split,
                    download=False,  # Don't auto-download, we'll handle it separately
                    annotation_level="variant",  # Use variant-level (102 classes)
                    transform=None,  # We'll apply transform ourselves
                )
                self.samples = [(img_path, label) for img_path, label in zip(
                    self.dataset._image_files, 
                    self.dataset._labels
                )]
                self.classes = self.dataset.classes
                self.num_classes = len(self.classes)
            except Exception as e:
                print(f"Warning: torchvision FGVCAircraft loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
                self._load_custom()
        else:
            self._load_custom()
    
    def _load_custom(self):
        """Load dataset from custom directory structure."""
        # Expected structure:
        # root/
        #   train/
        #     class_001/
        #     class_002/
        #     ...
        #   test/ or val/
        #     class_001/
        #     class_002/
        #     ...
        
        split_dir = os.path.join(self.root, self.split)
        if not os.path.exists(split_dir):
            # Try alternative: data/images with split files
            split_dir = os.path.join(self.root, "data", "images")
            if os.path.exists(split_dir):
                self._load_from_split_files()
                return
        
        if not os.path.exists(split_dir):
            raise RuntimeError(f"Split directory not found: {split_dir}")
        
        # Load class directories
        class_dirs = sorted([d for d in os.listdir(split_dir) 
                           if os.path.isdir(os.path.join(split_dir, d))])
        
        if not class_dirs:
            raise RuntimeError(f"No class directories found in {split_dir}")
        
        # Build class mapping
        self.classes = [d.replace("class_", "").replace("_", " ").strip() 
                       for d in class_dirs]
        self.num_classes = len(self.classes)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all samples
        self.samples = []
        for class_dir in class_dirs:
            class_path = os.path.join(split_dir, class_dir)
            class_name = class_dir.replace("class_", "").replace("_", " ").strip()
            class_idx = class_to_idx[class_name]
            
            # Find all images in this class directory
            for img_file in sorted(os.listdir(class_path)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    self.samples.append((img_path, class_idx))
        
        if not self.samples:
            raise RuntimeError(f"No images found for {self.split} split")
    
    def _load_from_split_files(self):
        """Load dataset using split files (images_variant_train.txt, etc.)."""
        data_dir = os.path.join(self.root, "data")
        images_dir = os.path.join(data_dir, "images")
        
        # Load variant names (classes)
        variants_file = os.path.join(data_dir, "variants.txt")
        if os.path.exists(variants_file):
            with open(variants_file, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            # Try to infer from split files
            self.classes = []
        
        # Load split file
        split_file = os.path.join(data_dir, f"images_variant_{self.split}.txt")
        if not os.path.exists(split_file):
            raise RuntimeError(f"Split file not found: {split_file}")
        
        # Parse split file: format is "image_id variant_name"
        samples = []
        seen_variants = set()
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    image_id, variant_name = parts
                    if variant_name not in seen_variants:
                        seen_variants.add(variant_name)
                        if variant_name not in self.classes:
                            self.classes.append(variant_name)
                    
                    variant_idx = self.classes.index(variant_name)
                    img_path = os.path.join(images_dir, f"{image_id}.jpg")
                    if os.path.exists(img_path):
                        samples.append((img_path, variant_idx))
        
        if not samples:
            raise RuntimeError(f"No images found for {self.split} split")
        
        self.samples = samples
        self.num_classes = len(self.classes)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new("RGB", (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


def get_fgvc_aircraft(
    root: str,
    split: Literal["train", "test", "val"],
    transform=None,
    use_torchvision: bool = True,
) -> Dataset:
    """
    Get FGVC Aircraft dataset for the specified split.
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "test", or "val")
        transform: Optional image transform (defaults to CLIP preprocessing)
        use_torchvision: If True, try to use torchvision's FGVCAircraft loader
    
    Returns:
        Dataset instance
    """
    return FGVCAircraftDataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
    )


def extract_fgvc_aircraft_metadata(root: str) -> list:
    """
    Extract metadata for FGVC Aircraft dataset.
    
    Creates metadata with aircraft-specific fields:
    - variant: aircraft variant name (e.g., "Boeing 737-800")
    - manufacturer: aircraft manufacturer (e.g., "Boeing")
    - model: aircraft model (e.g., "737")
    - variant_name: specific variant (e.g., "737-800")
    - full_name: complete aircraft name
    
    Also includes legacy fields for compatibility:
    - species, genus, family, order, scientific_name
    
    Args:
        root: Root directory containing the dataset
    
    Returns:
        List of metadata dictionaries (one for each class)
    """
    # Try to load the dataset to get class names
    try:
        train_ds = FGVCAircraftDataset(root=root, split="train", use_torchvision=True)
        class_names = train_ds.classes
    except:
        try:
            train_ds = FGVCAircraftDataset(root=root, split="train", use_torchvision=False)
            class_names = train_ds.classes
        except:
            # Try to load from variants.txt
            try:
                variants_file = os.path.join(root, "data", "variants.txt")
                if os.path.exists(variants_file):
                    with open(variants_file, 'r') as f:
                        class_names = [line.strip() for line in f if line.strip()]
                else:
                    # Last resort: use generic names
                    class_names = [f"aircraft variant {i+1}" for i in range(102)]
            except:
                class_names = [f"aircraft variant {i+1}" for i in range(102)]
    
    # Map of model codes to manufacturers (common aircraft)
    manufacturer_map = {
        "707": "Boeing", "727": "Boeing", "737": "Boeing", "747": "Boeing",
        "757": "Boeing", "767": "Boeing", "777": "Boeing", "787": "Boeing",
        "a300": "Airbus", "a310": "Airbus", "a318": "Airbus", "a319": "Airbus",
        "a320": "Airbus", "a321": "Airbus", "a330": "Airbus", "a340": "Airbus",
        "a350": "Airbus", "a380": "Airbus",
        "md": "McDonnell Douglas", "dc": "Douglas",
        "emb": "Embraer", "crj": "Bombardier", "atr": "ATR",
        "an": "Antonov", "il": "Ilyushin", "tu": "Tupolev",
    }
    
    metadata = []
    for class_name in class_names:
        # Parse aircraft variant name
        # Format is typically model-variant codes like "707-320", "737-800", "A320-200"
        variant_name = class_name.strip()
        
        # Extract model number (first part before dash or first word)
        if '-' in variant_name:
            model_code = variant_name.split('-')[0].lower()
            variant_code = variant_name.split('-', 1)[1]
            model = variant_name.split('-')[0]  # Keep original case
            variant = "-" + variant_code
        else:
            model_code = variant_name.lower().split()[0] if variant_name.split() else variant_name.lower()
            model = variant_name.split()[0] if variant_name.split() else variant_name
            variant = ""
        
        # Find manufacturer from model code
        manufacturer = "Unknown"
        
        # Check for multi-word names (e.g., "Cessna 560", "McDonnell Douglas MD-90")
        variant_name_lower = variant_name.lower()
        if variant_name_lower.startswith("cessna"):
            manufacturer = "Cessna"
            # Update model to be the part after "Cessna"
            if len(variant_name.split()) > 1:
                model = " ".join(variant_name.split()[1:])
                variant = ""
        elif variant_name_lower.startswith("mcdonnell") or variant_name_lower.startswith("md-"):
            manufacturer = "McDonnell Douglas"
            if variant_name_lower.startswith("md-"):
                model = "MD" + variant_name.split('-')[0].replace("MD", "").strip()
        else:
            # Check manufacturer map
            for code, manu in manufacturer_map.items():
                if model_code.startswith(code):
                    manufacturer = manu
                    break
            
            # If no match, try to infer from first letter
            if manufacturer == "Unknown" and variant_name:
                first_char = variant_name[0].upper()
                if first_char == 'A' and variant_name[0:2].upper() not in ['A3', 'A4', 'A5', 'A7', 'A8']:
                    manufacturer = "Airbus"  # Likely Airbus if starts with A
                elif first_char in ['7', '8']:
                    manufacturer = "Boeing"  # Likely Boeing for 7xx series
        
        # Infer aircraft category
        variant_lower = variant_name.lower()
        model_code_lower = model_code.lower()
        
        if any(code in model_code_lower for code in ["737", "747", "777", "787", "757", "767", "707", "727"]):
            category = "commercial_jet"
            aircraft_type = "airliner"
        elif any(code in model_code_lower for code in ["a300", "a310", "a318", "a319", "a320", "a321", "a330", "a340", "a350", "a380"]):
            category = "commercial_jet"
            aircraft_type = "airliner"
        elif model_code_lower.startswith("f-") or "fighter" in variant_lower:
            category = "military"
            aircraft_type = "fighter"
        elif model_code_lower.startswith("c-") or "cargo" in variant_lower or "transport" in variant_lower:
            category = "military"
            aircraft_type = "transport"
        elif model_code_lower.startswith("b-") or "bomber" in variant_lower:
            category = "military"
            aircraft_type = "bomber"
        elif "helicopter" in variant_lower or "rotorcraft" in variant_lower:
            category = "rotorcraft"
            aircraft_type = "helicopter"
        else:
            category = "commercial_jet"  # Most variants are commercial
            aircraft_type = "airliner"
        
        # Create metadata with aircraft-specific fields
        metadata.append({
            # Aircraft-specific fields (primary)
            "variant": variant_name,
            "manufacturer": manufacturer,
            "model": model,
            "variant_name": variant,
            "full_name": variant_name,
            "category": category,
            "type": aircraft_type,
            # Legacy fields for backward compatibility
            "species": model.lower() + variant.lower() if variant else model.lower(),
            "genus": manufacturer.lower(),
            "family": aircraft_type.lower(),
            "order": category.lower(),
            "scientific_name": variant_name.lower(),
        })
    
    return metadata


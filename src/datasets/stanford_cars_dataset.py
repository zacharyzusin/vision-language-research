"""
Dataset loader for Stanford Cars dataset.

The Stanford Cars dataset contains 196 classes of cars with 8,144 training images
and 8,041 test images. Each class represents a specific make, model, and year.

Dataset structure (torchvision format):
    stanford_cars/
        car_ims/          # All images
        cars_annos.mat     # Annotations file
        cars_meta.mat      # Metadata file

Or custom format:
    stanford_cars/
        train/
            class_001/     # Images for class 1
            class_002/     # Images for class 2
            ...
        test/
            class_001/
            class_002/
            ...
"""

import os
import json
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's StanfordCars dataset
try:
    from torchvision.datasets import StanfordCars
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


class StanfordCarsDataset(Dataset):
    """
    Custom dataset loader for Stanford Cars dataset.
    
    Supports both torchvision's StanfordCars format and custom directory structures.
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
            split: Dataset split ("train", "test", or "val" - val maps to test)
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's StanfordCars loader
        """
        self.root = root
        self.transform = transform if transform is not None else _make_transform()
        self.use_torchvision = use_torchvision and TORCHVISION_AVAILABLE
        
        # For Stanford Cars, the test set doesn't have labels (competition format)
        # So we'll use a split of the training set for validation
        # Map "val" to use a portion of training data, "test" uses the original test set
        if split == "val":
            # Use training data split for validation (since test has no labels)
            self.split = "train"
            self.use_train_split_for_val = True
        else:
            self.split = split
            self.use_train_split_for_val = False
        
        # Try torchvision loader first
        if self.use_torchvision:
            try:
                # torchvision uses "train" and "test" splits
                # Note: torchvision's StanfordCars download is broken, so we skip automatic download
                tv_split = "train" if self.split == "train" else "test"
                self.base_dataset = StanfordCars(
                    root=root,
                    split=tv_split,
                    transform=None,  # We'll apply transform ourselves
                    download=False,  # Disabled - use manual download script instead
                )
                self.samples = [(idx, label) for idx, (_, label) in enumerate(self.base_dataset)]
                self.num_classes = len(self.base_dataset.classes)
                self.classes = self.base_dataset.classes
                print(f"Stanford Cars ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (using torchvision)")
                return
            except Exception as e:
                print(f"Warning: torchvision StanfordCars loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader: try to find class directories
        split_dir = os.path.join(root, self.split)
        if not os.path.exists(split_dir):
            # Try alternative locations
            if os.path.exists(os.path.join(root, "car_ims")):
                # torchvision format: all images in car_ims/, need annotations
                self._load_from_torchvision_format(root, self.split)
                return
            else:
                raise FileNotFoundError(
                    f"Could not find {self.split} split directory: {split_dir}\n"
                    f"Expected structure: {root}/{self.split}/ or {root}/car_ims/"
                )
        
        # Load from directory structure: split/class_XXX/images
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        class_dirs.sort()
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {split_dir}")
        
        # Build class mapping
        self.class_to_idx = {}
        self.idx_to_class = {}
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir] = idx
            self.idx_to_class[idx] = class_dir
        
        self.num_classes = len(class_dirs)
        self.classes = [self.idx_to_class[i] for i in range(self.num_classes)]
        
        # Build samples list
        self.samples = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir]
            class_path = os.path.join(split_dir, class_dir)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, filename)
                    self.samples.append((img_path, label))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}")
        
        print(f"Stanford Cars ({self.split}): {len(self.samples)} samples, {self.num_classes} classes")
    
    def _load_from_torchvision_format(self, root: str, split: str = None):
        """
        Load dataset from torchvision format (car_ims/ directory with annotations).
        This requires scipy to read .mat files.
        """
        if split is None:
            split = self.split
        
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError(
                "scipy is required to load Stanford Cars from .mat annotation files. "
                "Install with: pip install scipy"
            )
        
        # Load annotations
        annos_path = os.path.join(root, "cars_annos.mat")
        meta_path = os.path.join(root, "cars_meta.mat")
        
        if not os.path.exists(annos_path):
            raise FileNotFoundError(f"Could not find annotations file: {annos_path}")
        
        annos = sio.loadmat(annos_path, squeeze_me=True)
        annotations = annos['annotations']
        
        # Load metadata for class names
        if os.path.exists(meta_path):
            meta = sio.loadmat(meta_path, squeeze_me=True)
            class_names = [str(name) for name in meta['class_names']]
        else:
            # Fallback: use class indices
            class_names = [f"class_{i:03d}" for i in range(196)]
        
        # Filter by split
        all_train_samples = []
        all_test_samples = []
        
        # Determine image directory structure
        # Try standard torchvision format first (all images in car_ims/)
        image_dir = os.path.join(root, "car_ims")
        cars_train_dir = os.path.join(root, "cars_train")
        cars_test_dir = os.path.join(root, "cars_test")
        
        # Check if images are in separate train/test directories
        use_separate_dirs = os.path.exists(cars_train_dir) and os.path.exists(cars_test_dir)
        
        # annotations is a structured array or list of dicts
        if isinstance(annotations, dict):
            # Handle different annotation formats
            if 'annotations' in annotations:
                annotations = annotations['annotations']
        
        # Check if annotations have a 'test' field
        has_test_field = False
        if len(annotations) > 0:
            first_ann = annotations[0]
            if hasattr(first_ann, 'dtype') and first_ann.dtype.names:
                has_test_field = 'test' in first_ann.dtype.names
            elif isinstance(first_ann, dict):
                has_test_field = 'test' in first_ann or 'is_test' in first_ann
            elif hasattr(first_ann, 'test'):
                has_test_field = True
        
        # Process annotations - collect train and test separately
        for ann in annotations:
            # Handle different annotation formats
            fname = None
            class_idx = None
            test = None
            
            # Try structured numpy array format (from scipy .mat file)
            if hasattr(ann, 'dtype') and ann.dtype.names:
                # Structured array with named fields
                fname = str(ann['fname']) if 'fname' in ann.dtype.names else None
                class_idx = int(ann['class']) if 'class' in ann.dtype.names else None
                if 'test' in ann.dtype.names:
                    test = bool(ann['test'])
                else:
                    # If no test field, assume all are training images
                    test = False
            # Try dict format
            elif isinstance(ann, dict):
                fname = ann.get('fname', ann.get('relative_im_path', ''))
                class_idx = ann.get('class', ann.get('class_id', 0))
                test = ann.get('test', ann.get('is_test', False))
            # Try object with attributes
            elif hasattr(ann, 'fname'):
                fname = ann.fname
                class_idx = getattr(ann, 'class', 1)
                test = getattr(ann, 'test', False)
            # Try tuple/array format - scipy saves as tuple-like structured array
            elif hasattr(ann, '__len__'):
                try:
                    ann_len = len(ann)
                    # Format from our organize script: (fname, class, test, x1, y1, x2, y2)
                    # Standard format: (x1, y1, x2, y2, class, fname) - no test field
                    if ann_len >= 6:
                        # Standard Stanford Cars format: (x1, y1, x2, y2, class, fname)
                        fname = str(ann[5])
                        class_idx = int(ann[4])
                        test = False  # No test field in standard format
                    elif ann_len >= 3:
                        # Custom format with test field: (fname, class, test, ...)
                        fname = str(ann[0])
                        class_idx = int(ann[1])
                        test = bool(ann[2])
                    elif ann_len >= 2:
                        fname = str(ann[0])
                        class_idx = int(ann[1])
                        test = False
                    else:
                        continue
                except (TypeError, ValueError, IndexError):
                    continue
            
            if fname is None or class_idx is None:
                continue
            
            # Default to False (training) if test is still None
            if test is None:
                test = False
            
            # Convert class from 1-indexed to 0-indexed (if needed)
            # Our combined format already uses 0-indexed, but check if it's 1-indexed
            if class_idx > 0:
                class_idx = class_idx - 1
            
            # Try to find the image in multiple possible locations
            img_path = None
            if use_separate_dirs:
                # Try separate train/test directories first
                if test:
                    img_path = os.path.join(cars_test_dir, fname)
                else:
                    img_path = os.path.join(cars_train_dir, fname)
            
            # If not found in separate dirs or separate dirs don't exist, try car_ims/
            if img_path is None or not os.path.exists(img_path):
                img_path = os.path.join(image_dir, fname)
            
            if not os.path.exists(img_path):
                continue
            
            # Collect samples by split
            is_test = bool(test)
            if is_test:
                all_test_samples.append((img_path, class_idx))
            else:
                all_train_samples.append((img_path, class_idx))
        
        # Handle train/val split for validation
        # For Stanford Cars, we split the training data 80/20 for train/val
        # since the test set doesn't have labels
        if self.split == "train" or self.use_train_split_for_val:
            import random
            random.seed(42)  # For reproducibility - same seed ensures consistent split
            random.shuffle(all_train_samples)
            val_size = int(len(all_train_samples) * 0.2)
            
            if self.use_train_split_for_val:
                # Validation: use 20% of training data
                self.samples = all_train_samples[:val_size]
            else:
                # Training: use 80% of training data
                self.samples = all_train_samples[val_size:]
        elif self.split == "test":
            # Test set (no labels, but keep for compatibility)
            self.samples = all_test_samples
        else:
            self.samples = []
        
        # Build class mapping
        self.num_classes = len(class_names)
        self.classes = class_names
        self.idx_to_class = {i: name for i, name in enumerate(class_names)}
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for {split} split")
        
        # Display correct split name
        display_split = "val" if self.use_train_split_for_val else self.split
        print(f"Stanford Cars ({display_split}): {len(self.samples)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_torchvision:
            # Use torchvision dataset
            base_idx, label = self.samples[idx]
            img, _ = self.base_dataset[base_idx]
        else:
            # Use custom loader
            img_path, label = self.samples[idx]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                img = Image.new("RGB", (224, 224))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_stanford_cars(
    root: str,
    split: Literal["train", "test", "val"],
    transform=None,
    use_torchvision: bool = True,
):
    """
    Get Stanford Cars dataset for specified split.
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "test", or "val" - val maps to test)
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's StanfordCars loader
    
    Returns:
        StanfordCarsDataset instance
    """
    return StanfordCarsDataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
    )


def extract_stanford_cars_metadata(root: str) -> list:
    """
    Extract metadata for Stanford Cars dataset.
    
    Creates metadata with car-specific fields:
    - make: car manufacturer (e.g., "tesla")
    - model: car model name (e.g., "model s")
    - type: car body type (e.g., "sedan", "suv")
    - category: car category (e.g., "luxury", "sports")
    - full_name: complete car name (e.g., "2012 tesla model s")
    
    Also includes legacy fields for compatibility:
    - species, genus, family, order, scientific_name (mapped from car fields)
    
    Args:
        root: Root directory containing the dataset
    
    Returns:
        List of metadata dictionaries (one for each class)
    """
    # Try to load the dataset to get class names
    try:
        train_ds = StanfordCarsDataset(root=root, split="train", use_torchvision=True)
        class_names = train_ds.classes
    except:
        try:
            train_ds = StanfordCarsDataset(root=root, split="train", use_torchvision=False)
            class_names = train_ds.classes
        except:
            # Fallback: try to load from metadata file
            try:
                import scipy.io as sio
                meta_path = os.path.join(root, "cars_meta.mat")
                if os.path.exists(meta_path):
                    meta = sio.loadmat(meta_path, squeeze_me=True)
                    class_names = [str(name) for name in meta['class_names']]
                else:
                    # Last resort: use generic names
                    class_names = [f"car class {i+1}" for i in range(196)]
            except:
                class_names = [f"car class {i+1}" for i in range(196)]
    
    metadata = []
    for class_name in class_names:
        # Parse car name (format is typically "YYYY Make Model" or "Make Model")
        parts = class_name.strip().split()
        
        # Try to extract make and model
        if len(parts) >= 3 and parts[0].isdigit():
            # Format: "2012 Tesla Model S"
            year = parts[0]
            make = parts[1]
            model = " ".join(parts[2:])
            full_name = class_name
        elif len(parts) >= 2:
            # Format: "Tesla Model S"
            make = parts[0]
            model = " ".join(parts[1:])
            full_name = class_name
        else:
            # Fallback
            make = parts[0] if parts else "car"
            model = class_name
            full_name = class_name
        
        # Infer car type from model name (simple heuristics)
        model_lower = model.lower()
        full_name_lower = full_name.lower()
        
        if any(word in full_name_lower for word in ["suv", "crossover"]):
            car_type = "suv"
            category = "suv"
        elif any(word in full_name_lower for word in ["coupe", "roadster", "convertible", "cabriolet"]):
            car_type = "coupe"
            category = "sports"
        elif any(word in full_name_lower for word in ["sedan", "saloon"]):
            car_type = "sedan"
            category = "sedan"
        elif any(word in full_name_lower for word in ["wagon", "estate", "touring"]):
            car_type = "wagon"
            category = "wagon"
        elif any(word in full_name_lower for word in ["truck", "pickup"]):
            car_type = "truck"
            category = "truck"
        elif any(word in full_name_lower for word in ["van", "minivan"]):
            car_type = "van"
            category = "van"
        else:
            car_type = "sedan"  # Default
            category = "automobile"
        
        # Create metadata with car-specific fields
        metadata.append({
            # Car-specific fields (primary)
            "make": make,
            "model": model,
            "type": car_type,
            "category": category,
            "full_name": full_name,
            # Legacy fields for backward compatibility (if needed)
            "species": model.lower(),
            "genus": make.lower(),
            "family": car_type.lower(),
            "order": category.lower(),
            "scientific_name": full_name.lower(),
        })
    
    return metadata


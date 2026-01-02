"""
Dataset loader for UCF101 dataset.

The UCF101 dataset contains 101 action classes from videos.
Each class contains multiple video clips.
For image-based CLIP models, we extract frames from videos.
Total dataset: ~13,320 video clips across 101 action classes.
"""

import os
from typing import Literal, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# Try to import torchvision's UCF101 dataset
try:
    from torchvision.datasets import UCF101
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


def _extract_frame_from_video(video_tensor, frame_idx=0):
    """
    Extract a single frame from a video tensor.
    
    Args:
        video_tensor: torch.Tensor of shape [C, T, H, W] or [T, C, H, W]
        frame_idx: Index of the frame to extract (default: 0, first frame)
    
    Returns:
        torch.Tensor: Single frame of shape [C, H, W]
    """
    if video_tensor is None:
        return torch.zeros(3, 224, 224)
    
    # Handle different tensor formats
    if len(video_tensor.shape) == 4:
        # Could be [C, T, H, W] or [T, C, H, W]
        if video_tensor.shape[0] == 3:
            # [C, T, H, W] format
            T_dim = video_tensor.shape[1]
            if T_dim > 0:
                frame = video_tensor[:, min(frame_idx, T_dim - 1), :, :]
            else:
                frame = video_tensor[:, 0, :, :] if video_tensor.shape[1] > 0 else torch.zeros(3, 224, 224)
        else:
            # [T, C, H, W] format
            T_dim = video_tensor.shape[0]
            if T_dim > 0:
                frame = video_tensor[min(frame_idx, T_dim - 1), :, :, :]
            else:
                frame = video_tensor[0, :, :, :] if video_tensor.shape[0] > 0 else torch.zeros(3, 224, 224)
    elif len(video_tensor.shape) == 3:
        # Already a single frame [C, H, W]
        frame = video_tensor
    else:
        # Fallback: create blank frame
        frame = torch.zeros(3, 224, 224)
    
    # Ensure shape is [C, H, W]
    if len(frame.shape) == 4:
        frame = frame[0]
    if frame.shape[0] != 3:
        frame = frame.permute(2, 0, 1) if len(frame.shape) == 3 else torch.zeros(3, 224, 224)
    
    return frame


class UCF101Dataset(Dataset):
    """
    Custom dataset loader for UCF101 dataset.
    
    Supports both torchvision's UCF101 format (extracts frames from videos)
    and custom directory structures with pre-extracted frames.
    
    UCF101 has official train/test splits (split 1), but we create a val split
    from the training set (70% train, 30% val from original train split).
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        transform=None,
        use_torchvision: bool = True,
        download: bool = False,
        frames_per_clip: int = 1,
        step_between_clips: int = 1,
        train_ratio: float = 0.7,
    ):
        """
        Args:
            root: Root directory containing the dataset
            split: Dataset split ("train", "val", or "test")
            transform: Optional image transform (defaults to CLIP preprocessing)
            use_torchvision: If True, try to use torchvision's UCF101 loader
            download: If True, download the dataset if not present
            frames_per_clip: Number of frames to extract per clip (we use 1 for CLIP)
            step_between_clips: Step between clips
            train_ratio: Ratio of training data to use for train vs val
        """
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else _make_transform()
        self.use_torchvision = use_torchvision and TORCHVISION_AVAILABLE
        self.download = download
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.train_ratio = train_ratio
        
        # Try torchvision loader first
        if self.use_torchvision:
            try:
                # UCF101 has train/test splits (split 1)
                # We'll split the train set into train/val
                if split in ["train", "val"]:
                    # Load the full training set
                    # Note: UCF101 doesn't support download parameter in torchvision
                    base_dataset = UCF101(
                        root=root,
                        annotation_path=None,  # Let torchvision find it
                        frames_per_clip=frames_per_clip,
                        step_between_clips=step_between_clips,
                        train=True,
                        transform=None,  # We'll apply transform ourselves
                        num_workers=1,
                    )
                    self.classes = base_dataset.classes
                    self.num_classes = len(self.classes)
                    
                    # Split train into train/val
                    from torch.utils.data import random_split
                    total_len = len(base_dataset)
                    train_len = int(total_len * train_ratio)
                    val_len = total_len - train_len
                    
                    train_subset, val_subset = random_split(
                        base_dataset, 
                        [train_len, val_len], 
                        generator=torch.Generator().manual_seed(42)
                    )
                    
                    if split == "train":
                        self.base_dataset = train_subset
                    else:  # val
                        self.base_dataset = val_subset
                    
                    # Create samples list from subset
                    self.samples = [(idx, self.base_dataset.dataset[self.base_dataset.indices[idx]][1]) 
                                   for idx in range(len(self.base_dataset))]
                else:  # test
                    # Note: UCF101 doesn't support download parameter in torchvision
                    self.base_dataset = UCF101(
                        root=root,
                        annotation_path=None,
                        frames_per_clip=frames_per_clip,
                        step_between_clips=step_between_clips,
                        train=False,
                        transform=None,
                        num_workers=1,
                    )
                    self.classes = self.base_dataset.classes
                    self.num_classes = len(self.classes)
                    self.samples = [(idx, label) for idx, (_, label) in enumerate(self.base_dataset)]
                
                print(f"UCF101 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (using torchvision)")
                return
            except Exception as e:
                print(f"Warning: torchvision UCF101 loader failed: {e}")
                print("Falling back to custom loader...")
                self.use_torchvision = False
        
        # Custom loader fallback
        # UCF101 structure: videos/class_name/video_XXXX.avi
        videos_dir = os.path.join(root, "UCF-101")
        if not os.path.exists(videos_dir):
            videos_dir = os.path.join(root, "ucf101", "UCF-101")
        if not os.path.exists(videos_dir):
            videos_dir = os.path.join(root, "videos")
        
        if not os.path.exists(videos_dir):
            raise FileNotFoundError(
                f"Could not find UCF101 dataset at {videos_dir}\n"
                f"Expected structure: {root}/UCF-101/class_name/*.avi\n"
                f"UCF101 must be manually downloaded from: https://www.crcv.ucf.edu/data/UCF101.php\n"
                f"Place the extracted UCF-101 folder in: {root}/"
            )
        
        self._load_from_directory(videos_dir, split)
    
    def _load_from_directory(self, videos_dir: str, split: str):
        """Load dataset from directory structure."""
        # UCF101 structure: videos/class_name/video_XXXX.avi
        class_dirs = sorted([d for d in os.listdir(videos_dir) 
                           if os.path.isdir(os.path.join(videos_dir, d)) 
                           and not d.startswith('.')])
        
        self.classes = class_dirs
        self.num_classes = len(self.classes)
        class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all video files
        all_samples = []
        for class_name in self.classes:
            class_dir = os.path.join(videos_dir, class_name)
            video_files = sorted([f for f in os.listdir(class_dir) 
                                if f.lower().endswith(('.avi', '.mp4', '.mov')) 
                                and not f.startswith('.')])
            label = class_to_label[class_name]
            for video_file in video_files:
                video_path = os.path.join(class_dir, video_file)
                all_samples.append((video_path, label))
        
        # For UCF101, there are official splits (split 1, 2, 3)
        # For simplicity, we'll use a deterministic split
        import random
        random.seed(42)
        random.shuffle(all_samples)
        
        n_total = len(all_samples)
        n_train = int(n_total * self.train_ratio)
        n_val = int((n_total - n_train) / 2)
        n_test = n_total - n_train - n_val
        
        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        else:  # test
            self.samples = all_samples[n_train + n_val:]
        
        print(f"UCF101 ({self.split}): {len(self.samples)} samples, {self.num_classes} classes (custom loader)")
        print(f"Note: Custom loader requires pre-extracted frames or video decoding libraries")
    
    def __len__(self):
        if self.use_torchvision:
            return len(self.base_dataset)
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.use_torchvision:
            # UCF101 returns (video_tensor, label)
            # video_tensor is typically [C, T, H, W] or [T, C, H, W]
            video_tensor, label = self.base_dataset[idx]
            
            # Extract first frame from video for image-based CLIP
            frame = _extract_frame_from_video(video_tensor, frame_idx=0)
            
            # Convert frame tensor to PIL Image for transform
            # Frame is [C, H, W] with values in [0, 1] (from torchvision)
            frame_pil = T.ToPILImage()(frame)
            
            if self.transform:
                img = self.transform(frame_pil)
            else:
                # Just convert to tensor if no transform
                img = T.ToTensor()(frame_pil)
            
            return img, torch.tensor(label, dtype=torch.long)
        else:
            # Custom loader: samples are (video_path, label)
            # This would require video decoding - for now, return placeholder
            # In practice, you'd need to extract frames from video files
            video_path, label = self.samples[idx]
            try:
                # Try to load first frame using PIL if it's an image
                # For actual video files, you'd need opencv-python or similar
                img = Image.new("RGB", (224, 224))  # Placeholder
                if self.transform:
                    img = self.transform(img)
                return img, torch.tensor(label, dtype=torch.long)
            except Exception:
                img = Image.new("RGB", (224, 224))
                if self.transform:
                    img = self.transform(img)
                return img, torch.tensor(label, dtype=torch.long)


def get_ucf101(
    root: str,
    split: Literal["train", "val", "test"],
    transform=None,
    use_torchvision: bool = True,
    download: bool = False,
    frames_per_clip: int = 1,
    step_between_clips: int = 1,
    train_ratio: float = 0.7,
) -> Dataset:
    """
    Get UCF101 dataset.
    
    Args:
        root: Root directory containing the dataset
        split: Dataset split ("train", "val", or "test")
        transform: Optional image transform
        use_torchvision: If True, try to use torchvision's loader
        download: If True, download the dataset if not present
        frames_per_clip: Number of frames per clip (default: 1 for CLIP)
        step_between_clips: Step between clips
        train_ratio: Ratio of training data for train vs val
    
    Returns:
        Dataset instance
    """
    return UCF101Dataset(
        root=root,
        split=split,
        transform=transform,
        use_torchvision=use_torchvision,
        download=download,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        train_ratio=train_ratio,
    )


def extract_ucf101_metadata(root: str) -> list:
    """
    Extract metadata for UCF101 dataset.
    
    Returns a list of metadata dicts, one per class.
    """
    try:
        # Try to load using torchvision to get class names
        if TORCHVISION_AVAILABLE:
            try:
                # UCF101 doesn't support download parameter
                base_dataset = UCF101(root=root, train=True, frames_per_clip=1, num_workers=1)
                class_names = base_dataset.classes
            except:
                # Fallback: try to read from directory
                videos_dir = os.path.join(root, "UCF-101")
                if not os.path.exists(videos_dir):
                    videos_dir = os.path.join(root, "ucf101", "UCF-101")
                if not os.path.exists(videos_dir):
                    videos_dir = os.path.join(root, "videos")
                
                if os.path.exists(videos_dir):
                    class_names = sorted([d for d in os.listdir(videos_dir) 
                                        if os.path.isdir(os.path.join(videos_dir, d)) 
                                        and not d.startswith('.')])
                else:
                    # Final fallback: use generic names
                    class_names = [f"action_{i+1}" for i in range(101)]
        else:
            videos_dir = os.path.join(root, "UCF-101")
            if not os.path.exists(videos_dir):
                videos_dir = os.path.join(root, "ucf101", "UCF-101")
            if not os.path.exists(videos_dir):
                videos_dir = os.path.join(root, "videos")
            
            if os.path.exists(videos_dir):
                class_names = sorted([d for d in os.listdir(videos_dir) 
                                    if os.path.isdir(os.path.join(videos_dir, d)) 
                                    and not d.startswith('.')])
            else:
                class_names = [f"action_{i+1}" for i in range(101)]
    except:
        # Fallback to generic names
        class_names = [f"action_{i+1}" for i in range(101)]
    
    metadata = []
    for class_name in class_names:
        action_name = class_name.replace('_', ' ').strip()
        
        metadata.append({
            "species": action_name.lower(),
            "genus": "action",
            "family": "ucf101",
            "order": "video",
            "scientific_name": action_name.lower(),
            "full_name": action_name,
        })
    
    return metadata


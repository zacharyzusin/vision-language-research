#!/usr/bin/env python3
"""
Organize Stanford Cars dataset from Kaggle format to the expected pipeline format.
"""

import os
import shutil
import scipy.io as sio
from pathlib import Path

def organize_dataset():
    """Organize Kaggle Stanford Cars dataset into pipeline format."""
    
    project_root = Path(__file__).parent.parent
    target_dir = project_root / "data" / "stanford_cars"
    
    print("Organizing Stanford Cars dataset...")
    print(f"Target directory: {target_dir}")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    car_ims_dir = target_dir / "car_ims"
    car_ims_dir.mkdir(exist_ok=True)
    
    # Step 1: Copy all images to car_ims/
    print("\n1. Copying images to car_ims/...")
    
    source_dirs = [
        project_root / "cars_train" / "cars_train",
        project_root / "cars_test" / "cars_test",
    ]
    
    total_copied = 0
    for source_dir in source_dirs:
        if source_dir.exists():
            for img_file in source_dir.glob("*.jpg"):
                dest = car_ims_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    total_copied += 1
            print(f"  Copied from {source_dir.name}: {total_copied} images")
    
    print(f"  Total images in car_ims/: {len(list(car_ims_dir.glob('*.jpg')))}")
    
    # Step 2: Combine annotation files
    print("\n2. Combining annotation files...")
    
    devkit_dir = project_root / "car_devkit" / "devkit"
    train_annos_path = devkit_dir / "cars_train_annos.mat"
    test_annos_path = devkit_dir / "cars_test_annos.mat"
    meta_path = devkit_dir / "cars_meta.mat"
    
    if train_annos_path.exists() and test_annos_path.exists():
        # Load train and test annotations
        train_annos = sio.loadmat(str(train_annos_path), squeeze_me=True)
        test_annos = sio.loadmat(str(test_annos_path), squeeze_me=True)
        
        train_data = train_annos['annotations']
        test_data = test_annos['annotations']
        
        # Combine into single annotations array with test flag
        combined_annotations = []
        
        # Process train annotations (test=False)
        # Train format: (x1, y1, x2, y2, class, filename)
        for ann in train_data:
            if isinstance(ann, tuple) and len(ann) >= 6:
                # Tuple format: (x1, y1, x2, y2, class, filename)
                combined_ann = {
                    'fname': str(ann[5]),
                    'class': int(ann[4]),
                    'bbox_x1': int(ann[0]),
                    'bbox_y1': int(ann[1]),
                    'bbox_x2': int(ann[2]),
                    'bbox_y2': int(ann[3]),
                    'test': False,
                }
                combined_annotations.append(combined_ann)
            elif hasattr(ann, '__len__') and len(ann) > 0:
                # Try to handle as array/list
                try:
                    combined_ann = {
                        'fname': str(ann[-1]),  # Last element is filename
                        'class': int(ann[-2]) if len(ann) > 1 else 0,  # Second to last is class
                        'test': False,
                    }
                    if len(ann) >= 4:
                        combined_ann['bbox_x1'] = int(ann[0])
                        combined_ann['bbox_y1'] = int(ann[1])
                        combined_ann['bbox_x2'] = int(ann[2])
                        combined_ann['bbox_y2'] = int(ann[3])
                    combined_annotations.append(combined_ann)
                except (ValueError, TypeError, IndexError):
                    # Handle dict-like annotations
                    combined_ann = {
                        'fname': ann.get('fname', ann.get('relative_im_path', '')),
                        'class': int(ann.get('class', ann.get('class_id', 0))),
                        'test': False,
                    }
                    combined_annotations.append(combined_ann)
            else:
                # Handle dict-like annotations
                combined_ann = {
                    'fname': ann.get('fname', ann.get('relative_im_path', '')),
                    'class': int(ann.get('class', ann.get('class_id', 0))),
                    'test': False,
                }
                combined_annotations.append(combined_ann)
        
        # Process test annotations (test=True)
        # Test format: (x1, y1, x2, y2, filename) - no class in test set
        for ann in test_data:
            if isinstance(ann, tuple) and len(ann) >= 5:
                # Tuple format: (x1, y1, x2, y2, filename)
                # For test, we need to infer class from filename or use 0
                # Actually, test annotations don't have class labels - we'll need to handle this
                combined_ann = {
                    'fname': str(ann[4]),
                    'class': 0,  # Test set doesn't have class labels in annotations
                    'bbox_x1': int(ann[0]),
                    'bbox_y1': int(ann[1]),
                    'bbox_x2': int(ann[2]),
                    'bbox_y2': int(ann[3]),
                    'test': True,
                }
                combined_annotations.append(combined_ann)
            elif hasattr(ann, '__len__') and len(ann) > 0:
                try:
                    combined_ann = {
                        'fname': str(ann[-1]),  # Last element is filename
                        'class': 0,  # Test set doesn't have class labels
                        'test': True,
                    }
                    if len(ann) >= 4:
                        combined_ann['bbox_x1'] = int(ann[0])
                        combined_ann['bbox_y1'] = int(ann[1])
                        combined_ann['bbox_x2'] = int(ann[2])
                        combined_ann['bbox_y2'] = int(ann[3])
                    combined_annotations.append(combined_ann)
                except (ValueError, TypeError, IndexError):
                    combined_ann = {
                        'fname': ann.get('fname', ann.get('relative_im_path', '')),
                        'class': int(ann.get('class', ann.get('class_id', 0))),
                        'test': True,
                    }
                    combined_annotations.append(combined_ann)
            else:
                combined_ann = {
                    'fname': ann.get('fname', ann.get('relative_im_path', '')),
                    'class': int(ann.get('class', ann.get('class_id', 0))),
                    'test': True,
                }
                combined_annotations.append(combined_ann)
        
        # Save combined annotations
        combined_dict = {'annotations': combined_annotations}
        sio.savemat(str(target_dir / "cars_annos.mat"), combined_dict)
        print(f"  Created cars_annos.mat with {len(combined_annotations)} annotations")
        print(f"    Train: {len(train_data)}, Test: {len(test_data)}")
    else:
        print("  Warning: Annotation files not found!")
    
    # Step 3: Copy metadata file
    print("\n3. Copying metadata file...")
    if meta_path.exists():
        shutil.copy2(meta_path, target_dir / "cars_meta.mat")
        print("  Copied cars_meta.mat")
    else:
        print("  Warning: cars_meta.mat not found!")
    
    # Step 4: Verify structure
    print("\n4. Verifying dataset structure...")
    required_files = [
        car_ims_dir,
        target_dir / "cars_annos.mat",
    ]
    
    all_good = True
    for item in required_files:
        if item.exists():
            if item.is_dir():
                img_count = len(list(item.glob("*.jpg")))
                print(f"  ✓ {item.name}/ ({img_count} images)")
            else:
                size = item.stat().st_size / 1024
                print(f"  ✓ {item.name} ({size:.1f} KB)")
        else:
            print(f"  ✗ {item.name} - MISSING")
            all_good = False
    
    if target_dir / "cars_meta.mat" in [target_dir / "cars_meta.mat"]:
        if (target_dir / "cars_meta.mat").exists():
            print(f"  ✓ cars_meta.mat (optional)")
    
    print("\n" + "="*50)
    if all_good:
        print("✓ Dataset organized successfully!")
        print(f"\nDataset location: {target_dir}")
        print("You can now run training with:")
        print("  python train.py --config configs/stanford_cars.yaml")
    else:
        print("✗ Dataset organization incomplete. Please check missing files.")
    print("="*50)
    
    return all_good

if __name__ == "__main__":
    import sys
    success = organize_dataset()
    sys.exit(0 if success else 1)


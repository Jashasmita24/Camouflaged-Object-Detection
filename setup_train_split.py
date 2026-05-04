"""
Create a train/test split from COD10K-v3 test data.
This is a workaround when full training data is not available.
"""
import os
import shutil
from pathlib import Path
import random

def setup_training_split():
    """Create Train/Test split from available data."""
    
    # Source and destination paths
    source_base = Path("data/COD10K/COD10K-v3")
    dest_base = Path("data/COD10K")
    
    if not source_base.exists():
        print(f"ERROR: {source_base} not found!")
        return False
    
    # Check what GT data we have
    test_dirs = source_base / "Test"
    if not test_dirs.exists():
        print(f"ERROR: {test_dirs} not found!")
        return False
    
    # Use GT_Object as the main ground truth
    print("Setting up training directory structure...")
    
    # Create Train directories
    train_img_dir = dest_base / "Train" / "Images"
    train_gt_dir = dest_base / "Train" / "GT"
    test_img_dir = dest_base / "Test" / "Images"
    test_gt_dir = dest_base / "Test" / "GT"
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_gt_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from source test
    source_img_dir = test_dirs / "Image"
    source_gt_dir = test_dirs / "GT_Object"
    
    if not source_img_dir.exists():
        print(f"ERROR: {source_img_dir} not found!")
        return False
    
    if not source_gt_dir.exists():
        print(f"ERROR: {source_gt_dir} not found! Using GT_Edge instead...")
        source_gt_dir = test_dirs / "GT_Edge"
    
    if not source_gt_dir.exists():
        print(f"ERROR: {source_gt_dir} still not found!")
        return False
    
    # Get all image files
    image_files = sorted([f for f in source_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    print(f"Found {len(image_files)} images")
    
    if not image_files:
        print("ERROR: No images found!")
        return False
    
    # Shuffle and split (80% train, 20% test)
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(0.8 * len(image_files))
    train_images = image_files[:split_idx]
    test_images = image_files[split_idx:]
    
    print(f"Train: {len(train_images)} images")
    print(f"Test:  {len(test_images)} images")
    
    # Setup training set
    print("\nSetting up training set...")
    for i, img_file in enumerate(train_images):
        # Copy image
        dest_img = train_img_dir / img_file.name
        if not dest_img.exists():
            shutil.copy2(img_file, dest_img)
        
        # Copy corresponding GT
        gt_name = img_file.stem + ".png"
        gt_file = source_gt_dir / gt_name
        if gt_file.exists():
            dest_gt = train_gt_dir / gt_name
            if not dest_gt.exists():
                shutil.copy2(gt_file, dest_gt)
        else:
            print(f"  WARNING: GT not found for {img_file.name}")
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(train_images)}")
    
    # Setup test set
    print("\nSetting up test set...")
    for i, img_file in enumerate(test_images):
        # Symlink or copy image
        dest_img = test_img_dir / img_file.name
        if not dest_img.exists():
            shutil.copy2(img_file, dest_img)
        
        # Copy corresponding GT
        gt_name = img_file.stem + ".png"
        gt_file = source_gt_dir / gt_name
        if gt_file.exists():
            dest_gt = test_gt_dir / gt_name
            if not dest_gt.exists():
                shutil.copy2(gt_file, dest_gt)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_images)}")
    
    print("\n✓ Setup complete!")
    print(f"  Train: {len(list(train_img_dir.glob('*')))} images")
    print(f"  Test:  {len(list(test_img_dir.glob('*')))} images")
    
    return True

if __name__ == "__main__":
    setup_training_split()

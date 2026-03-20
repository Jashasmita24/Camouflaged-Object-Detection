"""
Smart training launcher that:
1. Auto-detects the COD10K dataset folder structure (handles different layouts)
2. Updates config paths accordingly
3. Starts training immediately

Run after downloading: python start_training.py
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "data", "COD10K")


def find_split_dirs(base):
    """Recursively find Train/Test image+GT directory pairs."""
    for dirpath, dirnames, filenames in os.walk(base):
        # Look for a Train folder
        if "Train" in dirnames or "train" in dirnames:
            train_sub = os.path.join(dirpath, "Train" if "Train" in dirnames else "train")
            test_sub  = os.path.join(dirpath, "Test"  if "Test"  in dirnames else "test")
            return dirpath, train_sub, test_sub
    return None, None, None


def find_images_and_gt(split_dir):
    """Find Images/ and GT/ (or equivalent) inside a split directory."""
    img_dir = gt_dir = None
    if not os.path.isdir(split_dir):
        return None, None
    for name in os.listdir(split_dir):
        lower = name.lower()
        if lower in ("image", "images", "img", "imgs"):
            img_dir = os.path.join(split_dir, name)
        if lower in ("gt", "mask", "masks", "groundtruth", "ground_truth", "annotation"):
            gt_dir = os.path.join(split_dir, name)
    return img_dir, gt_dir


def count_images(d):
    if not d or not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def detect_dataset_structure():
    """Detect the actual folder structure and return (train_img, train_gt, test_img, test_gt)."""
    print(f"\nScanning dataset directory: {DATA_ROOT}")

    if not os.path.isdir(DATA_ROOT):
        print(f"  ERROR: Data directory not found: {DATA_ROOT}")
        print("  Has the download completed? Check the download terminal.")
        return None

    # List top-level contents
    contents = os.listdir(DATA_ROOT)
    print(f"  Top-level contents: {contents[:10]}")

    # Try to find Train/Test structure (may be nested)
    base, train_dir, test_dir = find_split_dirs(DATA_ROOT)
    if base is None:
        # Maybe images are directly in COD10K/
        # Check for any images
        jpgs = [f for f in os.listdir(DATA_ROOT) if f.endswith(".jpg")]
        if jpgs:
            print("  Found flat image layout (no Train/Test split).")
            return DATA_ROOT, DATA_ROOT, DATA_ROOT, DATA_ROOT
        print("  Could not detect structure. Contents:", contents)
        return None

    print(f"  Dataset base: {base}")
    train_img, train_gt = find_images_and_gt(train_dir)
    test_img,  test_gt  = find_images_and_gt(test_dir)

    # Fallbacks if GT not found under Image/GT directly
    if train_img and not train_gt:
        # Try sibling dirs
        parent = os.path.dirname(train_img)
        for name in os.listdir(parent):
            if name.lower() in ("gt", "mask", "masks"):
                train_gt = os.path.join(parent, name)

    print(f"  Train images: {train_img} ({count_images(train_img)} files)")
    print(f"  Train GT:     {train_gt} ({count_images(train_gt)} files)")
    print(f"  Test images:  {test_img}  ({count_images(test_img)} files)")
    print(f"  Test GT:      {test_gt}   ({count_images(test_gt)} files)")

    return train_img, train_gt, test_img, test_gt


def patch_config(train_img, train_gt, test_img, test_gt):
    """Update config.py with detected paths."""
    cfg_path = os.path.join(ROOT, "config", "config.py")
    with open(cfg_path, encoding="utf-8") as f:
        src = f.read()

    def replace_path(src, var, new_val):
        import re
        pattern = rf'^({var}\s*=\s*).*$'
        escaped_val = new_val.replace('\\', '\\\\')
        replacement = rf'\g<1>r"{escaped_val}"'
        return re.sub(pattern, replacement, src, flags=re.MULTILINE)

    src = replace_path(src, "TRAIN_IMAGE_DIR", train_img)
    src = replace_path(src, "TRAIN_GT_DIR",    train_gt)
    src = replace_path(src, "TEST_IMAGE_DIR",  test_img)
    src = replace_path(src, "TEST_GT_DIR",     test_gt)

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"\n  Config updated: {cfg_path}")


def main():
    print("=" * 60)
    print("  CODNet — Smart Training Launcher")
    print("=" * 60)

    result = detect_dataset_structure()
    if result is None:
        print("\nERROR: Could not detect dataset. Make sure download is complete.")
        sys.exit(1)

    train_img, train_gt, test_img, test_gt = result

    if not train_img or not train_gt:
        print("\nERROR: Could not find Train images/GT directories.")
        print("Please check the data/COD10K folder structure.")
        sys.exit(1)

    # Patch config
    print("\nUpdating configuration...")
    patch_config(train_img, train_gt, test_img or test_img, test_gt or train_gt)

    # Start training
    python = os.path.join(ROOT, "venv", "Scripts", "python.exe")
    train_script = os.path.join(ROOT, "train.py")

    print("\n" + "=" * 60)
    print("  Starting CODNet Training!")
    print("  This will train for 100 epochs.")
    print("  Watch TensorBoard: tensorboard --logdir runs/")
    print("=" * 60 + "\n")

    cmd = [python, train_script,
           "--epochs", "100",
           "--batch_size", "8",
           "--backbone", "resnet50"]

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

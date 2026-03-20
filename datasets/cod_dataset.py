"""
COD Dataset Loader for Camouflaged Object Detection.

Supports COD10K, CAMO, CHAMELEON, and NC4K benchmark datasets.
Structure expected:
  dataset_root/
    Train/
      Images/     *.jpg
      GT/         *.png (binary masks)
    Test/
      Images/
      GT/
"""
import os
import glob
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .transforms import get_train_transforms, get_test_transforms


class CODDataset(Dataset):
    """
    Dataset for Camouflaged Object Detection.

    Args:
        root:       Path to dataset root (e.g., 'data/COD10K')
        split:      'train' or 'test'
        image_size: Target image size (default 352)
        transforms: Optional albumentations transforms
    """

    def __init__(self, root, split="train", image_size=352, transforms=None):
        super().__init__()
        self.image_size = image_size
        self.split = split

        # Resolve paths
        if split == "train":
            img_dir = os.path.join(root, "Train", "Images")
            gt_dir = os.path.join(root, "Train", "GT")
        else:
            img_dir = os.path.join(root, "Test", "Images")
            gt_dir = os.path.join(root, "Test", "GT")

        # Find all images
        self.images = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
            + glob.glob(os.path.join(img_dir, "*.bmp"))
        )

        # Match ground truth masks
        self.gts = []
        for img_path in self.images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, name + ".png")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, name + ".jpg")
            self.gts.append(gt_path)

        # Setup transforms
        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_test_transforms(image_size)

        print(f"[CODDataset] Loaded {len(self.images)} {split} samples from {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.gts[idx]).convert("L"))

        # Normalize mask to [0, 1]
        if mask.max() > 1:
            mask = mask / 255.0

        # Apply augmentations
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)  # [3, H, W], normalized to [0, 1]
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
            
        mask = mask.float()
            
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [1, H, W]

        # Normalize image with ImageNet stats
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask

    def get_image_path(self, idx):
        """Return original image path for visualization."""
        return self.images[idx]

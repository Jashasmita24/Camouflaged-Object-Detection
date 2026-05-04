"""
Animals Dataset Loader — extends the base CODDataset to support
the CAMO Kaggle dataset (camouflaged animals) with flexible folder structures.

Supported layouts:
  Layout A (CAMO-style):
    animals/
      Image/   *.jpg
      GT/      *.png

  Layout B (flat folder of images):
    animals/
      *.jpg
"""
import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .transforms import get_train_transforms, get_test_transforms


class AnimalsCAMODataset(Dataset):
    """
    Dataset for the CAMO Camouflaged Animals Kaggle dataset.
    Supports both paired (image + mask) and image-only modes.
    """

    def __init__(self, root, split="train", image_size=352, transforms=None):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.split = split

        # Detect layout
        self.paired = False
        img_dir = None
        gt_dir = None

        # Layout A: Image/ + GT/
        for img_sub in ["Image", "Images", "imgs", "images"]:
            candidate = os.path.join(root, img_sub)
            if os.path.isdir(candidate):
                img_dir = candidate
                break

        for gt_sub in ["GT", "Mask", "masks", "gt", "annotation"]:
            candidate = os.path.join(root, gt_sub)
            if os.path.isdir(candidate):
                gt_dir = candidate
                self.paired = True
                break

        # Collect images
        if img_dir:
            self.images = sorted(
                glob.glob(os.path.join(img_dir, "*.jpg"))
                + glob.glob(os.path.join(img_dir, "*.png"))
                + glob.glob(os.path.join(img_dir, "*.bmp"))
            )
        else:
            # Flat layout — images directly in root
            self.images = sorted(
                glob.glob(os.path.join(root, "*.jpg"))
                + glob.glob(os.path.join(root, "*.png"))
            )
            img_dir = root

        # Pair with masks if available
        self.gts = []
        if self.paired and gt_dir:
            for img_path in self.images:
                name = os.path.splitext(os.path.basename(img_path))[0]
                for ext in [".png", ".jpg", ".bmp"]:
                    gt_path = os.path.join(gt_dir, name + ext)
                    if os.path.exists(gt_path):
                        self.gts.append(gt_path)
                        break
                else:
                    self.gts.append(None)

            # Filter out unmatched pairs
            valid = [(img, gt) for img, gt in zip(self.images, self.gts) if gt is not None]
            if valid:
                self.images, self.gts = zip(*valid)
                self.images, self.gts = list(self.images), list(self.gts)
        else:
            self.paired = False

        # Transforms
        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_test_transforms(image_size)

        mode_str = "paired (image+mask)" if self.paired else "image-only"
        print(f"[AnimalsCAMO] {len(self.images)} images loaded from {root} [{mode_str}]")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        if self.paired:
            mask = np.array(Image.open(self.gts[idx]).convert("L"))
            mask = (mask / 255.0).astype(np.float32) if mask.max() > 1 else mask.astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Augment
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # To tensor
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image, mask

    def get_image_path(self, idx):
        return self.images[idx]

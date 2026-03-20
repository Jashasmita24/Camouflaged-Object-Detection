"""
Data Augmentation Transforms for COD Training and Testing.

Uses albumentations for efficient CPU-based augmentation.
Falls back to basic resize if albumentations is not available.
"""
import os
import numpy as np

# Suppress albumentations version check warning which times out
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[Warning] albumentations not installed. Using basic transforms.")

from PIL import Image
import torchvision.transforms as T


def get_train_transforms(image_size=352):
    """Training augmentation pipeline."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return BasicTransform(image_size, is_train=True)


def get_test_transforms(image_size=352):
    """Test/validation transform pipeline (no augmentation)."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return BasicTransform(image_size, is_train=False)


class BasicTransform:
    """Fallback transform when albumentations is not available."""

    def __init__(self, image_size, is_train=False):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, image, mask):
        # Resize
        image = np.array(
            Image.fromarray(image).resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        )
        mask = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
        ).astype(np.float32) / 255.0

        # Basic augmentation for training
        if self.is_train:
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

        return {"image": image, "mask": mask}

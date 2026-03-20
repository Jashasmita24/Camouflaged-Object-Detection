"""
Inference & Visualization Script for CODNet.

Run camouflaged object detection on individual images or a folder,
with side-by-side visualization of Input | Predicted Mask | Overlay.

Usage:
    python inference.py --image path/to/image.jpg --checkpoint checkpoints/codnet_best.pth
    python inference.py --image_dir path/to/folder/ --checkpoint checkpoints/codnet_best.pth
"""
import os
import sys
import argparse
import glob
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.cod_net import CODNet


def parse_args():
    parser = argparse.ArgumentParser(description="CODNet Inference")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to a folder of images")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "codnet_best.pth"))
    parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(config.RESULT_DIR, "inference"))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary threshold for mask")
    parser.add_argument("--show", action="store_true",
                        help="Display results interactively")
    return parser.parse_args()


def preprocess_image(image_path, image_size=352):
    """Load and preprocess a single image for inference."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    # Resize
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)
    tensor = TF.to_tensor(image_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return tensor.unsqueeze(0), np.array(image), original_size


@torch.no_grad()
def predict(model, image_tensor, device, original_size):
    """Run inference and return predicted mask at original resolution."""
    image_tensor = image_tensor.to(device)
    main_pred, _, _ = model(image_tensor)

    # Sigmoid activation
    pred = torch.sigmoid(main_pred).squeeze().cpu().numpy()

    # Resize back to original dimensions
    pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
    pred_pil = pred_pil.resize((original_size[0], original_size[1]), Image.BILINEAR)
    pred_mask = np.array(pred_pil).astype(np.float32) / 255.0

    return pred_mask


def create_overlay(image, mask, alpha=0.5, color=(0, 255, 0)):
    """Create an overlay of the predicted mask on the original image."""
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_binary = mask > 0.5

    for c in range(3):
        mask_colored[:, :, c] = color[c]

    overlay[mask_binary] = (
        alpha * mask_colored[mask_binary]
        + (1 - alpha) * overlay[mask_binary]
    ).astype(np.uint8)

    return overlay


def visualize_result(image, pred_mask, threshold=0.5, save_path=None, show=False):
    """Create side-by-side visualization."""
    binary_mask = (pred_mask > threshold).astype(np.float32)
    overlay = create_overlay(image, binary_mask, alpha=0.4, color=(0, 255, 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Predicted Camouflage Map", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Detection Overlay", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Please train the model first with: python train.py")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CODNet(
        backbone_name=config.BACKBONE,
        pretrained=False,
        channel_dim=config.CHANNEL_DIM,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Model loaded successfully!\n")

    # Collect images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        for ext in ["*.jpg", "*.png", "*.bmp", "*.jpeg"]:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths = sorted(image_paths)
    else:
        print("Please provide --image or --image_dir")
        return

    print(f"Processing {len(image_paths)} image(s)...\n")

    for i, img_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1}/{len(image_paths)}] {name}")

        # Preprocess
        image_tensor, original_image, original_size = preprocess_image(
            img_path, args.image_size
        )

        # Predict
        pred_mask = predict(model, image_tensor, device, original_size)

        # Save prediction mask
        mask_save_path = os.path.join(args.output_dir, f"{name}_mask.png")
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_save_path)

        # Visualize
        vis_save_path = os.path.join(args.output_dir, f"{name}_result.png")
        visualize_result(
            original_image, pred_mask,
            threshold=args.threshold,
            save_path=vis_save_path,
            show=args.show,
        )

    print(f"\n{'='*50}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

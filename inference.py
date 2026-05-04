"""
Inference & Visualization Script — CODNet, UNet, and YOLOv8.

Run camouflaged object detection on individual images or a folder,
with side-by-side visualization of Input | Predicted Mask/Boxes | Overlay.

Usage:
    # CODNet segmentation (default)
    python inference.py --image path/to/image.jpg --checkpoint checkpoints/codnet_best.pth

    # UNet segmentation
    python inference.py --model unet --image path/to/image.jpg
    python inference.py --model unet --image path/to/image.jpg --checkpoint checkpoints/unet/unet_best.pth

    # YOLOv8 object detection
    python inference.py --model yolo --image path/to/image.jpg
    python inference.py --model yolo --image_dir path/to/folder/ --yolo_model yolov8n --conf 0.25

    # Folder of images with CODNet
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
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config


# ─── Argument Parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="COD Inference — CODNet, UNet, or YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Source
    parser.add_argument("--image",     type=str, default=None, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to a folder of images")

    # Model selection
    parser.add_argument(
        "--model", type=str, default="codnet",
        choices=["codnet", "unet", "yolo"],
        help="Detection model: 'codnet' (seg), 'unet' (seg), or 'yolo' (bbox). Default: codnet",
    )

    # CODNet / UNet options
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="[CODNet/UNet] Path to model checkpoint (auto-detected if omitted)",
    )
    parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE,
                        help="[CODNet/UNet] Input resolution")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="[CODNet/UNet] Binary mask threshold")

    # YOLOv8 options
    parser.add_argument(
        "--yolo_model", type=str, default=getattr(config, "YOLO_MODEL", "yolov8n"),
        help="[YOLOv8] Model variant or path to .pt checkpoint",
    )
    parser.add_argument(
        "--conf", type=float, default=getattr(config, "YOLO_CONFIDENCE", 0.25),
        help="[YOLOv8] Confidence threshold",
    )
    parser.add_argument(
        "--iou", type=float, default=getattr(config, "YOLO_IOU", 0.45),
        help="[YOLOv8] NMS IoU threshold",
    )

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(config.RESULT_DIR, "inference"),
                        help="Directory to save results")
    parser.add_argument("--show", action="store_true", help="Display results interactively")

    return parser.parse_args()


# ─── CODNet helpers ───────────────────────────────────────────────────────────
def preprocess_image(image_path, image_size=352):
    """Load and preprocess a single image for CODNet inference."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    image_resized = image.resize((image_size, image_size), Image.BILINEAR)
    tensor = TF.to_tensor(image_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0), np.array(image), original_size


@torch.no_grad()
def codnet_predict(model, image_tensor, device, original_size):
    """Run CODNet inference and return predicted mask at original resolution."""
    image_tensor = image_tensor.to(device)
    main_pred, _, _ = model(image_tensor)
    pred = torch.sigmoid(main_pred).squeeze().cpu().numpy()
    pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
    pred_pil = pred_pil.resize((original_size[0], original_size[1]), Image.BILINEAR)
    return np.array(pred_pil).astype(np.float32) / 255.0


def create_overlay(image, mask, alpha=0.5, color=(0, 255, 0)):
    """Create a mask overlay on the original image."""
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


def visualize_codnet(image, pred_mask, threshold=0.5, save_path=None, show=False):
    """Create side-by-side visualization for CODNet."""
    binary_mask = (pred_mask > threshold).astype(np.float32)
    overlay = create_overlay(image, binary_mask, alpha=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image);       axes[0].set_title("Input Image",        fontsize=14, fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Predicted Camouflage Map", fontsize=14, fontweight="bold"); axes[1].axis("off")
    axes[2].imshow(overlay);     axes[2].set_title("Detection Overlay",   fontsize=14, fontweight="bold"); axes[2].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show() if show else plt.close()


# ─── YOLOv8 helpers ───────────────────────────────────────────────────────────
def visualize_yolo(pil_img, annotated_img, detections, save_path=None, show=False):
    """Create side-by-side visualization for YOLOv8."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(np.array(pil_img))
    axes[0].set_title("Input Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.array(annotated_img))
    axes[1].set_title(f"YOLOv8 Detections ({len(detections)} objects)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show() if show else plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model.upper()}\n")

    # ── Collect image paths ───────────────────────────────────────────────────
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

    if not image_paths:
        print("No images found.")
        return
    print(f"Processing {len(image_paths)} image(s)...\n")

    # ══════════════════════════════════════════════════════════════════════════
    # CODNet MODE
    # ══════════════════════════════════════════════════════════════════════════
    if args.model == "codnet":
        from models.cod_net import CODNet

        checkpoint = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "codnet_best.pth")
        if not os.path.exists(checkpoint):
            print(f"ERROR: Checkpoint not found: {checkpoint}")
            print("Train first: python train.py")
            return

        print(f"Loading CODNet from: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model = CODNet(
            backbone_name=config.BACKBONE,
            pretrained=False,
            channel_dim=config.CHANNEL_DIM,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print("CODNet loaded.\n")

        for i, img_path in enumerate(image_paths):
            name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[{i+1}/{len(image_paths)}] {name}")

            image_tensor, original_image, original_size = preprocess_image(img_path, args.image_size)
            pred_mask = codnet_predict(model, image_tensor, device, original_size)

            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                os.path.join(args.output_dir, f"{name}_mask.png")
            )
            visualize_codnet(
                original_image, pred_mask,
                threshold=args.threshold,
                save_path=os.path.join(args.output_dir, f"{name}_result.png"),
                show=args.show,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # UNet MODE  (same segmentation pipeline as CODNet)
    # ══════════════════════════════════════════════════════════════════════════
    elif args.model == "unet":
        from models.unet import UNet

        checkpoint = args.checkpoint or os.path.join(
            getattr(config, "UNET_CHECKPOINT_DIR", config.CHECKPOINT_DIR), "unet_best.pth"
        )
        if not os.path.exists(checkpoint):
            print(f"ERROR: Checkpoint not found: {checkpoint}")
            print("Train first: python train.py --model_type unet")
            return

        print(f"Loading UNet from: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        encoder = getattr(config, "UNET_ENCODER", "resnet50")
        model = UNet(
            encoder=encoder,
            pretrained=False,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print("UNet loaded.\n")

        for i, img_path in enumerate(image_paths):
            name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[{i+1}/{len(image_paths)}] {name}")

            image_tensor, original_image, original_size = preprocess_image(img_path, args.image_size)
            # UNet's forward returns (main_pred, [], []), same as codnet_predict expects
            pred_mask = codnet_predict(model, image_tensor, device, original_size)

            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                os.path.join(args.output_dir, f"{name}_mask.png")
            )
            visualize_codnet(
                original_image, pred_mask,
                threshold=args.threshold,
                save_path=os.path.join(args.output_dir, f"{name}_unet_result.png"),
                show=args.show,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # YOLOv8 MODE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        from models.yolo_detector import YOLODetector

        print(f"Loading YOLOv8 model: {args.yolo_model} (conf={args.conf}, iou={args.iou})")
        detector = YOLODetector(
            model_name=args.yolo_model,
            confidence=args.conf,
            iou_threshold=args.iou,
        )
        print("YOLOv8 loaded.\n")

        for i, img_path in enumerate(image_paths):
            name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"[{i+1}/{len(image_paths)}] {name}")

            pil_img = Image.open(img_path).convert("RGB")
            detections, inf_ms = detector.detect(pil_img)
            annotated = detector.draw_detections(pil_img, detections)

            print(f"  → {len(detections)} object(s) detected in {inf_ms:.0f}ms")
            for d in detections:
                print(f"     {d['class_name']:20s} conf={d['confidence']:.3f}  bbox={d['bbox']}")

            # Save annotated image
            annotated.save(os.path.join(args.output_dir, f"{name}_yolo.png"))

            # Save visualization
            visualize_yolo(
                pil_img, annotated, detections,
                save_path=os.path.join(args.output_dir, f"{name}_yolo_result.png"),
                show=args.show,
            )

    print(f"\n{'='*50}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

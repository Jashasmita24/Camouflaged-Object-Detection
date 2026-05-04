"""
Training Script for Camouflaged Object Detection (CODNet / UNet).

Usage:
    python train.py
    python train.py --model_type codnet --epochs 50 --batch_size 4 --backbone resnet50
    python train.py --model_type unet --epochs 50 --batch_size 4
    python train.py --epochs 1 --max_iters 5 --batch_size 2  # Smoke test
"""
import os
import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.cod_net import CODNet
from models.unet import UNet, build_unet
from datasets.cod_dataset import CODDataset
from utils.losses import CODLoss, UNetLoss
from utils.metrics import MetricCalculator


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CODNet / UNet")
    parser.add_argument("--model_type", type=str,
                        default=getattr(config, "MODEL_TYPE", "codnet"),
                        choices=["codnet", "unet"],
                        help="Model architecture: codnet or unet")
    parser.add_argument("--backbone", type=str, default=config.BACKBONE,
                        choices=["resnet50", "res2net50"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(config.DATA_DIR, "COD10K"))
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Max iterations per epoch (-1 for all)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision training")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch, args):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    loss_details = {"bce": 0, "iou": 0, "boundary": 0, "total": 0}
    num_batches = 0

    for i, (images, masks) in enumerate(loader):
        if args.max_iters > 0 and i >= args.max_iters:
            break

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        use_amp = not args.no_amp and device.type == "cuda"
        with autocast(enabled=use_amp):
            main_pred, coarse_preds, boundary_maps = model(images)
            loss, details = criterion(main_pred, coarse_preds, boundary_maps, masks)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        for k in loss_details:
            loss_details[k] += details[k]
        num_batches += 1

        # Print progress
        if (i + 1) % config.PRINT_FREQ == 0 or i == 0:
            avg_loss = running_loss / num_batches
            print(
                f"  Epoch [{epoch+1}/{args.epochs}] "
                f"Iter [{i+1}/{len(loader)}] "
                f"Loss: {avg_loss:.4f} "
                f"(BCE: {details['bce']:.4f}, "
                f"IoU: {details['iou']:.4f}, "
                f"Boundary: {details['boundary']:.4f})"
            )

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, loader, device, args):
    """Validate on test set."""
    model.eval()
    metric_calc = MetricCalculator()

    for i, (images, masks) in enumerate(loader):
        if args.max_iters > 0 and i >= args.max_iters:
            break

        images = images.to(device)
        main_pred, _, _ = model(images)

        # Convert to numpy for metrics
        pred = torch.sigmoid(main_pred).squeeze(1).cpu().numpy()  # [B, H, W]
        gt = masks.squeeze(1).numpy()  # [B, H, W]

        for j in range(pred.shape[0]):
            metric_calc.update(pred[j], gt[j])

    return metric_calc.get_results()


def main():
    args = parse_args()
    set_seed(config.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    if args.model_type == "unet":
        encoder = getattr(config, "UNET_ENCODER", "resnet50")
        print(f"\nBuilding UNet with {encoder} encoder...")
        model = UNet(
            encoder=encoder,
            pretrained=True,
        ).to(device)
    else:
        print(f"\nBuilding CODNet with {args.backbone} backbone...")
        model = CODNet(
            backbone_name=args.backbone,
            pretrained=True,
            channel_dim=config.CHANNEL_DIM,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    train_dataset = CODDataset(
        root=args.data_dir, split="train", image_size=args.image_size
    )
    test_dataset = CODDataset(
        root=args.data_dir, split="test", image_size=args.image_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Loss, optimizer, scheduler
    if args.model_type == "unet":
        criterion = UNetLoss(
            bce_weight=config.BCE_WEIGHT,
            iou_weight=config.IOU_WEIGHT,
            boundary_weight=config.BOUNDARY_WEIGHT,
        ).to(device)
    else:
        criterion = CODLoss(
            bce_weight=config.BCE_WEIGHT,
            iou_weight=config.IOU_WEIGHT,
            boundary_weight=config.BOUNDARY_WEIGHT,
        ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=config.MIN_LR
    )
    scaler = GradScaler(enabled=not args.no_amp and device.type == "cuda")

    # Resume from checkpoint
    start_epoch = 0
    best_mae = float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", float("inf"))

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.LOG_DIR)
        print(f"TensorBoard logs: {config.LOG_DIR}")
    except ImportError:
        writer = None
        print("TensorBoard not available. Skipping logging.")

    # Training loop
    model_name = args.model_type  # "codnet" or "unet"
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs  [{model_name.upper()}]")
    print(f"Image size: {args.image_size}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )

        # Validate
        metrics = validate(model, test_loader, device, args)
        epoch_time = time.time() - start_time

        # Update learning rate
        scheduler.step()

        # Print epoch results
        print(f"\n{'-'*60}")
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f} | "
              f"S-measure: {metrics['S-measure']:.4f} | "
              f"E-measure: {metrics['E-measure']:.4f} | "
              f"wF-measure: {metrics['wF-measure']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'-'*60}\n")

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            for k, v in metrics.items():
                writer.add_scalar(f"Metrics/{k}", v, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save checkpoints
        is_best = metrics["MAE"] < best_mae
        if is_best:
            best_mae = metrics["MAE"]

        # Select checkpoint directory based on model type
        ckpt_dir = (config.UNET_CHECKPOINT_DIR if model_name == "unet"
                    else config.CHECKPOINT_DIR)

        if (epoch + 1) % config.SAVE_EVERY == 0 or is_best:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mae": best_mae,
                "metrics": metrics,
                "model_type": model_name,
            }
            save_path = os.path.join(
                ckpt_dir,
                f"{model_name}_epoch{epoch+1}.pth"
            )
            torch.save(ckpt, save_path)
            print(f"Saved checkpoint: {save_path}")

            if is_best:
                best_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
                torch.save(ckpt, best_path)
                print(f"  * New best MAE: {best_mae:.4f}")

    print(f"\nTraining complete! Best MAE: {best_mae:.4f}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()

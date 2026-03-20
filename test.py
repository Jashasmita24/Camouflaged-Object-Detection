"""
Test/Evaluation Script for CODNet.

Evaluates a trained model on the test set and reports standard COD metrics.

Usage:
    python test.py --checkpoint checkpoints/codnet_best.pth
    python test.py --checkpoint checkpoints/codnet_best.pth --save_results
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.cod_net import CODNet
from datasets.cod_dataset import CODDataset
from utils.metrics import MetricCalculator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CODNet")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(config.DATA_DIR, "COD10K"))
    parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_results", action="store_true",
                        help="Save prediction masks to results/")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, save_dir=None, dataset=None):
    """Evaluate model and optionally save prediction masks."""
    model.eval()
    metric_calc = MetricCalculator()

    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        main_pred, _, _ = model(images)

        # Sigmoid + convert to numpy
        pred = torch.sigmoid(main_pred).squeeze(1).cpu().numpy()
        gt = masks.squeeze(1).numpy()

        for j in range(pred.shape[0]):
            metric_calc.update(pred[j], gt[j])

            # Save prediction mask
            if save_dir and dataset:
                idx = i * loader.batch_size + j
                img_path = dataset.get_image_path(idx)
                name = os.path.splitext(os.path.basename(img_path))[0]

                pred_mask = (pred[j] * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(
                    os.path.join(save_dir, f"{name}.png")
                )

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(loader)} batches...")

    return metric_calc.get_results()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = CODNet(
        backbone_name=config.BACKBONE,
        pretrained=False,
        channel_dim=config.CHANNEL_DIM,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from epoch {ckpt['epoch']+1}")

    # Dataset
    test_dataset = CODDataset(
        root=args.data_dir, split="test", image_size=args.image_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Save directory
    save_dir = None
    if args.save_results:
        save_dir = os.path.join(config.RESULT_DIR, "predictions")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving predictions to: {save_dir}")

    # Evaluate
    print(f"\nEvaluating on {len(test_dataset)} test images...")
    metrics = evaluate(
        model, test_loader, device,
        save_dir=save_dir, dataset=test_dataset
    )

    # Print results
    print(f"\n{'='*50}")
    print("  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  MAE:        {metrics['MAE']:.4f}")
    print(f"  S-measure:  {metrics['S-measure']:.4f}")
    print(f"  E-measure:  {metrics['E-measure']:.4f}")
    print(f"  wF-measure: {metrics['wF-measure']:.4f}")
    print(f"{'='*50}")

    if "metrics" in ckpt:
        print(f"\n  (Training best MAE: {ckpt.get('best_mae', 'N/A')})")


if __name__ == "__main__":
    main()

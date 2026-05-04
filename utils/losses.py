"""
Loss Functions for Camouflaged Object Detection.

Combined loss: Weighted BCE + IoU + Boundary Loss
Supports deep supervision with coarse predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """Weighted Binary Cross-Entropy with Logits."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, reduction="mean")


class IoULoss(nn.Module):
    """IoU (Intersection over Union) Loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that penalizes errors near object boundaries.
    Extracts boundaries using Laplacian-like operation and applies
    higher weight to boundary regions.
    """

    def __init__(self):
        super().__init__()
        # Laplacian kernel for boundary extraction
        laplacian = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian", laplacian)

    def forward(self, pred, target):
        # Extract boundaries from ground truth
        boundary = F.conv2d(target, self.laplacian.to(target.device), padding=1)
        boundary = boundary.abs()
        boundary = (boundary > 0.1).float()

        # Dilate boundary slightly for broader supervision
        boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)

        # Apply boundary-weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Higher weight on boundary regions
        weight = 1.0 + boundary * 4.0  # 5x weight on boundaries
        weighted_bce = (bce * weight).mean()

        return weighted_bce


class CODLoss(nn.Module):
    """
    Combined loss for CODNet with deep supervision.

    L_total = bce_w * BCE + iou_w * IoU + bound_w * Boundary
            + deep_supervision_losses (weighted by 0.5 each)
    """

    def __init__(self, bce_weight=1.0, iou_weight=1.0, boundary_weight=0.5):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.iou_loss = IoULoss()
        self.boundary_loss = BoundaryLoss()

        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight

    def forward(self, main_pred, coarse_preds, boundary_maps, target):
        """
        Args:
            main_pred:     [B, 1, H, W] final prediction (logits)
            coarse_preds:  List of [B, 1, H, W] deep supervision predictions (logits)
            boundary_maps: List of boundary maps from SGFL
            target:        [B, 1, H, W] ground truth mask
        """
        # Main prediction loss
        loss_bce = self.bce_loss(main_pred, target)
        loss_iou = self.iou_loss(main_pred, target)
        loss_boundary = self.boundary_loss(main_pred, target)

        total_loss = (
            self.bce_weight * loss_bce
            + self.iou_weight * loss_iou
            + self.boundary_weight * loss_boundary
        )

        # Deep supervision losses (coarse predictions)
        ds_weight = 0.5
        for coarse_pred in coarse_preds:
            total_loss += ds_weight * (
                self.bce_loss(coarse_pred, target)
                + self.iou_loss(coarse_pred, target)
            )

        # Boundary map supervision
        # Generate boundary GT from target mask
        boundary_gt = self._extract_boundary(target)
        bm_weight = 0.3
        for bmap in boundary_maps:
            bmap_resized = F.interpolate(
                bmap, size=target.shape[2:], mode="bilinear", align_corners=False
            )
            total_loss += bm_weight * F.binary_cross_entropy_with_logits(
                bmap_resized, boundary_gt
            )

        return total_loss, {
            "bce": loss_bce.item(),
            "iou": loss_iou.item(),
            "boundary": loss_boundary.item(),
            "total": total_loss.item(),
        }

    def _extract_boundary(self, mask):
        """Extract boundary from binary mask using erosion-dilation difference."""
        kernel_size = 3
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
        boundary = dilated - eroded
        return boundary.clamp(0, 1)


class UNetLoss(nn.Module):
    """
    Loss for UNet — simplified version without deep supervision or boundary maps.

    L_total = bce_w * BCE + iou_w * IoU + bound_w * Boundary

    Returns the same (loss, details_dict) signature as CODLoss for pipeline
    compatibility, but ignores the (empty) coarse_preds and boundary_maps.
    """

    def __init__(self, bce_weight=1.0, iou_weight=1.0, boundary_weight=0.5):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.iou_loss = IoULoss()
        self.boundary_loss = BoundaryLoss()

        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight

    def forward(self, main_pred, coarse_preds, boundary_maps, target):
        """
        Args:
            main_pred:     [B, 1, H, W] segmentation logits from UNet
            coarse_preds:  Ignored (expected to be [])
            boundary_maps: Ignored (expected to be [])
            target:        [B, 1, H, W] ground truth mask
        """
        loss_bce = self.bce_loss(main_pred, target)
        loss_iou = self.iou_loss(main_pred, target)
        loss_boundary = self.boundary_loss(main_pred, target)

        total_loss = (
            self.bce_weight * loss_bce
            + self.iou_weight * loss_iou
            + self.boundary_weight * loss_boundary
        )

        return total_loss, {
            "bce": loss_bce.item(),
            "iou": loss_iou.item(),
            "boundary": loss_boundary.item(),
            "total": total_loss.item(),
        }


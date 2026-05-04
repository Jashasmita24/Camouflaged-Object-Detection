"""
Boundary-Guided Decoder for Camouflaged Object Detection.

Progressive upsampling decoder with skip connections and boundary guidance.
Produces coarse-to-fine prediction maps with deep supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Single decoder block: upsample + skip fusion + conv refinement."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Skip connection fusion
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Refinement after fusion
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        """
        Args:
            x:    Deeper level feature to upsample [B, C_in, H, W]
            skip: Same-level skip feature [B, C_skip, 2H, 2W]
        """
        x = self.upsample(x)
        # Ensure spatial sizes match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = self.input_conv(x)
        skip = self.skip_conv(skip)
        fused = x + skip  # Additive fusion
        return self.refine(fused)


class BoundaryGuidedDecoder(nn.Module):
    """
    Progressive decoder taking 4-level features and producing
    multi-scale predictions for deep supervision.

    Flow: f5 → decode4(f5, f4) → decode3(*, f3) → decode2(*, f2) → output
    """

    def __init__(self, channels=64):
        super().__init__()
        # Decoder blocks (deepest to shallowest)
        self.decode4 = DecoderBlock(channels, channels, channels)
        self.decode3 = DecoderBlock(channels, channels, channels)
        self.decode2 = DecoderBlock(channels, channels, channels)

        # Multi-scale prediction heads (for deep supervision)
        self.pred_head5 = nn.Conv2d(channels, 1, 1)  # 1/32
        self.pred_head4 = nn.Conv2d(channels, 1, 1)  # 1/16
        self.pred_head3 = nn.Conv2d(channels, 1, 1)  # 1/8
        self.pred_head2 = nn.Conv2d(channels, 1, 1)  # 1/4

        # Final upsampling to full resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, features, input_size):
        """
        Args:
            features: [f2, f3, f4, f5] refined features from SGFL
            input_size: (H, W) original input spatial size
        Returns:
            main_pred:    Final prediction at full resolution [B, 1, H, W]
            coarse_preds: List of coarse predictions for deep supervision
        """
        f2, f3, f4, f5 = features

        # Coarse prediction at deepest level
        pred5 = self.pred_head5(f5)

        # Progressive decoding
        d4 = self.decode4(f5, f4)
        pred4 = self.pred_head4(d4)

        d3 = self.decode3(d4, f3)
        pred3 = self.pred_head3(d3)

        d2 = self.decode2(d3, f2)
        pred2 = self.pred_head2(d2)

        # Final prediction at full resolution
        main_feat = F.interpolate(d2, size=input_size, mode="bilinear", align_corners=False)
        main_pred = self.final_conv(main_feat)

        # Upsample all coarse predictions to full resolution for loss computation
        coarse_preds = [
            F.interpolate(p, size=input_size, mode="bilinear", align_corners=False)
            for p in [pred5, pred4, pred3, pred2]
        ]

        return main_pred, coarse_preds

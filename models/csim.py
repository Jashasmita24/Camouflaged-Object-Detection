"""
Cross-Scale Interaction Module (CSIM) for Camouflaged Object Detection.

This module performs feature alignment and fusion across adjacent scales
to capture both local texture details and broader contextual information.
Uses channel attention + spatial attention for adaptive cross-scale fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism using global average & max pooling."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism using channel-wise pooling."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class CrossScaleInteractionBlock(nn.Module):
    """
    Fuses features from two adjacent scales (high-res and low-res).
    1. Upsample low-res feature to match high-res spatial size
    2. Reduce both to same channel dim
    3. Apply channel & spatial attention on concatenated features
    4. Output fused feature
    """

    def __init__(self, high_channels, low_channels, out_channels):
        super().__init__()
        # Channel reduction for each input
        self.reduce_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion convolution (concat → fuse)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Attention modules
        self.channel_attn = ChannelAttention(out_channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, high_feat, low_feat):
        """
        Args:
            high_feat: Higher resolution feature [B, C_high, H, W]
            low_feat:  Lower resolution feature [B, C_low, H/2, W/2]
        Returns:
            Fused feature [B, out_channels, H, W]
        """
        # Upsample low-res to match high-res spatial size
        low_feat_up = F.interpolate(
            low_feat, size=high_feat.shape[2:], mode="bilinear", align_corners=False
        )

        # Reduce channels
        high_feat = self.reduce_high(high_feat)
        low_feat_up = self.reduce_low(low_feat_up)

        # Concatenate and fuse
        fused = torch.cat([high_feat, low_feat_up], dim=1)
        fused = self.fuse_conv(fused)

        # Apply attention
        fused = fused * self.channel_attn(fused)
        fused = fused * self.spatial_attn(fused)

        return fused


class CSIM(nn.Module):
    """
    Cross-Scale Interaction Module.
    Takes 4-level backbone features and produces 4 cross-scale fused features.
    Each level is fused with its adjacent deeper level.
    """

    def __init__(self, in_channels_list=(256, 512, 1024, 2048), out_channels=64):
        super().__init__()
        c2, c3, c4, c5 = in_channels_list

        # Cross-scale interaction: fuse each level with the next deeper level
        self.csib_2_3 = CrossScaleInteractionBlock(c2, c3, out_channels)
        self.csib_3_4 = CrossScaleInteractionBlock(c3, c4, out_channels)
        self.csib_4_5 = CrossScaleInteractionBlock(c4, c5, out_channels)

        # Reduce C5 (deepest level, no deeper neighbor)
        self.reduce_c5 = nn.Sequential(
            nn.Conv2d(c5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        """
        Args:
            features: List of [C2, C3, C4, C5] from backbone
        Returns:
            List of 4 fused features, all with out_channels
        """
        c2, c3, c4, c5 = features

        f2 = self.csib_2_3(c2, c3)   # Fuses C2 + C3 → [B, out_ch, H/4, W/4]
        f3 = self.csib_3_4(c3, c4)   # Fuses C3 + C4 → [B, out_ch, H/8, W/8]
        f4 = self.csib_4_5(c4, c5)   # Fuses C4 + C5 → [B, out_ch, H/16, W/16]
        f5 = self.reduce_c5(c5)      # Reduce C5     → [B, out_ch, H/32, W/32]

        return [f2, f3, f4, f5]

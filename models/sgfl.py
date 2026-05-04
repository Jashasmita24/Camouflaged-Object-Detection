"""
Semantic Guided Feature Learning (SGFL) Module for Camouflaged Object Detection.

Pure CNN-based module that refines features through:
1. Global Context Modeling — using global average pooling + FC layers
2. Foreground Enhancement — amplify foreground regions using semantic cues
3. Background Suppression — reduce false positives from background clutter
4. Boundary Feature Extraction — extract edge/boundary features for sharper segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalContextBlock(nn.Module):
    """
    CNN-based global context modeling.
    Uses global average pooling to capture scene-level semantics,
    then broadcasts back to spatial dimensions.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.fc(self.gap(x))
        return x * weight


class ForegroundEnhancementBlock(nn.Module):
    """
    Enhances foreground features by learning discriminative spatial masks.
    Uses dilated convolutions to capture multi-scale context.
    """

    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return out + x  # Residual connection


class BoundaryExtractionBlock(nn.Module):
    """
    Extracts boundary features using a learned edge detector.
    Uses Laplacian-like convolutions + learned refinement.
    """

    def __init__(self, channels):
        super().__init__()
        # Edge detection path
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Boundary prediction head (produces 1-channel boundary map)
        self.boundary_head = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        """
        Returns:
            boundary_feat: Enhanced boundary features [B, C, H, W]
            boundary_map:  Predicted boundary map [B, 1, H, W]
        """
        edge = self.edge_conv1(x)
        edge = self.edge_conv2(edge)
        boundary_feat = edge + x  # Residual
        boundary_map = self.boundary_head(edge)
        return boundary_feat, boundary_map


class SGFLBlock(nn.Module):
    """Single SGFL processing block for one feature level."""

    def __init__(self, channels):
        super().__init__()
        self.global_ctx = GlobalContextBlock(channels)
        self.fg_enhance = ForegroundEnhancementBlock(channels)
        self.boundary = BoundaryExtractionBlock(channels)

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Returns:
            refined_feat: Semantically guided feature [B, C, H, W]
            boundary_map: Predicted boundary [B, 1, H, W]
        """
        # Global context
        ctx = self.global_ctx(x)

        # Foreground enhancement
        fg = self.fg_enhance(ctx)

        # Boundary extraction
        boundary_feat, boundary_map = self.boundary(x)

        # Combine foreground + boundary features
        combined = torch.cat([fg, boundary_feat], dim=1)
        refined = self.refine(combined)

        return refined, boundary_map


class SGFL(nn.Module):
    """
    Semantic Guided Feature Learning Module.
    Applies SGFL processing to each feature level independently.
    """

    def __init__(self, channels=64, num_levels=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            SGFLBlock(channels) for _ in range(num_levels)
        ])

    def forward(self, features):
        """
        Args:
            features: List of 4 features from CSIM [B, C, H_i, W_i]
        Returns:
            refined_features: List of refined features
            boundary_maps: List of boundary predictions
        """
        refined_features = []
        boundary_maps = []

        for feat, block in zip(features, self.blocks):
            refined, boundary = block(feat)
            refined_features.append(refined)
            boundary_maps.append(boundary)

        return refined_features, boundary_maps

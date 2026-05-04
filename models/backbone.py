"""
Backbone Feature Extractor for Camouflaged Object Detection.

Supports ResNet50 (default) and Res2Net50 backbones.
Extracts multi-scale features from 4 stages:
  - C2: 1/4  resolution, 256 channels
  - C3: 1/8  resolution, 512 channels
  - C4: 1/16 resolution, 1024 channels
  - C5: 1/32 resolution, 2048 channels
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone that returns multi-scale features."""

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Stage 0: conv1 + bn1 + relu + maxpool (1/4 resolution)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # Stages 1-4
        self.layer1 = resnet.layer1   # C2: 256 channels,  1/4
        self.layer2 = resnet.layer2   # C3: 512 channels,  1/8
        self.layer3 = resnet.layer3   # C4: 1024 channels, 1/16
        self.layer4 = resnet.layer4   # C5: 2048 channels, 1/32

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            List of features [C2, C3, C4, C5]
        """
        x = self.layer0(x)
        c2 = self.layer1(x)   # [B, 256, H/4, W/4]
        c3 = self.layer2(c2)  # [B, 512, H/8, W/8]
        c4 = self.layer3(c3)  # [B, 1024, H/16, W/16]
        c5 = self.layer4(c4)  # [B, 2048, H/32, W/32]
        return [c2, c3, c4, c5]


def get_backbone(name="resnet50", pretrained=True):
    """Factory function for backbone selection."""
    if name == "resnet50":
        return ResNet50Backbone(pretrained=pretrained)
    elif name == "res2net50":
        try:
            import timm
            backbone = timm.create_model(
                "res2net50_26w_4s",
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4)
            )
            return backbone
        except ImportError:
            print("timm not installed. Falling back to ResNet50.")
            return ResNet50Backbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}")

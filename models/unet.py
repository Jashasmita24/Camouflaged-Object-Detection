"""
U-Net — Encoder-Decoder Network for Camouflaged Object Detection.

A standard U-Net architecture with:
  - Encoder: 4 down-sampling stages with pretrained ResNet50 feature maps
    OR a lightweight custom encoder for faster training.
  - Decoder: 4 up-sampling stages with skip connections from the encoder,
    progressively recovering spatial resolution.
  - Final 1×1 conv producing a single-channel segmentation logit map.

The forward() returns (main_pred, [], []) so that it is drop-in compatible
with the existing CODNet training / inference pipeline, which expects:
    main_pred, coarse_preds, boundary_maps = model(x)

Outputs:
  - main_pred:     [B, 1, H, W] — full-resolution segmentation logits
  - coarse_preds:  [] (empty list — no deep supervision)
  - boundary_maps: [] (empty list — no boundary heads)

Usage:
    from models.unet import UNet, build_unet
    model = UNet()                        # custom lightweight encoder
    model = UNet(encoder="resnet50")      # pretrained ResNet50 encoder
    model = build_unet(config)            # build from project config
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive 3×3 Conv-BN-ReLU layers (standard U-Net block)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: upsample → concatenate skip → ConvBlock."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle spatial size mismatch (pad if needed)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── Custom Lightweight Encoder ───────────────────────────────────────────────

class CustomEncoder(nn.Module):
    """
    A simple 4-stage down-sampling encoder (no pretraining).
    Each stage: ConvBlock → MaxPool.

    Output channels per stage: 64, 128, 256, 512
    Bottleneck: 1024
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)

        # Expose channel counts for the decoder
        self.skip_channels = [64, 128, 256, 512]
        self.bottleneck_channels = 1024

    def forward(self, x):
        s1 = self.enc1(x)            # [B, 64,  H,   W]
        s2 = self.enc2(self.pool(s1)) # [B, 128, H/2, W/2]
        s3 = self.enc3(self.pool(s2)) # [B, 256, H/4, W/4]
        s4 = self.enc4(self.pool(s3)) # [B, 512, H/8, W/8]
        bn = self.bottleneck(self.pool(s4))  # [B, 1024, H/16, W/16]
        return [s1, s2, s3, s4], bn


# ─── ResNet50-based Encoder ───────────────────────────────────────────────────

class ResNet50Encoder(nn.Module):
    """
    Pretrained ResNet50 repurposed as a U-Net encoder.

    Skip connections are tapped at 4 stages:
      s1 — after layer0 (conv1+bn+relu)     :  64 ch, H/2
      s2 — after layer1 (+ maxpool)         : 256 ch, H/4
      s3 — after layer2                     : 512 ch, H/8
      s4 — after layer3                     : 1024 ch, H/16
    Bottleneck — after layer4               : 2048 ch, H/32
    """

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # H/2
        self.pool  = resnet.maxpool                                        # H/4
        self.layer1 = resnet.layer1   # 256 ch, H/4
        self.layer2 = resnet.layer2   # 512 ch, H/8
        self.layer3 = resnet.layer3   # 1024 ch, H/16
        self.layer4 = resnet.layer4   # 2048 ch, H/32

        self.skip_channels = [64, 256, 512, 1024]
        self.bottleneck_channels = 2048

    def forward(self, x):
        s1 = self.conv1(x)            # [B, 64,   H/2, W/2]
        s2 = self.layer1(self.pool(s1))  # [B, 256,  H/4, W/4]
        s3 = self.layer2(s2)          # [B, 512,  H/8, W/8]
        s4 = self.layer3(s3)          # [B, 1024, H/16, W/16]
        bn = self.layer4(s4)          # [B, 2048, H/32, W/32]
        return [s1, s2, s3, s4], bn


# ─── U-Net ────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for Camouflaged Object Detection.

    Args:
        encoder:    "custom" (lightweight, no pretraining) or
                    "resnet50" (ImageNet-pretrained ResNet50 encoder).
        pretrained: Whether to use pretrained weights (only for resnet50).
        in_channels: Number of input channels (default 3 for RGB).
    """

    def __init__(self, encoder="resnet50", pretrained=True, in_channels=3):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        if encoder == "resnet50":
            self.encoder = ResNet50Encoder(pretrained=pretrained)
        else:
            self.encoder = CustomEncoder(in_channels=in_channels)

        sc = self.encoder.skip_channels     # e.g. [64, 256, 512, 1024]
        bc = self.encoder.bottleneck_channels  # e.g. 2048

        # ── Decoder ───────────────────────────────────────────────────────────
        # Mirrors encoder in reverse order
        self.up4 = UpBlock(bc,    sc[3], sc[3])   # 2048 → 1024
        self.up3 = UpBlock(sc[3], sc[2], sc[2])   # 1024 → 512
        self.up2 = UpBlock(sc[2], sc[1], sc[1])   # 512  → 256
        self.up1 = UpBlock(sc[1], sc[0], sc[0])   # 256  → 64

        # ── Segmentation Head ─────────────────────────────────────────────────
        self.seg_head = nn.Conv2d(sc[0], 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]
        Returns:
            main_pred:     [B, 1, H, W] segmentation logits
            coarse_preds:  [] (empty — no deep supervision)
            boundary_maps: [] (empty — no boundary heads)
        """
        input_size = x.shape[2:]  # (H, W)

        # Encode
        skips, bottleneck = self.encoder(x)  # skips = [s1, s2, s3, s4]

        # Decode (deepest → shallowest)
        d4 = self.up4(bottleneck, skips[3])
        d3 = self.up3(d4, skips[2])
        d2 = self.up2(d3, skips[1])
        d1 = self.up1(d2, skips[0])

        # Upsample to original resolution (encoder may have halved it)
        if d1.shape[2:] != input_size:
            d1 = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)

        main_pred = self.seg_head(d1)

        # Return in the same format as CODNet for pipeline compatibility
        return main_pred, [], []


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_unet(config=None):
    """Build a UNet model from config or with sensible defaults."""
    if config is not None:
        encoder = getattr(config, "UNET_ENCODER", "resnet50")
        pretrained = getattr(config, "PRETRAINED", True)
    else:
        encoder = "resnet50"
        pretrained = True
    return UNet(encoder=encoder, pretrained=pretrained)

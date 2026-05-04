"""
CODNet - Complete Camouflaged Object Detection Network.

Assembles the full pipeline:
  Backbone (ResNet50) → CSIM → SGFL → Boundary-Guided Decoder

Outputs:
  - main_pred:     Final segmentation map [B, 1, H, W]
  - coarse_preds:  Deep supervision predictions (list)
  - boundary_maps: Boundary predictions from SGFL (list)
"""
import torch
import torch.nn as nn

from .backbone import get_backbone
from .csim import CSIM
from .sgfl import SGFL
from .decoder import BoundaryGuidedDecoder


class CODNet(nn.Module):
    """
    Deep Learning based Camouflaged Object Detection Network (CNN-based).
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        channel_dim=64,
        in_channels_list=(256, 512, 1024, 2048),
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # 1. Backbone (multi-scale feature extraction)
        self.backbone = get_backbone(backbone_name, pretrained=pretrained)

        # 2. Cross-Scale Interaction Module
        self.csim = CSIM(
            in_channels_list=in_channels_list,
            out_channels=channel_dim,
        )

        # 3. Semantic Guided Feature Learning
        self.sgfl = SGFL(channels=channel_dim, num_levels=4)

        # 4. Boundary-Guided Decoder
        self.decoder = BoundaryGuidedDecoder(channels=channel_dim)

    def forward(self, x):
        """
        Args:
            x: Input RGB image [B, 3, H, W]
        Returns:
            main_pred:     [B, 1, H, W] final segmentation prediction
            coarse_preds:  List of [B, 1, H, W] for deep supervision
            boundary_maps: List of [B, 1, H_i, W_i] boundary predictions
        """
        input_size = x.shape[2:]  # (H, W)

        # Stage 1: Backbone feature extraction
        if self.backbone_name == "resnet50":
            features = self.backbone(x)  # [C2, C3, C4, C5]
        else:
            # timm-based backbone (res2net)
            features = self.backbone(x)

        # Stage 2: Cross-scale interaction
        fused_features = self.csim(features)  # 4 features, all channel_dim channels

        # Stage 3: Semantic guided feature learning
        refined_features, boundary_maps = self.sgfl(fused_features)

        # Stage 4: Boundary-guided decoding
        main_pred, coarse_preds = self.decoder(refined_features, input_size)

        return main_pred, coarse_preds, boundary_maps


def build_model(config=None):
    """Build CODNet from config or with defaults."""
    if config is not None:
        return CODNet(
            backbone_name=getattr(config, "BACKBONE", "resnet50"),
            pretrained=getattr(config, "PRETRAINED", True),
            channel_dim=getattr(config, "CHANNEL_DIM", 64),
        )
    return CODNet()

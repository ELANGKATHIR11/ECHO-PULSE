"""
EchoPhys-X: Decoupled Detection Head with Explicit Gating Flags
==============================================================
Provides decoupled detection branches:
  - Objectness Logits (1 channel)
  - Classification Logits (num_classes channels)
  - Bounding Box Offsets (4 channels: Softplus LTRB distances)

Unlabelled extension heads (Biofouling, Natural Coral/Rock Mimic, Acoustic Shadow,
Calibrated Uncertainty) are placed behind explicit boolean flags and disabled by default.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.echophys.models.backbone import DSConv, DirectionalStateSpaceMixer


class DecoupledDetectionHead(nn.Module):
    def __init__(
        self,
        in_c: int = 128,
        num_classes: int = 4,
        enable_biofouling: bool = False,
        enable_mimic: bool = False,
        enable_shadow: bool = False,
        enable_uncertainty: bool = False
    ):
        super().__init__()
        self.in_c = in_c
        self.num_classes = num_classes
        self.enable_biofouling = enable_biofouling
        self.enable_mimic = enable_mimic
        self.enable_shadow = enable_shadow
        self.enable_uncertainty = enable_uncertainty

        # Shared feature refinement
        self.stem = nn.Sequential(
            DSConv(in_c, in_c),
            DirectionalStateSpaceMixer(in_c)
        )

        # Core object detection heads
        self.obj_head = nn.Conv2d(in_c, 1, kernel_size=1)
        self.cls_head = nn.Conv2d(in_c, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(in_c, 4, kernel_size=1)

        # Optional auxiliary heads (only initialized if explicitly enabled)
        if self.enable_mimic:
            self.mimic_head = nn.Conv2d(in_c, 1, kernel_size=1)
        if self.enable_biofouling:
            self.bio_head = nn.Conv2d(in_c, 1, kernel_size=1)
        if self.enable_shadow:
            self.shadow_head = nn.Conv2d(in_c, 1, kernel_size=1)
        if self.enable_uncertainty:
            self.unc_head = nn.Conv2d(in_c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.stem(x)

        outputs = {
            "obj": self.obj_head(feat),
            "cls": self.cls_head(feat),
            "box": F.softplus(self.box_head(feat))
        }

        if self.enable_mimic:
            mimic_logits = self.mimic_head(feat)
            outputs["mimic_logits"] = mimic_logits
            outputs["p_mimic"] = torch.sigmoid(mimic_logits)
            
        if self.enable_biofouling:
            outputs["bio_ratio"] = torch.sigmoid(self.bio_head(feat))
            
        if self.enable_shadow:
            outputs["shadow_len"] = F.softplus(self.shadow_head(feat))
            
        if self.enable_uncertainty:
            outputs["uncertainty"] = self.unc_head(feat)

        return outputs

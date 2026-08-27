"""
EchoPhys-X: Master Model Implementations
========================================
- EchoPhysX_SSS640: Validated single-frequency dataset model (NUM_CLASSES configurable).
- EchoPhysX_Physics: Model accepting physical environmental telemetry (T, S, depth, frequency).
- EchoPhysX_Bio: Model enabling biofouling / shadow extension branches.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from src.echophys.models.backbone import EchoPhysBackbone
from src.echophys.models.neck import BiFPNNeck
from src.echophys.models.head import DecoupledDetectionHead
from src.echophys.physics.acoustic_proxies import (
    make_acoustic_proxy_tensor,
    make_physical_conditioning_tensor
)


class EchoPhysX_SSS640(nn.Module):
    """
    EchoPhys-X-SSS640: Validated Single-Frequency Sonar Object Detector.
    Processes 640x640 single-frequency acoustic imagery using 5 acoustic proxy channels,
    a multi-scale directional state-space backbone, weighted BiFPN, and decoupled heads.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.backbone = EchoPhysBackbone(in_channels=in_channels)
        self.neck = BiFPNNeck(in_channels_list=[96, 160, 224], out_dim=128)

        self.head_p3 = DecoupledDetectionHead(in_c=128, num_classes=num_classes)
        self.head_p4 = DecoupledDetectionHead(in_c=128, num_classes=num_classes)
        self.head_p5 = DecoupledDetectionHead(in_c=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        # x is (B, 1, H, W) or (B, in_channels, H, W)
        if x.shape[1] == 1:
            x = make_acoustic_proxy_tensor(x)

        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)

        return {
            "p3": self.head_p3(f3), # 80x80 (stride 8)
            "p4": self.head_p4(f4), # 40x40 (stride 16)
            "p5": self.head_p5(f5)  # 20x20 (stride 32)
        }


class EchoPhysX_Physics(nn.Module):
    """
    EchoPhys-X-Physics: Model conditioned on real physical environmental telemetry.
    Input channels: 8 (5 base proxies + sound speed field + transmission loss + grazing angle).
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = 8

        self.backbone = EchoPhysBackbone(in_channels=8)
        self.neck = BiFPNNeck(in_channels_list=[96, 160, 224], out_dim=128)

        self.head_p3 = DecoupledDetectionHead(in_c=128, num_classes=num_classes)
        self.head_p4 = DecoupledDetectionHead(in_c=128, num_classes=num_classes)
        self.head_p5 = DecoupledDetectionHead(in_c=128, num_classes=num_classes)

    def forward(
        self,
        x: torch.Tensor,
        temp_c: Optional[float] = None,
        salinity_ppt: Optional[float] = None,
        depth_m: Optional[float] = None,
        freq_khz: Optional[float] = None,
        altitude_m: Optional[float] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if x.shape[1] == 1 or x.shape[1] == 5:
            im_raw = x[:, 0:1] if x.shape[1] == 5 else x
            x, _ = make_physical_conditioning_tensor(
                im_raw,
                temp_c=temp_c,
                salinity_ppt=salinity_ppt,
                depth_m=depth_m,
                freq_khz=freq_khz,
                altitude_m=altitude_m
            )

        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)

        return {
            "p3": self.head_p3(f3),
            "p4": self.head_p4(f4),
            "p5": self.head_p5(f5)
        }

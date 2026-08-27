"""
EchoPhys-X: Multi-Scale BiFPN Feature Fusion Neck (Memory-Optimized)
===================================================================
Fuses P3 (80x80), P4 (40x40), and P5 (20x20) with fast normalized learnable
weights and depthwise separable convolutions to preserve high-resolution small
objects without loss of spatial context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.echophys.models.backbone import ConvBNAct, DSConv


class BiFPNNeck(nn.Module):
    """
    Weighted Bi-Directional Feature Pyramid Network (BiFPN).
    Provides top-down and bottom-up cross-scale feature aggregation.
    """
    def __init__(self, in_channels_list=[96, 160, 224], out_dim: int = 128):
        super().__init__()
        c3, c4, c5 = in_channels_list
        self.p3_proj = ConvBNAct(c3, out_dim, 1)
        self.p4_proj = ConvBNAct(c4, out_dim, 1)
        self.p5_proj = ConvBNAct(c5, out_dim, 1)

        # Learnable positive weights for fast normalized fusion
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.w_p3_td = nn.Parameter(torch.ones(2))
        self.w_p4_bu = nn.Parameter(torch.ones(3))
        self.w_p5_bu = nn.Parameter(torch.ones(2))

        self.conv_p4_td = DSConv(out_dim, out_dim)
        self.conv_p3_out = DSConv(out_dim, out_dim)
        self.conv_p4_out = DSConv(out_dim, out_dim)
        self.conv_p5_out = DSConv(out_dim, out_dim)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        p3_in = self.p3_proj(p3)
        p4_in = self.p4_proj(p4)
        p5_in = self.p5_proj(p5)

        # Top-down pathway
        w_td_4 = F.relu(self.w_p4_td) / (torch.sum(F.relu(self.w_p4_td)) + 1e-4)
        p4_td = self.conv_p4_td(
            w_td_4[0] * p4_in + w_td_4[1] * F.interpolate(p5_in, scale_factor=2, mode="nearest")
        )

        w_td_3 = F.relu(self.w_p3_td) / (torch.sum(F.relu(self.w_p3_td)) + 1e-4)
        p3_out = self.conv_p3_out(
            w_td_3[0] * p3_in + w_td_3[1] * F.interpolate(p4_td, scale_factor=2, mode="nearest")
        )

        # Bottom-up pathway
        w_bu_4 = F.relu(self.w_p4_bu) / (torch.sum(F.relu(self.w_p4_bu)) + 1e-4)
        p4_out = self.conv_p4_out(
            w_bu_4[0] * p4_in + w_bu_4[1] * p4_td + w_bu_4[2] * F.interpolate(p3_out, scale_factor=0.5, mode="nearest")
        )

        w_bu_5 = F.relu(self.w_p5_bu) / (torch.sum(F.relu(self.w_p5_bu)) + 1e-4)
        p5_out = self.conv_p5_out(
            w_bu_5[0] * p5_in + w_bu_5[1] * F.interpolate(p4_out, scale_factor=0.5, mode="nearest")
        )

        return p3_out, p4_out, p5_out

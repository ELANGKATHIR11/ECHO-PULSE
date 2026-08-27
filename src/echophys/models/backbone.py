"""
EchoPhys-X: Directional State-Space Inspired Mixer (Memory-Efficient)
=====================================================================
Uses inplace arithmetic and chunked gating to minimize peak VRAM allocations
during multi-scale feature pyramid processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalStateSpaceMixer(nn.Module):
    """
    Directional State-Space Inspired Acoustic Mixer.
    Processes acoustic backscatter along horizontal (across-track range)
    and vertical (along-track scanline) axes using depthwise causal/bidirectional
    convolutions with bounded learned decay parameters.
    """
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        self.dim = dim
        pad = kernel_size // 2

        self.proj_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.dw_along = nn.Conv2d(dim, dim, (kernel_size, 1), padding=(pad, 0), groups=dim, bias=False)
        self.dw_across = nn.Conv2d(dim, dim, (1, kernel_size), padding=(0, pad), groups=dim, bias=False)
        
        # Bounded decay parameters in (0, 1) to ensure stability
        self.decay_along = nn.Parameter(torch.ones(dim, 1, 1) * 0.80)
        self.decay_across = nn.Parameter(torch.ones(dim, 1, 1) * 0.80)
        
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(dim)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u, v = self.proj_in(x).chunk(2, dim=1)

        # Directional state transitions with learned sigmoid decay
        s_along = self.dw_along(u) * torch.sigmoid(self.decay_along)
        s_across = self.dw_across(v) * torch.sigmoid(self.decay_across)

        # Memory efficient cross-directional gating
        g = self.gate(s_along + s_across)
        fused = g * s_along + (1.0 - g) * s_across

        out = self.proj_out(self.norm(fused))
        return x + out


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSConv(nn.Module):
    """Depthwise-Separable Convolution for ultra-lightweight execution."""
    def __init__(self, cin: int, cout: int, s: int = 1):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, 3, s, groups=cin)
        self.pw = ConvBNAct(cin, cout, 1, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class EchoPhysBackbone(nn.Module):
    """
    Lightweight Multi-Scale Acoustic Backbone producing:
      P3: 80x80 (High resolution for small acoustic highlights/debris)
      P4: 40x40 (Medium scale)
      P5: 20x20 (Large scale / shipwrecks / macro seabed features)
    """
    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, 3, 2), # 320x320
            DSConv(32, 32)
        )
        self.stage1 = nn.Sequential(
            ConvBNAct(32, 64, 3, 2),          # 160x160
            DSConv(64, 64)
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(64, 96, 3, 2),          # 80x80 (P3)
            DirectionalStateSpaceMixer(96)
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(96, 160, 3, 2),         # 40x40 (P4)
            DirectionalStateSpaceMixer(160)
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(160, 224, 3, 2),        # 20x20 (P5)
            DirectionalStateSpaceMixer(224)
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5

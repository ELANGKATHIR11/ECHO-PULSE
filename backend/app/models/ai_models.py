import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

class LightweightSonarUNet(nn.Module):
    """PyTorch U-Net for Acoustic Shadow & Object Boundary Segmentation."""
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))


class SonarAutoencoder(nn.Module):
    """Convolutional Autoencoder for Normal Seabed Baseline & Anomaly Discovery."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 128 -> 64
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MultiFactorFusion:
    """Combines YOLO Detection, U-Net Shadow, Anomaly Autoencoder, and Geometric Scores."""
    
    WEIGHTS = {
        "detector": 0.40,
        "shadow": 0.25,
        "geometry": 0.15,
        "anomaly": 0.10,
        "quality": 0.10
    }

    @staticmethod
    def fuse(
        detector_score: float,
        shadow_score: float,
        geometry_score: float,
        anomaly_score: float,
        quality_score: float
    ) -> float:
        w = MultiFactorFusion.WEIGHTS
        raw_fused = (
            detector_score * w["detector"] +
            shadow_score * w["shadow"] +
            geometry_score * w["geometry"] +
            anomaly_score * w["anomaly"] +
            quality_score * w["quality"]
        )
        return float(np.clip(raw_fused, 0.0, 0.99))


class HomoscedasticMultiTaskLoss(nn.Module):
    """
    Homoscedastic Uncertainty-Weighted Multi-Task Loss for Marine Acoustic Perception:
    Balances Detection Loss (L_det), UNet Shadow Segmentation Loss (L_shadow), and Bathymetric Depth Loss (L_depth)
    Loss = 0.5 * exp(-s1) * L_det + 0.5 * exp(-s2) * L_shadow + 0.5 * exp(-s3) * L_depth + 0.5 * (s1 + s2 + s3)
    where s_i = log(sigma_i^2) are learnable homoscedastic variance parameters.
    """
    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Learnable log variances
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, requires_grad=True))

    def forward(self, loss_det: torch.Tensor, loss_shadow: torch.Tensor, loss_depth: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        precision_det = torch.exp(-self.log_vars[0])
        precision_shadow = torch.exp(-self.log_vars[1])
        precision_depth = torch.exp(-self.log_vars[2])

        total_loss = (
            0.5 * precision_det * loss_det +
            0.5 * precision_shadow * loss_shadow +
            0.5 * precision_depth * loss_depth +
            0.5 * (self.log_vars[0] + self.log_vars[1] + self.log_vars[2])
        )

        weights = {
            "weight_det": float(precision_det.detach().cpu().item()),
            "weight_shadow": float(precision_shadow.detach().cpu().item()),
            "weight_depth": float(precision_depth.detach().cpu().item())
        }
        return total_loss, weights


class AcousticAngularReflectanceAttention(nn.Module):
    """
    Acoustic Angular Reflectance Attention (AARA Head):
    Modulates backbone feature activations dynamically based on slant-range grazing angle:
    theta_grazing = arccos(H_alt / R_slant)
    x_modulated = gamma(theta) * F_backbone + beta(theta)
    Suppresses acoustic falloff speckle noise in far-range sonar boundaries.
    """
    def __init__(self, in_features: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, in_features * 2) # Outputs gamma (scale) and beta (shift)
        )

    def forward(self, x: torch.Tensor, grazing_angle_rad: torch.Tensor) -> torch.Tensor:
        # grazing_angle_rad: [B, 1]
        scale_shift = self.mlp(grazing_angle_rad) # [B, 2 * in_features]
        gamma, beta = torch.chunk(scale_shift, 2, dim=-1)
        # Reshape to broadcast over [B, C, H, W]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ShadowHighlightCrossAttention(nn.Module):
    """
    Shadow-Highlight Dual-Stream Cross-Attention Module:
    Co-models bright acoustic highlight returns with corresponding dark trailing acoustic shadows:
    Attention(Q_highlight, K_shadow, V_shadow) = softmax(Q_h * K_s^T / sqrt(d_k)) * V_s
    Eliminates false positives on isolated high-reflectivity seabed rocks lacking true acoustic shadows.
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, highlight_feat: torch.Tensor, shadow_feat: torch.Tensor) -> torch.Tensor:
        # highlight_feat, shadow_feat: [B, N, C]
        norm_h = self.norm1(highlight_feat)
        norm_s = self.norm1(shadow_feat)
        attn_out, _ = self.mha(query=norm_h, key=norm_s, value=norm_s)
        x = highlight_feat + attn_out
        x = x + self.ffn(self.norm2(x))
        return x



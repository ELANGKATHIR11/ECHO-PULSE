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

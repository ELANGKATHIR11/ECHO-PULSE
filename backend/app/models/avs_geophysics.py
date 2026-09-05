"""
AVS-GeoPhysics-X: Probabilistic Spherical DOA + Range + Geolocation Engine
EchoPulseNet Marine Sonar Intelligence Platform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


class AVSGeoPhysicsX(nn.Module):
    """
    AVS-GeoPhysics-X:
    Probabilistic 3D Direction of Arrival, Transmission Loss Ranging,
    and WGS-84 Geolocation under Spatially-Varying Sound Speed Profiles.
    """

    def __init__(self, in_channels: int = 4, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # 4-Channel AVS Vector Intensity Encoder (P, Ux, Uy, Uz)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Environmental Conditioning (Sound Speed, Depth, Salinity, Temperature)
        self.env_proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        # 1. Probabilistic Spherical DOA Head: Unit Vector [ux, uy, uz] + Angular Variance
        self.spherical_doa_head = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.SiLU(),
            nn.Linear(64, 4) # [ux, uy, uz, log_var_angle]
        )

        # 2. Heteroscedastic Range Estimator: [Range (meters), log_var_range]
        self.range_head = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )

    def forward(self, avs_4ch: torch.Tensor, env_params: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        avs_4ch: (B, 4, L) 4-channel acoustic vector sensor recordings
        env_params: (B, 4) [c_mps, depth_m, salinity_psu, temp_c]
        """
        B = avs_4ch.shape[0]
        feat = self.encoder(avs_4ch).squeeze(-1) # (B, d_model)

        if env_params is None:
            env_params = torch.tensor([[1500.0, 50.0, 35.0, 18.0]], device=avs_4ch.device).expand(B, -1)

        env_feat = self.env_proj(env_params)
        fused = torch.cat([feat, env_feat], dim=-1)

        doa_out = self.spherical_doa_head(fused)
        vec_raw = doa_out[:, :3]
        vec_norm = F.normalize(vec_raw, p=2, dim=-1) # Unit vector on S^2 sphere
        log_var_angle = doa_out[:, 3:4]
        sigma_angle_deg = torch.exp(0.5 * log_var_angle) * 15.0 # Degrees uncertainty

        # Decompose into spherical angles: Azimuth theta, Elevation phi
        ux = vec_norm[:, 0]
        uy = vec_norm[:, 1]
        uz = vec_norm[:, 2]
        azimuth_rad = torch.atan2(uy, ux)
        azimuth_deg = torch.remainder(torch.rad2deg(azimuth_rad), 360.0)

        horiz_mag = torch.sqrt(ux ** 2 + uy ** 2) + 1e-9
        elevation_rad = torch.atan2(uz, horiz_mag)
        elevation_deg = torch.rad2deg(elevation_rad)

        # Range regression
        range_out = self.range_head(fused)
        range_norm = F.softplus(range_out[:, 0:1])
        range_m = range_norm * 1200.0 # Physical range scale
        sigma_range_m = torch.exp(0.5 * range_out[:, 1:2]) * 80.0

        return {
            "spherical_doa_vector": vec_norm,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "angular_uncertainty_deg": sigma_angle_deg.squeeze(-1),
            "range_meters": range_m.squeeze(-1),
            "range_uncertainty_meters": sigma_range_m.squeeze(-1)
        }

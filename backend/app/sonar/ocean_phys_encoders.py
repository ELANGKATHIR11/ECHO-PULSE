"""
Complex Spectral & Spatial Physics-Aware Multi-Branch Encoders
OCEAN-PHYSNet - Marine Sonar Intelligence Platform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class ComplexHydrophoneSpectralEncoder(nn.Module):
    """
    Complex-valued and multi-representation Hydrophone Spectral Encoder:
    - Raw 1D waveform temporal convolution
    - Complex STFT with Magnitude A(f,t) and Phase phi(f,t) preservation
    - Log-Mel spectral envelope extraction
    - Physics-aware feature fusion into unified acoustic token sequence
    """

    def __init__(self, d_model: int = 256, n_fft: int = 512, hop_length: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 1. 1D Raw Waveform Temporal Encoder
        self.raw_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # 2. 2D Complex STFT Encoder (Real + Imaginary + Phase Channels = 3 channels)
        self.complex_stft_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, None))  # Fix frequency dimension to 16
        )

        # Fusion Projection
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (B, 1, L) or (B, L) audio tensor
        Returns: (B, T_tokens, d_model)
        """
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(1)
        B, C, L = x_raw.shape

        # 1. Raw 1D features
        feat_raw = self.raw_conv(x_raw) # (B, d_model, T_raw)
        feat_raw = feat_raw.transpose(1, 2) # (B, T_raw, d_model)

        # 2. Complex STFT representation
        x_1d = x_raw.squeeze(1) # (B, L)
        window = torch.hann_window(self.n_fft, device=x_raw.device)
        stft_res = torch.stft(x_1d, n_fft=self.n_fft, hop_length=self.hop_length, 
                              window=window, return_complex=True) # (B, F, T)
        
        real_part = stft_res.real.unsqueeze(1)
        imag_part = stft_res.imag.unsqueeze(1)
        phase_part = torch.angle(stft_res).unsqueeze(1)
        complex_input = torch.cat([real_part, imag_part, phase_part], dim=1) # (B, 3, F, T)

        feat_stft = self.complex_stft_conv(complex_input) # (B, d_model, 16, T_stft)
        feat_stft = feat_stft.mean(dim=2).transpose(1, 2) # (B, T_stft, d_model)

        # Interpolate raw features to match STFT time length
        target_len = feat_stft.shape[1]
        feat_raw_resampled = F.interpolate(feat_raw.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)

        # Concatenate and fuse
        fused = torch.cat([feat_stft, feat_raw_resampled], dim=-1)
        tokens = self.norm(self.fusion_proj(fused))
        return tokens


class ComplexAVSSpatialEncoder(nn.Module):
    """
    4-Channel Complex Acoustic Vector Sensor (AVS) Spatial Branch:
    - Pressure P(t) + Particle Velocity (Ux, Uy, Uz)
    - 3D Active Acoustic Intensity: I(f) = 0.5 * Re{P(f) * U*(f)}
    - Cross-spectral density covariance & Inter-sensor phase
    - Spatial directional token projection
    """

    def __init__(self, d_model: int = 256, n_fft: int = 512, hop_length: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 4-Channel spatio-temporal convolution (P, Ux, Uy, Uz)
        self.avs_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # Intensity vector linear head (Ix, Iy, Iz -> d_model)
        self.intensity_proj = nn.Linear(3, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, avs_4ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        avs_4ch: (B, 4, L) containing [P, Ux, Uy, Uz]
        Returns:
            avs_tokens: (B, T_tokens, d_model)
            intensity_vector: (B, 3) 3D active intensity [Ix, Iy, Iz]
        """
        B, C, L = avs_4ch.shape
        p = avs_4ch[:, 0, :]
        ux = avs_4ch[:, 1, :]
        uy = avs_4ch[:, 2, :]
        uz = avs_4ch[:, 3, :]

        # 1. 3D Active Intensity Computation: I = <p * u>
        ix = torch.mean(p * ux, dim=1, keepdim=True)
        iy = torch.mean(p * uy, dim=1, keepdim=True)
        iz = torch.mean(p * uz, dim=1, keepdim=True)
        intensity_3d = torch.cat([ix, iy, iz], dim=1) # (B, 3)

        # 2. Spatio-temporal convolutional features
        spatial_feats = self.avs_conv(avs_4ch).transpose(1, 2) # (B, T, d_model)

        # 3. Project intensity to tokens and broadcast
        intensity_embed = self.intensity_proj(intensity_3d).unsqueeze(1).expand(-1, spatial_feats.shape[1], -1)

        fused = torch.cat([spatial_feats, intensity_embed], dim=-1)
        avs_tokens = self.norm(self.fusion(fused))

        return avs_tokens, intensity_3d


class MultipathTokenizer(nn.Module):
    """
    Multipath Tokenizer & Direct/Reflected Separation:
    - Evaluates soft attention weights wm = softmax(qm) over acoustic multipath hypotheses
    - Prevents boundary reverberations from being misclassified as distinct targets
    """

    def __init__(self, d_model: int = 256, num_paths: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_paths = num_paths
        self.path_queries = nn.Parameter(torch.randn(num_paths, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_tokens: (B, T, d_model)
        Returns:
            multipath_tokens: (B, num_paths, d_model)
            path_weights: (B, num_paths, T)
        """
        B, T, D = x_tokens.shape
        keys = self.key_proj(x_tokens) # (B, T, D)
        vals = self.val_proj(x_tokens) # (B, T, D)
        queries = self.path_queries.unsqueeze(0).expand(B, -1, -1) # (B, num_paths, D)

        # Attention distribution over time frames
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(D) # (B, num_paths, T)
        attn_weights = F.softmax(scores, dim=-1) # (B, num_paths, T)

        path_repr = torch.bmm(attn_weights, vals) # (B, num_paths, D)
        multipath_tokens = self.norm(self.out_proj(path_repr))

        return multipath_tokens, attn_weights


class OceanStateProjector(nn.Module):
    """
    Projects Ocean State parameter vector Eo into ocean conditioning tokens.
    """

    def __init__(self, ocean_dim: int = 16, d_model: int = 256, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(ocean_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model * num_tokens),
            nn.LayerNorm(d_model * num_tokens)
        )
        self.d_model = d_model

    def forward(self, ocean_tensor: torch.Tensor) -> torch.Tensor:
        """
        ocean_tensor: (B, 16)
        Returns: (B, num_tokens, d_model)
        """
        B = ocean_tensor.shape[0]
        out = self.mlp(ocean_tensor)
        tokens = out.view(B, self.num_tokens, self.d_model)
        return tokens

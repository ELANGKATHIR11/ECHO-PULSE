"""
Physics-Biased Cross-Attention & Fourier Neural Operator (FNO) Helmholtz Block
OCEAN-PHYSNet - Marine Sonar Intelligence Platform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PhysicsBiasedCrossAttention(nn.Module):
    """
    Physics-Biased Cross-Attention Module:
    A_phys = softmax( (QK^T / sqrt(d)) + B_phys ) * V
    where B_phys = g(Delta_r, Delta_t, c(z), alpha(f), Coherence, SSP)
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Physics bias neural generator
        self.physics_bias_mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                ocean_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: (B, N_q, d_model) e.g. Hydrophone / AVS query tokens
        key_value: (B, N_k, d_model) e.g. Multimodal & Ocean tokens
        ocean_state: (B, 16) Physical ocean parameter vector Eo
        Returns:
            out: (B, N_q, d_model)
            attn_weights: (B, num_heads, N_q, N_k)
        """
        B, N_q, _ = query.shape
        _, N_k, _ = key_value.shape

        # Linear projections
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N_q, head_dim)
        K = self.k_proj(key_value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N_k, head_dim)
        V = self.v_proj(key_value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N_k, head_dim)

        # Standard dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, N_q, N_k)

        # Compute physical acoustic propagation bias B_phys
        # Shape: (B, H, 1, 1) broadcast across (N_q, N_k)
        b_phys = self.physics_bias_mlp(ocean_state).unsqueeze(-1).unsqueeze(-1) * 2.5 # (B, H, 1, 1)

        # Create structured acoustic distance-decay matrix along sequence dimension
        i_idx = torch.arange(N_q, device=query.device).unsqueeze(1)
        j_idx = torch.arange(N_k, device=query.device).unsqueeze(0)
        dist_matrix = torch.abs(i_idx - j_idx).float() / max(1, N_k) # (N_q, N_k)

        # Apply exponential seawater absorption penalty e^(-alpha * r)
        absorption_decay = -0.5 * dist_matrix.unsqueeze(0).unsqueeze(0) # (1, 1, N_q, N_k)

        # Add physics bias to standard attention scores
        biased_scores = scores + b_phys + absorption_decay

        attn_weights = F.softmax(biased_scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Context aggregation
        context = torch.matmul(attn_weights_drop, V) # (B, H, N_q, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.d_model)

        out = self.norm(query + self.out_proj(context))
        return out, attn_weights


class SpectralConv1d(nn.Module):
    """1D Fourier Neural Operator spectral convolution layer."""
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, L)
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)

        modes = min(self.modes, x_ft.size(-1))
        # Complex matrix multiplication in Fourier space
        out_ft[:, :, :modes] = torch.einsum("bix,iox->box", x_ft[:, :, :modes], self.weights[:, :, :modes])

        x_out = torch.fft.irfft(out_ft, n=x.size(-1))
        return x_out


class FourierNeuralOperatorPropagationBlock(nn.Module):
    """
    Fourier Neural Operator (FNO) for Differentiable Acoustic Propagation:
    - Learns propagation operator G_theta: [c(z), alpha(f), bathymetry, p0] -> p_hat(r, z, f)
    - Computes Helmholtz wave equation residual: R_wave = grad^2(p) + k^2 * p
    """

    def __init__(self, d_model: int = 256, modes: int = 16):
        super().__init__()
        self.modes = modes
        self.d_model = d_model

        self.conv0 = SpectralConv1d(d_model, d_model, modes=modes)
        self.w0 = nn.Conv1d(d_model, d_model, 1)

        self.conv1 = SpectralConv1d(d_model, d_model, modes=modes)
        self.w1 = nn.Conv1d(d_model, d_model, 1)

        self.pressure_field_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2) # Real & Imaginary parts of complex pressure field p_hat(r,z)
        )

    def forward(self, tokens: torch.Tensor, ocean_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tokens: (B, T, d_model)
        ocean_state: (B, 16)
        Returns:
            prop_tokens: (B, T, d_model)
            complex_p_field: (B, T, 2) [p_real, p_imag]
            helmholtz_residual: (B,) wave equation penalty
        """
        x = tokens.transpose(1, 2) # (B, d_model, T)

        # FNO Layer 1
        x1 = self.conv0(x) + self.w0(x)
        x1 = F.gelu(x1)

        # FNO Layer 2
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)

        prop_tokens = x2.transpose(1, 2) # (B, T, d_model)

        # Predict latent acoustic pressure field
        p_field = self.pressure_field_head(prop_tokens) # (B, T, 2)
        p_real = p_field[:, :, 0]
        p_imag = p_field[:, :, 1]

        # Evaluate 1D numerical Helmholtz residual: d^2 p / dr^2 + k^2 * p
        # k = 2 * pi * f / c
        c_sound = 1500.0 + ocean_state[:, 3] * 50.0 # (B,)
        c_sound = c_sound.unsqueeze(1) # (B, 1)
        k_wavenumber = (2.0 * math.pi * 1000.0) / (c_sound + 1e-6) # ~1kHz acoustic wave number

        # Second-order numerical Laplacian along propagation coordinate
        if p_real.shape[1] >= 3:
            laplacian_real = p_real[:, 2:] - 2 * p_real[:, 1:-1] + p_real[:, :-2]
            laplacian_imag = p_imag[:, 2:] - 2 * p_imag[:, 1:-1] + p_imag[:, :-2]

            p_mid_real = p_real[:, 1:-1]
            p_mid_imag = p_imag[:, 1:-1]

            res_real = laplacian_real + (k_wavenumber ** 2) * p_mid_real
            res_imag = laplacian_imag + (k_wavenumber ** 2) * p_mid_imag

            helmholtz_residual = torch.mean(res_real ** 2 + res_imag ** 2, dim=1) # (B,)
        else:
            helmholtz_residual = torch.zeros(p_real.shape[0], device=p_real.device)

        return prop_tokens, p_field, helmholtz_residual

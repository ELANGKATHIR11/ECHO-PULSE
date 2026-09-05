"""
BEATs Transformer Acoustic Encoder for EchoPulseNet
Bidirectional Encoder representation from Audio Transformers (BEATs)
Self-supervised acoustic representation learning with discrete acoustic tokenization
for underwater hydrophone signal understanding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class BEATsPatchEmbedding(nn.Module):
    """
    Patch extraction module for 2D acoustic time-frequency spectrograms (F x T),
    conforming to BEATs standard 16x16 / 8x8 patch projection into d_model.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 256, patch_size: Tuple[int, int] = (16, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, Freq, Time)
        x = self.proj(x)                 # (B, embed_dim, F', T')
        x = x.flatten(2).transpose(1, 2) # (B, N_patches, embed_dim)
        x = self.norm(x)
        return x


class BEATsAcousticTransformerEncoder(nn.Module):
    """
    BEATs-style Self-Supervised Acoustic Transformer Encoder:
    - Discrete acoustic patch embedding on complex spectrogram representation
    - Multi-head Self-Attention Transformer encoder with pre-LN
    - Acoustic CLS summary token extraction
    - Direct projection into OCEAN-PHYSNet-X latent token sequence
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_patches: int = 512,
        n_fft: int = 512,
        hop_length: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 1. 2D Spectrogram Patch Embedding
        self.patch_embed = BEATsPatchEmbedding(in_channels=1, embed_dim=d_model, patch_size=(16, 8))

        # 2. Learnable Tokens & Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches + 1, d_model) * 0.02)
        self.pos_drop = nn.Dropout(p=0.1)

        # 3. Transformer Encoder Blocks (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 4. Fusion Layer to blend with raw physical temporal features
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.stem_proj = nn.Linear(d_model, d_model)

    def forward(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_raw: (B, 1, L) Raw hydrophone audio waveform.
        Returns:
            acoustic_tokens: (B, T, d_model) High-level acoustic sequence for OCEAN-PHYSNet cross-attention
            cls_token: (B, d_model) Global soundscape summary embedding
        """
        B, _, L = x_raw.shape
        device = x_raw.device

        # 1. Compute time-frequency magnitude spectrogram
        window = torch.hann_window(self.n_fft, device=device)
        stft_complex = torch.stft(
            x_raw.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        mag_spec = torch.abs(stft_complex).unsqueeze(1) # (B, 1, Freq, Time)
        log_spec = torch.log1p(mag_spec)

        # Ensure minimal dimensions for patch projection (Freq >= 16, Time >= 8)
        F_dim, T_dim = log_spec.shape[2], log_spec.shape[3]
        if F_dim < 16 or T_dim < 8:
            log_spec = F.interpolate(log_spec, size=(max(16, F_dim), max(8, T_dim)), mode='bilinear', align_corners=False)

        # 2. BEATs Patch Projection
        patches = self.patch_embed(log_spec) # (B, N, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1) # (B, N+1, d_model)

        # Add learned positional encoding
        num_tokens = x.shape[1]
        if num_tokens > self.pos_embed.shape[1]:
            # Interpolate pos embedding if sequence exceeds pre-allocated size
            pos_emb = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=num_tokens,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embed[:, :num_tokens, :]

        x = self.pos_drop(x + pos_emb)

        # 3. Transformer Self-Attention Blocks
        x = self.blocks(x)
        x = self.norm(x)

        cls_summary = x[:, 0]       # (B, d_model)
        patch_tokens = x[:, 1:]     # (B, N, d_model)

        # 4. Hybridize with fast 1D temporal convolutions
        t_feat = self.temporal_stem(x_raw).transpose(1, 2) # (B, T_temp, d_model)
        t_feat = self.stem_proj(t_feat)

        # Align lengths and residual blend
        min_len = min(patch_tokens.shape[1], t_feat.shape[1])
        fused_tokens = patch_tokens[:, :min_len, :] + t_feat[:, :min_len, :]

        return fused_tokens, cls_summary

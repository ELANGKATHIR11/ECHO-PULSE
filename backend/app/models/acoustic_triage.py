"""
Acoustic-Triage-Transformer-X: Fast Hierarchical Acoustic Classification Model
EchoPulseNet Marine Sonar Intelligence Platform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


class AcousticTriageTransformerX(nn.Module):
    """
    Acoustic-Triage-Transformer-X:
    Ultra-low-latency (<2ms) hierarchical acoustic classifier for early alert triage.
    Hierarchy:
      - Level 1: Macro Domain (Biophonic, Anthropogenic, Geophonic, Tactical Intruder)
      - Level 2: Threat Severity (Critical, Warning, Normal, Background)
      - Level 3: 17 Fine-grained Marine Acoustic Subclasses
    """

    MACRO_CLASSES = ["Biophonic", "Anthropogenic", "Geophonic", "Tactical Intruder"]
    SEVERITY_LEVELS = ["BACKGROUND", "NORMAL", "WARNING", "CRITICAL"]

    def __init__(self, in_features: int = 128, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        # Multi-scale 1D Convolutional Front-End for Spectrogram Frames
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_features, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # Fast Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling & Classification Heads
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 1. Macro Domain Head
        self.macro_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, len(self.MACRO_CLASSES))
        )

        # 2. Threat Severity Head
        self.severity_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, len(self.SEVERITY_LEVELS))
        )

        # 3. Subclass Head
        self.subclass_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 17)
        )

        # 4. Out-of-Distribution / Novelty Uncertainty Head
        self.ood_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, spec_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        spec_frames: (B, in_features, time_steps) or (B, time_steps, in_features)
        """
        if spec_frames.dim() == 2:
            spec_frames = spec_frames.unsqueeze(1)

        if spec_frames.shape[1] != self.d_model and spec_frames.shape[-1] == 128:
            spec_frames = spec_frames.transpose(1, 2)

        x = self.patch_embed(spec_frames) # (B, d_model, T)
        x_trans = x.transpose(1, 2)       # (B, T, d_model)
        feat_seq = self.transformer(x_trans) # (B, T, d_model)
        feat = feat_seq.transpose(1, 2)
        pooled = self.pool(feat).squeeze(-1) # (B, d_model)

        macro_logits = self.macro_head(pooled)
        severity_logits = self.severity_head(pooled)
        subclass_logits = self.subclass_head(pooled)
        ood_score = self.ood_head(pooled)

        macro_probs = F.softmax(macro_logits, dim=-1)
        severity_probs = F.softmax(severity_logits, dim=-1)
        subclass_probs = F.softmax(subclass_logits, dim=-1)

        return {
            "macro_probs": macro_probs,
            "severity_probs": severity_probs,
            "subclass_probs": subclass_probs,
            "ood_uncertainty": ood_score,
            "features": pooled
        }

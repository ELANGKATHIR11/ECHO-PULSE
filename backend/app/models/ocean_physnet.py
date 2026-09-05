"""
OCEAN-PHYSNet: Ocean-Conditioned Physics-Constrained Multimodal Acoustic Intelligence Network
EchoPulseNet Marine Sonar Intelligence Platform
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from ..sonar.ocean_state import OceanStateEngine
from ..sonar.ocean_phys_encoders import (
    ComplexHydrophoneSpectralEncoder,
    ComplexAVSSpatialEncoder,
    MultipathTokenizer,
    OceanStateProjector
)
from ..sonar.beats_encoder import BEATsAcousticTransformerEncoder
from ..sonar.physics_attention import (
    PhysicsBiasedCrossAttention,
    FourierNeuralOperatorPropagationBlock
)


class OCEANPhysNet(nn.Module):
    """
    OCEAN-PHYSNet Master Architecture:
    - BEATs Acoustic Transformer Encoder: Self-supervised discrete acoustic tokenization
    - Ocean State Conditioning [T(z), S(z), P(z), c(z), alpha(f,z), H, B]
    - Complex Multi-Representation Hydrophone & AVS Encoders
    - Multipath Hypothesis Separation
    - Physics-Biased Cross-Attention Engine
    - Differentiable Fourier Neural Operator (FNO) Helmholtz Wave Propagation
    - Periodic Trigonometric DOA, Heteroscedastic Range, and Mahalanobis OOD Anomaly Detection
    """

    CATEGORIES = ["Biophonic", "Anthropogenic", "Geophonic", "Tactical Intruder"]
    NUM_CLASSES = 4
    NUM_SUBCLASSES = 17

    def __init__(self, d_model: int = 256, num_heads: int = 8, num_paths: int = 4, use_beats: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_beats = use_beats

        # 1. Encoders: BEATs Transformer Acoustic Encoder + AVS Spatial + Ocean Projector
        if use_beats:
            self.hydro_encoder = BEATsAcousticTransformerEncoder(d_model=d_model, num_heads=num_heads, num_layers=4)
        else:
            self.hydro_encoder = ComplexHydrophoneSpectralEncoder(d_model=d_model)

        self.avs_encoder = ComplexAVSSpatialEncoder(d_model=d_model)
        self.ocean_projector = OceanStateProjector(ocean_dim=16, d_model=d_model, num_tokens=4)
        self.multipath_tokenizer = MultipathTokenizer(d_model=d_model, num_paths=num_paths)

        # 2. Physics Cross-Attention & FNO Blocks
        self.cross_attention = PhysicsBiasedCrossAttention(d_model=d_model, num_heads=num_heads)
        self.fno_propagation = FourierNeuralOperatorPropagationBlock(d_model=d_model, modes=16)

        # 3. Task Heads
        # A. Classification Head (Multi-label Sigmoid)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.NUM_CLASSES)
        )

        # Subclass Head
        self.subclass_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, self.NUM_SUBCLASSES)
        )

        # B. Periodic Trigonometric DOA Head: [sin(theta), cos(theta), sin(phi), cos(phi), log_sigma_theta^2]
        self.doa_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 5)
        )

        # C. Heteroscedastic Range Head: [range_norm, log_sigma_R^2]
        self.range_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

        # D. Latent Embedding for Mahalanobis OOD Anomaly Detection
        self.embedding_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64)
        )

        # Register running mean and covariance for Mahalanobis OOD
        self.register_buffer("ood_mean", torch.zeros(64))
        self.register_buffer("ood_var", torch.ones(64))

    def forward(self, x_hydro: torch.Tensor, avs_4ch: torch.Tensor, 
                ocean_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_hydro: (B, 1, L) Hydrophone raw waveform
        avs_4ch: (B, 4, L) AVS [P, Ux, Uy, Uz]
        ocean_state: (B, 16) Ocean parameter vector Eo
        """
        B = x_hydro.shape[0]

        # 1. Multi-Branch Encoders
        if self.use_beats:
            hydro_tokens, beats_cls = self.hydro_encoder(x_hydro) # (B, T_h, d_model), (B, d_model)
        else:
            hydro_tokens = self.hydro_encoder(x_hydro) # (B, T_h, d_model)
            beats_cls = None

        avs_tokens, intensity_3d = self.avs_encoder(avs_4ch) # (B, T_a, d_model), (B, 3)
        ocean_tokens = self.ocean_projector(ocean_state) # (B, 4, d_model)

        # 2. Multipath direct/reflected separation
        multipath_tokens, path_weights = self.multipath_tokenizer(hydro_tokens) # (B, num_paths, d_model)

        # 3. Concatenate multimodal key-value memory: [AVS + Ocean + Multipath]
        kv_memory = torch.cat([avs_tokens, ocean_tokens, multipath_tokens], dim=1) # (B, N_kv, d_model)

        # 4. Physics-Biased Cross-Attention
        attended_tokens, attn_weights = self.cross_attention(hydro_tokens, kv_memory, ocean_state)

        # 5. Differentiable FNO Helmholtz Wave Field Propagation
        prop_tokens, p_field, helmholtz_res = self.fno_propagation(attended_tokens, ocean_state)

        # Global average pooling over sequence tokens
        pooled = torch.mean(prop_tokens, dim=1) # (B, d_model)

        # 6. Multi-Task Output Predictions
        # A. Classification
        class_logits = self.classifier_head(pooled) # (B, 4)
        class_probs = torch.sigmoid(class_logits)
        subclass_logits = self.subclass_head(pooled) # (B, 17)
        subclass_probs = F.softmax(subclass_logits, dim=-1)

        # B. Periodic DOA Distribution
        doa_raw = self.doa_head(pooled) # (B, 5)
        sin_theta = torch.tanh(doa_raw[:, 0])
        cos_theta = torch.tanh(doa_raw[:, 1])
        sin_phi = torch.tanh(doa_raw[:, 2])
        cos_phi = torch.tanh(doa_raw[:, 3])
        log_sigma_theta_sq = doa_raw[:, 4]

        # Reconstruct Azimuth and Elevation in radians and degrees
        theta_rad = torch.atan2(sin_theta, cos_theta) # [-pi, pi]
        theta_deg = (torch.rad2deg(theta_rad) + 360.0) % 360.0 # [0, 360)
        phi_rad = torch.atan2(sin_phi, cos_phi)
        phi_deg = torch.clamp(torch.rad2deg(phi_rad), -90.0, 90.0)
        sigma_theta_deg = torch.exp(0.5 * log_sigma_theta_sq) * 5.0 # Degree scale uncertainty

        # C. Heteroscedastic Range
        range_raw = self.range_head(pooled) # (B, 2)
        norm_range = F.softplus(range_raw[:, 0]) # Positive range
        range_meters = norm_range * 3000.0 # Scale to meters
        log_sigma_r_sq = range_raw[:, 1]
        sigma_r_meters = torch.exp(0.5 * log_sigma_r_sq) * 200.0

        # D. Latent Embedding & Mahalanobis Distance for OOD Detection
        latent_z = self.embedding_head(pooled) # (B, 64)
        mahalanobis_dist = torch.sqrt(torch.sum(((latent_z - self.ood_mean) ** 2) / (self.ood_var + 1e-6), dim=-1))
        is_novel_event = mahalanobis_dist > 3.5 # 3.5-sigma threshold

        return {
            "class_logits": class_logits,
            "class_probs": class_probs,
            "subclass_logits": subclass_logits,
            "subclass_probs": subclass_probs,
            "azimuth_deg": theta_deg,
            "elevation_deg": phi_deg,
            "sigma_theta_deg": sigma_theta_deg,
            "range_meters": range_meters,
            "sigma_range_meters": sigma_r_meters,
            "intensity_3d": intensity_3d,
            "helmholtz_residual": helmholtz_res,
            "mahalanobis_ood_distance": mahalanobis_dist,
            "is_novel_event": is_novel_event,
            "beats_cls_embedding": beats_cls,
            "pressure_field": p_field,
            "attention_weights": attn_weights,
            "multipath_weights": path_weights,
            "latent_embedding": latent_z
        }

    def compute_physics_loss(self, preds: Dict[str, torch.Tensor], 
                             targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluates the combined Multi-Task Physics-Constrained Loss:
        L = L_task + lambda1*L_DOA + lambda2*L_range + lambda3*L_wave + lambda4*L_AVS
        """
        # 1. Multi-Label Classification Loss
        if "labels" in targets:
            if "class_logits" in preds:
                loss_task = F.binary_cross_entropy_with_logits(preds["class_logits"], targets["labels"].float())
            else:
                loss_task = F.binary_cross_entropy(preds["class_probs"].float(), targets["labels"].float())
        else:
            loss_task = torch.tensor(0.0, device=preds["azimuth_deg"].device)

        # 2. Discontinuity-Free Periodic Trigonometric DOA Loss
        if "true_azimuth_deg" in targets:
            true_theta_rad = torch.deg2rad(targets["true_azimuth_deg"])
            pred_theta_rad = torch.deg2rad(preds["azimuth_deg"])
            loss_doa = torch.mean(
                (torch.sin(pred_theta_rad) - torch.sin(true_theta_rad)) ** 2 +
                (torch.cos(pred_theta_rad) - torch.cos(true_theta_rad)) ** 2
            )
        else:
            loss_doa = torch.tensor(0.0, device=preds["azimuth_deg"].device)

        # 3. Heteroscedastic Range Loss
        if "true_range_m" in targets:
            true_r = targets["true_range_m"]
            pred_r = preds["range_meters"]
            sigma_r = preds["sigma_range_meters"]
            loss_range = torch.mean(((true_r - pred_r) ** 2) / (2 * (sigma_r ** 2) + 1e-6) + torch.log(sigma_r + 1e-6))
        else:
            loss_range = torch.tensor(0.0, device=preds["azimuth_deg"].device)

        # 4. Helmholtz Wave Equation Residual Penalty
        loss_wave = torch.mean(preds["helmholtz_residual"])

        # Total Loss
        total_loss = loss_task + 1.2 * loss_doa + 0.8 * loss_range + 0.5 * loss_wave

        return {
            "total_loss": total_loss,
            "loss_task": loss_task,
            "loss_doa": loss_doa,
            "loss_range": loss_range,
            "loss_wave": loss_wave
        }

"""
EchoPhys-X: Physics Conditioning & Acoustic Proxies
====================================================
Rigorous implementation separating heuristic acoustic proxies (for single-frequency
imagery without environmental sensors) from real ocean physics calculations (when
T, S, P, frequency, altitude, or range metadata are provided).
"""

import math
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_acoustic_proxy_tensor(im_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes a 5-channel acoustic tensor proxy for single-frequency SSS imagery:
      Channel 0: Raw Calibrated Acoustic Backscatter I (normalized to [0, 1])
      Channel 1: Low-Frequency (LF) Substrate Reverberation Proxy (AvgPool 9x9)
      Channel 2: High-Frequency (HF) Specular Highlight Residual Proxy (I - LF + 0.5)
      Channel 3: Local Texture Contrast Proxy (|I - LF_coarse| * 3.0, clamped)
      Channel 4: Normalized Cross-Track Range Coordinate Proxy (linear across swath)

    Input:
      im_tensor: (B, 1, H, W) in range [0.0, 1.0]
    Output:
      (B, 5, H, W) float tensor
    """
    B, C, H, W = im_tensor.shape
    device = im_tensor.device
    dtype = im_tensor.dtype

    # 1. LF proxy: Smooth background reverberation
    lf = F.avg_pool2d(im_tensor, kernel_size=9, stride=1, padding=4)

    # 2. HF proxy: High-pass detail / specular highlights
    hf = torch.clamp(im_tensor - lf + 0.5, 0.0, 1.0)

    # 3. Local contrast proxy: multi-scale variation
    lf_coarse = F.avg_pool2d(im_tensor, kernel_size=17, stride=1, padding=8)
    local_contrast = torch.clamp(torch.abs(im_tensor - lf_coarse) * 3.0, 0.0, 1.0)

    # 4. Normalized cross-track range proxy: (0.0 to 1.0 across scan lines)
    range_coord = (
        torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        .view(1, 1, 1, W)
        .expand(B, 1, H, W)
    )

    return torch.cat([im_tensor, lf, hf, local_contrast, range_coord], dim=1)


def compute_mackenzie_sound_speed(
    temp_c: float,
    salinity_ppt: float,
    depth_m: float
) -> float:
    """
    Mackenzie (1981) Nine-term Sound Speed Formula in seawater:
    c(T, S, D) = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
                 + 1.340*(S - 35) + 1.630e-2*D + 1.675e-7*D^2
                 - 1.025e-2*T*(S - 35) - 7.139e-13*T*D^3
    Valid for: 0 <= T <= 30 deg C, 25 <= S <= 40 ppt, 0 <= D <= 8000 m.
    """
    T = temp_c
    S = salinity_ppt
    D = depth_m

    c = (
        1448.96
        + 4.591 * T
        - 5.304e-2 * (T ** 2)
        + 2.374e-4 * (T ** 3)
        + 1.340 * (S - 35.0)
        + 1.630e-2 * D
        + 1.675e-7 * (D ** 2)
        - 1.025e-2 * T * (S - 35.0)
        - 7.139e-13 * T * (D ** 3)
    )
    return float(c)


def compute_ainslie_mccolm_attenuation(
    freq_khz: float,
    temp_c: float = 10.0,
    salinity_ppt: float = 35.0,
    depth_m: float = 100.0,
    ph: float = 8.0
) -> float:
    """
    Ainslie and McColm (1998) simplified sound absorption in seawater alpha (dB/km).
    Includes Boric acid relaxation, Magnesium sulfate relaxation, and pure water viscosity.
    """
    f = freq_khz
    T = temp_c
    D = depth_m / 1000.0  # depth in km
    S = salinity_ppt

    # Relaxation frequencies (kHz)
    f1 = 0.78 * math.sqrt(S / 35.0) * math.exp(T / 26.0)  # Boric acid
    f2 = 42.0 * math.exp(T / 17.0)                         # MgSO4

    # Boric acid contribution
    a1 = 0.106 * (f1 * f ** 2) / (f ** 2 + f1 ** 2) * math.exp((ph - 8.0) / 0.56)
    # MgSO4 contribution
    a2 = 0.52 * (1.0 + T / 43.0) * (S / 35.0) * (f2 * f ** 2) / (f ** 2 + f2 ** 2) * math.exp(-D / 6.0)
    # Pure water viscosity
    a3 = 0.00049 * (f ** 2) * math.exp(-(T / 27.0 + D / 17.0))

    alpha_db_per_km = a1 + a2 + a3
    return float(alpha_db_per_km)


def make_physical_conditioning_tensor(
    im_tensor: torch.Tensor,
    temp_c: Optional[float] = None,
    salinity_ppt: Optional[float] = None,
    depth_m: Optional[float] = None,
    freq_khz: Optional[float] = None,
    altitude_m: Optional[float] = None,
    max_range_m: Optional[float] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Constructs physical conditioning tensor when genuine oceanographic metadata exists.
    If physical quantities are unavailable, marks them explicitly as UNKNOWN and falls
    back to proxy values while reporting metadata provenance.
    """
    B, _, H, W = im_tensor.shape
    device = im_tensor.device
    dtype = im_tensor.dtype

    metadata_status = {
        "temperature_c": temp_c if temp_c is not None else "UNKNOWN",
        "salinity_ppt": salinity_ppt if salinity_ppt is not None else "UNKNOWN",
        "depth_m": depth_m if depth_m is not None else "UNKNOWN",
        "freq_khz": freq_khz if freq_khz is not None else "UNKNOWN",
        "altitude_m": altitude_m if altitude_m is not None else "UNKNOWN",
        "max_range_m": max_range_m if max_range_m is not None else "UNKNOWN",
        "mode": "Physical_Conditioning" if (temp_c is not None and depth_m is not None) else "Proxy_Acoustic_Fallback"
    }

    # Base 5-channel proxies
    base_proxies = make_acoustic_proxy_tensor(im_tensor)

    # Derived physics fields
    t_val = temp_c if temp_c is not None else 10.0
    s_val = salinity_ppt if salinity_ppt is not None else 35.0
    d_val = depth_m if depth_m is not None else 50.0
    f_val = freq_khz if freq_khz is not None else 450.0
    alt_val = altitude_m if altitude_m is not None else 15.0
    r_max = max_range_m if max_range_m is not None else 150.0

    # 1. Mackenzie sound speed field
    c_ocean = compute_mackenzie_sound_speed(t_val, s_val, d_val)
    c_norm = torch.full((B, 1, H, W), float(c_ocean / 1600.0), device=device, dtype=dtype)

    # 2. Transmission Loss TL(r) = 20*log10(r) + alpha*r (normalized to [0, 1])
    alpha_db_km = compute_ainslie_mccolm_attenuation(f_val, t_val, s_val, d_val)
    alpha_per_m = alpha_db_km / 1000.0
    r_norm = torch.linspace(0.05, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    r_m = r_norm * r_max
    tl = (20.0 * torch.log10(torch.clamp(r_m, min=1.0)) + alpha_per_m * r_m) / 80.0
    tl_norm = torch.clamp(tl, 0.0, 1.0)

    # 3. Grazing angle field gamma(r, alt) = atan2(altitude, ground_range)
    grazing = torch.atan(alt_val / torch.clamp(r_m, min=1.0)) / (math.pi / 2.0)

    physics_tensor = torch.cat([base_proxies, c_norm, tl_norm, grazing], dim=1)
    return physics_tensor, metadata_status

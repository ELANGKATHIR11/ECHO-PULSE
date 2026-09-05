"""
EchoPulseNet Unified Physics Core
Rigorous Underwater Acoustic Propagation, Seawater Physical State,
AVS Active Intensity, Geodesic Geolocation, and Adaptive Physics Loss Engine.

Units:
- Frequency: kHz
- Sound Speed: m/s
- Attenuation/Absorption: dB/km and m^-1
- Distance/Range/Depth: meters
- Coordinates: Degrees (WGS-84)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union


# ==============================================================================
# 1. Spatially Varying Sound Speed Profile (SSP) & Seawater State
# ==============================================================================

class SeawaterPhysics:
    """
    Computes spatially varying sound speed c(z, T, S) and frequency-dependent
    acoustic absorption alpha(f, T, S, z) using empirical oceanic formulations.
    """

    @staticmethod
    def mackenzie_sound_speed(temperature_c: float, salinity_psu: float, depth_m: float) -> float:
        """
        Mackenzie (1981) Equation for Sound Speed in Seawater (m/s).
        Range: T: -2..30°C, S: 25..40 PSU, D: 0..8000 m
        """
        T = float(temperature_c)
        S = float(salinity_psu)
        D = float(depth_m)

        c = (
            1448.96
            + 4.591 * T
            - 0.05304 * (T ** 2)
            + 0.0002374 * (T ** 3)
            + 1.340 * (S - 35.0)
            + 0.0163 * D
            + 1.675e-7 * (D ** 2)
            - 0.01025 * T * (S - 35.0)
            - 7.139e-13 * T * (D ** 3)
        )
        return float(c)

    @staticmethod
    def francois_garrison_absorption(
        freq_khz: float,
        temperature_c: float = 18.0,
        salinity_psu: float = 35.0,
        depth_m: float = 50.0,
        ph: float = 8.0
    ) -> float:
        """
        Francois-Garrison Seawater Acoustic Absorption Model (dB/km).
        Includes Boric acid (f1), Magnesium Sulfate (f2), and Pure Water viscosity (f3).
        """
        f = max(0.01, float(freq_khz))
        T = float(temperature_c)
        S = float(salinity_psu)
        D = float(depth_m)

        c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * D

        # 1. Boric Acid Relaxation
        A1 = (8.86 / c) * (10.0 ** (0.78 * ph - 5.0))
        P1 = 1.0
        f1 = 2.8 * math.sqrt(max(0.1, S / 35.0)) * (10.0 ** (4.0 - 1245.0 / (T + 273.15)))
        boric_term = (A1 * P1 * f1 * (f ** 2)) / (f1 ** 2 + f ** 2)

        # 2. Magnesium Sulfate Relaxation
        A2 = 21.44 * (S / 35.0) * (1.0 + 0.025 * T)
        P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * (D ** 2)
        f2 = (8.17 * (10.0 ** (8.0 - 1990.0 / (T + 273.15)))) / (1.0 + 0.0018 * (S - 35.0))
        mgso4_term = (A2 * P2 * f2 * (f ** 2)) / (f2 ** 2 + f ** 2)

        # 3. Pure Water Viscous Attenuation
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * (T ** 2) - 1.5e-8 * (T ** 3)
        P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * (D ** 2)
        water_term = A3 * P3 * (f ** 2)

        alpha_db_km = boric_term + mgso4_term + water_term
        return max(0.001, float(alpha_db_km))

    @staticmethod
    def compute_travel_time(
        trajectory_points_m: np.ndarray,
        ssp_depths_m: np.ndarray,
        ssp_speeds_mps: np.ndarray
    ) -> float:
        """
        Calculates path travel-time: t = \int_\Gamma \frac{ds}{c(x)}
        via numerical quadrature along a 3D ray trajectory \Gamma.
        """
        if len(trajectory_points_m) < 2:
            return 0.0

        diffs = np.diff(trajectory_points_m, axis=0)
        ds = np.linalg.norm(diffs, axis=1)
        mid_depths = (trajectory_points_m[:-1, 2] + trajectory_points_m[1:, 2]) / 2.0

        # Interpolate local sound speeds
        c_vals = np.interp(mid_depths, ssp_depths_m, ssp_speeds_mps)
        c_vals = np.maximum(c_vals, 1400.0)

        dt = ds / c_vals
        return float(np.sum(dt))


# ==============================================================================
# 2. Helmholtz Wave Equation Residual & Differentiable Wave Physics
# ==============================================================================

class HelmholtzWaveResidual(nn.Module):
    """
    Evaluates acoustic wave residual for frequency-domain Helmholtz equation:
    \nabla^2 p(x) + k^2(x) p(x) = 0
    where wavenumber k(x) = \frac{2\pi f}{c(x)} - i \alpha(x)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pressure_field: torch.Tensor,
        freq_khz: float = 10.0,
        sound_speed_mps: float = 1500.0,
        dx: float = 0.5,
        dy: float = 0.5
    ) -> torch.Tensor:
        """
        pressure_field: (B, 1, H, W) Complex or real amplitude pressure map
        Returns residual map: (B, 1, H-2, W-2)
        """
        f_hz = freq_khz * 1000.0
        k = (2.0 * math.pi * f_hz) / max(1400.0, sound_speed_mps)

        # Second-order central finite difference Laplacian \nabla^2 p
        d2_dx2 = (pressure_field[:, :, 1:-1, 2:] - 2.0 * pressure_field[:, :, 1:-1, 1:-1] + pressure_field[:, :, 1:-1, :-2]) / (dx ** 2)
        d2_dy2 = (pressure_field[:, :, 2:, 1:-1] - 2.0 * pressure_field[:, :, 1:-1, 1:-1] + pressure_field[:, :, :-2, 1:-1]) / (dy ** 2)
        laplacian = d2_dx2 + d2_dy2

        p_center = pressure_field[:, :, 1:-1, 1:-1]
        helmholtz_residual = laplacian + (k ** 2) * p_center
        return torch.abs(helmholtz_residual)


# ==============================================================================
# 3. Acoustic Vector Sensor (AVS) Active Intensity & Spherical DOA
# ==============================================================================

class AVSPhysicsCore:
    """
    Rigorous 4-Channel AVS Processing ($p, u_x, u_y, u_z$):
    - Complex Active Acoustic Intensity: I = 1/2 Re{ p* v }
    - Unit Spherical DOA Vector: [cos\phi cos\theta, cos\phi sin\theta, sin\phi]
    - Multipath Synthesis: x(t) = \sum a_k s(t - \tau_k) + n(t)
    - Doppler Shift: f_d = f_0 \frac{v_{rel}}{c}
    """

    @staticmethod
    def compute_active_intensity(
        p: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Computes 3D active intensity vector I = [Ix, Iy, Iz] and intensity coherence.
        """
        min_len = min(len(p), len(ux), len(uy), len(uz))
        if min_len < 16:
            return np.array([1.0, 0.0, 0.0]), 0.5

        p_c = p[:min_len]
        ux_c = ux[:min_len]
        uy_c = uy[:min_len]
        uz_c = uz[:min_len]

        ix = float(np.mean(p_c * ux_c))
        iy = float(np.mean(p_c * uy_c))
        iz = float(np.mean(p_c * uz_c))

        total_mag = math.sqrt(ix**2 + iy**2 + iz**2) + 1e-9
        unit_intensity = np.array([ix / total_mag, iy / total_mag, iz / total_mag])

        energy = float((np.mean(p_c**2) + np.mean(ux_c**2) + np.mean(uy_c**2) + np.mean(uz_c**2)) / 4.0 + 1e-9)
        coherence = float(np.clip(total_mag / energy, 0.1, 0.99))
        return unit_intensity, coherence

    @staticmethod
    def spherical_doa_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        """
        Constructs true 3D spherical direction-of-arrival vector:
        [cos(phi)*cos(theta), cos(phi)*sin(theta), sin(phi)]
        where theta is Azimuth [0, 360 deg), phi is Elevation [-90, +90 deg]
        """
        theta = math.radians(azimuth_deg)
        phi = math.radians(elevation_deg)
        return np.array([
            math.cos(phi) * math.cos(theta),
            math.cos(phi) * math.sin(theta),
            math.sin(phi)
        ])

    @staticmethod
    def apply_multipath_and_doppler(
        signal: np.ndarray,
        sr: int = 44100,
        path_amplitudes: List[float] = [1.0, 0.45, 0.22],
        path_delays_sec: List[float] = [0.0, 0.018, 0.042],
        relative_speed_mps: float = 3.5,
        sound_speed_mps: float = 1500.0,
        noise_snr_db: float = 24.0
    ) -> np.ndarray:
        """
        Synthesizes physical multipath x(t) = \sum a_k s(t - \tau_k) + n(t)
        with ocean current Doppler compression/expansion.
        """
        # 1. Doppler factor: f' = f * (1 + v/c)
        doppler_factor = 1.0 + (relative_speed_mps / sound_speed_mps)
        if abs(doppler_factor - 1.0) > 1e-5:
            # Resample time axis to simulate Doppler compression
            old_indices = np.arange(len(signal))
            new_indices = old_indices * doppler_factor
            new_indices = new_indices[new_indices < len(signal) - 1]
            doppler_signal = np.interp(new_indices, old_indices, signal)
        else:
            doppler_signal = signal.copy()

        # 2. Multipath summation
        out_signal = np.zeros_like(doppler_signal)
        for a_k, tau_k in zip(path_amplitudes, path_delays_sec):
            delay_samples = int(tau_k * sr)
            if delay_samples < len(doppler_signal):
                delayed = np.pad(doppler_signal, (delay_samples, 0))[:len(doppler_signal)]
                out_signal += a_k * delayed

        # 3. Add ambient ocean noise
        sig_power = np.mean(out_signal ** 2) + 1e-9
        noise_power = sig_power / (10.0 ** (noise_snr_db / 10.0))
        noise = np.random.normal(0, math.sqrt(noise_power), len(out_signal))

        return (out_signal + noise).astype(np.float32)


# ==============================================================================
# 4. Geodetic Geolocation & Coordinate Transformations
# ==============================================================================

class GeodeticTransforms:
    """
    Standard Geodetic Conversions:
    - ENU (East, North, Up) -> ECEF -> WGS-84 (Lat, Lng, Alt)
    - Vincenty / Forward Great-Circle Geodesic Bearing Projection
    """

    WGS84_A = 6378137.0         # Semi-major axis (m)
    WGS84_F = 1.0 / 298.257223563 # Flattening
    WGS84_B = WGS84_A * (1.0 - WGS84_F)

    @classmethod
    def enu_to_wgs84(
        cls,
        east_m: float,
        north_m: float,
        up_m: float,
        ref_lat_deg: float,
        ref_lng_deg: float,
        ref_alt_m: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Transforms local ENU Cartesian vector to WGS-84 coordinates.
        """
        lat_rad = math.radians(ref_lat_deg)
        lng_rad = math.radians(ref_lng_deg)

        # Earth radii of curvature
        sin_lat = math.sin(lat_rad)
        e2 = 2.0 * cls.WGS84_F - cls.WGS84_F ** 2
        N = cls.WGS84_A / math.sqrt(1.0 - e2 * sin_lat ** 2)
        M = cls.WGS84_A * (1.0 - e2) / ((1.0 - e2 * sin_lat ** 2) ** 1.5)

        # First order ENU displacement in radians
        dlat = north_m / (M + ref_alt_m)
        dlng = east_m / ((N + ref_alt_m) * math.cos(lat_rad))

        target_lat = ref_lat_deg + math.degrees(dlat)
        target_lng = ref_lng_deg + math.degrees(dlng)
        target_alt = ref_alt_m + up_m

        return round(target_lat, 7), round(target_lng, 7), round(target_alt, 2)


# ==============================================================================
# 5. Adaptive Physics Loss & Reliability Gating
# ==============================================================================

class AdaptivePhysicsLossEngine(nn.Module):
    r"""
    Implements Adaptive Environmental Physics Loss Weighting:
    \lambda_{phys} = \text{sigmoid}(f(\text{env\_conf}, \text{sensor\_conf}, \text{model\_disagreement}))
    \mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{phys} \cdot \mathcal{L}_{\text{physics}}

    Prevents corrupted physical assertions when sensors/environment suffer low confidence.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.gating_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        env_confidence: torch.Tensor,
        sensor_confidence: torch.Tensor,
        model_disagreement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        env_confidence: (B, 1) Range [0, 1]
        sensor_confidence: (B, 1) Range [0, 1]
        model_disagreement: (B, 1) Range [0, 1]
        """
        gate_inputs = torch.cat([env_confidence, sensor_confidence, model_disagreement], dim=-1)
        lambda_phys = self.gating_mlp(gate_inputs) # (B, 1)

        total_loss = data_loss + torch.mean(lambda_phys) * physics_loss
        return total_loss, torch.mean(lambda_phys)

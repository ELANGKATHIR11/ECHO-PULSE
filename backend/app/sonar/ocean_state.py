"""
Ocean State & Seawater Physics Environment Engine
OCEAN-PHYSNet - Marine Sonar Intelligence Platform
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class OceanStateEngine:
    """
    Computes rigorous seawater physical propagation parameters:
    - Mackenzie empirical Sound Speed Profile c(T, S, D)
    - Francois-Garrison frequency-dependent underwater acoustic absorption alpha(f, T, S, D)
    - Sound Speed Profile (SSP) depth gradient dc/dz
    - Ocean state parameter vector Eo = [T(z), S(z), P(z), c(z), alpha(f,z), H, B]
    """

    @staticmethod
    def mackenzie_sound_speed(temperature_c: float, salinity_psu: float, depth_m: float) -> float:
        """
        Mackenzie (1981) Equation for Sound Speed in Seawater.
        Valid for:
            Temperature: -2 to 30 deg C
            Salinity: 25 to 40 PSU
            Depth: 0 to 8000 m
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
    def francois_garrison_absorption(freq_khz: float, temperature_c: float, salinity_psu: float, depth_m: float, ph: float = 8.0) -> float:
        """
        Francois-Garrison Seawater Acoustic Absorption Model (dB/km).
        Includes Boric acid (f1), Magnesium Sulfate (f2), and Pure Water viscosity (f3) relaxation terms.
        """
        f = max(0.01, float(freq_khz))
        T = float(temperature_c)
        S = float(salinity_psu)
        D = float(depth_m)

        # Sound speed
        c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * D

        # 1. Boric Acid Relaxation (f < 1 kHz)
        A1 = (8.86 / c) * (10.0 ** (0.78 * ph - 5.0))
        P1 = 1.0
        f1 = 2.8 * math.sqrt(max(0.1, S / 35.0)) * (10.0 ** (4.0 - 1245.0 / (T + 273.15)))
        boric_term = (A1 * P1 * f1 * (f ** 2)) / (f1 ** 2 + f ** 2)

        # 2. Magnesium Sulfate Relaxation (f ~ 10-100 kHz)
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

    @classmethod
    def compute_sound_speed_profile(cls, surface_temp_c: float = 26.5, bottom_temp_c: float = 14.0,
                                   salinity_psu: float = 34.8, max_depth_m: float = 500.0,
                                   num_points: int = 50) -> Dict[str, Any]:
        """
        Generates realistic layered Sound Speed Profile (SSP) across thermocline and deep ocean layer.
        """
        depths = np.linspace(0.0, max_depth_m, num_points).tolist()
        temps = []
        sound_speeds = []
        absorptions_1khz = []
        absorptions_10khz = []

        # Mixed layer depth ~40m, thermocline ~40-200m, deep layer >200m
        for d in depths:
            if d < 40.0:
                # Isothermal surface mixed layer
                temp = surface_temp_c
            elif d < 200.0:
                # Thermocline steep gradient
                frac = (d - 40.0) / 160.0
                temp = surface_temp_c - frac * (surface_temp_c - bottom_temp_c)
            else:
                # Deep cold isothermal layer
                temp = bottom_temp_c - 0.005 * (d - 200.0)

            c = cls.mackenzie_sound_speed(temp, salinity_psu, d)
            a1 = cls.francois_garrison_absorption(1.0, temp, salinity_psu, d)
            a10 = cls.francois_garrison_absorption(10.0, temp, salinity_psu, d)

            temps.append(round(temp, 2))
            sound_speeds.append(round(c, 2))
            absorptions_1khz.append(round(a1, 4))
            absorptions_10khz.append(round(a10, 4))

        # Sound channel axis depth (minimum sound speed)
        min_c_idx = int(np.argmin(sound_speeds))
        channel_axis_depth = depths[min_c_idx]

        return {
            "depths_m": [round(d, 1) for d in depths],
            "temperatures_c": temps,
            "sound_speeds_mps": sound_speeds,
            "absorption_1khz_db_km": absorptions_1khz,
            "absorption_10khz_db_km": absorptions_10khz,
            "surface_sound_speed_mps": sound_speeds[0],
            "bottom_sound_speed_mps": sound_speeds[-1],
            "sound_channel_axis_depth_m": channel_axis_depth,
            "sound_channel_axis_speed_mps": sound_speeds[min_c_idx],
            "salinity_psu": salinity_psu,
            "max_depth_m": max_depth_m
        }

    @classmethod
    def construct_ocean_state_tensor(cls, temperature_c: float, salinity_psu: float, depth_m: float,
                                     bathymetry_depth_m: float = 200.0, sea_state_beaufort: int = 2) -> np.ndarray:
        """
        Constructs normalized Ocean State parameter vector Eo in R^16 for neural network injection.
        """
        c_local = cls.mackenzie_sound_speed(temperature_c, salinity_psu, depth_m)
        a_1k = cls.francois_garrison_absorption(1.0, temperature_c, salinity_psu, depth_m)
        a_4k = cls.francois_garrison_absorption(4.0, temperature_c, salinity_psu, depth_m)
        a_10k = cls.francois_garrison_absorption(10.0, temperature_c, salinity_psu, depth_m)

        # Normalize features to [-1.0, 1.0] / standard physical ranges
        tensor = np.array([
            (temperature_c - 15.0) / 15.0,        # Temp [-1, 1]
            (salinity_psu - 35.0) / 5.0,          # Salinity [-1, 1]
            (depth_m - 250.0) / 250.0,            # Depth [-1, 1]
            (c_local - 1500.0) / 50.0,            # Sound Speed [-1, 1]
            math.log10(max(1e-4, a_1k)),          # Log Absorption 1kHz
            math.log10(max(1e-4, a_4k)),          # Log Absorption 4kHz
            math.log10(max(1e-4, a_10k)),         # Log Absorption 10kHz
            (bathymetry_depth_m - 500.0) / 500.0, # Bathymetry depth
            (sea_state_beaufort - 3.0) / 3.0,     # Sea State scale
            math.sin(2 * math.pi * depth_m / 1000.0),
            math.cos(2 * math.pi * depth_m / 1000.0),
            (c_local / 1500.0) ** 2,              # Refractive index squared n^2
            a_1k * (depth_m / 1000.0),            # Integrated 1kHz column attenuation
            a_4k * (depth_m / 1000.0),            # Integrated 4kHz column attenuation
            1.0 if depth_m > 100.0 else 0.0,      # Deep water flag
            1.0 if sea_state_beaufort >= 4 else 0.0 # High sea-state ambient noise flag
        ], dtype=np.float32)

        return tensor

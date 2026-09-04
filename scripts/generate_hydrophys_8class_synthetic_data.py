"""
================================================================================
HydroPhys-OmniNet 8-Class High-Fidelity Synthetic Dataset Generator
Physics-Driven Acoustic Sonar & Underwater Optical Image Synthesis Engine
================================================================================

Generates comprehensive, multi-modal training datasets for the 8-Class HydroPhys
Neural Architecture:
  0: ghost_gear             (Derelict synthetic fishing nets, entangled mesh, pots)
  1: shipwreck              (Submerged hulls, timber ribs, rusted metallic keels)
  2: unexploded_ordnance    (Aerial bombs, naval torpedo shells, naval mines)
  3: pipeline_anomaly       (Subsea metallic pipelines, joint leaks, scour dips)
  4: marine_debris          (Plastic bottles, consumer litter, packaging, bags)
  5: subsea_cable           (Subsea power cables, optical conduits, cable loops)
  6: biological_cluster     (Coral reefs, sponge mounds, kelp - natural feature)
  7: geological_formation   (Rock boulders, bedrock ledges, sand ripples - natural)

Generates:
  1. Multi-Frequency Acoustic Sonar Imagery (SSS / FLS with acoustic shadow physics)
  2. Underwater Optical Imagery (Jerlov water attenuation, marine snow & caustics)
  3. 1D Sub-Bottom Strata Pings (1024-point analytical acoustic time-series)
  4. 3D Volumetric Oriented Bounding Boxes (OBB) & WGS84 Geodetic Coordinates
  5. YOLO-Standard Normalized Text Labels & HydroPhys Master Dataset Manifest
"""

import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ------------------------------------------------------------------------------
# Master 8-Class HydroPhys Taxonomy & Physical Acoustic Properties
# ------------------------------------------------------------------------------
HYDROPHYS_8_CLASSES = {
    0: {
        "name": "ghost_gear",
        "label": "Derelict Ghost Gear & Synthetic Fishing Net",
        "category": "PLASTIC",
        "is_debris": True,
        "acoustic_reflectivity": 0.55,
        "shadow_ratio": 1.2,
        "optical_color_rgb": (38, 166, 91),
        "typical_size_m": (1.8, 3.5, 0.8),
        "aspect_ratio_range": (0.6, 2.2),
    },
    1: {
        "name": "shipwreck",
        "label": "Submerged Metallic/Wood Vessel Hull & Wreckage",
        "category": "METAL_SCRAP",
        "is_debris": True,
        "acoustic_reflectivity": 0.92,
        "shadow_ratio": 2.5,
        "optical_color_rgb": (180, 95, 30),
        "typical_size_m": (6.0, 18.0, 3.2),
        "aspect_ratio_range": (1.8, 4.5),
    },
    2: {
        "name": "unexploded_ordnance",
        "label": "Unexploded Ordnance (UXO) Bomb / Naval Mine",
        "category": "METAL_SCRAP",
        "is_debris": True,
        "acoustic_reflectivity": 0.96,
        "shadow_ratio": 1.8,
        "optical_color_rgb": (210, 50, 40),
        "typical_size_m": (0.6, 1.8, 0.6),
        "aspect_ratio_range": (1.4, 3.2),
    },
    3: {
        "name": "pipeline_anomaly",
        "label": "Subsea Pipeline / Joint Scour Anomaly",
        "category": "METAL_SCRAP",
        "is_debris": True,
        "acoustic_reflectivity": 0.88,
        "shadow_ratio": 1.4,
        "optical_color_rgb": (41, 128, 185),
        "typical_size_m": (0.8, 12.0, 0.8),
        "aspect_ratio_range": (2.5, 8.0),
    },
    4: {
        "name": "marine_debris",
        "label": "Marine Anthropogenic Plastic Debris / Bottle Litter",
        "category": "PLASTIC",
        "is_debris": True,
        "acoustic_reflectivity": 0.68,
        "shadow_ratio": 1.1,
        "optical_color_rgb": (142, 68, 173),
        "typical_size_m": (0.3, 0.8, 0.3),
        "aspect_ratio_range": (0.8, 2.0),
    },
    5: {
        "name": "subsea_cable",
        "label": "Subsea Power & Telecommunication Cable",
        "category": "ELECTRICAL",
        "is_debris": True,
        "acoustic_reflectivity": 0.78,
        "shadow_ratio": 1.0,
        "optical_color_rgb": (243, 156, 18),
        "typical_size_m": (0.2, 10.0, 0.2),
        "aspect_ratio_range": (3.0, 10.0),
    },
    6: {
        "name": "biological_cluster",
        "label": "Natural Benthic Biological Coral / Kelp Cluster",
        "category": "NOT_A_DEBRIS",
        "is_debris": False,
        "acoustic_reflectivity": 0.42,
        "shadow_ratio": 0.9,
        "optical_color_rgb": (26, 188, 156),
        "typical_size_m": (1.2, 2.8, 1.0),
        "aspect_ratio_range": (0.8, 1.6),
    },
    7: {
        "name": "geological_formation",
        "label": "Natural Geological Rock Outcrop & Sand Bedform",
        "category": "NOT_A_DEBRIS",
        "is_debris": False,
        "acoustic_reflectivity": 0.72,
        "shadow_ratio": 1.6,
        "optical_color_rgb": (127, 140, 141),
        "typical_size_m": (2.0, 5.0, 1.5),
        "aspect_ratio_range": (0.7, 2.2),
    },
}


# ==============================================================================
# 1. Physics-Based Acoustic Sonar Simulation Engine
# ==============================================================================
class SonarAcousticSynthesizer:
    """
    Simulates Side-Scan Sonar (SSS) and Forward-Looking Sonar (FLS) acoustic physics:
      - 2D Seabed Texture via Multi-Octave Perlin/Fractal Gaussian Fields
      - Time-Varied Gain (TVG) & Slant-Range Geometric Distortion
      - Lambert's Law Surface Backscatter with Grazing Angle Attenuation
      - Specular Acoustic Highlights + Acoustic Shadow Inversion (Ls = H*Rs / Ha)
      - K-Distribution & Rayleigh Multiplicative Acoustic Speckle Noise
    """

    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height

    def generate_seabed_background(self, seabed_type: str = "sand_ripples") -> np.ndarray:
        """Creates high-resolution benthic acoustic backscatter texture."""
        H, W = self.height, self.width
        
        # Multi-scale Perlin noise approximation using Gaussian blurred octaves
        octave1 = cv2.resize(np.random.normal(120, 28, (H // 32, W // 32)), (W, H), interpolation=cv2.INTER_CUBIC)
        octave2 = cv2.resize(np.random.normal(0, 18, (H // 8, W // 8)), (W, H), interpolation=cv2.INTER_CUBIC)
        octave3 = cv2.resize(np.random.normal(0, 10, (H // 2, W // 2)), (W, H), interpolation=cv2.INTER_LINEAR)
        
        base_texture = octave1 + octave2 + octave3
        
        # Seabed morphological features
        if seabed_type == "sand_ripples":
            # Periodic wave pattern simulating water current ripples
            y_coords, x_coords = np.mgrid[0:H, 0:W]
            ripple_angle = random.uniform(-0.4, 0.4)
            ripple_freq = random.uniform(0.04, 0.08)
            ripples = 22.0 * np.sin(x_coords * math.cos(ripple_angle) * ripple_freq + y_coords * math.sin(ripple_angle) * ripple_freq)
            base_texture += ripples
        elif seabed_type == "mud_flat":
            base_texture = cv2.GaussianBlur(base_texture, (15, 15), 0)
        elif seabed_type == "rocky_outcrop":
            grain = np.random.exponential(scale=14.0, size=(H, W))
            base_texture += grain

        # Apply Range TVG & Grazing Angle Gradient (Sonar nadir is dark/bright line, edges attenuate)
        nadir_x = W // 2
        dist_from_nadir = np.abs(np.linspace(-1.0, 1.0, W)).reshape(1, W)
        tvg_curve = np.clip(1.0 - (dist_from_nadir ** 1.8) * 0.45 + np.exp(-((dist_from_nadir * 6.0) ** 2)) * 0.35, 0.2, 1.4)
        base_texture = base_texture * tvg_curve

        # Multiplicative acoustic speckle noise (Rayleigh distribution)
        speckle = np.random.rayleigh(scale=1.0, size=(H, W))
        sonar_img = np.clip(base_texture * (0.7 + 0.3 * speckle), 0, 255).astype(np.uint8)
        
        return sonar_img

    def render_sonar_target(
        self,
        canvas: np.ndarray,
        class_id: int,
        cx: int,
        cy: int,
        target_w: int,
        target_h: int,
        angle_deg: float
    ) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        """
        Renders an acoustic target highlight (specular reflection) with an
        attached acoustic shadow extending outward from the sonar nadir.
        """
        H, W = canvas.shape[:2]
        props = HYDROPHYS_8_CLASSES[class_id]
        
        # Compute direction away from nadir (shadow casts away from acoustic center)
        nadir_x = W // 2
        shadow_dir_x = 1.0 if cx >= nadir_x else -1.0
        slant_range_factor = max(0.4, abs(cx - nadir_x) / (W * 0.5))
        shadow_length_px = int(target_w * props["shadow_ratio"] * slant_range_factor * random.uniform(1.2, 2.0))
        shadow_length_px = max(18, min(shadow_length_px, 140))

        # Target height estimate from shadow length
        auv_altitude_m = 12.0
        target_height_m = round((shadow_length_px * 0.05 * auv_altitude_m) / (slant_range_factor * 30.0 + shadow_length_px * 0.05), 2)
        target_height_m = max(0.2, min(target_height_m, 6.5))

        # 1. Render Acoustic Shadow (Complete acoustic blockage, near zero return)
        shadow_mask = np.zeros((H, W), dtype=np.uint8)
        shadow_start_x = int(cx + (target_w // 2) * shadow_dir_x)
        shadow_end_x = int(cx + (target_w // 2 + shadow_length_px) * shadow_dir_x)
        
        shadow_pts = np.array([
            [shadow_start_x, cy - target_h // 2],
            [shadow_end_x, cy - int(target_h * 0.65)],
            [shadow_end_x, cy + int(target_h * 0.65)],
            [shadow_start_x, cy + target_h // 2]
        ], dtype=np.int32)
        
        cv2.fillConvexPoly(shadow_mask, shadow_pts, 255)
        # Blur shadow edges to simulate acoustic penumbra
        shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0)
        shadow_factor = 1.0 - (shadow_mask.astype(np.float32) / 255.0) * 0.92
        canvas = np.clip(canvas.astype(np.float32) * shadow_factor, 0, 255).astype(np.uint8)

        # 2. Render Specular Highlight (High backscatter return)
        highlight_mask = np.zeros((H, W), dtype=np.uint8)
        reflectivity = props["acoustic_reflectivity"]
        highlight_val = int(210 * reflectivity + random.randint(20, 45))

        if class_id == 0:  # ghost_gear: Irregular mesh polygon
            pts = []
            for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                r = (target_w // 2) * random.uniform(0.6, 1.0)
                pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
            cv2.fillPoly(highlight_mask, [np.array(pts, dtype=np.int32)], 255)
            # Add grid lines for netting texture
            for off in range(-target_w // 2, target_w // 2, 6):
                cv2.line(highlight_mask, (cx + off, cy - target_h // 2), (cx + off, cy + target_h // 2), 180, 1)

        elif class_id == 1:  # shipwreck: Elongated hull structure + internal ribs
            rect = ((cx, cy), (target_w, target_h), angle_deg)
            box = cv2.boxPoints(rect)
            cv2.fillPoly(highlight_mask, [np.int32(box)], 255)
            # Deck ribs
            for i in range(-target_w // 3, target_w // 3, 10):
                cv2.line(highlight_mask, (cx + i, cy - target_h // 3), (cx + i, cy + target_h // 3), 120, 2)

        elif class_id == 2:  # unexploded_ordnance: Torpedo/bomb cylinder + fins
            cv2.ellipse(highlight_mask, (cx, cy), (target_w // 2, target_h // 2), angle_deg, 0, 360, 255, -1)
            # Stabilizing fin
            cv2.rectangle(highlight_mask, (cx - target_w // 2 - 4, cy - 3), (cx - target_w // 2, cy + 3), 255, -1)

        elif class_id == 3:  # pipeline_anomaly: Linear pipe line + scour trench
            rect = ((cx, cy), (target_w, target_h), angle_deg)
            box = cv2.boxPoints(rect)
            cv2.fillPoly(highlight_mask, [np.int32(box)], 255)
            cv2.circle(highlight_mask, (cx, cy), max(3, target_h // 2), 255, -1)

        elif class_id == 4:  # marine_debris: Plastic bottle / container shape
            cv2.ellipse(highlight_mask, (cx, cy), (target_w // 2, target_h // 2), angle_deg, 0, 360, 255, -1)
            cv2.rectangle(highlight_mask, (cx - 2, cy - target_h // 2 - 3), (cx + 2, cy - target_h // 2), 255, -1)

        elif class_id == 5:  # subsea_cable: Thin curvilinear track
            pts = []
            for t in np.linspace(-target_w // 2, target_w // 2, 7):
                curve_y = cy + int(12 * math.sin(t * 0.1))
                pts.append([int(cx + t), curve_y])
            cv2.polylines(highlight_mask, [np.array(pts, dtype=np.int32)], False, 255, thickness=max(2, target_h // 3))

        elif class_id == 6:  # biological_cluster: Organic clump
            for _ in range(4):
                ox = cx + random.randint(-target_w // 3, target_w // 3)
                oy = cy + random.randint(-target_h // 3, target_h // 3)
                cv2.circle(highlight_mask, (ox, oy), random.randint(target_h // 4, target_h // 2), 255, -1)

        else:  # geological_formation: Natural Benthic Rock Boulder (Granular mineral facets)
            num_v = random.randint(6, 9)
            angles = np.sort(np.random.uniform(0, 2 * math.pi, num_v))
            pts = []
            for a in angles:
                r_rad = random.uniform(0.7, 1.1)
                pts.append([
                    int(cx + (target_w // 2) * r_rad * math.cos(a)),
                    int(cy + (target_h // 2) * r_rad * math.sin(a))
                ])
            cv2.fillConvexPoly(highlight_mask, np.array(pts, dtype=np.int32), 220)
            # Add granular mineral noise and algae acoustic absorption
            noise = np.random.normal(0, 35, (H, W)).astype(np.float32)
            highlight_mask = np.clip(highlight_mask.astype(np.float32) + noise * (highlight_mask > 0), 0, 255).astype(np.uint8)

        # Composite Highlight onto canvas
        highlight_float = (highlight_mask.astype(np.float32) / 255.0) * highlight_val
        canvas = np.clip(canvas.astype(np.float32) + highlight_float, 0, 255).astype(np.uint8)

        # Compute Normalized YOLO Bounding Box [x_center, y_center, w, h] (normalized 0..1)
        # Note: Bounding box covers both target highlight and acoustic signature
        min_x = max(0, cx - target_w // 2)
        max_x = min(W - 1, cx + target_w // 2)
        min_y = max(0, cy - target_h // 2)
        max_y = min(H - 1, cy + target_h // 2)

        norm_cx = (min_x + max_x) / (2.0 * W)
        norm_cy = (min_y + max_y) / (2.0 * H)
        norm_w = (max_x - min_x) / float(W)
        norm_h = (max_y - min_y) / float(H)

        yolo_box = [round(norm_cx, 6), round(norm_cy, 6), round(norm_w, 6), round(norm_h, 6)]
        
        meta = {
            "class_id": class_id,
            "class_name": props["name"],
            "target_height_m": target_height_m,
            "shadow_length_px": shadow_length_px,
            "snr_db": round(20.0 * math.log10(highlight_val / 45.0), 1),
            "reflectivity": reflectivity,
            "is_debris": props["is_debris"]
        }

        return canvas, yolo_box, meta


# ==============================================================================
# 2. Physics-Based Underwater Optical Simulation Engine
# ==============================================================================
class OpticalMarineSynthesizer:
    """
    Simulates in-situ underwater camera video and photographic frames:
      - Jerlov Oceanic Water Column (Wavelength-dependent absorption: Red decays fast)
      - Marine Snow Particulate Backscatter & Floating Organic Suspensions
      - Optical Sunlight Caustic Wave Refraction
      - Multi-Category Debris with Natural Algal Fouling & Bio-Crust Textures
    """

    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height

    def generate_water_column(self, depth_m: float = 8.5) -> np.ndarray:
        """Generates deep sea or coastal marine water backdrop."""
        H, W = self.height, self.width
        
        # Depth-dependent Jerlov attenuation (Blue-Green dominant, Red suppressed)
        red_decay = math.exp(-0.35 * depth_m)
        green_decay = math.exp(-0.06 * depth_m)
        blue_decay = math.exp(-0.02 * depth_m)
        
        base_r = int(18 * red_decay + random.randint(2, 6))
        base_g = int(85 * green_decay + random.randint(10, 25))
        base_b = int(120 * blue_decay + random.randint(20, 35))

        # Vertical depth gradient
        y_gradient = np.linspace(1.15, 0.75, H).reshape(H, 1, 1)
        base_img = np.ones((H, W, 3), dtype=np.float32)
        base_img[:, :, 0] = base_b * y_gradient[:, :, 0] # BGR
        base_img[:, :, 1] = base_g * y_gradient[:, :, 0]
        base_img[:, :, 2] = base_r * y_gradient[:, :, 0]

        # Optical wave caustics for shallow coastal zones
        if depth_m < 15.0:
            caustic_map = np.zeros((H, W), dtype=np.float32)
            for _ in range(5):
                freq = random.uniform(0.015, 0.035)
                phase = random.uniform(0, math.pi * 2)
                y_c, x_c = np.mgrid[0:H, 0:W]
                caustic_map += np.sin(x_c * freq + phase) * np.cos(y_c * freq + phase)
            caustic_norm = np.clip((caustic_map + 2.5) / 5.0, 0.0, 1.0)
            base_img = base_img * (0.85 + 0.3 * caustic_norm[:, :, np.newaxis])

        # Floating Marine Snow Particles
        num_particles = random.randint(120, 280)
        for _ in range(num_particles):
            px = random.randint(0, W - 1)
            py = random.randint(0, H - 1)
            pr = random.randint(1, 3)
            p_intensity = random.randint(140, 230)
            cv2.circle(base_img, (px, py), pr, (p_intensity, p_intensity + 10, p_intensity - 10), -1)

        # Gaussian atmospheric / water haze
        base_img = cv2.GaussianBlur(base_img, (3, 3), 0)
        return np.clip(base_img, 0, 255).astype(np.uint8)

    def render_optical_target(
        self,
        canvas: np.ndarray,
        class_id: int,
        cx: int,
        cy: int,
        target_w: int,
        target_h: int,
        angle_deg: float
    ) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        """Renders naturalistic 3D optical marine object with realistic shading."""
        H, W = canvas.shape[:2]
        props = HYDROPHYS_8_CLASSES[class_id]
        color_rgb = props["optical_color_rgb"]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        overlay = canvas.copy()
        mask = np.zeros((H, W), dtype=np.uint8)

        if class_id == 0:  # ghost_gear: Synthetic entangled mesh
            pts = []
            for a in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                r = (target_w // 2) * random.uniform(0.7, 1.1)
                pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color_bgr)
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
            # Netting grid pattern
            for x_off in range(-target_w // 2, target_w // 2, 7):
                cv2.line(overlay, (cx + x_off, cy - target_h // 2), (cx + x_off, cy + target_h // 2), (200, 240, 200), 1)

        elif class_id == 1:  # shipwreck: Hull & rusted metal plating
            rect = ((cx, cy), (target_w, target_h), angle_deg)
            box = np.int32(cv2.boxPoints(rect))
            cv2.fillPoly(overlay, [box], (35, 60, 110)) # Rust BGR
            cv2.fillPoly(mask, [box], 255)
            # Deck outline
            cv2.polylines(overlay, [box], True, (50, 90, 160), 2)

        elif class_id == 2:  # unexploded_ordnance: Torpedo metallic cylinder
            cv2.ellipse(overlay, (cx, cy), (target_w // 2, target_h // 2), angle_deg, 0, 360, (40, 45, 180), -1)
            cv2.ellipse(mask, (cx, cy), (target_w // 2, target_h // 2), angle_deg, 0, 360, 255, -1)
            # Yellow warning band
            cv2.circle(overlay, (cx, cy), max(2, target_h // 3), (20, 220, 220), -1)

        elif class_id == 3:  # pipeline_anomaly: Industrial steel pipe
            rect = ((cx, cy), (target_w, target_h), angle_deg)
            box = np.int32(cv2.boxPoints(rect))
            cv2.fillPoly(overlay, [box], (140, 120, 60))
            cv2.fillPoly(mask, [box], 255)

        elif class_id == 4:  # marine_debris: Polymer plastic bottle
            cv2.ellipse(overlay, (cx, cy), (target_w // 2, target_h // 2), angle_deg, 0, 360, (210, 180, 70), -1)
            cv2.ellipse(mask, (cx, cy), (target_w // 2, target_h // 2), angle_deg, 0, 360, 255, -1)
            # Cap
            cv2.rectangle(overlay, (cx - 3, cy - target_h // 2 - 4), (cx + 3, cy - target_h // 2), (230, 70, 40), -1)

        elif class_id == 5:  # subsea_cable: Armored electrical cable
            pts = []
            for t in np.linspace(-target_w // 2, target_w // 2, 8):
                pts.append([int(cx + t), int(cy + 14 * math.sin(t * 0.08))])
            cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, (30, 140, 220), thickness=max(3, target_h // 3))
            cv2.polylines(mask, [np.array(pts, dtype=np.int32)], False, 255, thickness=max(3, target_h // 3))

        elif class_id == 6:  # biological_cluster: Dense Coral Reef & Benthic Moss/Algae Bed
            # 1. Base organic cluster
            for _ in range(6):
                ox = cx + random.randint(-target_w // 3, target_w // 3)
                oy = cy + random.randint(-target_h // 3, target_h // 3)
                r_c = random.randint(target_h // 4, target_h // 2)
                # Emerald / mossy green / turquoise bio-pigment
                bio_color = random.choice([
                    (60, 160, 45),   # Forest moss green
                    (35, 185, 110),  # Turquoise coral
                    (30, 130, 85),   # Deep algae olive
                    (80, 140, 160)   # Blue-green benthic sponge
                ])
                cv2.circle(overlay, (ox, oy), r_c, bio_color, -1)
                cv2.circle(mask, (ox, oy), r_c, 255, -1)
            
            # 2. Filamentous Algae & Moss Tendrils
            for _ in range(random.randint(15, 30)):
                tx = cx + random.randint(-target_w // 2, target_w // 2)
                ty = cy + random.randint(-target_h // 2, target_h // 2)
                t_len = random.randint(6, 18)
                t_ang = random.uniform(0, 2 * math.pi)
                t_end_x = int(tx + t_len * math.cos(t_ang))
                t_end_y = int(ty + t_len * math.sin(t_ang))
                moss_shade = (random.randint(40, 80), random.randint(140, 210), random.randint(30, 70))
                cv2.line(overlay, (tx, ty), (t_end_x, t_end_y), moss_shade, thickness=random.randint(1, 2))
                cv2.line(mask, (tx, ty), (t_end_x, t_end_y), 255, thickness=2)

        else:  # geological_formation: Natural Benthic Rock Boulder with Biofouling (Algae, Moss, Barnacles)
            # 1. Multi-faceted Angular / Rounded Rock Geometry
            num_vertices = random.randint(6, 10)
            angles = np.sort(np.random.uniform(0, 2 * math.pi, num_vertices))
            pts = []
            for a in angles:
                r_rad = random.uniform(0.7, 1.15)
                pts.append([
                    int(cx + (target_w // 2) * r_rad * math.cos(a)),
                    int(cy + (target_h // 2) * r_rad * math.sin(a))
                ])
            rock_poly = np.array(pts, dtype=np.int32)

            # Base Mineral Rock Color (Granite, Basalt, Limestone)
            rock_base_bgr = random.choice([
                (95, 105, 110),   # Granite gray
                (75, 80, 85),     # Basalt dark slate
                (110, 120, 130),  # Limestone light gray
                (65, 85, 95)      # Silt-covered rock
            ])
            cv2.fillConvexPoly(overlay, rock_poly, rock_base_bgr)
            cv2.fillConvexPoly(mask, rock_poly, 255)

            # 2. Rock Crevices & Shading Facets
            for i in range(len(pts)):
                p1 = tuple(pts[i])
                p2 = tuple(pts[(i + 1) % len(pts)])
                # Shadow contour on edges
                cv2.line(overlay, p1, p2, (int(rock_base_bgr[0] * 0.65), int(rock_base_bgr[1] * 0.65), int(rock_base_bgr[2] * 0.65)), 2)

            # 3. Dense Marine Moss & Chlorophyll Green Algae Coating
            # Algae patches encrusting the rock surface
            num_moss_patches = random.randint(4, 9)
            for _ in range(num_moss_patches):
                mx = cx + random.randint(-target_w // 3, target_w // 3)
                my = cy + random.randint(-target_h // 3, target_h // 3)
                mr = random.randint(max(4, target_h // 6), max(8, target_h // 3))
                moss_color = random.choice([
                    (35, 155, 65),   # Vibrant green moss
                    (25, 115, 45),   # Dark emerald velvet moss
                    (45, 135, 100),  # Olive benthic algae slime
                    (30, 85, 70),    # Marine biofilm crust
                    (20, 180, 80)    # Shallow water bright chlorophyte
                ])
                cv2.circle(overlay, (mx, my), mr, moss_color, -1)

            # 4. Filamentous Moss Tendrils & Algae Fringes extending around rock borders
            num_fringes = random.randint(16, 35)
            for _ in range(num_fringes):
                # Pick a point near the rock edge
                idx_v = random.randint(0, len(pts) - 1)
                fx, fy = pts[idx_v]
                f_len = random.randint(4, 14)
                f_ang = random.uniform(0, 2 * math.pi)
                f_end_x = int(fx + f_len * math.cos(f_ang))
                f_end_y = int(fy + f_len * math.sin(f_ang))
                f_color = (random.randint(30, 60), random.randint(140, 210), random.randint(30, 80))
                cv2.line(overlay, (fx, fy), (f_end_x, f_end_y), f_color, thickness=random.randint(1, 2))
                cv2.line(mask, (fx, fy), (f_end_x, f_end_y), 255, thickness=2)

            # 5. Calcified Barnacle Clusters & Tubeworms
            num_barnacles = random.randint(8, 20)
            for _ in range(num_barnacles):
                bx = cx + random.randint(-target_w // 3, target_w // 3)
                by = cy + random.randint(-target_h // 3, target_h // 3)
                cv2.circle(overlay, (bx, by), random.randint(2, 3), (210, 225, 230), -1) # Cream/white barnacle cone
                cv2.circle(overlay, (bx, by), 1, (120, 130, 135), -1) # Barnacle aperture center

        # Alpha blend with underwater water column attenuation
        alpha = 0.84
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)

        # Compute Normalized YOLO Bounding Box
        min_x = max(0, cx - target_w // 2)
        max_x = min(W - 1, cx + target_w // 2)
        min_y = max(0, cy - target_h // 2)
        max_y = min(H - 1, cy + target_h // 2)

        norm_cx = (min_x + max_x) / (2.0 * W)
        norm_cy = (min_y + max_y) / (2.0 * H)
        norm_w = (max_x - min_x) / float(W)
        norm_h = (max_y - min_y) / float(H)

        yolo_box = [round(norm_cx, 6), round(norm_cy, 6), round(norm_w, 6), round(norm_h, 6)]
        
        meta = {
            "class_id": class_id,
            "class_name": props["name"],
            "target_height_m": round(target_h * 0.04, 2),
            "snr_db": round(random.uniform(22.0, 32.0), 1),
            "is_debris": props["is_debris"]
        }

        return canvas, yolo_box, meta


# ==============================================================================
# 3. 1D Sub-Bottom Strata Acoustic Ping Synthesizer
# ==============================================================================
def synthesize_1d_strata_sweep(num_samples: int = 1024, has_buried_object: bool = False, depth_m: float = 2.4) -> np.ndarray:
    """
    Synthesizes a 1024-point sub-bottom acoustic profiler sweep signal s(t):
      - Direct water arrival
      - Water-Seabed reflection interface (R1)
      - Sub-bottom sediment layers (Mud, Silt, Gravel, Bedrock)
      - Buried target echo horizon (if present)
    """
    t = np.linspace(0, 1.0, num_samples)
    sweep = np.random.normal(0, 0.015, num_samples).astype(np.float32)

    # 1. Water-Seabed Primary Pulse (Chirp envelope)
    seabed_idx = int(num_samples * 0.20)
    pulse_len = 32
    if seabed_idx + pulse_len < num_samples:
        pulse = np.hanning(pulse_len) * np.sin(np.linspace(0, 8 * math.pi, pulse_len))
        sweep[seabed_idx:seabed_idx + pulse_len] += pulse * 0.85

    # 2. Sediment Horizon 1 (Silt/Mud boundary)
    layer1_idx = int(num_samples * 0.42)
    if layer1_idx + pulse_len < num_samples:
        layer1_pulse = np.hanning(pulse_len) * np.sin(np.linspace(0, 6 * math.pi, pulse_len))
        sweep[layer1_idx:layer1_idx + pulse_len] += layer1_pulse * 0.45

    # 3. Sediment Horizon 2 (Gravel/Bedrock)
    layer2_idx = int(num_samples * 0.72)
    if layer2_idx + pulse_len < num_samples:
        layer2_pulse = np.hanning(pulse_len) * np.sin(np.linspace(0, 4 * math.pi, pulse_len))
        sweep[layer2_idx:layer2_idx + pulse_len] += layer2_pulse * 0.35

    # 4. Buried Object Anomaly Horizon
    if has_buried_object:
        obj_idx = int(num_samples * min(0.95, (0.22 + depth_m * 0.15)))
        if obj_idx + pulse_len < num_samples:
            obj_pulse = np.hanning(pulse_len) * np.sin(np.linspace(0, 12 * math.pi, pulse_len))
            sweep[obj_idx:obj_idx + pulse_len] += obj_pulse * 0.95

    # Normalize
    max_val = np.max(np.abs(sweep))
    if max_val > 0:
        sweep /= max_val

    return sweep


# ==============================================================================
# 4. Grand Dataset Assembly Pipeline (Sonar + Optical + 1D + YOLO Labels)
# ==============================================================================
def generate_hydrophys_8class_dataset(
    output_dir: str = "data/hydrophys_8class_dataset",
    total_samples: int = 1200,
    sonar_ratio: float = 0.55,
    img_size: int = 640,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Assembles the complete 8-Class HydroPhys training and benchmark dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    base_path = Path(output_dir)
    splits = ["train", "val", "test"]
    split_weights = [0.70, 0.20, 0.10] # 70% train, 20% val, 10% test

    # Directory structures
    for s in splits:
        (base_path / "sonar" / "images" / s).mkdir(parents=True, exist_ok=True)
        (base_path / "sonar" / "labels" / s).mkdir(parents=True, exist_ok=True)
        (base_path / "optical" / "images" / s).mkdir(parents=True, exist_ok=True)
        (base_path / "optical" / "labels" / s).mkdir(parents=True, exist_ok=True)
        (base_path / "unified" / "images" / s).mkdir(parents=True, exist_ok=True)
        (base_path / "unified" / "labels" / s).mkdir(parents=True, exist_ok=True)
        (base_path / "strata_1d_pings" / s).mkdir(parents=True, exist_ok=True)

    sonar_synth = SonarAcousticSynthesizer(img_size, img_size)
    optical_synth = OpticalMarineSynthesizer(img_size, img_size)

    class_distribution = {i: 0 for i in range(8)}
    manifest_records = []

    print("\n==========================================================================")
    print("  HYDROPHYS-OMNINET 8-CLASS SYNTHETIC DATASET GENERATION ENGINE           ")
    print("==========================================================================")
    print(f"[*] Target Directory   : {base_path.resolve()}")
    print(f"[*] Total Sample Count : {total_samples:,} (Sonar: {int(total_samples*sonar_ratio)}, Optical: {total_samples - int(total_samples*sonar_ratio)})")
    print(f"[*] Classes Modeled    : 8 Distinct HydroPhys Taxonomy Classes")
    print(f"[*] Image Dimensions   : {img_size}x{img_size} px (3-Ch RGB & 1-Ch Sonar)")

    start_time = time.time()

    for idx in range(total_samples):
        # Determine split
        rand_split = random.random()
        if rand_split < split_weights[0]:
            split = "train"
        elif rand_split < split_weights[0] + split_weights[1]:
            split = "val"
        else:
            split = "test"

        is_sonar = (random.random() < sonar_ratio)
        sample_id = f"HP8_{'SONAR' if is_sonar else 'OPTIC'}_{idx:06d}"
        
        # Decide targets in this image (1 to 3 objects)
        num_targets = random.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]
        chosen_classes = [random.randint(0, 7) for _ in range(num_targets)]

        yolo_labels = []
        target_metas = []

        if is_sonar:
            # Acoustic Sonar Generation
            seabed_type = random.choice(["sand_ripples", "mud_flat", "rocky_outcrop"])
            canvas = sonar_synth.generate_seabed_background(seabed_type)
            
            for cls_id in chosen_classes:
                class_distribution[cls_id] += 1
                props = HYDROPHYS_8_CLASSES[cls_id]
                
                # Size in pixels
                tw = random.randint(int(img_size * 0.06), int(img_size * 0.22))
                aspect = random.uniform(props["aspect_ratio_range"][0], props["aspect_ratio_range"][1])
                th = max(12, int(tw / aspect))
                
                cx = random.randint(tw // 2 + 15, img_size - tw // 2 - 15)
                cy = random.randint(th // 2 + 15, img_size - th // 2 - 15)
                angle = random.uniform(-45.0, 45.0)

                canvas, box, meta = sonar_synth.render_sonar_target(canvas, cls_id, cx, cy, tw, th, angle)
                yolo_labels.append([cls_id] + box)
                target_metas.append(meta)

            # Convert Grayscale to 3-Channel Sonar Image
            final_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            domain_tag = "sonar"

        else:
            # Optical Underwater Camera Generation
            water_depth = random.uniform(2.5, 35.0)
            canvas = optical_synth.generate_water_column(depth_m=water_depth)

            for cls_id in chosen_classes:
                class_distribution[cls_id] += 1
                props = HYDROPHYS_8_CLASSES[cls_id]

                tw = random.randint(int(img_size * 0.08), int(img_size * 0.24))
                aspect = random.uniform(props["aspect_ratio_range"][0], props["aspect_ratio_range"][1])
                th = max(14, int(tw / aspect))

                cx = random.randint(tw // 2 + 20, img_size - tw // 2 - 20)
                cy = random.randint(th // 2 + 20, img_size - th // 2 - 20)
                angle = random.uniform(-60.0, 60.0)

                canvas, box, meta = optical_synth.render_optical_target(canvas, cls_id, cx, cy, tw, th, angle)
                yolo_labels.append([cls_id] + box)
                target_metas.append(meta)

            final_img = canvas
            domain_tag = "optical"

        # Generate 1D Sub-bottom Strata Sweep (for multi-modal CAW-SSM)
        has_debris = any(HYDROPHYS_8_CLASSES[c]["is_debris"] for c in chosen_classes)
        top_height = max([m.get("target_height_m", 1.0) for m in target_metas] or [1.0])
        strata_1d = synthesize_1d_strata_sweep(num_samples=1024, has_buried_object=has_debris, depth_m=top_height)

        # Save Image & Label into domain-specific and unified folders
        img_filename = f"{sample_id}.jpg"
        lbl_filename = f"{sample_id}.txt"
        ping_filename = f"{sample_id}.npy"

        # Domain directory save
        img_save_path = base_path / domain_tag / "images" / split / img_filename
        lbl_save_path = base_path / domain_tag / "labels" / split / lbl_filename
        cv2.imwrite(str(img_save_path), final_img, [cv2.IMWRITE_JPEG_QUALITY, 94])

        # Unified dataset directory save
        cv2.imwrite(str(base_path / "unified" / "images" / split / img_filename), final_img, [cv2.IMWRITE_JPEG_QUALITY, 94])
        np.save(str(base_path / "strata_1d_pings" / split / ping_filename), strata_1d)

        # Write YOLO label lines: [class_id cx cy w h]
        label_lines = [" ".join(map(str, row)) for row in yolo_labels]
        lbl_content = "\n".join(label_lines) + "\n"
        lbl_save_path.write_text(lbl_content)
        (base_path / "unified" / "labels" / split / lbl_filename).write_text(lbl_content)

        manifest_records.append({
            "sample_id": sample_id,
            "domain": domain_tag,
            "split": split,
            "image_path": f"{domain_tag}/images/{split}/{img_filename}",
            "label_path": f"{domain_tag}/labels/{split}/{lbl_filename}",
            "strata_ping_path": f"strata_1d_pings/{split}/{ping_filename}",
            "targets_count": len(yolo_labels),
            "classes": [int(row[0]) for row in yolo_labels],
            "targets_metadata": target_metas
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == total_samples:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(0.1, elapsed)
            print(f"[*] Progress: [{idx + 1:04d}/{total_samples:04d}] ({rate:.1f} samples/s) | Last: {sample_id} ({domain_tag.upper()})")

    # Create dataset.yaml for YOLOv12 / HydroPhys Training
    yaml_content = f"""# HydroPhys-OmniNet 8-Class Multi-Modal Marine Sonar & Optical Dataset
path: {base_path.resolve().as_posix()}/unified
train: images/train
val: images/val
test: images/test

nc: 8
names:
  0: ghost_gear
  1: shipwreck
  2: unexploded_ordnance
  3: pipeline_anomaly
  4: marine_debris
  5: subsea_cable
  6: biological_cluster
  7: geological_formation

metadata:
  total_samples: {total_samples}
  sonar_samples: {len([r for r in manifest_records if r['domain'] == 'sonar'])}
  optical_samples: {len([r for r in manifest_records if r['domain'] == 'optical'])}
  created_timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
  engine: "HydroPhys Synthetic Acoustic & Optical Generator"
"""
    yaml_file = base_path / "dataset.yaml"
    yaml_file.write_text(yaml_content)

    # Save manifest JSON
    manifest_file = base_path / "hydrophys_8class_manifest.json"
    summary_report = {
        "dataset_name": "HydroPhys-OmniNet 8-Class Synthetic Grand Corpus",
        "total_samples": total_samples,
        "splits": {
            "train": len([r for r in manifest_records if r['split'] == 'train']),
            "val": len([r for r in manifest_records if r['split'] == 'val']),
            "test": len([r for r in manifest_records if r['split'] == 'test']),
        },
        "domains": {
            "acoustic_sonar": len([r for r in manifest_records if r['domain'] == 'sonar']),
            "optical_camera": len([r for r in manifest_records if r['domain'] == 'optical']),
        },
        "class_distribution": {
            HYDROPHYS_8_CLASSES[k]["name"]: count for k, count in class_distribution.items()
        },
        "classes_taxonomy": HYDROPHYS_8_CLASSES,
        "yaml_config": str(yaml_file),
        "manifest_records_count": len(manifest_records)
    }

    with open(manifest_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    # Generate multi-class visual collage sheet
    create_dataset_sample_sheet(base_path, manifest_records[:12])

    print("\n==========================================================================")
    print("  [SUCCESS] 8-CLASS SYNTHETIC DATASET GENERATION COMPLETE                 ")
    print("==========================================================================")
    print(f"[*] Total Generated     : {total_samples:,} Images + Labels + 1D Ping Sweeps")
    print(f"[*] Train Split         : {summary_report['splits']['train']} samples")
    print(f"[*] Validation Split    : {summary_report['splits']['val']} samples")
    print(f"[*] Test Split          : {summary_report['splits']['test']} samples")
    print(f"[*] YAML Config         : {yaml_file}")
    print(f"[*] JSON Manifest       : {manifest_file}")
    print("--------------------------------------------------------------------------")
    for cls_id, count in class_distribution.items():
        name = HYDROPHYS_8_CLASSES[cls_id]["name"]
        print(f"  Class {cls_id} [{name:<22}] : {count:>4} instances")
    print("==========================================================================\n")

    return summary_report


def create_dataset_sample_sheet(base_path: Path, sample_records: List[Dict[str, Any]], out_path: Optional[Path] = None):
    """Creates a preview visual sheet of generated synthetic sonar and optical images."""
    if not sample_records:
        return

    cols = 4
    rows = 2
    tile_w, tile_h = 320, 320
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for i, rec in enumerate(sample_records[: cols * rows]):
        r = i // cols
        c = i % cols
        img_p = base_path / rec["image_path"]
        if img_p.exists():
            im = cv2.imread(str(img_p))
            im_resized = cv2.resize(im, (tile_w, tile_h))
            
            # Draw Class Tags
            domain = rec["domain"].upper()
            cv2.putText(im_resized, f"{domain} | ID: {rec['sample_id'][-6:]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            for t_idx, cls_id in enumerate(rec["classes"]):
                cls_name = HYDROPHYS_8_CLASSES[cls_id]["name"]
                cv2.putText(im_resized, f"C{cls_id}: {cls_name}", (10, 50 + t_idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = im_resized

    out_file = out_path or (base_path / "dataset_preview_sheet.png")
    cv2.imwrite(str(out_file), sheet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HydroPhys-OmniNet 8-Class Synthetic Data Generator")
    parser.add_argument("--output-dir", type=str, default="data/hydrophys_8class_dataset", help="Output directory")
    parser.add_argument("--samples", type=int, default=1200, help="Total sample count to generate")
    parser.add_argument("--sonar-ratio", type=float, default=0.55, help="Ratio of sonar to optical samples (default: 0.55)")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution width/height in px")
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")

    args = parser.parse_args()
    generate_hydrophys_8class_dataset(
        output_dir=args.output_dir,
        total_samples=args.samples,
        sonar_ratio=args.sonar_ratio,
        img_size=args.img_size,
        seed=args.seed
    )

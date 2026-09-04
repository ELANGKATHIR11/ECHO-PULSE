"""
================================================================================
HydroPhys-OmniNet Dedicated Biofouled Debris vs. Natural Mimics Dataset Generator
Specialized for Discriminating Biofouled Metals, Plastics & Ghost Gear
from Natural Biofouled Rocks, Boulders, Algae Beds & Corals
================================================================================

Generates targeted, high-realism training datasets with explicit biofouling modeling:

TARGET DEBRIS CLASSES (is_debris = True):
  - Biofouled Metal Scrap (Rusted steel plates, metal drums, wreckage with algae/barnacles) -> Class 1 (shipwreck) / 3 (pipeline_anomaly)
  - Biofouled Plastics (Sunken polymer bottles, crates, packaging with biofilm/moss)       -> Class 4 (marine_debris)
  - Biofouled Ghost Gear (Algae-entangled synthetic nets, traps, nylon mesh)             -> Class 0 (ghost_gear)
  - Biofouled Subsea Cables (Power conduits encrusted with marine growth)                -> Class 5 (subsea_cable)
  - Biofouled UXO Munitions (Ordnance shells with calcified marine crust)                -> Class 2 (unexploded_ordnance)

NATURAL MIMICS & HABITAT (is_debris = False, ALERT SUPPRESSED):
  - Biofouled Rocks & Boulders (Granite/basalt with dense green moss, algae & barnacles)  -> Class 7 (geological_formation)
  - Biofouled Corals & Sponges (Benthic coral reefs, sponge mounds, kelp beds)            -> Class 6 (biological_cluster)

Generates:
  1. Optical Underwater Frames (Jerlov absorption, marine snow, moss tufts, barnacles)
  2. Acoustic Sonar Swaths (Specular metallic returns vs diffuse granular rock scatter)
  3. YOLO Bounding Box Text Labels [class_id cx cy w h]
  4. Comprehensive JSON Dataset Catalog with fouling ratios & material physical properties
"""

import os
import sys

# Prevent OpenBLAS thread memory contention
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Master Taxonomy Definitions
BIOFOULED_TAXONOMY = {
    0: {"name": "ghost_gear", "label": "Biofouled Ghost Net & Fishing Trap", "is_debris": True, "category": "PLASTIC", "z_impedance": 2.8, "base_bgr": (40, 140, 60)},
    1: {"name": "shipwreck", "label": "Biofouled Metallic Shipwreck & Steel Scrap", "is_debris": True, "category": "METAL_SCRAP", "z_impedance": 45.0, "base_bgr": (35, 60, 120)},
    2: {"name": "unexploded_ordnance", "label": "Biofouled UXO Ordnance / Shell", "is_debris": True, "category": "HAZARD_UXO", "z_impedance": 42.0, "base_bgr": (40, 45, 175)},
    3: {"name": "pipeline_anomaly", "label": "Biofouled Subsea Pipeline / Flange", "is_debris": True, "category": "METAL_SCRAP", "z_impedance": 40.0, "base_bgr": (120, 110, 50)},
    4: {"name": "marine_debris", "label": "Biofouled Plastic Bottle & Consumer Litter", "is_debris": True, "category": "PLASTIC", "z_impedance": 2.4, "base_bgr": (190, 160, 60)},
    5: {"name": "subsea_cable", "label": "Biofouled Subsea Power / Telecom Cable", "is_debris": True, "category": "ELECTRICAL", "z_impedance": 35.0, "base_bgr": (25, 135, 215)},
    6: {"name": "biological_cluster", "label": "Natural Coral Reef, Kelp & Sponge Bed", "is_debris": False, "category": "BIOLOGICAL", "z_impedance": 1.9, "base_bgr": (30, 175, 105)},
    7: {"name": "geological_formation", "label": "Natural Biofouled Rock Boulder / Bedrock", "is_debris": False, "category": "GEOLOGICAL", "z_impedance": 12.0, "base_bgr": (90, 100, 105)},
}


# ==============================================================================
# Optical Biofouling Marine Synthesizer
# ==============================================================================
class BiofoulingOpticalSynthesizer:
    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height

    def generate_water_background(self, depth_m: float = 10.0) -> np.ndarray:
        H, W = self.height, self.width
        
        # Jerlov spectral absorption
        r_val = int(12 * math.exp(-0.30 * depth_m) + random.randint(2, 5))
        g_val = int(75 * math.exp(-0.05 * depth_m) + random.randint(10, 20))
        b_val = int(115 * math.exp(-0.02 * depth_m) + random.randint(15, 25))

        y_grad = np.linspace(1.12, 0.80, H).reshape(H, 1, 1)
        bg = np.ones((H, W, 3), dtype=np.float32)
        bg[:, :, 0] = b_val * y_grad[:, :, 0]
        bg[:, :, 1] = g_val * y_grad[:, :, 0]
        bg[:, :, 2] = r_val * y_grad[:, :, 0]

        # Caustics
        if depth_m < 18.0:
            c_map = np.zeros((H, W), dtype=np.float32)
            for _ in range(4):
                f = random.uniform(0.015, 0.035)
                ph = random.uniform(0, 6.28)
                y_c, x_c = np.mgrid[0:H, 0:W]
                c_map += np.sin(x_c * f + ph) * np.cos(y_c * f + ph)
            c_norm = np.clip((c_map + 2.0) / 4.0, 0.0, 1.0)
            bg = bg * (0.88 + 0.25 * c_norm[:, :, np.newaxis])

        # Marine snow floating particulates
        for _ in range(random.randint(100, 220)):
            px = random.randint(0, W - 1)
            py = random.randint(0, H - 1)
            pr = random.randint(1, 3)
            int_val = random.randint(150, 230)
            cv2.circle(bg, (px, py), pr, (int_val, int_val + 8, int_val - 8), -1)

        return np.clip(bg, 0, 255).astype(np.uint8)

    def apply_biofouling_layers(
        self,
        overlay: np.ndarray,
        mask: np.ndarray,
        pts: List[Tuple[int, int]],
        cx: int,
        cy: int,
        tw: int,
        th: int,
        fouling_ratio: float,
        is_rock_or_coral: bool
    ):
        """Applies realistic moss, algae, biofilm, and barnacle crusts."""
        # 1. Algae Patches (Green / Olive Bio-film)
        num_patches = int(fouling_ratio * random.randint(6, 12))
        for _ in range(num_patches):
            px = cx + random.randint(-tw // 3, tw // 3)
            py = cy + random.randint(-th // 3, th // 3)
            pr = random.randint(max(3, th // 8), max(6, th // 3))
            algae_color = random.choice([
                (35, 160, 60),   # Vibrant green moss
                (20, 115, 40),   # Dark velvet moss
                (40, 130, 95),   # Olive biofilm
                (25, 90, 65)     # Decaying sediment crust
            ])
            cv2.circle(overlay, (px, py), pr, algae_color, -1)

        # 2. Filamentous Moss & Algae Tendrils (Organic fuzzy fringes)
        num_tendrils = int(fouling_ratio * random.randint(18, 40))
        for _ in range(num_tendrils):
            if pts:
                idx = random.randint(0, len(pts) - 1)
                bx, by = pts[idx]
            else:
                bx = cx + random.randint(-tw // 2, tw // 2)
                by = cy + random.randint(-th // 2, th // 2)

            t_len = random.randint(4, 15)
            t_ang = random.uniform(0, math.pi * 2)
            ex = int(bx + t_len * math.cos(t_ang))
            ey = int(by + t_len * math.sin(t_ang))
            t_col = (random.randint(30, 65), random.randint(140, 210), random.randint(30, 75))
            cv2.line(overlay, (bx, by), (ex, ey), t_col, thickness=random.randint(1, 2))
            cv2.line(mask, (bx, by), (ex, ey), 255, thickness=2)

        # 3. Calcified Barnacle Clusters & Tubeworms
        num_barnacles = int(fouling_ratio * random.randint(8, 22))
        for _ in range(num_barnacles):
            bx = cx + random.randint(-tw // 3, tw // 3)
            by = cy + random.randint(-th // 3, th // 3)
            cv2.circle(overlay, (bx, by), random.randint(2, 4), (215, 230, 235), -1)
            cv2.circle(overlay, (bx, by), 1, (110, 120, 125), -1)

    def render_object(
        self,
        canvas: np.ndarray,
        class_id: int,
        cx: int,
        cy: int,
        tw: int,
        th: int,
        angle_deg: float,
        fouling_ratio: float = 0.75
    ) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        H, W = canvas.shape[:2]
        props = BIOFOULED_TAXONOMY[class_id]
        overlay = canvas.copy()
        mask = np.zeros((H, W), dtype=np.uint8)
        pts_list = []

        is_debris = props["is_debris"]

        if class_id == 1:  # Biofouled Metal Scrap / Shipwreck Hull
            # Rectilinear metallic plate with structural welds
            rect = ((cx, cy), (tw, th), angle_deg)
            box = np.int32(cv2.boxPoints(rect))
            pts_list = [tuple(p) for p in box]
            cv2.fillPoly(overlay, [box], (35, 55, 115)) # Rust metallic BGR
            cv2.fillPoly(mask, [box], 255)
            # Metallic structural ribs
            for i in range(-tw // 3, tw // 3, 12):
                cv2.line(overlay, (cx + i, cy - th // 3), (cx + i, cy + th // 3), (60, 90, 160), 2)
            # Apply biofouling over metal
            self.apply_biofouling_layers(overlay, mask, pts_list, cx, cy, tw, th, fouling_ratio, is_rock_or_coral=False)

        elif class_id == 4:  # Biofouled Marine Plastic Container
            cv2.ellipse(overlay, (cx, cy), (tw // 2, th // 2), angle_deg, 0, 360, (200, 170, 60), -1)
            cv2.ellipse(mask, (cx, cy), (tw // 2, th // 2), angle_deg, 0, 360, 255, -1)
            cv2.rectangle(overlay, (cx - 3, cy - th // 2 - 4), (cx + 3, cy - th // 2), (220, 60, 40), -1)
            self.apply_biofouling_layers(overlay, mask, [], cx, cy, tw, th, fouling_ratio, is_rock_or_coral=False)

        elif class_id == 0:  # Biofouled Ghost Gear / Entangled Net
            net_pts = []
            for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                r = (tw // 2) * random.uniform(0.7, 1.1)
                net_pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
            cv2.fillPoly(overlay, [np.array(net_pts, dtype=np.int32)], (40, 150, 70))
            cv2.fillPoly(mask, [np.array(net_pts, dtype=np.int32)], 255)
            # Netting grid
            for x_off in range(-tw // 2, tw // 2, 7):
                cv2.line(overlay, (cx + x_off, cy - th // 2), (cx + x_off, cy + th // 2), (180, 235, 180), 1)
            self.apply_biofouling_layers(overlay, mask, net_pts, cx, cy, tw, th, fouling_ratio, is_rock_or_coral=False)

        elif class_id == 7:  # Biofouled Natural Rock Boulder (Non-Debris)
            num_v = random.randint(7, 11)
            angles = np.sort(np.random.uniform(0, 2 * math.pi, num_v))
            pts_list = []
            for a in angles:
                r_rad = random.uniform(0.7, 1.15)
                pts_list.append((int(cx + (tw // 2) * r_rad * math.cos(a)), int(cy + (th // 2) * r_rad * math.sin(a))))
            rock_poly = np.array(pts_list, dtype=np.int32)
            cv2.fillConvexPoly(overlay, rock_poly, (90, 100, 105)) # Granite gray
            cv2.fillConvexPoly(mask, rock_poly, 255)
            # Mineral fracture lines
            for i in range(len(pts_list)):
                cv2.line(overlay, pts_list[i], pts_list[(i + 1) % len(pts_list)], (60, 70, 75), 2)
            # Heavy moss, algae, and barnacle crusts on rock
            self.apply_biofouling_layers(overlay, mask, pts_list, cx, cy, tw, th, fouling_ratio, is_rock_or_coral=True)

        elif class_id == 6:  # Biofouled Coral Reef & Sponge Mound (Non-Debris)
            for _ in range(6):
                ox = cx + random.randint(-tw // 3, tw // 3)
                oy = cy + random.randint(-th // 3, th // 3)
                r_c = random.randint(th // 4, th // 2)
                cv2.circle(overlay, (ox, oy), r_c, (35, 185, 110), -1)
                cv2.circle(mask, (ox, oy), r_c, 255, -1)
            self.apply_biofouling_layers(overlay, mask, [], cx, cy, tw, th, fouling_ratio, is_rock_or_coral=True)

        else:  # UXO, Cable, Pipeline
            rect = ((cx, cy), (tw, th), angle_deg)
            box = np.int32(cv2.boxPoints(rect))
            cv2.fillPoly(overlay, [box], props["base_bgr"])
            cv2.fillPoly(mask, [box], 255)
            self.apply_biofouling_layers(overlay, mask, [tuple(p) for p in box], cx, cy, tw, th, fouling_ratio, is_rock_or_coral=False)

        # Alpha composite
        alpha = 0.85
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)

        # Bounding box
        min_x = max(0, cx - tw // 2)
        max_x = min(W - 1, cx + tw // 2)
        min_y = max(0, cy - th // 2)
        max_y = min(H - 1, cy + th // 2)

        norm_cx = (min_x + max_x) / (2.0 * W)
        norm_cy = (min_y + max_y) / (2.0 * H)
        norm_w = (max_x - min_x) / float(W)
        norm_h = (max_y - min_y) / float(H)

        yolo_box = [round(norm_cx, 6), round(norm_cy, 6), round(norm_w, 6), round(norm_h, 6)]
        
        meta = {
            "class_id": class_id,
            "class_name": props["name"],
            "is_debris": is_debris,
            "category": props["category"],
            "fouling_ratio": round(fouling_ratio, 2),
            "z_impedance": props["z_impedance"]
        }

        return canvas, yolo_box, meta


# ==============================================================================
# Sonar Biofouling Acoustic Synthesizer
# ==============================================================================
class BiofoulingSonarSynthesizer:
    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height

    def generate_seabed(self) -> np.ndarray:
        H, W = self.height, self.width
        octave1 = cv2.resize(np.random.normal(120, 26, (H // 32, W // 32)), (W, H), interpolation=cv2.INTER_CUBIC)
        octave2 = cv2.resize(np.random.normal(0, 16, (H // 8, W // 8)), (W, H), interpolation=cv2.INTER_CUBIC)
        base = octave1 + octave2
        
        # TVG
        dist = np.abs(np.linspace(-1.0, 1.0, W)).reshape(1, W)
        tvg = np.clip(1.0 - (dist ** 1.8) * 0.45 + np.exp(-((dist * 6.0) ** 2)) * 0.35, 0.2, 1.4)
        base = base * tvg
        
        # Speckle
        speckle = np.random.rayleigh(scale=1.0, size=(H, W))
        return np.clip(base * (0.75 + 0.25 * speckle), 0, 255).astype(np.uint8)

    def render_sonar_target(
        self,
        canvas: np.ndarray,
        class_id: int,
        cx: int,
        cy: int,
        tw: int,
        th: int,
        angle_deg: float,
        fouling_ratio: float = 0.75
    ) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        H, W = canvas.shape[:2]
        props = BIOFOULED_TAXONOMY[class_id]
        is_debris = props["is_debris"]
        
        # Shadow projection
        nadir_x = W // 2
        shadow_dir = 1.0 if cx >= nadir_x else -1.0
        slant = max(0.4, abs(cx - nadir_x) / (W * 0.5))
        shadow_len = int(tw * 1.6 * slant * random.uniform(1.2, 1.8))
        shadow_len = max(20, min(shadow_len, 120))

        # Shadow mask
        shadow_mask = np.zeros((H, W), dtype=np.uint8)
        s_start = int(cx + (tw // 2) * shadow_dir)
        s_end = int(cx + (tw // 2 + shadow_len) * shadow_dir)
        s_pts = np.array([
            [s_start, cy - th // 2],
            [s_end, cy - int(th * 0.65)],
            [s_end, cy + int(th * 0.65)],
            [s_start, cy + th // 2]
        ], dtype=np.int32)
        cv2.fillConvexPoly(shadow_mask, s_pts, 255)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0)
        
        # Sharpness of shadow: Metal debris has crisp cutoff; Rocks have diffuse penumbra
        shadow_atten = 0.95 if is_debris else 0.85
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - (shadow_mask.astype(np.float32) / 255.0) * shadow_atten), 0, 255).astype(np.uint8)

        # Highlight reflection
        highlight_mask = np.zeros((H, W), dtype=np.uint8)
        
        if is_debris:
            # Metal / Polymer debris: High specular return through biofouling
            h_val = int(220 + random.randint(10, 30))
            rect = ((cx, cy), (tw, th), angle_deg)
            box = cv2.boxPoints(rect)
            cv2.fillPoly(highlight_mask, [np.int32(box)], h_val)
        else:
            # Natural Rock / Coral: Diffuse granular acoustic scattering
            h_val = int(160 + random.randint(10, 25))
            num_v = random.randint(7, 10)
            angles = np.sort(np.random.uniform(0, 2 * math.pi, num_v))
            pts = []
            for a in angles:
                r_rad = random.uniform(0.7, 1.1)
                pts.append([int(cx + (tw // 2) * r_rad * math.cos(a)), int(cy + (th // 2) * r_rad * math.sin(a))])
            cv2.fillConvexPoly(highlight_mask, np.array(pts, dtype=np.int32), h_val)
            # Granular texture
            noise = np.random.normal(0, 25, (H, W)).astype(np.float32)
            highlight_mask = np.clip(highlight_mask.astype(np.float32) + noise * (highlight_mask > 0), 0, 255).astype(np.uint8)

        canvas = np.clip(canvas.astype(np.float32) + highlight_mask.astype(np.float32), 0, 255).astype(np.uint8)

        min_x = max(0, cx - tw // 2)
        max_x = min(W - 1, cx + tw // 2)
        min_y = max(0, cy - th // 2)
        max_y = min(H - 1, cy + th // 2)

        yolo_box = [
            round((min_x + max_x) / (2.0 * W), 6),
            round((min_y + max_y) / (2.0 * H), 6),
            round((max_x - min_x) / float(W), 6),
            round((max_y - min_y) / float(H), 6)
        ]

        meta = {
            "class_id": class_id,
            "class_name": props["name"],
            "is_debris": is_debris,
            "category": props["category"],
            "fouling_ratio": round(fouling_ratio, 2),
            "z_impedance": props["z_impedance"]
        }

        return canvas, yolo_box, meta


# ==============================================================================
# Master Biofouling Dataset Generation Routine
# ==============================================================================
def generate_biofouled_debris_and_mimics_dataset(
    output_dir: str = "data/hydrophys_8class_dataset/biofouled_expert_corpus",
    samples: int = 1500,
    img_size: int = 640
):
    base_path = Path(output_dir)
    for s in ["train", "val", "test"]:
        (base_path / "images" / s).mkdir(parents=True, exist_ok=True)
        (base_path / "labels" / s).mkdir(parents=True, exist_ok=True)

    optical_synth = BiofoulingOpticalSynthesizer(img_size, img_size)
    sonar_synth = BiofoulingSonarSynthesizer(img_size, img_size)

    splits = ["train", "val", "test"]
    split_weights = [0.70, 0.20, 0.10]

    records = []
    class_counts = {i: 0 for i in range(8)}

    print("\n==========================================================================")
    print("  EXPERT BIOFOULED DEBRIS VS NATURAL MIMICS DATASET GENERATOR             ")
    print("==========================================================================")
    print(f"[*] Target Directory   : {base_path.resolve()}")
    print(f"[*] Total Sample Count : {samples:,} Expert Multi-Modal Frames")
    print(f"[*] Focus              : Heavy Algae/Moss/Barnacle Scraps vs Natural Rocks & Corals")

    for idx in range(samples):
        r_sp = random.random()
        split = "train" if r_sp < 0.70 else ("val" if r_sp < 0.90 else "test")
        is_sonar = (random.random() < 0.50)
        sample_id = f"BIOFOUL_{'SONAR' if is_sonar else 'OPTIC'}_{idx:06d}"

        # Balanced object selection (50% genuine biofouled debris, 50% natural biofouled rocks/corals)
        num_targets = random.choice([1, 2])
        chosen_classes = []
        for _ in range(num_targets):
            if random.random() < 0.50:
                # Genuine Debris (Metal, Plastic, Net, Cable, UXO)
                c = random.choice([0, 1, 2, 3, 4, 5])
            else:
                # Natural Mimics (Biofouled Rock, Coral, Algae Bed)
                c = random.choice([6, 7])
            chosen_classes.append(c)

        yolo_labels = []
        metas = []

        fouling_ratio = random.uniform(0.60, 0.95)

        if is_sonar:
            canvas = sonar_synth.generate_seabed()
            for cls_id in chosen_classes:
                class_counts[cls_id] += 1
                tw = random.randint(int(img_size * 0.08), int(img_size * 0.20))
                th = max(14, int(tw / random.uniform(1.2, 3.0)))
                cx = random.randint(tw // 2 + 20, img_size - tw // 2 - 20)
                cy = random.randint(th // 2 + 20, img_size - th // 2 - 20)
                ang = random.uniform(-45.0, 45.0)

                canvas, box, meta = sonar_synth.render_sonar_target(canvas, cls_id, cx, cy, tw, th, ang, fouling_ratio)
                yolo_labels.append([cls_id] + box)
                metas.append(meta)

            final_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        else:
            canvas = optical_synth.generate_water_background(depth_m=random.uniform(3.0, 25.0))
            for cls_id in chosen_classes:
                class_counts[cls_id] += 1
                tw = random.randint(int(img_size * 0.09), int(img_size * 0.22))
                th = max(16, int(tw / random.uniform(1.0, 2.5)))
                cx = random.randint(tw // 2 + 25, img_size - tw // 2 - 25)
                cy = random.randint(th // 2 + 25, img_size - th // 2 - 25)
                ang = random.uniform(-60.0, 60.0)

                canvas, box, meta = optical_synth.render_object(canvas, cls_id, cx, cy, tw, th, ang, fouling_ratio)
                yolo_labels.append([cls_id] + box)
                metas.append(meta)

            final_img = canvas

        # Save Image & Label
        img_path = base_path / "images" / split / f"{sample_id}.jpg"
        lbl_path = base_path / "labels" / split / f"{sample_id}.txt"
        cv2.imwrite(str(img_path), final_img, [cv2.IMWRITE_JPEG_QUALITY, 94])

        label_lines = [" ".join(map(str, row)) for row in yolo_labels]
        lbl_path.write_text("\n".join(label_lines) + "\n")

        records.append({
            "sample_id": sample_id,
            "split": split,
            "domain": "sonar" if is_sonar else "optical",
            "targets": metas
        })

    # Summary manifest
    manifest_file = base_path / "biofouled_expert_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({
            "name": "HydroPhys Expert Biofouled Debris vs Natural Mimics Corpus",
            "total_samples": samples,
            "class_distribution": {BIOFOULED_TAXONOMY[k]["name"]: c for k, c in class_counts.items()},
            "records_count": len(records)
        }, f, indent=2)

    print("\n==========================================================================")
    print("  [SUCCESS] EXPERT BIOFOULED DATASET GENERATION COMPLETE                   ")
    print("==========================================================================")
    print(f"[*] Total Generated     : {samples:,} Biofouled Frames + YOLO Labels")
    print(f"[*] Output Directory    : {base_path}")
    print(f"[*] Manifest Catalog    : {manifest_file}")
    for cid, cnt in class_counts.items():
        name = BIOFOULED_TAXONOMY[cid]["name"]
        deb = "DEBRIS" if BIOFOULED_TAXONOMY[cid]["is_debris"] else "PROTECTED NATURAL"
        print(f"  Class {cid} [{name:<22}] ({deb:<18}): {cnt:>4} instances")
    print("==========================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/hydrophys_8class_dataset/biofouled_expert_corpus")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    generate_biofouled_debris_and_mimics_dataset(
        output_dir=args.output_dir,
        samples=args.samples,
        img_size=args.img_size
    )

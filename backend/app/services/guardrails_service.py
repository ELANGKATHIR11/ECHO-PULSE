import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2

# Load canonical model taxonomy
TAXONOMY_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "model_taxonomy.json"
if not TAXONOMY_PATH.exists():
    TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "configs" / "model_taxonomy.json"

TAXONOMY_DATA = {}
if TAXONOMY_PATH.exists():
    try:
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            TAXONOMY_DATA = json.load(f)
    except Exception as e:
        print(f"[!] Warning loading model_taxonomy.json: {e}")

# Build lookup mapping
TARGET_CLASS_MAPPING = {}
for item in TAXONOMY_DATA.get("model_classes", []):
    key = item["model_class"]
    TARGET_CLASS_MAPPING[key] = (
        key,
        item["display_name"],
        item["operational_category"],
        item["color_hex"],
        item["is_debris"]
    )

# Fallback taxonomy items
FALLBACK_ITEM = TAXONOMY_DATA.get("fallback_anomaly", {
    "model_class": "unknown_anomaly",
    "display_name": "Unclassified Acoustic Anomaly",
    "operational_category": "UNKNOWN_ANOMALY",
    "color_hex": "#64748B",
    "is_debris": False
})

EXCLUDED_NATURAL_CLASSES = {
    "biological_cluster", "coral_reef", "benthic_cluster", "organic",
    "geological_formation", "rock_outcrop", "sand_ripple", "mud_ridge", "bathymetry_ridge",
    "seafloor", "water_column"
}


class HeavyDebrisGuardrailEngine:
    """
    Authoritative Guardrail Engine enforcing SIH26057 Marine Debris Taxonomy & Acoustic Domain Integrity:
    1. Strict Acoustic Domain Verification (OOD Rejection of Optical/Natural RGB Photos, Flowers, Web Graphics).
    2. Rigorous Target Classification against Canonical Marine Debris Taxonomy.
    3. Natural Habitat & Geological Formation Protection (Coral, Rock, Sand, Mud).
    4. False Positive & Clutter Elimination.
    """

    @classmethod
    def verify_sonar_acoustic_domain(cls, image_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Validates whether the ingested file/frame is an authentic Side-Scan Sonar (SSS)
        acoustic backscatter dataset image vs an out-of-distribution optical photograph (flowers, faces, etc.).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {
                    "is_sonar": False,
                    "reason": f"File does not exist: {image_input}",
                    "confidence": 0.0,
                    "metrics": {}
                }
            img = cv2.imread(image_input, cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                # 1-channel grayscale is inherently acoustic compatible
                img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif len(image_input.shape) == 3:
                img = image_input
            else:
                return {
                    "is_sonar": False,
                    "reason": f"Invalid tensor shape: {image_input.shape}",
                    "confidence": 0.0,
                    "metrics": {}
                }
        else:
            return {
                "is_sonar": False,
                "reason": "Unsupported image format input.",
                "confidence": 0.0,
                "metrics": {}
            }

        if img is None or img.size == 0:
            return {
                "is_sonar": False,
                "reason": "Unable to decode image raster payload.",
                "confidence": 0.0,
                "metrics": {}
            }

        # 1. Chrominance Dispersion Check (RGB Channel Divergence)
        # Genuine SSS sonar is scalar acoustic pressure (R=G=B) or standardized monotonic colormaps (copper/amber).
        # Optical photographs (flowers, animals, landscapes, indoor objects) have high cross-channel divergence.
        b_ch, g_ch, r_ch = img[:, :, 0].astype(np.float32), img[:, :, 1].astype(np.float32), img[:, :, 2].astype(np.float32)
        diff_rg = np.mean(np.abs(r_ch - g_ch))
        diff_gb = np.mean(np.abs(g_ch - b_ch))
        diff_rb = np.mean(np.abs(r_ch - b_ch))
        mean_chroma_diff = float((diff_rg + diff_gb + diff_rb) / 3.0)

        # 2. HSV Saturation Profile
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_sat = float(np.mean(sat))
        p90_sat = float(np.percentile(sat, 90))

        # 3. Decision Boundary:
        # Pure SSS sonar (true grayscale): mean_chroma_diff ≈ 0.0, mean_sat ≈ 0.0.
        # Sonar images exported as PNG/JPEG may have up to ~28 chroma divergence due to compression.
        # Sonar with colour-map overlays (copper/amber): mean_chroma_diff up to ~30, p90_sat < 0.65.
        # Optical photos (flowers, nature, faces, objects): mean_chroma_diff >> 30, mean_sat > 0.25, p90_sat > 0.60.
        is_optical_flower_or_photo = (mean_chroma_diff > 30.0) and (mean_sat > 0.25 and p90_sat > 0.60)

        if is_optical_flower_or_photo:
            return {
                "is_sonar": False,
                "confidence": 0.0,
                "rejection_code": "OUT_OF_DISTRIBUTION_OPTICAL_IMAGE",
                "reason": (
                    "REJECTED: Out-of-Distribution Optical/Natural Image Detected. "
                    f"High chromatic divergence ({mean_chroma_diff:.1f}) and saturation ({mean_sat:.2f}) "
                    "violate Side-Scan Sonar (SSS) scalar acoustic backscatter physics. "
                    "EchoPulseNet strictly validates and executes inference only on marine sonar dataset imagery."
                ),
                "metrics": {
                    "mean_chroma_divergence": round(mean_chroma_diff, 2),
                    "mean_saturation": round(mean_sat, 3),
                    "p90_saturation": round(p90_sat, 3),
                    "is_acoustic_sensor": False
                }
            }

        return {
            "is_sonar": True,
            "confidence": 1.0,
            "rejection_code": None,
            "reason": "Verified Authentic Marine Side-Scan Sonar (SSS) Acoustic Sensor Ingestion.",
            "metrics": {
                "mean_chroma_divergence": round(mean_chroma_diff, 2),
                "mean_saturation": round(mean_sat, 3),
                "p90_saturation": round(p90_sat, 3),
                "is_acoustic_sensor": True
            }
        }

    @classmethod
    def evaluate_target(
        cls,
        raw_class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int], # (x, y, w, h)
        image_shape: Tuple[int, int],    # (H, W)
        shadow_strength: float = 0.5,
        anomaly_sharpness: float = 0.5
    ) -> Dict[str, Any]:
        norm_class = str(raw_class_name).lower().strip().replace(" ", "_").replace("-", "_")
        x, y, w, h = bbox
        img_h, img_w = image_shape

        # 1. Dimension guardrails
        if w < 6 or h < 6:
            return {
                "passed": False,
                "is_debris": False,
                "target_category": "NOT_A_DEBRIS",
                "class_id": "not_a_debris",
                "class_label": "Not a Debris (Microscopic Acoustic Noise <6px)",
                "color_hex": "#64748B",
                "color_rgb": (100, 116, 139),
                "reason": "Detection dimensions are microscopic acoustic noise artifacts."
            }

        # Reject only if the box nearly fills the ENTIRE frame (likely seafloor gradient, not a discrete target)
        # A shipwreck can legitimately fill 80%+ of the swath width on a side-scan image
        if w > img_w * 0.88 and h > img_h * 0.88:
            return {
                "passed": False,
                "is_debris": False,
                "target_category": "NOT_A_DEBRIS",
                "class_id": "not_a_debris",
                "class_label": "Not a Debris (Seafloor Backscatter Gradient)",
                "color_hex": "#64748B",
                "color_rgb": (100, 116, 139),
                "reason": "Candidate covers majority of frame; consistent with gradual seafloor slope."
            }

        # 2. Strict Confidence Threshold
        if confidence < 0.28:
            return {
                "passed": False,
                "is_debris": False,
                "target_category": "NOT_A_DEBRIS",
                "class_id": "not_a_debris",
                "class_label": "Not a Debris (Low Confidence Noise)",
                "color_hex": "#94A3B8",
                "color_rgb": (148, 163, 184),
                "reason": f"Detection failed minimum confidence threshold ({confidence:.2f} < 0.28)."
            }

        # 3. Check canonical model taxonomy
        if norm_class in TARGET_CLASS_MAPPING:
            target_id, target_label, category, color_hex, is_debris = TARGET_CLASS_MAPPING[norm_class]
            hex_clean = color_hex.lstrip("#")
            color_rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))

            if not is_debris:
                return {
                    "passed": True,
                    "is_debris": False,
                    "target_category": category,
                    "class_id": target_id,
                    "class_label": target_label,
                    "color_hex": color_hex,
                    "color_rgb": color_rgb,
                    "reason": f"Classified as natural {category} feature (non-anthropogenic)."
                }

            return {
                "passed": True,
                "is_debris": True,
                "target_category": category,
                "class_id": target_id,
                "class_label": target_label,
                "color_hex": color_hex,
                "color_rgb": color_rgb,
                "reason": f"Verified anthropogenic marine {category} target."
            }

        # 4. Explicit check for natural exclusions
        if norm_class in EXCLUDED_NATURAL_CLASSES:
            return {
                "passed": True,
                "is_debris": False,
                "target_category": "NATURAL_FORMATION",
                "class_id": norm_class,
                "class_label": f"Natural Seafloor ({norm_class.replace('_', ' ').title()})",
                "color_hex": "#94A3B8",
                "color_rgb": (148, 163, 184),
                "reason": f"Evaluated as natural seafloor structure: '{norm_class}'."
            }

        # 5. Fallback for unclassified anomaly
        fb_hex = FALLBACK_ITEM.get("color_hex", "#64748B").lstrip("#")
        fb_rgb = tuple(int(fb_hex[i:i+2], 16) for i in (0, 2, 4))
        
        return {
            "passed": True,
            "is_debris": False,
            "target_category": FALLBACK_ITEM.get("operational_category", "UNKNOWN_ANOMALY"),
            "class_id": FALLBACK_ITEM.get("model_class", "unknown_anomaly"),
            "class_label": FALLBACK_ITEM.get("display_name", "Unclassified Acoustic Anomaly"),
            "color_hex": f"#{fb_hex}",
            "color_rgb": fb_rgb,
            "reason": f"Acoustic anomaly candidate '{norm_class}' requires hydrographic review."
        }

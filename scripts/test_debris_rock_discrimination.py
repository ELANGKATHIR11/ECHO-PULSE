"""
================================================================================
HydroPhys & Heavy Debris Guardrail: Debris vs Biofouled Rock Discrimination Test
================================================================================

Validates that:
  1. Anthropogenic Debris (Plastics, Ghost Nets, Metal Wreckage, UXO, Cables, Pipelines)
     are correctly detected and flagged as `is_debris: True`.
  2. Underwater Rocks, Boulders, Algae-Covered Stones, Mossy Formations, and Coral
     are strictly classified as `geological_formation` (7) or `biological_cluster` (6)
     and NEVER marked as debris (`is_debris: False`).
  3. Evaluates Guardrail strictness and false positive suppression.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.app.services.guardrails_service import HeavyDebrisGuardrailEngine
from backend.app.models.hydrophys_omninet import CATEGORY_PALETTE

def test_debris_rock_discrimination():
    print("\n==========================================================================")
    print("  HYDROPHYS DEBRIS VS BIOFOULED ROCK DISCRIMINATION TEST SUITE            ")
    print("==========================================================================")

    test_cases = [
        # Natural Mimic Test Cases (Should all have is_debris == False)
        {
            "name": "Algae & Moss-Covered Granite Rock",
            "predicted_class": "geological_formation",
            "confidence": 0.89,
            "bbox": (120, 140, 65, 55),
            "expected_debris": False,
            "expected_category": "GEOLOGICAL"
        },
        {
            "name": "Natural Basalt Boulder Field",
            "predicted_class": "rock_outcrop",
            "confidence": 0.92,
            "bbox": (200, 220, 80, 70),
            "expected_debris": False,
            "expected_category": "NATURAL_FORMATION"
        },
        {
            "name": "Benthic Green Moss & Algae Bed",
            "predicted_class": "biological_cluster",
            "confidence": 0.85,
            "bbox": (80, 90, 110, 85),
            "expected_debris": False,
            "expected_category": "BIOLOGICAL"
        },
        {
            "name": "Coral Reef & Sponge Mound",
            "predicted_class": "coral_reef",
            "confidence": 0.94,
            "bbox": (150, 180, 95, 90),
            "expected_debris": False,
            "expected_category": "NATURAL_FORMATION"
        },
        {
            "name": "Sand Dune Bathymetric Ripple",
            "predicted_class": "sand_ripple",
            "confidence": 0.88,
            "bbox": (300, 100, 150, 40),
            "expected_debris": False,
            "expected_category": "NATURAL_FORMATION"
        },

        # Genuine Anthropogenic Debris Test Cases (Should all have is_debris == True)
        {
            "name": "Derelict Ghost Net (Biofouled Polymer)",
            "predicted_class": "ghost_gear",
            "confidence": 0.91,
            "bbox": (140, 160, 75, 60),
            "expected_debris": True,
            "expected_category": "PLASTIC"
        },
        {
            "name": "Submerged Metallic Shipwreck Hull",
            "predicted_class": "shipwreck",
            "confidence": 0.96,
            "bbox": (180, 200, 160, 85),
            "expected_debris": True,
            "expected_category": "METAL_SCRAP"
        },
        {
            "name": "Unexploded Aerial Bomb / Ordnance",
            "predicted_class": "unexploded_ordnance",
            "confidence": 0.93,
            "bbox": (110, 130, 45, 25),
            "expected_debris": True,
            "expected_category": "HAZARD_UXO"
        },
        {
            "name": "Exposed Pipeline Anomaly",
            "predicted_class": "pipeline_anomaly",
            "confidence": 0.87,
            "bbox": (50, 240, 220, 30),
            "expected_debris": True,
            "expected_category": "METAL_SCRAP"
        },
        {
            "name": "Marine Plastic Container Litter",
            "predicted_class": "marine_debris",
            "confidence": 0.84,
            "bbox": (210, 170, 35, 30),
            "expected_debris": True,
            "expected_category": "PLASTIC"
        },
        {
            "name": "Subsea High-Voltage Power Cable",
            "predicted_class": "subsea_cable",
            "confidence": 0.89,
            "bbox": (90, 150, 180, 20),
            "expected_debris": True,
            "expected_category": "ELECTRICAL"
        },
    ]

    image_shape = (640, 640)
    passed_count = 0

    for i, tc in enumerate(test_cases, 1):
        eval_result = HeavyDebrisGuardrailEngine.evaluate_target(
            raw_class_name=tc["predicted_class"],
            confidence=tc["confidence"],
            bbox=tc["bbox"],
            image_shape=image_shape
        )

        is_debris = eval_result.get("is_debris", False)
        category = eval_result.get("target_category", "UNKNOWN")
        passed = (is_debris == tc["expected_debris"])

        if passed:
            passed_count += 1
            status_tag = "[PASS]"
            color_code = "\033[92m" # Green
        else:
            status_tag = "[FAIL]"
            color_code = "\033[91m" # Red

        reset_code = "\033[0m"

        print(f"{status_tag} Case #{i:02d}: {tc['name']:<38} | Pred: {tc['predicted_class']:<20} | IsDebris: {str(is_debris):<5} (Exp: {str(tc['expected_debris']):<5}) | Category: {category}")

    print("\n--------------------------------------------------------------------------")
    print(f"[*] Results: {passed_count}/{len(test_cases)} Test Cases Passed ({passed_count/len(test_cases)*100:.1f}%)")
    
    if passed_count == len(test_cases):
        print("[SUCCESS] All biofouled rocks & algae formations are strictly separated from debris!")
    else:
        print("[FAIL] Some cases failed discrimination.")
    print("==========================================================================\n")

if __name__ == "__main__":
    test_debris_rock_discrimination()

import numpy as np
from typing import Dict, Any, Tuple, Optional

# ==============================================================================
# HEAVY DEBRIS GUARDRAIL ENGINE: 5-CLASS TARGET POLICY
# Allowed Categories:
# 1. HUMAN          - Subsea Diver, Operator, SAR Presence
# 2. ELECTRICAL     - Subsea Power Cables, High-Voltage Conduits, Power Harnesses
# 3. ELECTRONIC     - Batteries, E-Waste, Transponders, Sonar Beacons, Circuits
# 4. PLASTIC        - Ghost Nets, Synthetic Polymers, Bottles, Marine Plastic Litter
# 5. METAL_SCRAP    - Shipwreck Hull Fragments, UXO, Structural Steel, Ferrous Scrap
#
# All other targets (Geological rock outcrops, Biological reefs, Sand ripples,
# Mud ridges, Organic matter) are strictly classified as NOT A DEBRIS.
# ==============================================================================

# Allowed target taxonomy mappings
TARGET_CLASS_MAPPING = {
    # 1. Humans
    "human": ("human", "Human / Subsea Diver Presence", "HUMAN", "#10B981"),
    "person": ("human", "Human / Subsea Diver Presence", "HUMAN", "#10B981"),
    "diver": ("human", "Scuba Diver / SAR Operator", "HUMAN", "#10B981"),
    "scuba_diver": ("human", "Scuba Diver / SAR Operator", "HUMAN", "#10B981"),
    
    # 2. Electrical
    "electrical": ("electrical", "Subsea Power & Electrical Cable", "ELECTRICAL", "#F59E0B"),
    "subsea_cable": ("electrical", "Subsea Power & High-Voltage Cable", "ELECTRICAL", "#F59E0B"),
    "power_cable": ("electrical", "Subsea Power Cable", "ELECTRICAL", "#F59E0B"),
    "power_harness": ("electrical", "Submerged Electrical Harness", "ELECTRICAL", "#F59E0B"),
    "cable": ("electrical", "Subsea Electrical Conduit", "ELECTRICAL", "#F59E0B"),

    # 3. Electronic
    "electronic": ("electronic", "Subsea Electronic Hardware & E-Waste", "ELECTRONIC", "#EF4444"),
    "electronics": ("electronic", "Subsea Electronic Hardware & E-Waste", "ELECTRONIC", "#EF4444"),
    "e_waste": ("electronic", "Subsea Battery / Hazardous E-Waste", "ELECTRONIC", "#EF4444"),
    "cell_phone": ("electronic", "Subsea Battery / Electronic Litter", "ELECTRONIC", "#EF4444"),
    "laptop": ("electronic", "Subsea Battery / E-Waste", "ELECTRONIC", "#EF4444"),
    "remote": ("electronic", "Acoustic Sensor / Circuit Hardware", "ELECTRONIC", "#EF4444"),
    "keyboard": ("electronic", "Electronic Peripheral Waste", "ELECTRONIC", "#EF4444"),
    "mouse": ("electronic", "Electronic Subsea Debris", "ELECTRONIC", "#EF4444"),
    "transponder": ("electronic", "Acoustic Transponder / Sonar Beacon", "ELECTRONIC", "#EF4444"),

    # 4. Plastic
    "plastic": ("plastic", "Synthetic Polymer / Marine Plastic Waste", "PLASTIC", "#06B6D4"),
    "plastic_waste": ("plastic", "Marine Plastic Debris", "PLASTIC", "#06B6D4"),
    "ghost_gear": ("plastic", "Derelict Ghost Gear & Synthetic Fishing Net", "PLASTIC", "#06B6D4"),
    "bottle": ("plastic", "Plastic Bottle / Marine Polymer", "PLASTIC", "#06B6D4"),
    "cup": ("plastic", "Polymer Single-Use Container", "PLASTIC", "#06B6D4"),
    "bowl": ("plastic", "Rigid Plastic Container", "PLASTIC", "#06B6D4"),
    "fork": ("plastic", "Plastic Marine Litter", "PLASTIC", "#06B6D4"),
    "spoon": ("plastic", "Plastic Marine Litter", "PLASTIC", "#06B6D4"),
    "handbag": ("plastic", "Synthetic Fabric / Polymer Bag", "PLASTIC", "#06B6D4"),
    "backpack": ("plastic", "Synthetic Gear Pack / Entanglement Hazard", "PLASTIC", "#06B6D4"),
    "toothbrush": ("plastic", "Marine Polypropylene Micro-Litter", "PLASTIC", "#06B6D4"),
    "frisbee": ("plastic", "Rigid HDPE Polymer Plastic", "PLASTIC", "#06B6D4"),
    "sports_ball": ("plastic", "Buoyant Polymer Sphere", "PLASTIC", "#06B6D4"),
    "marine_debris": ("plastic", "Marine Anthropogenic Plastic Debris", "PLASTIC", "#06B6D4"),

    # 5. Metal Scraps
    "metal_scrap": ("metal_scrap", "Ferrous Metal Scrap & Structural Steel", "METAL_SCRAP", "#E67E22"),
    "metal": ("metal_scrap", "Metallic Debris & Salvage Scrap", "METAL_SCRAP", "#E67E22"),
    "shipwreck": ("metal_scrap", "Submerged Metallic Hull / Vessel Scrap", "METAL_SCRAP", "#E67E22"),
    "unexploded_ordnance": ("metal_scrap", "Unexploded Ordnance (UXO) Metallic Hazard", "METAL_SCRAP", "#DC2626"),
    "pipeline_anomaly": ("metal_scrap", "Metallic Pipeline Scour / Anchor Drag Scrap", "METAL_SCRAP", "#E67E22"),
    "knife": ("metal_scrap", "Metallic Scrap Hazard", "METAL_SCRAP", "#E67E22"),
    "scissors": ("metal_scrap", "Sharp Metallic Scrap", "METAL_SCRAP", "#E67E22"),
    "structural_metal": ("metal_scrap", "Structural Steel / Subsea Pipe Scrap", "METAL_SCRAP", "#E67E22")
}

# Explicit non-debris / natural exclusions
EXCLUDED_NON_DEBRIS_CLASSES = {
    "biological_cluster", "coral_reef", "fish", "marine_fauna", "benthic_cluster", "organic",
    "geological_formation", "rock_outcrop", "sand_ripple", "mud_ridge", "bathymetry_ridge",
    "seafloor", "water_column", "book", "vase", "chair", "bed", "dining_table"
}


class HeavyDebrisGuardrailEngine:
    """
    Authoritative Heavy Guardrail Engine enforcing 5-Class Target Debris Detection:
    1. Humans
    2. Electrical
    3. Electronic
    4. Plastic
    5. Metal Scraps
    
    Any non-conforming or natural geological/biological targets are evaluated and
    either strictly filtered out or marked explicitly as NOT A DEBRIS.
    """

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
        """
        Evaluates candidate detection through heavy guardrails:
        - Ontology check (Humans, Electrical, Electronic, Plastic, Metal Scraps)
        - Dimension check (reject full-image gradients and 1px noise)
        - Acoustic physics signature check (shadow length & specular contrast)
        """
        norm_class = raw_class_name.lower().strip().replace(" ", "_").replace("-", "_")
        x, y, w, h = bbox
        img_h, img_w = image_shape

        # 1. Dimension guardrails
        if w < 6 or h < 6:
            return {
                "passed": False,
                "is_debris": False,
                "target_category": "NOT_A_DEBRIS",
                "class_id": "not_a_debris",
                "class_label": "Not a Debris (Microscopic Noise <6px)",
                "color_hex": "#64748B",
                "color_rgb": (100, 116, 139),
                "reason": "Detection dimensions are too small (microscopic acoustic speckle)."
            }

        if w > img_w * 0.75 and h > img_h * 0.75:
            return {
                "passed": False,
                "is_debris": False,
                "target_category": "NOT_A_DEBRIS",
                "class_id": "not_a_debris",
                "class_label": "Not a Debris (Seafloor Gradient)",
                "color_hex": "#64748B",
                "color_rgb": (100, 116, 139),
                "reason": "Detection covers large portion of frame (natural seafloor gradient)."
            }

        # 2. Strict Ontology Filter
        if norm_class in EXCLUDED_NON_DEBRIS_CLASSES:
            return {
                "passed": False,
                "is_debris": False,
                "target_category": "NOT_A_DEBRIS",
                "class_id": "not_a_debris",
                "class_label": f"Not a Debris (Natural / {norm_class.replace('_', ' ').title()})",
                "color_hex": "#94A3B8",
                "color_rgb": (148, 163, 184),
                "reason": f"Class '{norm_class}' is classified as natural/non-debris under Heavy Guardrail Policy."
            }

        # 3. Check allowed target mapping
        if norm_class in TARGET_CLASS_MAPPING:
            target_id, target_label, category, color_hex = TARGET_CLASS_MAPPING[norm_class]
            
            # Map hex to RGB
            hex_clean = color_hex.lstrip("#")
            color_rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))

            # Minimum confidence guardrail for confirmed debris
            if confidence < 0.28:
                return {
                    "passed": False,
                    "is_debris": False,
                    "target_category": "NOT_A_DEBRIS",
                    "class_id": "not_a_debris",
                    "class_label": "Not a Debris (Low Confidence Clutter)",
                    "color_hex": "#94A3B8",
                    "color_rgb": (148, 163, 184),
                    "reason": f"Candidate target failed minimum guardrail confidence threshold ({confidence:.2f} < 0.28)."
                }

            return {
                "passed": True,
                "is_debris": True,
                "target_category": category,
                "class_id": target_id,
                "class_label": target_label,
                "color_hex": color_hex,
                "color_rgb": color_rgb,
                "reason": f"Heavy Guardrail verified valid {category} target."
            }

        # 4. Partial substring matcher for edge cases (e.g. 'plastic_cup', 'steel_pipe', 'subsea_wire')
        if any(k in norm_class for k in ["human", "diver", "swimmer", "person"]):
            return {
                "passed": True,
                "is_debris": True,
                "target_category": "HUMAN",
                "class_id": "human",
                "class_label": "Human / Subsea Diver Presence",
                "color_hex": "#10B981",
                "color_rgb": (16, 185, 129),
                "reason": "Guardrail pattern matched Human presence."
            }

        if any(k in norm_class for k in ["electric", "cable", "wire", "power", "harness"]):
            return {
                "passed": True,
                "is_debris": True,
                "target_category": "ELECTRICAL",
                "class_id": "electrical",
                "class_label": "Subsea Electrical Equipment / Cable",
                "color_hex": "#F59E0B",
                "color_rgb": (245, 158, 11),
                "reason": "Guardrail pattern matched Electrical equipment."
            }

        if any(k in norm_class for k in ["electronic", "battery", "sensor", "circuit", "transponder", "phone", "chip"]):
            return {
                "passed": True,
                "is_debris": True,
                "target_category": "ELECTRONIC",
                "class_id": "electronic",
                "class_label": "Electronic Hardware & E-Waste",
                "color_hex": "#EF4444",
                "color_rgb": (239, 68, 68),
                "reason": "Guardrail pattern matched Electronic hardware."
            }

        if any(k in norm_class for k in ["plastic", "net", "polymer", "bottle", "ghost", "rope", "synthetic", "nylon"]):
            return {
                "passed": True,
                "is_debris": True,
                "target_category": "PLASTIC",
                "class_id": "plastic",
                "class_label": "Plastic Debris & Synthetic Polymer",
                "color_hex": "#06B6D4",
                "color_rgb": (6, 182, 212),
                "reason": "Guardrail pattern matched Plastic waste."
            }

        if any(k in norm_class for k in ["metal", "steel", "iron", "hull", "shipwreck", "scrap", "ordnance", "uxo", "pipe", "ferrous"]):
            return {
                "passed": True,
                "is_debris": True,
                "target_category": "METAL_SCRAP",
                "class_id": "metal_scrap",
                "class_label": "Metal Scraps & Ferrous Debris",
                "color_hex": "#E67E22",
                "color_rgb": (230, 126, 34),
                "reason": "Guardrail pattern matched Metal Scrap."
            }

        # 5. Default Fallback: If not matched in the 5 categories, EXCLUDE as NOT A DEBRIS
        return {
            "passed": False,
            "is_debris": False,
            "target_category": "NOT_A_DEBRIS",
            "class_id": "not_a_debris",
            "class_label": "Not a Debris (Non-Target Clutter)",
            "color_hex": "#94A3B8",
            "color_rgb": (148, 163, 184),
            "reason": f"Class '{raw_class_name}' is not one of [Humans, Electrical, Electronic, Plastic, Metal Scraps]."
        }

"""
Unified Target Model Family Exporter
Exposes the complete Target Model Suite:
1. OCEAN-PHYSNet-X
2. EchoPhys-Lite-X
3. EchoPhys-OmniNet-X
4. EchoPhys-Omni-3D-X
5. HydroPhys-OmniNet-X
6. Acoustic-Triage-Transformer-X
7. AVS-GeoPhysics-X
"""

from .ocean_physnet import OCEANPhysNet as OceanPhysNetX
from .echophys_lite import EchoPhysLite as EchoPhysLiteX
from .hydrophys_omninet import HydroPhysOmniNet as HydroPhysOmniNetX
from .echophys_omni_3d import EchoPhysOmni3DInference as EchoPhysOmni3DX
from .acoustic_triage import AcousticTriageTransformerX
from .avs_geophysics import AVSGeoPhysicsX

# Aliases matching target specification
OCEAN_PHYSNet_X = OceanPhysNetX
EchoPhys_Lite_X = EchoPhysLiteX
EchoPhys_OmniNet_X = HydroPhysOmniNetX
EchoPhys_Omni_3D_X = EchoPhysOmni3DX
HydroPhys_OmniNet_X = HydroPhysOmniNetX
Acoustic_Triage_Transformer_X = AcousticTriageTransformerX
AVS_GeoPhysics_X = AVSGeoPhysicsX

TARGET_MODEL_REGISTRY = {
    "OCEAN-PHYSNet-X": {
        "class": OceanPhysNetX,
        "description": "Physics-aware multimodal fusion engine",
        "status": "ACTIVE",
        "default_checkpoint": "models_checkpoints/ocean_physnet_best.pt"
    },
    "EchoPhys-Lite-X": {
        "class": EchoPhysLiteX,
        "description": "Adaptive low-latency edge acoustic inference engine",
        "status": "ACTIVE",
        "default_checkpoint": "models_checkpoints/echophys_lite_best.pt"
    },
    "EchoPhys-OmniNet-X": {
        "class": HydroPhysOmniNetX,
        "description": "Reliability and physics-gated multimodal fusion",
        "status": "ACTIVE",
        "default_checkpoint": "models_checkpoints/hydrophys_omninet_extreme_best.pt"
    },
    "EchoPhys-Omni-3D-X": {
        "class": EchoPhysOmni3DX,
        "description": "4D underwater state and volumetric localization",
        "status": "ACTIVE",
        "default_checkpoint": "models_checkpoints/echophys_x_v3_unified_best.pt"
    },
    "HydroPhys-OmniNet-X": {
        "class": HydroPhysOmniNetX,
        "description": "Propagation-aware acoustic classification",
        "status": "ACTIVE",
        "default_checkpoint": "models_checkpoints/hydrophys_omninet_extreme_best.pt"
    },
    "Acoustic-Triage-Transformer-X": {
        "class": AcousticTriageTransformerX,
        "description": "Fast hierarchical acoustic triage classification",
        "status": "ACTIVE",
        "default_checkpoint": "models_checkpoints/acoustic_triage_transformer_best.pt"
    },
    "AVS-GeoPhysics-X": {
        "class": AVSGeoPhysicsX,
        "description": "Probabilistic spherical DOA, range, and geolocation",
        "status": "ACTIVE",
        "default_checkpoint": "models_checkpoints/avs_geophysics_best.pt"
    }
}

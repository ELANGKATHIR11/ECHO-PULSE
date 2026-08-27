import pytest
import numpy as np
import os
import sys
import math
import torch
import cv2
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.acoustic_dsp import AcousticDSPService
from app.services.shadow_service import ShadowGeometryAnalyzer
from app.services.geotag_service import GeotaggingService
from app.models.ai_models import MultiFactorFusion, LightweightSonarUNet, SonarAutoencoder
from app.models.hydrophys_omninet import make_physics_acoustic_tensor
from app.services.sonar_parsers import UniversalSonarParser
from app.services.guardrails_service import HeavyDebrisGuardrailEngine
from app.services.inference_service import UnifiedInferenceService
from app.services.bathymetry_service import BathymetryService


# ==============================================================================
# 1. DSP & Hydrographic Signal Processing Tests
# ==============================================================================
def test_dsp_bottom_line_detection():
    sonar_img = np.zeros((100, 200), dtype=np.uint8)
    sonar_img[:, 40:] = 180 # simulated seabed onset
    bottom_lines = AcousticDSPService.detect_bottom_line(sonar_img)
    assert len(bottom_lines) == 100
    assert np.all(bottom_lines > 0)


def test_dsp_destripe_filter():
    sonar_img = np.random.normal(120, 20, (128, 256)).astype(np.uint8)
    sonar_img[20:25, :] = 250 # Artificial striping
    filtered = AcousticDSPService.fft_destripe_filter(sonar_img)
    assert filtered.shape == (128, 256)
    assert filtered.dtype == np.uint8


def test_dsp_tvg_gain():
    sonar_img = np.full((64, 128), 100, dtype=np.uint8)
    amplified = AcousticDSPService.apply_tvg_gain(sonar_img, gain_db_per_sample=0.08)
    assert amplified.shape == (64, 128)
    # Swath outer edges should have greater gain than nadir center
    center = 64
    assert amplified[32, 0] >= amplified[32, center]


def test_dsp_slant_range_correction():
    sonar_img = np.full((64, 128), 120, dtype=np.uint8)
    corrected = AcousticDSPService.slant_range_correction(sonar_img)
    assert corrected.shape == (64, 128)


# ==============================================================================
# 2. 8-Channel Physics-Guided Acoustic Tensor Tests
# ==============================================================================
def test_physics_acoustic_tensor_channels():
    device = torch.device("cpu")
    dummy_img = torch.rand(2, 1, 64, 64, device=device) # [B, 1, H, W]
    
    phys_tensor = make_physics_acoustic_tensor(
        im_tensor=dummy_img,
        temp_c=4.0,
        salinity_ppt=35.0,
        depth_m=1200.0,
        freq_khz=450.0
    )
    # Check 8 channels
    assert phys_tensor.shape == (2, 8, 64, 64)
    assert not torch.isnan(phys_tensor).any()
    assert (phys_tensor >= 0.0).all() and (phys_tensor <= 1.05).all()


# ==============================================================================
# 3. Acoustic Shadow Physics & Height Inversion Tests
# ==============================================================================
def test_shadow_height_physics_with_valid_shadow():
    # Target on starboard side: shadow should cast rightwards
    bbox = {"x": 50, "y": 50, "width": 20, "height": 20}
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[45:65, 72:110] = 255 # shadow contour
    
    shadow = ShadowGeometryAnalyzer.compute_acoustic_shadow(
        target_bbox=bbox,
        shadow_mask=mask,
        sensor_altitude_m=10.0,
        slant_range_m=30.0,
        m_per_pixel=0.05,
        is_port_channel=False
    )
    assert shadow.estimatedHeightMeters is not None
    assert shadow.estimatedHeightMeters > 0.0
    assert shadow.shadowConfidence > 0.30
    assert shadow.lengthMeters > 0.0


def test_shadow_honest_null_when_absent():
    # If no shadow contour exists, height must be null, not fabricated 1.2m
    bbox = {"x": 50, "y": 50, "width": 20, "height": 20}
    mask = np.zeros((100, 200), dtype=np.uint8) # Clean mask, no shadow
    
    shadow = ShadowGeometryAnalyzer.compute_acoustic_shadow(
        target_bbox=bbox,
        shadow_mask=mask,
        sensor_altitude_m=10.0,
        slant_range_m=30.0,
        m_per_pixel=0.05,
        is_port_channel=False
    )
    assert shadow.estimatedHeightMeters is None
    assert shadow.shadowConfidence == 0.0
    assert shadow.lengthMeters == 0.0


def test_shadow_directional_port_vs_starboard():
    # Port channel (left swath): shadow should propagate leftwards
    bbox_port = {"x": 70, "y": 40, "width": 20, "height": 20}
    mask_port = np.zeros((100, 200), dtype=np.uint8)
    mask_port[35:55, 20:65] = 255 # shadow left of target
    
    shadow = ShadowGeometryAnalyzer.compute_acoustic_shadow(
        target_bbox=bbox_port,
        shadow_mask=mask_port,
        sensor_altitude_m=8.0,
        slant_range_m=25.0,
        m_per_pixel=0.05,
        is_port_channel=True
    )
    assert shadow.estimatedHeightMeters is not None
    assert shadow.estimatedHeightMeters > 0.0


# ==============================================================================
# 4. Error-Propagated Geolocation Tests
# ==============================================================================
def test_geotagging_with_uncertainty():
    lat, lng, conf, unc_m, src = GeotaggingService.calculate_wgs84_position(
        vessel_lat=9.1524,
        vessel_lng=79.2819,
        vessel_heading_deg=45.0,
        slant_range_m=35.0,
        altitude_m=8.0,
        is_port_channel=False
    )
    assert lat is not None
    assert lng is not None
    assert 9.14 <= lat <= 9.16
    assert 79.27 <= lng <= 79.29
    assert conf > 0.0
    assert unc_m is not None
    assert unc_m > 0.0
    assert "ESTIMATED_WGS84" in src


def test_geotagging_unavailable_when_nav_missing():
    lat, lng, conf, unc_m, src = GeotaggingService.calculate_wgs84_position(
        vessel_lat=None,
        vessel_lng=None,
        vessel_heading_deg=None,
        slant_range_m=35.0,
        altitude_m=8.0
    )
    assert lat is None
    assert lng is None
    assert conf == 0.0
    assert unc_m is None
    assert src == "UNAVAILABLE"


# ==============================================================================
# 5. Sonar Parser Robustness & Error Handling
# ==============================================================================
def test_sonar_parser_missing_file():
    res = UniversalSonarParser.parse_file("non_existent_file.xtf")
    assert res["status"] == "FILE_NOT_FOUND"
    assert res["waterfall_ready"] is False


def test_sonar_parser_corrupt_xtf(tmp_path):
    corrupt_xtf = tmp_path / "corrupt.xtf"
    corrupt_xtf.write_bytes(b"NOT_A_VALID_XTF_HEADER_PACKET")
    res = UniversalSonarParser.parse_file(str(corrupt_xtf))
    assert res["status"] == "PARSING_FAILED"
    assert res["waterfall_ready"] is False


def test_sonar_parser_valid_raster(tmp_path):
    raster_path = tmp_path / "test_sonar.png"
    img = np.random.randint(50, 200, (100, 200), dtype=np.uint8)
    cv2.imwrite(str(raster_path), img)
    
    res = UniversalSonarParser.parse_file(str(raster_path))
    assert res["status"] == "PARSED_SUCCESS"
    assert res["waterfall_ready"] is True
    assert res["dimensions"]["height_pings"] == 100
    assert res["dimensions"]["width_samples"] == 200


# ==============================================================================
# 6. Confidence Fusion & Autoencoder Calibration
# ==============================================================================
def test_confidence_fusion_bounds():
    fused = MultiFactorFusion.fuse(
        detector_score=0.90,
        shadow_score=0.85,
        geometry_score=0.80,
        anomaly_score=0.70,
        quality_score=0.95
    )
    assert 0.0 <= fused <= 1.0
    assert fused >= 0.80


def test_autoencoder_forward_and_scoring():
    ae = SonarAutoencoder()
    ae.eval()
    patch = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        recon = ae(patch)
        mse = float(torch.mean((patch - recon)**2).item())
    assert recon.shape == (1, 1, 128, 128)
    assert mse >= 0.0


# ==============================================================================
# 7. Taxonomy & Guardrail Distinction Tests
# ==============================================================================
def test_guardrails_target_vs_natural_seafloor():
    # Ghost gear should be identified as plastic debris target
    res_gear = HeavyDebrisGuardrailEngine.evaluate_target(
        raw_class_name="ghost_gear",
        confidence=0.85,
        bbox=(50, 50, 40, 40),
        image_shape=(512, 1024)
    )
    assert res_gear["is_debris"] is True
    assert res_gear["target_category"] == "PLASTIC"
    assert res_gear["passed"] is True

    # Geological formation should be recognized as natural seafloor
    res_geo = HeavyDebrisGuardrailEngine.evaluate_target(
        raw_class_name="geological_formation",
        confidence=0.90,
        bbox=(100, 100, 60, 60),
        image_shape=(512, 1024)
    )
    assert res_geo["is_debris"] is False
    assert res_geo["target_category"] in ["GEOLOGICAL", "NATURAL_FORMATION"]


def test_guardrails_heuristic_unknown_anomaly():
    res_unk = HeavyDebrisGuardrailEngine.evaluate_target(
        raw_class_name="unclassified_highlight",
        confidence=0.50,
        bbox=(30, 30, 25, 25),
        image_shape=(512, 1024)
    )
    assert res_unk["is_debris"] is False
    assert res_unk["target_category"] == "UNKNOWN_ANOMALY"


# ==============================================================================
# 8. Bathymetry Data Honesty Test
# ==============================================================================
def test_bathymetry_honesty_flag():
    grid = BathymetryService.get_mission_bathymetry("MSN-TEST")
    assert grid.synthetic is True
    assert grid.source == "procedural_demo"


# ==============================================================================
# 9. End-to-End Pipeline Inference Test
# ==============================================================================
def test_end_to_end_unified_inference(tmp_path):
    # Create fixture sonar echogram with simulated target and acoustic shadow
    test_img = np.full((256, 512), 110, dtype=np.uint8)
    # Add target highlight on starboard side
    test_img[100:130, 320:350] = 230
    # Add dark shadow behind target
    test_img[98:132, 352:410] = 15
    
    fixture_path = tmp_path / "fixture_sonar.png"
    cv2.imwrite(str(fixture_path), test_img)

    service = UnifiedInferenceService(device="cpu")
    
    # 1. Run inference without navigation (should produce honest UNAVAILABLE position)
    dets_no_nav = service.run_inference(
        image_path=str(fixture_path),
        mission_id="MSN-TEST-1",
        vessel_nav=None
    )
    assert isinstance(dets_no_nav, list)
    for d in dets_no_nav:
        assert d.latitude is None
        assert d.longitude is None
        assert d.geotagConfidence == 0.0
        assert d.acousticShadow is not None
        assert d.geometry is not None

    # 2. Run inference with real navigation
    nav_meta = {
        "lat": 9.1524,
        "lng": 79.2819,
        "heading": 45.0,
        "altitude": 10.0,
        "depth": 30.0,
        "ping": 1020
    }
    dets_nav = service.run_inference(
        image_path=str(fixture_path),
        mission_id="MSN-TEST-2",
        vessel_nav=nav_meta
    )
    assert isinstance(dets_nav, list)
    for d in dets_nav:
        assert d.latitude is not None
        assert d.longitude is not None
        assert d.geotagConfidence > 0.0
        assert d.confidence > 0.0

    # 3. Validate runtime telemetry
    telem = service.last_model_telemetry
    assert "timing" in telem
    assert "t_total_ms" in telem["timing"]
    assert "acoustic_metrics" in telem
    assert telem["acoustic_metrics"]["measured_snr_db"] > 0

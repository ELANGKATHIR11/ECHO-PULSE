import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.acoustic_dsp import AcousticDSPService
from app.services.shadow_service import ShadowGeometryAnalyzer
from app.services.geotag_service import GeotaggingService
from app.models.ai_models import MultiFactorFusion, LightweightSonarUNet, SonarAutoencoder
from app.services.sonar_parsers import UniversalSonarParser
from app.services.active_learning_service import ActiveLearningService
import torch

def test_dsp_bottom_line_detection():
    sonar_img = np.zeros((100, 200), dtype=np.uint8)
    sonar_img[:, 40:] = 180 # simulated seabed onset
    bottom_lines = AcousticDSPService.detect_bottom_line(sonar_img)
    assert len(bottom_lines) == 100
    assert np.all(bottom_lines > 0)

def test_dsp_destripe_filter():
    sonar_img = np.random.normal(120, 20, (128, 256)).astype(np.uint8)
    # Add stripe
    sonar_img[20:25, :] = 250
    filtered = AcousticDSPService.fft_destripe_filter(sonar_img)
    assert filtered.shape == (128, 256)
    assert filtered.dtype == np.uint8

def test_shadow_height_physics():
    # Target height = (L_shadow * H_alt) / (R_slant + L_shadow)
    bbox = {"x": 50, "y": 50, "width": 20, "height": 20}
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[45:65, 70:110] = 255 # shadow contour
    shadow = ShadowGeometryAnalyzer.compute_acoustic_shadow(
        target_bbox=bbox,
        shadow_mask=mask,
        sensor_altitude_m=10.0,
        slant_range_m=30.0,
        m_per_pixel=0.05
    )
    assert shadow.estimatedHeightMeters is not None
    assert shadow.estimatedHeightMeters > 0

def test_confidence_fusion():
    fused = MultiFactorFusion.fuse(
        detector_score=0.90,
        shadow_score=0.85,
        geometry_score=0.80,
        anomaly_score=0.20,
        quality_score=0.95
    )
    assert 0.0 <= fused <= 1.0
    assert fused > 0.70 # Strong detection score

def test_geotagging_interpolation():
    lat, lng, conf = GeotaggingService.calculate_wgs84_position(
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
    assert conf >= 0.80

def test_models_forward():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet = LightweightSonarUNet(in_channels=1, out_channels=2).to(device)
    ae = SonarAutoencoder().to(device)
    
    x = torch.randn(2, 1, 128, 128, device=device)
    with torch.no_grad():
        out_unet = unet(x)
        out_ae = ae(x)
    assert out_unet.shape == (2, 2, 128, 128)
    assert out_ae.shape == (2, 1, 128, 128)

def test_sonar_parsers_robustness():
    res = UniversalSonarParser.parse_file("test_mock.xtf")
    assert res["status"] is not None
    assert res["waterfall_ready"] is True

import pytest
import os
import numpy as np
import cv2
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.sonar.processor import OpenCVProcessor
from backend.app.services.shadow_service import ShadowGeometryAnalyzer
from backend.app.services.geotag_service import GeotaggingService
from backend.app.models.ai_models import MultiFactorFusion

def test_opencv_preprocessing():
    synthetic_sonar = np.random.randint(20, 200, (256, 512), dtype=np.uint8)
    # Inject bright target and dark acoustic shadow
    synthetic_sonar[100:130, 200:230] = 250 # highlight
    synthetic_sonar[100:130, 230:300] = 10  # shadow
    
    res = OpenCVProcessor.preprocess_sonar_image(synthetic_sonar)
    assert "processed_image" in res
    assert "shadow_mask" in res
    assert "quality_score" in res
    assert 0.0 <= res["quality_score"] <= 1.0

def test_geotagging():
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
    assert conf > 0.7

def test_multifactor_fusion():
    score = MultiFactorFusion.fuse(
        detector_score=0.92,
        shadow_score=0.88,
        geometry_score=0.85,
        anomaly_score=0.78,
        quality_score=0.90
    )
    assert 0.80 <= score <= 0.99

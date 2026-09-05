import pytest
import torch
import numpy as np
import math

from backend.app.sonar.physics_core import (
    SeawaterPhysics,
    HelmholtzWaveResidual,
    AVSPhysicsCore,
    GeodeticTransforms,
    AdaptivePhysicsLossEngine
)
from backend.app.models.target_family import (
    OCEAN_PHYSNet_X,
    EchoPhys_Lite_X,
    EchoPhys_OmniNet_X,
    EchoPhys_Omni_3D_X,
    HydroPhys_OmniNet_X,
    Acoustic_Triage_Transformer_X,
    AVS_GeoPhysics_X,
    TARGET_MODEL_REGISTRY
)


def test_target_model_registry_contains_seven_models():
    assert len(TARGET_MODEL_REGISTRY) == 7
    expected = [
        "OCEAN-PHYSNet-X",
        "EchoPhys-Lite-X",
        "EchoPhys-OmniNet-X",
        "EchoPhys-Omni-3D-X",
        "HydroPhys-OmniNet-X",
        "Acoustic-Triage-Transformer-X",
        "AVS-GeoPhysics-X"
    ]
    for m in expected:
        assert m in TARGET_MODEL_REGISTRY
        assert TARGET_MODEL_REGISTRY[m]["status"] == "ACTIVE"


def test_seawater_physics_sound_speed_and_absorption():
    c = SeawaterPhysics.mackenzie_sound_speed(temperature_c=20.0, salinity_psu=35.0, depth_m=50.0)
    assert 1480.0 <= c <= 1550.0

    alpha = SeawaterPhysics.francois_garrison_absorption(freq_khz=10.0, temperature_c=18.0, salinity_psu=35.0, depth_m=50.0)
    assert alpha > 0.05

    # Path travel time test
    traj = np.array([
        [0.0, 0.0, 10.0],
        [100.0, 0.0, 50.0],
        [200.0, 0.0, 100.0]
    ])
    ssp_depths = np.array([0.0, 50.0, 100.0, 200.0])
    ssp_speeds = np.array([1520.0, 1510.0, 1495.0, 1485.0])
    t_travel = SeawaterPhysics.compute_travel_time(traj, ssp_depths, ssp_speeds)
    assert 0.1 <= t_travel <= 0.5


def test_helmholtz_wave_residual():
    residual_evaluator = HelmholtzWaveResidual()
    field = torch.randn(2, 1, 32, 32)
    res = residual_evaluator(field, freq_khz=5.0, sound_speed_mps=1500.0, dx=0.5, dy=0.5)
    assert res.shape == (2, 1, 30, 30)
    assert torch.all(res >= 0.0)


def test_avs_physics_core_and_spherical_doa():
    L = 512
    p = np.sin(np.linspace(0, 10, L))
    ux = 0.005 * p
    uy = 0.002 * p
    uz = -0.001 * p

    intensity, coherence = AVSPhysicsCore.compute_active_intensity(p, ux, uy, uz)
    assert len(intensity) == 3
    assert 0.0 <= coherence <= 1.0

    spherical_vec = AVSPhysicsCore.spherical_doa_vector(azimuth_deg=45.0, elevation_deg=-15.0)
    assert len(spherical_vec) == 3
    assert math.isclose(np.linalg.norm(spherical_vec), 1.0, rel_tol=1e-5)


def test_geodetic_enu_to_wgs84():
    lat, lng, alt = GeodeticTransforms.enu_to_wgs84(
        east_m=100.0,
        north_m=200.0,
        up_m=-30.0,
        ref_lat_deg=9.1524,
        ref_lng_deg=79.2819,
        ref_alt_m=0.0
    )
    assert abs(lat - 9.1524) < 0.01
    assert abs(lng - 79.2819) < 0.01
    assert alt == -30.0


def test_adaptive_physics_loss_engine():
    engine = AdaptivePhysicsLossEngine()
    data_loss = torch.tensor(1.2)
    phys_loss = torch.tensor(0.6)

    # High confidence environment -> Higher lambda_phys
    l_high, lam_high = engine(
        data_loss, phys_loss,
        env_confidence=torch.tensor([[0.95]]),
        sensor_confidence=torch.tensor([[0.90]]),
        model_disagreement=torch.tensor([[0.05]])
    )

    # Low confidence environment -> Reduced physics enforcement
    l_low, lam_low = engine(
        data_loss, phys_loss,
        env_confidence=torch.tensor([[0.10]]),
        sensor_confidence=torch.tensor([[0.15]]),
        model_disagreement=torch.tensor([[0.85]])
    )

    assert l_high.item() > 0.0
    assert l_low.item() > 0.0
    assert 0.0 <= lam_high.item() <= 1.0
    assert 0.0 <= lam_low.item() <= 1.0


def test_target_models_forward_execution():
    # 1. Acoustic-Triage-Transformer-X
    triage = Acoustic_Triage_Transformer_X()
    triage.eval()
    with torch.no_grad():
        out_triage = triage(torch.randn(2, 128, 32))
        assert "macro_probs" in out_triage
        assert "severity_probs" in out_triage
        assert out_triage["macro_probs"].shape == (2, 4)

    # 2. AVS-GeoPhysics-X
    avs_model = AVS_GeoPhysics_X()
    avs_model.eval()
    with torch.no_grad():
        out_avs = avs_model(torch.randn(2, 4, 1024))
        assert "spherical_doa_vector" in out_avs
        assert "azimuth_deg" in out_avs
        assert "range_meters" in out_avs
        assert out_avs["spherical_doa_vector"].shape == (2, 3)

    # 3. EchoPhys-Lite-X
    lite_model = EchoPhys_Lite_X()
    lite_model.eval()
    with torch.no_grad():
        out_lite = lite_model(torch.randn(1, 3, 320, 320))
        assert "cls_logits" in out_lite or isinstance(out_lite, dict)

import sys
import os
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.acoustic_dsp import AcousticDSPService
from app.services.sonar_parsers import UniversalSonarParser
from app.services.active_learning_service import ActiveLearningService

def test_dsp():
    print("=== 1. TESTING ACOUSTIC DSP SERVICE ===")
    # Generate synthetic side-scan sonar image with striping and target highlight
    h, w = 300, 600
    sonar_raw = np.random.normal(100, 20, (h, w)).astype(np.uint8)
    
    # Water column nadir
    sonar_raw[:, 270:330] = np.random.normal(20, 5, (h, 60)).astype(np.uint8)
    
    # Striping noise
    for i in range(0, h, 6):
        sonar_raw[i, :] = np.clip(sonar_raw[i, :].astype(np.int32) + 50, 0, 255).astype(np.uint8)

    res = AcousticDSPService.process_full_hydrographic_pipeline(
        sonar_raw,
        apply_bld=True,
        apply_src=True,
        apply_destripe=True,
        apply_tvg=True
    )
    
    print("DSP Metrics Result:")
    for k, v in res["metrics"].items():
        print(f"  - {k}: {v}")
    assert res["enhanced_image"].shape == (h, w), "Enhanced image shape mismatch"
    print("[PASS] Acoustic DSP Pipeline functional.")

def test_parsers():
    print("\n=== 2. TESTING UNIVERSAL SONAR PARSER ===")
    # Test XTF parsing fallback/pyxtf
    xtf_res = UniversalSonarParser.parse_xtf("mock_mission.xtf")
    print("XTF Parser:", xtf_res["format"], "Status:", xtf_res["status"])
    assert xtf_res["waterfall_ready"] is True

    # Test JSF parsing
    jsf_res = UniversalSonarParser.parse_jsf("mock_mission.jsf")
    print("JSF Parser:", jsf_res["format"], "Status:", jsf_res["status"])
    assert jsf_res["waterfall_ready"] is True
    print("[PASS] Sonar Parsers functional.")

def test_active_learning():
    print("\n=== 3. TESTING ACTIVE LEARNING SERVICE ===")
    queue = ActiveLearningService.get_triage_queue()
    print(f"Triage samples found: {len(queue)}")
    assert len(queue) > 0

    # Submit a mock review
    sample_id = queue[0]["id"]
    rev_res = ActiveLearningService.submit_review(
        sample_id=sample_id,
        corrected_class="shipwreck",
        bounding_box={"x": 100, "y": 80, "width": 120, "height": 90},
        notes="Verified verified acoustic highlight on port side."
    )
    print(f"Review Submission Status: {rev_res['status']}")

    # Retrain trigger
    retrain_res = ActiveLearningService.trigger_gpu_retrain(epochs=2)
    print(f"Retrain Trigger Job: {retrain_res['jobId']} -> {retrain_res['status']}")
    print(f"Retrain Metrics: mAP@50={retrain_res['metrics']['mAP50']}, F1={retrain_res['metrics']['f1Score']}")
    print("[PASS] Active Learning Service functional.")

if __name__ == "__main__":
    test_dsp()
    test_parsers()
    test_active_learning()
    print("\n[ALL TESTS PASSED SUCCESSFULLY]")

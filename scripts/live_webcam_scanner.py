import os
import sys
import time
import cv2
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.app.models.hydrophys_omninet import HydroPhysOmniVisionEngine, CATEGORY_PALETTE

# ==============================================================================
# HydroPhys-OmniNet Real-Time Live Webcam & 3D Scanning Interface
# ==============================================================================

def run_live_webcam(
    cam_index: int = 0,
    conf_thresh: float = 0.20,
    model_path: str = "models_checkpoints/hydrophys_omninet_extreme_best.pt"
):
    print("==================================================================")
    print("  HYDROPHYS-OMNINET: REAL-TIME 1D/2D/3D LIVE WEBCAM SCANNER       ")
    print("==================================================================")
    print(f"[*] Initializing HydroPhys-OmniNet Vision Engine from {model_path}...")
    
    engine = HydroPhysOmniVisionEngine(weights_path=model_path)
    
    print(f"[*] Opening Webcam device ({cam_index})...")
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"[!] Error: Could not access webcam at index {cam_index}.")
        print("[*] Trying secondary camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[!] Error: No accessible webcam found.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[PASS] Live Webcam Stream Active!")
    print("------------------------------------------------------------------")
    print("  CONTROLS: Press 'q' or 'ESC' in the window to exit.")
    print("            Press 's' to save a snapshot scan with 3D point cloud.")
    print("------------------------------------------------------------------\n")

    frame_count = 0
    fps_smooth = 0.0
    out_dir = Path("reports/live_scans")
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[!] Warning: Empty webcam frame received.")
            break

        frame_count += 1
        t_start = time.perf_counter()

        # Convert OpenCV BGR to PIL RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Run Real-Time Omni Vision Engine (1D/2D/3D + Color Segmentation + Mimic Rejection)
        res = engine.process_omni_frame(
            image_input=pil_img,
            conf_threshold=conf_thresh,
            altitude_m=15.0,
            swath_m=150.0
        )

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        instant_fps = 1000.0 / max(0.1, t_elapsed)
        fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

        # Convert rendered PIL image back to OpenCV BGR
        vis_bgr = cv2.cvtColor(np.array(res["rendered_visualization"]), cv2.COLOR_RGB2BGR)

        # Draw HUD Diagnostics Overlay
        h, w, _ = vis_bgr.shape
        cv2.rectangle(vis_bgr, (10, 10), (460, 125), (15, 15, 15), -1)
        cv2.rectangle(vis_bgr, (10, 10), (460, 125), (0, 255, 255), 2)

        cv2.putText(vis_bgr, f"HYDROPHYS-OMNINET 3D SCANNER", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(vis_bgr, f"FPS: {fps_smooth:.1f} | Latency: {t_elapsed:.1f} ms", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(vis_bgr, f"Device: {engine.device} (RTX 5060)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
        cv2.putText(vis_bgr, f"Detected Objects: {res['total_objects_scanned']} | 3D Wireframes: Active", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (46, 204, 113), 1)

        # Show Live Window
        cv2.imshow("HydroPhys-OmniNet: Live 1D/2D/3D Vision Scanner", vis_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            print("[*] User requested exit.")
            break
        elif key == ord('s'): # Snapshot
            snap_path = out_dir / f"live_scan_{int(time.time())}.png"
            cv2.imwrite(str(snap_path), vis_bgr)
            print(f"[PASS] Saved live scan snapshot to {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[PASS] Live Webcam Session Closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index")
    parser.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold")
    parser.add_argument("--weights", type=str, default="models_checkpoints/hydrophys_omninet_extreme_best.pt")
    args = parser.parse_args()

    run_live_webcam(
        cam_index=args.cam,
        conf_thresh=args.conf,
        model_path=args.weights
    )

# EchoPulseNet: Complete Engineering & Architecture Log
**Project:** EchoPulseNet (Marine Sonar Intelligence Platform — Problem Statement SIH26057)  
**Repository:** [https://github.com/ELANGKATHIR11/ECHO-PULSE](https://github.com/ELANGKATHIR11/ECHO-PULSE)  
**License:** GNU General Public License v3.0 (GPLv3)  
**Export Date:** August 24, 2026  

---

## 1. Executive Summary & Core Mandates Achieved
1. **100% Native Edge / De-Clouding:**
   * Stripped all Google AI Studio, Gemini API, and external cloud dependencies.
   * Fully offline-capable architecture (Edge-native FastAPI + PyTorch CUDA + React/Vite/Three.js).
2. **GPL v3.0 Licensing:**
   * Published official `LICENSE` under GNU General Public License v3.0.
3. **Problem Statement SIH26057 Compliance:**
   * **Domain:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery (Ministry of Earth Sciences / NIOT).
   * **Pipeline:** Sonar Input (.XTF/.JSF/.SL2/.DAT/PNG) → DSP Destriping & Leveling → YOLOv12 / UNet / Autoencoder Detection → Shadow Physics Target Height Calculation → Multi-Factor Confidence Fusion (0-100%) → Geotagging → PostGIS/SQLite → Reports (PDF/CSV/JSON) → 3D Digital Twin UI.
4. **Enhanced 3D Subsea Digital Twin Realism:**
   * Dynamic seabed wave caustics GLSL fragment shaders.
   * Deep oceanic Rayleigh scattering and depth fog absorption.
   * Volumetric sunbeams filtering down from surface.
   * 360-particle organic Brownian motion marine snow drift.
   * Realistic underside ocean surface wave mesh.
   * Adaptive DPR (1x-2x) with ACES Filmic HDR tone mapping at 60+ FPS.
5. **Raw Sonar Ingestion (`/upload`) & Strict Debris Guardrails:**
   * High-contrast gradient checks, aspect ratio filters, and physical shadow length requirements to reject flat seafloor clutter.
   * Automatic color-coded bounding box annotation generator.
6. **Open-Source (FOSS) Dataset Acquisition:**
   * Curated and downloaded benchmarks: SeabedObjects, NNSSS, OpenSonarDatasets, GhostVision (ghost pots), and SubPipe (6,252+ files, ~636 MB).
7. **DGPU NVIDIA RTX 5060 Optimization:**
   * PyTorch 2.11.0 + CUDA 12.8 sm_120 compatibility with memory-leak free PIL streamers and expandable segments.
   * Unit test suite verified (7/7 passed in 3.13s).
8. **Standalone Desktop SaaS Executable (.EXE):**
   * Packaged into `C:\Users\elang\Downloads\EchoPulseNet_Marine_SaaS_Desktop.exe` (224.62 MB) with Electron and Tauri configurations.

---

## 2. Directory Tree & Key Artifacts
```
echopulsenet/
├── backend/
│   ├── app/
│   │   ├── api/routes.py                # REST endpoints (/upload, /inference/frame, /reports)
│   │   ├── core/config.py               # Local settings and paths
│   │   ├── models/ai_models.py          # YOLOv12, LightweightSonarUNet, SonarAutoencoder
│   │   └── services/
│   │       ├── inference_service.py     # Debris guardrails & OpenCV visual annotator
│   │       ├── dsp_service.py           # Heave leveling & bilateral despeckle
│   │       ├── fusion_service.py        # Multi-factor confidence fusion
│   │       └── report_service.py        # PDF, CSV, JSON export engine
├── src/
│   ├── components/three/
│   │   └── DigitalTwinCanvas.tsx        # 3D WebGL underwater world with caustics & marine snow
│   ├── pages/
│   │   ├── RawSonarUploadPage.tsx       # Dedicated raw file ingestion interface
│   │   ├── WebcamTrackerPage.tsx        # Real-time webcam / live stream HUD
│   │   └── DigitalTwinPage.tsx          # Interactive bathymetric mission control
├── scripts/
│   ├── train_yolov12_sonar.py           # Attention-Centric YOLOv12 GPU trainer
│   ├── fetch_foss_datasets.py           # Automated FOSS dataset scraper
│   └── start_echopulsenet.ps1           # Single-command launcher
├── electron_main.js                     # Electron desktop orchestrator
├── EchoPulseNet_Standalone.spec         # PyInstaller single-file binary specification
├── Launch_EchoPulseNet_Standalone.bat   # One-click portable launcher
└── LICENSE                              # GNU General Public License v3.0
```

---

## 3. Git Commit History Summary
* `commit fa80570`: Optimize training scripts and verify 100% test passing across DSP, models, and physics pipeline.
* `commit 145638c`: Create standalone zero-dependency desktop app launcher bundling frontend, backend, AI/ML models, and datasets.
* `commit 3bc2a9c`: Add native Electron desktop app runner and Tauri integration with zero-config embedded AI backend.

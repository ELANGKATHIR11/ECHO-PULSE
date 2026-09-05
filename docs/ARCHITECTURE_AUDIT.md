# EchoPulseNet Architecture Audit & Baseline System Topology

**Date**: September 4, 2026  
**Auditor**: Senior ML + Full-Stack + MLOps Lead  
**Repository**: `ELANGKATHIR11/ECHO-PULSE` (Branch: `main`)  
**Commit Baseline**: `006339d7`

---

## 1. Executive Architecture Overview

EchoPulseNet is a native edge deep ocean acoustic and side-scan-sonar intelligence platform. The repository is configured as a dual-runtime system:
- **Backend Core**: FastAPI Python service (`backend/app/main.py`) running on Port 8000.
- **Frontend Workstation**: React 19 + TypeScript + Vite + TailwindCSS + Three.js / React-Three-Fiber 3D ocean digital twin.
- **Desktop Packaging**: Electron desktop application shell (`electron_main.js`) with GPU acceleration (RTX 5060 + WebGL2 + Intel AI Boost NPU coprocessor integration).
- **Spatial Storage**: Spatially indexed PostgreSQL / PostGIS database (`echopulse_spatial.db` SQLite fallback and live PostgreSQL 18 PostGIS connector via `GeoAlchemy2` and `psycopg2-binary`).

---

## 2. Directory Tree Topology

```text
echopulsenet---marine-sonar-intelligence-platform/
├── .env, .env.example
├── Launch_EchoPulseNet.bat, Launch_EchoPulseNet_Standalone.bat
├── electron_main.js                     # Electron desktop main entry
├── vite.config.ts, package.json         # Node.js frontend configuration
├── backend/
│   ├── app/
│   │   ├── api/routes.py                # Core REST API route controllers
│   │   ├── core/                        # Config, database, NPU accelerator
│   │   ├── models/                      # PyTorch neural network architectures
│   │   │   ├── ocean_physnet.py         # OCEAN-PHYSNet multimodal model
│   │   │   ├── echophys_lite.py         # EchoPhys-Lite 3-ch Mamba model
│   │   │   ├── echophys_omni_3d.py      # EchoPhys-Omni-3D 1D/2D/3D model
│   │   │   ├── hydrophys_omninet.py     # HydroPhys-OmniNet CAW-SSM
│   │   │   └── ai_models.py             # U-Net, Autoencoder, MultiFactorFusion
│   │   ├── schemas/                     # Pydantic contracts & schemas
│   │   ├── services/                    # Inference, PostGIS, MPA, geotag, retrain
│   │   └── sonar/                       # DSP, audio processing, AVS locator, ocean state
│   ├── data/
│   ├── reports/
│   └── requirements.txt                 # Backend Python package requirements
├── data/                                # Training datasets & raw samples
│   ├── avs_vector_dataset/              # 4-channel AVS arrays
│   ├── hydrophone_acoustic_dataset/     # Acoustic WAV recordings
│   ├── hydrophys_8class_dataset/        # 8-class sonar data
│   ├── scraped_foss_sonar_images/       # Open seabed SSS imagery
│   └── yolo_sonar_dataset/              # YOLO formatted SSS labels
├── models_checkpoints/                  # Active and legacy weights (.pt, .onnx)
├── plots/                               # Confusion matrices & evaluation graphs
├── reports/                             # Stress benchmarks & training reports
├── scripts/                             # Training, scraping, benchmarks, packaging
├── src/                                 # React + TypeScript Frontend
│   ├── components/                      # UI components (3D canvas, glass cards, GIS)
│   ├── pages/                           # 15 interactive workstation views
│   └── index.css                        # Ocean-blue liquid glass & 3D styling
└── tests/                               # Unit and edge integration test suites
```

---

## 3. Dependency & Environment Constraints

### Python Environment
- **Python Version**: `3.12.13 (64-bit AMD64)` (Anaconda distribution)
- **PyTorch Runtime**: `2.11.0+cu128` (CUDA available: False on standard command shell; CPU fallback verified, CUDA/NPU accelerator dynamically checked in services)
- **Key Python Packages**: `fastapi>=0.115.0`, `torch>=2.1.0`, `torchvision>=0.16.0`, `ultralytics>=8.3.0`, `scipy>=1.11.0`, `GeoAlchemy2>=0.14.0`, `shapely>=2.0.0`, `pyxtf>=1.4.0`

### Node.js / Web Frontend Environment
- **Node.js**: Modern ES modules with Vite 6.2.3 and TypeScript 5.8.2.
- **UI Framework**: React 19.0.1, `@react-three/fiber 9.7.0`, `@react-three/drei 10.7.8`, `three 0.185.1`, `leaflet 1.9.4`.

---

## 4. Component Classification Audit

### A) Active Production Code
- `backend/app/main.py`: FastAPI server mounting APIs and static distribution.
- `backend/app/api/routes.py`: 1,470 lines covering inference, missions, telemetry, ocean-physnet, AVS localization, retraining.
- `backend/app/services/inference_service.py`: Orchestrates multi-modal sonar detection, shadow analysis, and multi-factor confidence fusion.
- `backend/app/sonar/ocean_state.py`: Seawater physics engine (Mackenzie SSP, Francois-Garrison acoustic absorption).
- `backend/app/sonar/avs_locator.py`: 4-channel active acoustic intensity and geodetic WGS-84 coordinate transformation.
- `src/components/glass/OceanLiquidCausticBackground.tsx`: Global 3D translucent liquid ocean background.
- `src/components/layout/CommandLayout.tsx`: Command shell layout.

### B) Reusable Models / Modules
- `backend/app/models/ocean_physnet.py`: Foundation for `OCEAN-PHYSNet-X`.
- `backend/app/models/echophys_lite.py`: Foundation for `EchoPhys-Lite-X`.
- `backend/app/models/hydrophys_omninet.py`: Foundation for `HydroPhys-OmniNet-X`.
- `backend/app/models/echophys_omni_3d.py`: Foundation for `EchoPhys-Omni-3D-X`.
- `backend/app/sonar/acoustic_classifier.py` & `audio_processor.py`: Foundation for `Acoustic-Triage-Transformer-X`.
- `backend/app/sonar/avs_locator.py`: Foundation for `AVS-GeoPhysics-X`.

### C) Duplicate / Obsolete / Candidate Legacy Models
- Old checkpoints and scripts requiring versioned archival:
  - `models_checkpoints/seabed_autoencoder.pt` & `seabed_autoencoder.onnx` (Legacy 2025 autoencoder baseline).
  - `models_checkpoints/unet_shadow_segmenter.pt` & `unet_shadow.onnx` (Legacy lightweight shadow segmenter).
  - `models_checkpoints/yolov12_echopulse_marine.pt` & `.onnx` (Baseline detector before physics-guided Mamba).
  - Historical training scripts: `scripts/train_echophys_x_v3.py`, `scripts/train_echophys_lite.py`, `scripts/train_dual_models.py`.

### D) Broken or Fragile Components
- Hardcoded device selection in legacy inference (`cuda:0` without robust CPU fallback).
- Inconsistent physics loss weighting: naive fixed weights in some scripts rather than adaptive environmental confidence:
  $$\lambda_{\text{phys}} = \sigma(f(\text{env\_conf}, \text{sensor\_conf}, \text{model\_disagree}))$$

### E) Missing Components
- Unified Physics Core module encapsulating spatially varying SSP, Helmholtz acoustic wave residuals, travel time integrals $\int_\Gamma \frac{ds}{c(x)}$, and Doppler/ocean-current compensations.
- Explicit `Acoustic-Triage-Transformer-X` fast hierarchical triage model.
- Explicit `AVS-GeoPhysics-X` probabilistic spherical DOA + range + geolocation model.
- Cohesive Model Registry documentation (`docs/MODEL_REGISTRY.md`).
- Versioned model archive structure (`archive/models/`).

### F) Current Data & Model Pipeline
1. **Sonar Stream**: Side-Scan Sonar (.xtf / raw image) $\rightarrow$ Preprocessing $\rightarrow$ Physics decomposition $\rightarrow$ Inverted shadow height $\rightarrow$ Anomaly score $\rightarrow$ WGS84 Geotag.
2. **Acoustic / Hydrophone Stream**: Audio WAV $\rightarrow$ STFT / Log-Mel $\rightarrow$ Acoustic classification $\rightarrow$ Confidence.
3. **AVS Stream**: 4-channel vector intensity $\rightarrow$ Azimuth / Elevation DOA $\rightarrow$ Transmission Loss range $\rightarrow$ Geolocation.

---

## 5. Audit Conclusion

The repository contains advanced marine acoustic and sonar implementations with strong domain specialization. Upgrading it in-place into the unified EchoPulseNet platform requires:
1. Creating the versioned archive for legacy models.
2. Building the centralized Physics Core with adaptive environmental confidence weighting.
3. Establishing the complete 7-model Target Model Family.
4. Integrating unified endpoints and UI bridges without disturbing existing functionality.

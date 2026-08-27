# 🌊 EchoPulseNet: Deep Ocean Sonar Intelligence & Defense-Grade Marine Digital Twin Platform

[![Platform](https://img.shields.io/badge/Platform-Native_Edge_AI-06b6d4?style=for-the-badge&logo=nvidia)](https://github.com/ELANGKATHIR11/ECHO-PULSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+cu128-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-HydroPhys--OmniNet_v4-38bdf8?style=for-the-badge)](https://github.com/ELANGKATHIR11/ECHO-PULSE)
[![Frontend](https://img.shields.io/badge/Frontend-React_19_|_Three.js-61DAFB?style=for-the-badge&logo=react)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-GPLv3_|_Proprietary_Models-blue?style=for-the-badge)](LICENSE)

**EchoPulseNet** is an offline-capable, 100% native edge marine intelligence system designed for real-time acoustic target detection, autonomous underwater vehicle (AUV/ROV) hydrographic surveys, interactive 3D bathymetric Digital Twin visualization, system GPS & IR distance sensor fusion, and active learning annotation.

Built for **Smart India Hackathon (SIH 2026)** under Problem Statement **ID: 26057** (*AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery*) for the **Ministry of Earth Sciences (MoES) / National Institute of Ocean Technology (NIOT)**.

---

## 🏛️ System Architecture & Block Diagram

> 📖 **Full Architectural Specification**: See [ARCHITECTURE_AND_BLOCK_DIAGRAM.md](file:///f:/echopulsenet---marine-sonar-intelligence-platform%20%281%29/ARCHITECTURE_AND_BLOCK_DIAGRAM.md) for the exhaustive hardware/software breakdown, PostGIS spatial schemas, and Fernet security models.

```mermaid
graph LR
    %% Data Ingestion & Sensors Block
    subgraph BLOCK_1 ["1. SONAR & SENSOR INGESTION LAYER"]
        direction TB
        B1_1["Side-Scan Sonar (SSS) / SAS\n(100 kHz – 900 kHz)"]
        B1_2["Raw Sonar Files\n(.XTF / .JSF / .SEGY / Binary Matrices)"]
        B1_3["AUV / Vessel INS Telemetry\n(GPS, Heading, Speed, Depth, Altitude)"]
        B1_4["Edge Optical / Optical-Acoustic Camera Feed"]
    end

    %% Presentation / Frontend UI Block
    subgraph BLOCK_2 ["2. CLIENT & PRESENTATION (React 18 + Vite)"]
        direction TB
        subgraph FE_VIEWS ["Interactive Workspaces & HUDs"]
            B2_1["Sonar Waterfall & Ingestion Viewer"]
            B2_2["Interactive MPA & Hazard Map (MapLibre)"]
            B2_3["3D Bathymetric Digital Twin (Three.js)"]
            B2_4["Telemetry & Active Learning Studio"]
        end
        subgraph FE_SVC ["Network Client (src/services/api.ts)"]
            B2_5["Unified API Dispatcher (fetchWithTimeout)"]
        end
        FE_VIEWS --> B2_5
    end

    %% Desktop Edge Runtime Wrapper
    subgraph BLOCK_WRAPPER ["NATIVE DESKTOP EDGE RUNTIME"]
        direction TB
        W1["Tauri Native Rust Shell (src-tauri)"]
        W2["Electron Process (electron_main.js)"]
        W3["PyInstaller Executable (desktop_app.py)"]
    end

    %% API Gateway & Server Core
    subgraph BLOCK_3 ["3. API GATEWAY & ROUTING (FastAPI)"]
        direction TB
        B3_1["FastAPI Application (:8000)\n(backend/app/main.py)"]
        B3_2["CORS Middleware & Static SPA Serving"]
        B3_3["API Routes: /api/v1 & /api\n(backend/app/api/routes.py)"]
        B3_1 --> B3_2
        B3_1 --> B3_3
    end

    %% Intelligence & Physics Processing Pipeline
    subgraph BLOCK_4 ["4. AI & ACOUSTIC PHYSICS ENGINE"]
        direction TB
        B4_1["HeavyDebrisGuardrailEngine\n(GLCM / FFT Domain Guard & Habitat Shield)"]
        B4_2["UnifiedInferenceService\n(YOLOv12-Sonar 9-Class Deep Detector)"]
        B4_3["EchoPhysOmni3D & HydroPhysOmniNet\n(Slant-Range & Shadow Raymarching)"]
        B4_4["BathymetryService & ReportGenerator"]
        B4_1 --> B4_2 --> B4_3 --> B4_4
    end

    %% Security & Encryption Block
    subgraph BLOCK_5 ["5. SECURITY & ENCRYPTION"]
        direction TB
        B5_1["Fernet Symmetric Cryptography Subsystem\n(backend/app/core/security.py)"]
    end

    %% Data Persistence & Spatial DB Block
    subgraph BLOCK_6 ["6. POSTGIS SPATIAL DATABASE LAYER"]
        direction TB
        B6_POOL["DatabaseManager & PostGISConnector"]
        subgraph PG_DB ["PostgreSQL 15+ with PostGIS"]
            T_DET[("sonar_spatial_detections")]
            T_MSN[("sonar_spatial_missions")]
            T_MPA[("mpa_geotags")]
            ST_FUNC["Spatial Functions (ST_DWithin, ST_ConcaveHull)"]
        end
        subgraph SQLITE_DB ["Fallback: Embedded SQLite WAL"]
            L_SQLITE[("echopulsenet.db")]
        end
        B6_POOL --> PG_DB
        B6_POOL --> SQLITE_DB
    end

    %% Flow Connections between Blocks
    BLOCK_1 ==> |"Raw Pings / GPS"| BLOCK_2
    BLOCK_WRAPPER -.-> BLOCK_2
    BLOCK_WRAPPER -.-> BLOCK_3
    B2_5 ==> |"REST HTTP / JSON / Multipart"| B3_1
    B3_3 ==> |"Dispatches Request"| B4_1
    BLOCK_5 -.-> |"Decrypted DB Credentials"| B6_POOL
    B3_3 ==> |"Read / Write Missions & Detections"| B6_POOL
    POSTGIS_SRV["PostGIS Spatial Analytics"] --- ST_FUNC

    %% Styling
    style BLOCK_1 fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style BLOCK_2 fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style BLOCK_WRAPPER fill:#1e1b4b,stroke:#818cf8,stroke-width:1px,stroke-dasharray: 4 4,color:#f8fafc
    style BLOCK_3 fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style BLOCK_4 fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#f8fafc
    style BLOCK_5 fill:#0f172a,stroke:#ec4899,stroke-width:2px,color:#f8fafc
    style BLOCK_6 fill:#0f172a,stroke:#06b6d4,stroke-width:2px,color:#f8fafc
```

---

## ✨ Key Capabilities

* **⚡ Native Edge Execution:** Zero cloud API dependencies; runs fully on local hardware accelerated by **NVIDIA CUDA** (RTX 5060 Laptop GPU) and **WebGPU**.
* **🎯 HydroPhys-OmniNet v4 Deep Learning Core:** Multi-task marine detector with homoscedastic uncertainty loss balancing detection ($\mathcal{L}_{\text{det}}$), shadow segmentation ($\mathcal{L}_{\text{shadow}}$), and bathymetric depth ($\mathcal{L}_{\text{depth}}$).
* **🌊 Hydrographic Digital Signal Processing (DSP):**
  * Automated Bottom-Line Detection (BLD) tracking water-column altitude.
  * Slant-Range to Ground-Range geometric correction ($Y_{\text{ground}} = \sqrt{R_{\text{slant}}^2 - H_{\text{alt}}^2}$).
  * 2D-FFT frequency domain notch filtering eliminating vessel thruster reverberation.
  * Time-Varied Gain (TVG) and Empirical Gain Normalization (EGN).
* **🌐 3D Subsea Digital Twin (Multi-Object Projection):**
  * Interactive Three.js seabed mesh rendering **multiple simultaneous objects in parallel**.
  * Real-time projection from **Webcam Streams + System GPS + IR Distance Sensor** into 3D glowing beacons with vertical laser guidance pins.
* **🎛️ Defense Command Center HUD (`/command-center`):** Synchronized 4-quadrant mission control featuring live Sonar Waterfall, 3D Digital Twin, PostGIS Spatial Hazard Matrix, and Target Lock HUD.
* **🔊 Subsea Acoustic Audio Synthesizer:** Authentic frequency-modulated underwater submarine chirp audio via Web Audio API.
* **🧠 Human-in-the-Loop Active Learning:** Triage queue for high-uncertainty detections with 1-click local GPU fine-tuning.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* **Node.js** v18+ & **npm**
* **Python** 3.10+ (with CUDA 12.x recommended for GPU acceleration)
* **PostgreSQL / PostGIS** (Optional for cloud GIS clustering)

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
API documentation available at: `http://127.0.0.1:8000/api/v1/docs`

### 3. Frontend Setup
```bash
# In the root directory
npm install
npm run dev
```
Open your browser at `http://localhost:3000/`

---

## 📊 Marine Sonar Taxonomy

| Class ID | Target Classification | Typical Acoustic Signature | Threat Level |
| :--- | :--- | :--- | :--- |
| `0` | **Derelict Ghost Gear & Net** | Diffuse acoustic backscatter with irregular trailing shadow | 🔴 CRITICAL |
| `1` | **Shipwreck / Submerged Hull** | High-backscatter metallic highlight + elongated hull shadow | 🟠 HIGH |
| `2` | **Unexploded Ordnance (UXO)** | Compact high-intensity target with geometric cylindrical shadow | 🔴 CRITICAL |
| `3` | **Pipeline Scour / Anomaly** | Linear corridor with free-span depth deviations | 🟠 HIGH |
| `4` | **Marine Anthropogenic Debris**| High acoustic reflectivity clustered anomalies | 🟡 MEDIUM |
| `5` | **Subsea Power & Data Cable** | Continuous linear seabed depression/trench | 🟠 HIGH |
| `6` | **Benthic Biological Cluster** | Low-contrast porous acoustic return (coral reefs) | 🟢 LOW |
| `7` | **Geological Outcrop / Ridge** | Broad acoustic backscatter with terrain relief | 🟢 LOW |

---

## 📜 Dual Licensing

This repository is governed by a **Dual Licensing Model**:

1. **Open Platform Components ([GNU General Public License v3.0 (GPLv3)](LICENSE))**:
   * The web application, UI/UX components, Three.js 3D viewers, DSP sonar processors, and REST APIs are licensed under the **GNU GPL v3.0**.
2. **Proprietary Deep Learning Models ([MODELS_LICENSE.md](MODELS_LICENSE.md))**:
   * The **HydroPhys-OmniNet v4** and **EchoPhys-X** neural architectures, custom mathematical loss formulations, and trained `.pt` weight checkpoints are proprietary intellectual property covered under the **Private Model License**.

# 🌊 EchoPulseNet: Deep Ocean Sonar Intelligence & Defense-Grade Marine Digital Twin Platform

[![Platform](https://img.shields.io/badge/Platform-Native_Edge_AI-06b6d4?style=for-the-badge&logo=nvidia)](https://github.com/ELANGKATHIR11/ECHO-PULSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+cu128-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-HydroPhys--OmniNet_v4-38bdf8?style=for-the-badge)](https://github.com/ELANGKATHIR11/ECHO-PULSE)
[![Frontend](https://img.shields.io/badge/Frontend-React_19_|_Three.js-61DAFB?style=for-the-badge&logo=react)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostGIS](https://img.shields.io/badge/PostGIS-3.4_Spatial_DB-336791?style=for-the-badge&logo=postgresql)](https://postgis.net/)
[![License](https://img.shields.io/badge/License-GPLv3_|_Proprietary_Models-blue?style=for-the-badge)](LICENSE)

**EchoPulseNet** is an offline-capable, 100% native edge marine intelligence system designed for real-time acoustic target detection, autonomous underwater vehicle (AUV/ROV) hydrographic surveys, interactive 3D bathymetric Digital Twin visualization, system GPS & IR distance sensor fusion, and active learning annotation.

Built for **Smart India Hackathon (SIH 2026)** under Problem Statement **ID: 26057** (*AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery*) for the **Ministry of Earth Sciences (MoES) / National Institute of Ocean Technology (NIOT)**.

---

## 🏛️ System Architecture & Dataflow Diagram

```mermaid
graph TD
    %% 1. Marine & Optical Sensor Ingestion Layer
    subgraph SENSOR_INGESTION["1. Marine & Optical Sensor Ingestion"]
        RawSonar["Acoustic Sonar Files (.XTF / .JSF / .SL2 / .DAT)"]
        WebcamFeed["Live Optical Environmental Camera Stream"]
        HardwareGps["System WGS84 GPS Telemetry + Gyro Heading"]
        HardwareIR["Hardware IR / ToF Distance Calibrator"]
    end

    %% 2. Hydrographic Digital Signal Processing (DSP)
    subgraph EDGE_DSP["2. Hydrographic Digital Signal Processing (DSP)"]
        BLD["Bottom-Line Detection\n(BLD) & Water-Column\nRemoval"]
        SRC["Slant-Range to Ground-\nRange Geometric Transform"]
        FFT["2D-FFT Notch De-striping\n& Empirical Gain\nNormalization (EGN)"]
        
        BLD --> SRC
        SRC --> FFT
    end

    %% 3. Deep Learning Inference Core (Proprietary Models)
    subgraph AI_CORE["3. Deep Learning Inference Core (Proprietary Models)"]
        EchoPhys["EchoPhys-X (Dual-Head U-\nNet Shadow & Seabed\nAutoencoder)"]
        HydroPhys["HydroPhys-OmniNet v4\n(Attention-Centric\nYOLOv12 Detector)"]
        MultiFusion["Homoscedastic Multi-Task\nUncertainty Loss & Fusion"]
        SensorFusion["Sensor Fusion & 3D Optical\nRay Triangulation Engine"]
        
        EchoPhys --> MultiFusion
        HydroPhys --> MultiFusion
    end

    %% 4. Spatial Geospatial Intelligence Database
    subgraph POSTGIS_DB["4. Spatial Geospatial Intelligence Database"]
        PostGIS["PostgreSQL 16 / PostGIS\n(Encrypted Connection)"]
        GeoJSON["Spatial Hazard Matrix &\nCoastal Geofencing"]
        
        PostGIS --> GeoJSON
    end

    %% 5. Interactive Mission Control HUD & Digital Twin
    subgraph MISSION_CONTROL["5. Interactive Mission Control HUD & Digital Twin"]
        HITL["Active Learning & Local\nGPU Retrain Studio"]
        CommandCenter["Defense Command Center\nHUD (4-Quadrant)"]
        Waterfall["60 FPS Cascading Sonar\nWaterfall & Calipers"]
        WebcamHUD["Webcam Real-Time 3D\nMulti-Object Projector"]
        Twin3D["3D Bathymetric Subsea\nDigital Twin Mesh (Three.js)"]
    end

    %% Cross-Subgraph Dataflow Links
    RawSonar --> BLD
    
    FFT --> EchoPhys
    FFT --> HydroPhys
    
    WebcamFeed --> SensorFusion
    HardwareGps --> SensorFusion
    HardwareIR --> SensorFusion
    
    MultiFusion --> PostGIS
    SensorFusion --> PostGIS
    
    MultiFusion --> HITL
    MultiFusion --> CommandCenter
    MultiFusion --> Waterfall
    
    SensorFusion --> WebcamHUD
    SensorFusion --> Twin3D
    
    GeoJSON --> Twin3D
    GeoJSON --> CommandCenter
```

---

## 📦 Project Packages & Dependencies Inventory

### 💻 Client & Frontend Stack (`package.json`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **react** / **react-dom** | `^19.0.1` | High-performance reactive mission control UI |
| **three** | `^0.185.1` | WebGL 3D Bathymetric Digital Twin rendering engine |
| **@react-three/fiber** | `^9.7.0` | Declarative React wrapper for Three.js scene management |
| **@react-three/drei** | `^10.7.8` | Advanced 3D camera controls, shaders, and ocean environment helpers |
| **leaflet** / **react-leaflet** | `^1.9.4` / `^5.0.0` | GIS mapping for Marine Protected Areas (MPA) & debris geotags |
| **@tensorflow/tfjs** | `^4.22.0` | Web-native edge optical camera object tracking |
| **@tensorflow-models/coco-ssd** | `^2.2.3` | Optical detection backbone for ray-cast 3D triangulation |
| **motion** (Framer Motion) | `^12.23.24` | Ultra-smooth telemetry HUD animations and transitions |
| **lucide-react** | `^0.546.0` | High-tech tactical military/oceanographic HUD iconography |
| **tailwindcss** | `^4.1.14` | Styling framework for responsive tactical dark-mode dashboard |
| **vite** | `^6.2.3` | Next-generation ultra-fast frontend build engine |
| **electron** / **electron-builder** | `^43.4.1` / `^26.15.3` | Standalone cross-platform desktop application packaging |
| **@tauri-apps/cli** | `^1.5` | Native Rust-backed ultra-lightweight edge desktop shell |

### 🐍 Backend & AI Engine Stack (`backend/requirements.txt`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **fastapi** | `>=0.115.0` | Asynchronous REST and WebSocket API gateway |
| **uvicorn[standard]** | `>=0.30.0` | Lightning-fast ASGI production web server |
| **torch** / **torchvision** | `>=2.1.0` / `>=0.16.0` | PyTorch GPU deep learning computation & tensor physics |
| **ultralytics** | `>=8.3.0` | SOTA YOLOv12 acoustic anomaly detection backbone |
| **opencv-python-headless** | `>=4.8.0` | Computer vision, 2D-FFT filtering, and sonar image slicing |
| **scipy** / **numpy** | `>=1.11.0` / `>=1.24.0` | Hydrographic DSP, slant-range transform & signal normalization |
| **scikit-learn** | `>=1.3.0` | Spatial clustering, density metrics & active learning triage |
| **pandas** | `>=2.0.0` | Survey mission telemetry time-series & spatial analytics |
| **SQLAlchemy** / **psycopg2-binary** | `>=2.0.0` / `>=2.9.9` | High-throughput PostgreSQL database pooling & ORM |
| **GeoAlchemy2** / **shapely** | `>=0.14.0` / `>=2.0.0` | PostGIS spatial geometry indexing, geofencing & distance calculations |
| **cryptography** | `>=42.0.0` | Fernet symmetric credential encryption for defense-grade security |
| **pydantic** / **python-multipart** | `>=2.0.0` / `>=0.0.9` | Request validation & large acoustic sonar file ingestion |
| **pytest** | `>=8.0.0` | Automated backend testing suite |

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


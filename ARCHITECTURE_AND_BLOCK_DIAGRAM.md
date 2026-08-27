# 🌊 EchoPulseNet — System Architecture & Block Diagram

This document provides a comprehensive technical reference for the **EchoPulseNet** Marine Sonar Intelligence Platform. It details the complete hardware/software block diagram, system component interactions, security and encryption mechanisms, API communication patterns, and PostgreSQL / PostGIS spatial database architectures.

---

## 1. Functional System Block Diagram

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

    %% Cross-Subgraph Dataflow Links (Matching Image Flow)
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

## 2. Complete End-to-End System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer ["1. CLIENT & DESKTOP RUNTIME LAYER"]
        subgraph Browsers ["Web Application (Browser)"]
            UI["React 18 + TypeScript + Vite SPA\n(Tailwind CSS / High-Precision HUD)"]
            subgraph Pages ["Front-End Workspaces"]
                P1["Dashboard & Telemetry Hub"]
                P2["Raw Sonar Waterfall & XTF Ingestion"]
                P3["Webcam / Edge Optical-Acoustic Tracker"]
                P4["MPA Geo-Spatial Debris & Hazard Map"]
                P5["3D Digital Twin & Bathymetric Mesh"]
                P6["Command Center & Active Learning HUD"]
            end
        end
        subgraph DesktopWrappers ["Desktop & Edge Native Clients"]
            ELEC["Electron Main Process (Node.js)"]
            TAURI["Tauri Native Rust Shell (src-tauri)"]
            PYSTANDALONE["PyInstaller Standalone Executable (desktop_app.py)"]
        end
    end

    subgraph APIGateway ["2. COMMUNICATION & API GATEWAY"]
        FE_API["Front-End Unified API Client (src/services/api.ts)\n• fetchWithTimeout (6s AbortController)\n• Dynamic VITE_API_URL / Proxy Router"]
        FAST_API["FastAPI High-Performance Async Server (backend/app/main.py)\n• CORS Middleware (Allowed Origins / Tauri / Localhost)\n• Static SPA Distribution Serving (/dist)\n• Static Sonar Upload Mounts (/uploads)\n• Prefixes: /api/v1 & /api"]
    end

    subgraph BackendCore ["3. BACKEND SERVICES & AI / PHYSICS CORE"]
        ROUTER["API Router (backend/app/api/routes.py)"]
        
        subgraph AIServices ["AI & Intelligence Engine"]
            INFER["UnifiedInferenceService\n(YOLOv12-Sonar / Deep ML Backbone)"]
            GUARD["HeavyDebrisGuardrailEngine\n• OOD Optical vs SSS Acoustic Rejector\n• Natural Habitat Protection (Corals/Rocks)\n• Canonical Taxonomy Validator"]
            ACTLEARN["ActiveLearningService\n• Human-in-the-Loop Feedback\n• Hard-Sample Mining & Retraining"]
        end

        subgraph PhysicsEngines ["Physics-Informed Acoustic Tensor Engine"]
            HYDRO["HydroPhysOmniNet / EchoPhysOmni3D\n• Multi-frequency Backscatter Tensor (100kHz-900kHz)\n• Slant-Range Geometric Correction (R = sqrt(H^2 + Y^2))\n• Shadow-Height Estimation (H_target = L_shadow * Alt / R)"]
            BATHY["BathymetryService & Topography Mesh Generator"]
            PARSERS["Sonar Parsers (XTF, JSF, SEG-Y, Raw Matrices)"]
        end

        subgraph SpatialReportEngines ["Spatial Analytics & Reporting"]
            REPORT["ReportGenerator (PDF, GeoJSON, Shapefile, CSV)"]
            POSTGIS_SRV["PostGIS Spatial Analytics Service"]
        end
    end

    subgraph SecurityModule ["4. SECURITY & CREDENTIAL ENGINE"]
        SEC["Fernet Cryptographic Subsystem (backend/app/core/security.py)\n• Secret Key: ECHOPULSENET_SECRET_KEY\n• Deterministic 32-byte Edge Salt\n• Dynamic Credential Encryption & Decryption"]
    end

    subgraph DataPersistence ["5. PERSISTENCE & SPATIAL DATABASE LAYER"]
        DB_MGR["DatabaseManager & PostGISConnector\n(Connection Pooling, Auto-Reconnect, Pre-ping)"]
        
        subgraph PrimaryDB ["PostgreSQL 15+ with PostGIS Extension"]
            T1[("sonar_spatial_detections\n(Detections, BBoxes, Scores, Geo Coordinates)")]
            T2[("sonar_spatial_missions\n(Mission Tracks, Ping Counts, Telemetry)")]
            T3[("mpa_geotags\n(Protected Zones, Threat Levels, Marine Labels)")]
            EXT["PostGIS Spatial Functions\n(ST_DWithin, ST_ConcaveHull, ST_Centroid)"]
        end

        subgraph LocalFallbackDB ["Edge Offline Fallback"]
            SQLITE[("Local SQLite WAL Database\n(echopulsenet.db with Spatial Emulation)")]
        end
    end

    %% Linkages
    UI --> Pages
    Pages --> FE_API
    DesktopWrappers -.-> UI
    DesktopWrappers -.-> FAST_API
    FE_API -- "REST HTTP / JSON / Multipart" --> FAST_API
    FAST_API --> ROUTER

    ROUTER --> INFER
    ROUTER --> GUARD
    ROUTER --> HYDRO
    ROUTER --> BATHY
    ROUTER --> PARSERS
    ROUTER --> ACTLEARN
    ROUTER --> REPORT
    ROUTER --> POSTGIS_SRV

    GUARD --> INFER
    INFER --> HYDRO

    SEC -.-> FAST_API
    SEC -.-> DB_MGR

    POSTGIS_SRV --> DB_MGR
    ROUTER --> DB_MGR

    DB_MGR -- "Primary (pool_size=10, timeout=3s)" --> PrimaryDB
    DB_MGR -- "Fallback on Disconnect" --> LocalFallbackDB
    PrimaryDB --- EXT
```

---

## 3. Component Details & Interactions

### 1. Ingestion & Pre-Processing
* **Sensors**: Side-Scan Sonar (SSS) operating at 100 kHz – 900 kHz, Synthetic Aperture Sonar (SAS), and Sub-Bottom Profilers.
* **Acoustic Formats**: Native parsers for `.XTF` (eXtended Triton Format), `.JSF` (EdgeTech), `.SEGY`, and raw acoustic matrix tensors.
* **Optical-Acoustic Camera Feeds**: USB / RTSP live camera frames transformed into simulated acoustic waterfall spectrograms.

### 2. Frontend Application (`src/`)
* **React 18 + TypeScript + Vite**: Provides high-performance component rendering with 60 FPS subsea waterfalls.
* **3D Subsea Digital Twin**: WebGL / Three.js seabed mesh rendering real-time target bounding boxes, beacon pins, and depth contours.
* **GIS Map Integration**: MapLibre GL and Leaflet mapping engines rendering Marine Protected Area (MPA) boundaries and hazard polygons.
* **Unified API Client (`src/services/api.ts`)**: Manages requests using `fetchWithTimeout` (6000ms threshold) and dynamic `VITE_API_URL` discovery.

### 3. FastAPI Gateway (`backend/app/main.py`)
* Serves production single-page application static files from `/dist`.
* Exposes RESTful endpoints at both `/api/v1` and `/api`.
* Hosts static cropped detection thumbnails at `/uploads`.

### 4. AI & Physics Services (`backend/app/services/`)
* **`HeavyDebrisGuardrailEngine`**: Uses Gray-Level Co-occurrence Matrix (GLCM) contrast and 2D-FFT frequency entropy to reject out-of-distribution optical images and avoid false detections on coral reefs or sand ripples.
* **`UnifiedInferenceService`**: Deep YOLOv12-Sonar model detecting 9 canonical marine debris classes.
* **`EchoPhysOmni3D` & `HydroPhysOmniNet`**: Applies multi-frequency backscatter tensors, slant-range unwarping, and shadow raymarching for 3D depth recovery:
  $$\text{Target Height } (H) = \frac{L_{\text{shadow}} \cdot \text{Altitude}}{\text{Slant Range}}$$
* **`BathymetryService`**: Synthesizes 3D digital elevation models (DEM) of the surveyed seafloor.
* **`ReportGenerator`**: Produces official PDF, GeoJSON, Shapefile, and CSV mission dossiers.

### 5. Cryptography & Security Subsystem (`backend/app/core/security.py`)
* Implements symmetric Fernet encryption via `cryptography.fernet.Fernet`.
* Protects connection credentials (`ENCRYPTED_DATABASE_URL`, `ENC_POSTGRES_USER`, `ENC_POSTGRES_PASSWORD`).
* Safely resolves database parameters in memory at startup.

---

## 4. PostgreSQL & PostGIS Spatial Persistence

### Entity Relationship Model

```mermaid
erDiagram
    SONAR_SPATIAL_MISSIONS ||--o{ SONAR_SPATIAL_DETECTIONS : "contains"
    MPA_GEOTAGS }|..|{ SONAR_SPATIAL_DETECTIONS : "spatial proximity"

    SONAR_SPATIAL_DETECTIONS {
        string id PK "DET-2026-XXXX"
        string mission_id FK "MSN-2026-XXXX"
        string mission_name "Mission Title"
        string target_class "ghost_net, uxo, shipwreck, etc."
        string class_name_label "Human readable display name"
        float confidence "Overall confidence [0.0 - 1.0]"
        float detector_score "YOLO detector raw score"
        float shadow_score "Acoustic shadow confidence"
        float geometry_score "Geometric contour match"
        float anomaly_score "Autoencoder anomaly index"
        float quality_score "SNR & ping clarity metric"
        float latitude "WGS84 Latitude"
        float longitude "WGS84 Longitude"
        float depth_meters "Water column depth"
        float slant_range_meters "Sonar vehicle range"
        float altitude_meters "AUV altitude above seafloor"
        float geotag_confidence "GPS/INS acoustic positioning score"
        int ping_index "Sequential ping number in survey"
        string model_version "YOLOv12-Sonar / OmniNet"
        string image_crop_url "/uploads/crops/..."
        string verification_status "VERIFIED, UNVERIFIED, REJECTED"
        string operator_notes "Manual analyst remarks"
        json bbox_json "Normalized BBox [x, y, w, h]"
        json geometry_meta "Contour length, width, area"
        json shadow_meta "Shadow length, calculated height"
        datetime created_at "Timestamp UTC"
    }

    SONAR_SPATIAL_MISSIONS {
        string id PK "MSN-2026-XXXX"
        string name "Survey Mission Name"
        string code_name "OPERATION NEPTUNE-SWEEP"
        string date "YYYY-MM-DD"
        string location "Gulf of Mannar Sector 4"
        float center_lat "Center latitude"
        float center_lng "Center longitude"
        string sonar_source "Side-Scan Sonar (SSS)"
        float frequency_khz "455 kHz / 900 kHz"
        float survey_distance_km "Total linear track km"
        float area_sq_km "Area swept in sq km"
        string status "Active, Completed, Paused"
        int duration_minutes "Mission duration"
        int ping_count "Total pings collected"
        string vessel_name "RV Sagar Nidhi"
        string vehicle_type "AUV DeepScan-4"
        string target_objective "Objective statement"
        json track_points_json "Array of GPS track points & pings"
        json summary_metrics_json "SNR, anomaly counts, FPS"
        datetime created_at "Timestamp UTC"
    }

    MPA_GEOTAGS {
        string id PK "MPA-GEO-XXXX"
        string official_ref "MoES-MPA-REF-09"
        string agency "NIOT / Coast Guard / Wildlife Trust"
        string mpa_id "GULF-OF-MANNAR"
        string mpa_name "Gulf of Mannar Biosphere Reserve"
        string target_class "ghost_net"
        string marine_label "Derelict Gillnet Entanglement"
        float latitude "WGS84 Latitude"
        float longitude "WGS84 Longitude"
        float depth_meters "Depth in meters"
        string threat_level "CRITICAL, HIGH, MEDIUM, LOW"
        datetime created_at "Timestamp UTC"
    }
```

---

## 5. Environment Variables & Credentials Reference

| Variable | Scope | Purpose | Default / Example Value |
| :--- | :--- | :--- | :--- |
| `VITE_API_URL` | Frontend Client | API Server Base URL | `http://localhost:8000/api/v1` or `/api/v1` |
| `DATABASE_URL` | Backend Server | Primary PostgreSQL connection URL | `postgresql://postgres:postgres@localhost:5432/echopulse_postgis` |
| `POSTGIS_DATABASE_URL` | Backend Server | PostGIS spatial database connection | `postgresql://postgres:postgres@localhost:5432/echopulse_postgis` |
| `ECHOPULSENET_SECRET_KEY` | Backend Server | Secret key for Fernet symmetric encryption | *32-byte Cryptographic String* |
| `ENCRYPTED_DATABASE_URL` | Backend Server | Encrypted DB connection string token | *Fernet Ciphertext Token* |
| `ENC_POSTGRES_USER` | Backend Server | Encrypted DB user | *Fernet Ciphertext Token* |
| `ENC_POSTGRES_PASSWORD` | Backend Server | Encrypted DB password | *Fernet Ciphertext Token* |
| `DATA_ROOT` | Backend Server | Sonar dataset storage directory | `data/` |
| `MODEL_ROOT` | Backend Server | PyTorch & ONNX checkpoints directory | `models_checkpoints/` |
| `CACHE_ROOT` | Backend Server | Cache directory for acoustic tiles | `cache/` |
| `REPORTS_ROOT` | Backend Server | Generated mission PDF/GeoJSON reports | `reports/` |
| `CONFIDENCE_THRESHOLD` | Backend Server | Minimum model confidence threshold | `0.50` |
| `DEVICE` | Backend Server | Compute device selector (`auto`, `cuda`, `cpu`) | `auto` |

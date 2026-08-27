# 🌊 EchoPulseNet — System Architecture & Block Diagram

This document provides a comprehensive technical reference for the **EchoPulseNet** Marine Sonar Intelligence Platform. It details the complete hardware/software block diagram, system component interactions, security and encryption mechanisms, API communication patterns, and PostgreSQL / PostGIS spatial database architectures.

---

## 1. Functional System Block Diagram

```mermaid
graph LR
    %% Data Ingestion & Sensors Block
    subgraph BLOCK_1 ["BLOCK 1: SONAR & SENSOR INGESTION LAYER"]
        direction TB
        B1_1["Side-Scan Sonar (SSS) / SAS\n(100 kHz – 900 kHz)"]
        B1_2["Raw Sonar Files\n(.XTF / .JSF / .SEGY / Binary Matrices)"]
        B1_3["AUV / Vessel INS Telemetry\n(GPS, Heading, Speed, Depth, Altitude)"]
        B1_4["Edge Optical / Optical-Acoustic Camera Feed"]
    end

    %% Presentation / Frontend UI Block
    subgraph BLOCK_2 ["BLOCK 2: PRESENTATION & CLIENT PLATFORM (React 18 + Vite)"]
        direction TB
        subgraph FE_VIEWS ["Interactive Workspaces & HUDs"]
            B2_1["Sonar Waterfall & Ingestion Viewer"]
            B2_2["Interactive MPA & Hazard Map (MapLibre / Leaflet)"]
            B2_3["3D Bathymetric Digital Twin (WebGL / Three.js)"]
            B2_4["Telemetry, Active Learning & Command HUD"]
        end
        subgraph FE_SVC ["Frontend Network Client (src/services/api.ts)"]
            B2_5["Unified API Dispatcher\n(fetchWithTimeout + AbortController + VITE_API_URL)"]
        end
        FE_VIEWS --> B2_5
    end

    %% Desktop Edge Runtime Wrapper
    subgraph BLOCK_WRAPPER ["NATIVE DESKTOP EDGE RUNTIME"]
        direction TB
        W1["Tauri Native Rust Shell\n(src-tauri)"]
        W2["Electron Process\n(electron_main.js)"]
        W3["PyInstaller Single Executable\n(desktop_app.py)"]
    end

    %% API Gateway & Server Core
    subgraph BLOCK_3 ["BLOCK 3: API GATEWAY & ROUTING (FastAPI Server)"]
        direction TB
        B3_1["FastAPI Application (Port: 8000)\n(backend/app/main.py)"]
        B3_2["CORS Middleware\n(Localhost, 127.0.0.1, Tauri Origins)"]
        B3_3["Static Assets & Artifact Mounts\n(/dist SPA + /uploads Sonar Crops)"]
        B3_4["API v1 & /api Endpoint Router\n(backend/app/api/routes.py)"]
        B3_1 --> B3_2
        B3_1 --> B3_3
        B3_1 --> B3_4
    end

    %% Intelligence & Physics Processing Pipeline
    subgraph BLOCK_4 ["BLOCK 4: INTELLIGENCE & PHYSICS ENGINE (backend/app/services)"]
        direction TB
        subgraph GUARD_ENGINE ["Guardrail Verification Subsystem"]
            B4_1["HeavyDebrisGuardrailEngine\n• Acoustic Domain OOD Check (GLCM / FFT Entropy)\n• Natural Benthic Filter (Coral/Rock Protection)"]
        end
        subgraph AI_ENGINE ["Detection & Classification Subsystem"]
            B4_2["UnifiedInferenceService\n• YOLOv12-Sonar Deep Neural Net\n• 9 Marine Debris Canonical Classes\n• Autoencoder Anomaly Scoring"]
        end
        subgraph PHYS_ENGINE ["Physics-Informed Acoustic Tensor Engine"]
            B4_3["EchoPhysOmni3D & HydroPhysOmniNet\n• Multi-Frequency Backscatter Tensor\n• Slant-Range Geometric Unwarping\n• Acoustic Shadow Raymarching (Target Height Recovery)"]
        end
        subgraph BATHY_REPORT ["Spatial Bathymetry & Reports"]
            B4_4["BathymetryService (Seabed Mesh Synthesis)"]
            B4_5["ReportGenerator (PDF / GeoJSON / Shapefile / CSV)"]
            B4_6["ActiveLearningService (Hard-sample Mining)"]
        end
        GUARD_ENGINE --> AI_ENGINE
        AI_ENGINE --> PHYS_ENGINE
        PHYS_ENGINE --> BATHY_REPORT
    end

    %% Security & Encryption Block
    subgraph BLOCK_5 ["BLOCK 5: SECURITY & CREDENTIAL ENGINE (backend/app/core/security.py)"]
        direction TB
        B5_1["Fernet Symmetric Cryptography Engine"]
        B5_2["Key Derivation: ECHOPULSENET_SECRET_KEY"]
        B5_3["Dynamic Decryptor: resolve_db_connection_url()"]
        B5_1 --- B5_2
        B5_2 --- B5_3
    end

    %% Data Persistence & Spatial DB Block
    subgraph BLOCK_6 ["BLOCK 6: PERSISTENCE & POSTGIS SPATIAL DATABASE LAYER"]
        direction TB
        B6_POOL["DatabaseManager & PostGISConnector\n(SQLAlchemy Connection Pool: size=10, timeout=3s)"]
        
        subgraph PG_DB ["Primary: PostgreSQL 15+ & PostGIS Extension"]
            T_DET[("sonar_spatial_detections\n• Target Class, Confidences\n• WGS84 Lat/Lng/Depth/Alt\n• BBox, Contour & Shadow Meta")]
            T_MSN[("sonar_spatial_missions\n• Survey Transects & Tracks\n• Ping Metrics, SNR, Vessel Info")]
            T_MPA[("mpa_geotags\n• MPA Zones & Official Refs\n• Threat Levels & Hazard Tags")]
            ST_FUNC["Spatial Analysis Engine\n• ST_DWithin (Radial Proximity Alerts)\n• Hazard Cluster Hull Generation"]
        end

        subgraph SQLITE_DB ["Fallback: Local Embedded SQLite WAL"]
            L_SQLITE[("echopulsenet.db\n(Zero-config Offline Edge Storage)")]
        end
        
        B6_POOL -- "Online Connection" --> PG_DB
        B6_POOL -- "Offline Fallback" --> SQLITE_DB
    end

    %% Flow Connections between Blocks
    BLOCK_1 ==> |"Raw Files / Streams / GPS"| BLOCK_2
    BLOCK_WRAPPER -.-> |"Hosts & Executes"| BLOCK_2
    BLOCK_WRAPPER -.-> |"Launches Backend Process"| BLOCK_3
    
    B2_5 ==> |"REST HTTP / Multipart Form Data"| B3_1
    B3_4 ==> |"Dispatches Request"| GUARD_ENGINE
    
    BLOCK_5 -.-> |"Decrypted DB Credentials"| B6_POOL
    B3_4 ==> |"Read / Write Missions & Detections"| B6_POOL
    B4_5 ==> |"Fetches Audit Records"| B6_POOL
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

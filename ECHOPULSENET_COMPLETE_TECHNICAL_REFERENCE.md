# EchoPulseNet — Marine Sonar Intelligence Platform
## **Complete Technical Reference**

> *Built for the Indian Ocean. Powered by Physics-Aware AI.*
> *Version 2.0 · September 2026*

---

## 📖 Table of Contents

| # | Section |
|---|---------|
| 1 | [Vision & Mission](#1-vision--mission) |
| 2 | [System Architecture Overview](#2-system-architecture-overview) |
| 3 | [Frontend Architecture](#3-frontend-architecture) |
| 4 | [Backend Architecture](#4-backend-architecture) |
| 5 | [AI/ML Model Suite & Channels](#5-aiml-model-suite--channels) |
| 6 | [Core Mathematical Formulas](#6-core-mathematical-formulas) |
| 7 | [Data Processing Pipeline](#7-data-processing-pipeline) |
| 8 | [Key Advantages Over Conventional Systems](#8-key-advantages-over-conventional-systems) |
| 9 | [India's Ocean, Environment & National Security](#9-indias-ocean-environment--national-security) |
| 10 | [Deployment & Edge Hardware](#10-deployment--edge-hardware) |
| 11 | [Future Roadmap](#11-future-roadmap) |

---

## 1. Vision & Mission

**EchoPulseNet** is a full-stack, real-time **Intelligent Hydrophone Acoustic Surveillance, Event Classification & AVS Underwater Drone Localisation Platform**. It transforms passive hydrophone audio and Acoustic Vector Sensor (AVS) recordings into **actionable maritime intelligence** using physics-constrained deep learning — without requiring any active sonar emission.

### Mission Pillars

| Pillar | Description |
|--------|-------------|
| 🔬 **Scientific** | Advance passive underwater acoustics research with physics-consistent AI |
| 🛡️ **Security** | Detect illegal AUV/UUV intrusions in India's Exclusive Economic Zone (EEZ) |
| 🌿 **Environmental** | Monitor marine biodiversity and anthropogenic noise pollution |
| 🎓 **Educational** | Provide a retrainable, open platform for Indian universities and research labs |
| 🚀 **Operational** | Run in real time on edge hardware aboard Indian Navy patrol vessels and coastal stations |

---

## 2. System Architecture Overview

```
+================================================================================+
|                        ECHOPULSENET PLATFORM                                   |
+==================+===============================+============================+
|   DATA SOURCES   |        PROCESSING CORE        |        OUTPUTS             |
+==================+===============================+============================+
| Hydrophone WAV/  |  FastAPI Backend (Python 3.11)|  React UI (TypeScript)     |
| FLAC recordings  |  +-------------------------+  |  +---------------------+   |
|                  |  | Audio Processor (DSP)   |  |  | Hydrophone Studio   |   |
| AVS 4-channel    |  | Ocean State Engine      |  |  | AVS Surveillance    |   |
| P,Ux,Uy,Uz       |  | AI/ML Model Zoo         |  |  | Ocean-PhysNet Studio|   |
|                  |  | Inference Service       |  |  | Command Centre      |   |
| Ocean Parameters |  | Retraining Service      |  |  | Analytics           |   |
| T, S, Depth, SSP |  | PostGIS Spatial Service |  |  | Digital Twin        |   |
|                  |  | MPA / Debris Map Service|  |  | MPA & Debris Map    |   |
| Webcam / CCTV    |  +-------------------------+  |  | Webcam Tracker      |   |
| (Surface tracks) |  PostgreSQL + SpatiaLite DB   |  +---------------------+   |
+==================+===============================+============================+
```

### Component Communication
```
Browser --(REST/JSON)--> FastAPI --> Audio Processor --> Model --> JSON Response --> UI
                                 +--> Ocean State Engine
                                 +--> PostGIS Service --> PostgreSQL (Spatial)
                                 +--> Retraining Service --> Celery + Redis
                                 +--> GPU/NPU Worker --> Intel OpenVINO
```

---

## 3. Frontend Architecture

### 3.1 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Framework** | React 18 (TypeScript) + Vite | Fast HMR, strong type safety, modular components |
| **Build Tool** | Vite | Sub-second builds, native ESM |
| **Styling** | Vanilla CSS with CSS Variables | Glassmorphism, dark mode, animated gradients |
| **Routing** | React Router v6 | Code-split pages, nested layouts |
| **State Management** | React Context + `useReducer` | Lightweight global state without Redux overhead |
| **Visualization** | Native canvas renderers + lucide-react | Custom SSP plots, DOA radar, intensity vector overlays |
| **Desktop Shell** | Tauri (Rust) + Electron (fallback) | Cross-platform native app packaging |
| **Icons** | lucide-react | Consistent maritime/science icon vocabulary |

### 3.2 Page Inventory (`src/pages/`)

| Page | File | Purpose |
|------|------|---------|
| **Home / Landing** | `HomePage.tsx` | Hero, feature cards, quick-start guide |
| **Dashboard** | `DashboardPage.tsx` | Live event feed, alert counters, map overview |
| **Hydrophone Studio** | `HydrophoneStudioPage.tsx` | Drag-and-drop WAV/FLAC upload, waveform viewer, spectrogram, MFCC heatmap, playback, classification result |
| **AVS Surveillance** | `AvsSurveillancePage.tsx` | 4-channel P/U input, intensity vector plot, real-time DOA compass, range estimate |
| **Raw Sonar Upload** | `RawSonarUploadPage.tsx` | Batch sonar file ingestion (SDF, SEGY, binary), format parser, preview |
| **Ocean-PhysNet Studio** | `OceanPhysNetStudioPage.tsx` | Ocean parameter sliders (T, S, D), SSP profile canvas, DOA radar, OOD gauge, full inference results |
| **Command Centre** | `CommandCenterPage.tsx` | Operator HQ — live threat board, alert priority queue, operator notes |
| **Digital Twin** | `DigitalTwinPage.tsx` | 3-D ocean acoustic simulation, ray-tracing visualisation, sensor placement optimizer |
| **Analytics** | `AnalyticsPage.tsx` | Historical classification trends, model confidence charts, geographic heat maps |
| **Detections** | `DetectionsPage.tsx` + `DetectionDetailPage.tsx` | Browse, filter, and audit all past detection events |
| **Model Retrain** | `ModelRetrainPage.tsx` | Upload new labelled datasets, monitor training job progress, swap model checkpoint |
| **MPA & Debris Map** | `MpaDebrisMapPage.tsx` | GIS overlay of Marine Protected Areas, pollution events, and acoustic anomaly clusters |
| **PostGIS Spatial Data** | `PostgresSpatialDataPage.tsx` | Raw spatial query explorer, GeoJSON exporter |
| **Webcam Tracker** | `WebcamTrackerPage.tsx` | Real-time surface vessel/drone detection via webcam (YOLO-based) with acoustic correlation |

### 3.3 Component Architecture (`src/components/`)

```
src/components/
+-- layout/
|   +-- Sidebar.tsx          <- Navigation with nested model routes
|   +-- Header.tsx           <- Alert badge, user profile, theme toggle
|   +-- PageContainer.tsx    <- Consistent page padding / max-width
+-- charts/
|   +-- SspProfileCanvas.tsx <- Sound-speed vs depth SVG/canvas plot
|   +-- DoaRadarCanvas.tsx   <- Polar radar for bearing/elevation
|   +-- OodGauge.tsx         <- Mahalanobis OOD score ring gauge
+-- audio/
|   +-- WaveformPlayer.tsx   <- Web Audio API playback + visual scrubber
|   +-- SpectrogramCanvas.tsx<- Real-time FFT waterfall
+-- shared/
    +-- ConfidenceBar.tsx     <- Colour-coded probability bars
    +-- AlertBadge.tsx        <- Pulsing threat-level indicator
    +-- OceanSliders.tsx      <- Synchronized T/S/Depth/SSP knobs
```

### 3.4 Design System

```css
/* Design tokens (index.css) */
--color-primary:      #00d4ff;   /* Sonar cyan */
--color-accent:       #7c3aed;   /* Deep violet */
--color-surface:      rgba(15, 23, 42, 0.8);  /* Ocean dark glass */
--color-threat-high:  #ef4444;
--color-threat-mid:   #f59e0b;
--color-threat-low:   #10b981;
--blur-glass:         blur(20px) saturate(180%);
--gradient-ocean:     linear-gradient(135deg, #0f172a 0%, #0c4a6e 50%, #0f172a 100%);
```

All pages use glassmorphism cards, animated scan-line overlays, and pulsing threat badges for an immersive maritime operations feel.

---

## 4. Backend Architecture

### 4.1 Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Web Framework** | FastAPI 0.111 (Python 3.11) | Async HTTP/WebSocket API, auto OpenAPI docs |
| **ASGI Server** | Uvicorn | High-performance event loop |
| **Database** | PostgreSQL 15 + PostGIS 3.4 | Spatial queries, detection event storage |
| **Embedded DB** | SQLite + SpatiaLite | Lightweight fallback for edge deployments |
| **ORM** | SQLAlchemy 2 (async) | Model-based DB access |
| **ML Runtime** | PyTorch 2.x + Intel OpenVINO | GPU training + NPU inference |
| **Task Queue** | Celery 5 + Redis | Async training jobs |
| **Audio DSP** | LibROSA, SciPy, NumPy | Waveform processing, feature extraction |
| **Spatial** | GeoPandas, Shapely, PyProj | GIS coordinate transforms |
| **Containerisation** | Docker + docker-compose | Reproducible deployment |

### 4.2 API Endpoint Map (`backend/app/api/routes.py`)

#### Audio Ingestion & Classification
| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/upload-audio` | Upload raw WAV/FLAC hydrophone recording |
| `POST` | `/classify` | Run acoustic classification on uploaded file |
| `POST` | `/analyze` | Full pipeline: DSP + features + all models + JSON result |
| `GET`  | `/detections` | Paginated list of all past detections |
| `GET`  | `/detections/{id}` | Single detection detail with full metadata |

#### Ocean-PhysNet Endpoints
| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/ocean-physnet/ssp-calc` | Compute SSP profile from T, S, Depth array |
| `POST` | `/ocean-physnet/infer` | Full physics-conditioned inference (hydrophone + AVS + ocean state) |

#### Model Retraining
| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/retrain/upload-dataset` | Upload new labelled audio dataset (ZIP) |
| `POST` | `/retrain/start` | Launch async fine-tuning job |
| `GET`  | `/retrain/status/{job_id}` | Poll training job progress |
| `POST` | `/retrain/activate/{checkpoint}` | Hot-swap active model checkpoint |

#### Spatial & GIS
| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET`  | `/spatial/detections` | GeoJSON feature collection of all events |
| `POST` | `/spatial/query` | Spatial query by bounding box |
| `GET`  | `/mpa/layers` | Marine Protected Area GeoJSON boundaries |
| `GET`  | `/debris/hotspots` | Debris cluster centroids |

#### AVS & DOA
| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/avs/locate` | DOA + range from 4-channel AVS vector |
| `GET`  | `/avs/live` | WebSocket stream for real-time AVS telemetry |

#### Security & Health
| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET`  | `/health` | System health check |
| `GET`  | `/metrics` | Prometheus-compatible inference metrics |

### 4.3 Core Backend Modules

#### `backend/app/sonar/audio_processor.py`
Raw audio to rich acoustic feature tensor. Extracts:

| Feature | Formula | Purpose |
|---------|---------|---------|
| **MFCC** | Mel-Frequency Cepstral Coefficients | Compact tonal fingerprint |
| **STFT Spectrogram** | Short-Time Fourier Transform | Time-frequency representation |
| **Mel Spectrogram** | Non-linear frequency warping | Matches mammal hearing |
| **Spectral Centroid** | sum(f * X(f)) / sum(X(f)) | Brightness of sound |
| **NDSI** | Normalized Difference Soundscape Index | Biophonic vs anthropogenic ratio |
| **ACI** | Acoustic Complexity Index | Biodiversity proxy |
| **ADI** | Acoustic Diversity Index | Spectral entropy diversity |
| **ZCR** | Zero Crossing Rate | Noise character |
| **RMS Energy** | sqrt(sum(x^2)/N) | Signal strength |

#### `backend/app/sonar/ocean_state.py`
Seawater physics engine — converts raw ocean measurements into an acoustic state tensor:
- `mackenzie_sound_speed(T, S, D)` — Mackenzie 1981 equation
- `francois_garrison_absorption(f, T, S, D)` — Frequency-dependent attenuation
- `build_ocean_state_tensor(T, S, D, f)` — normalised Eo vector
- `compute_ssp_profile(T_array, S_array, D_array)` — Full depth profile

#### `backend/app/sonar/ocean_phys_encoders.py`
- `ComplexSpectralEncoder` — Complex-valued STFT, preserves phase
- `ComplexAVSSpatialEncoder` — 4-channel P/U, active intensity, covariance
- `MultipathTokenizer` — Labels direct/surface-bounce/bottom-bounce paths

#### `backend/app/sonar/physics_attention.py`
- `PhysicsBiasedCrossAttention` — Injects travel-time + absorption into attention
- `FourierNeuralOperatorPropBlock` — FNO enforcing Helmholtz wave equation

#### `backend/app/services/inference_service.py`
Central inference orchestrator: model loading, batching, NPU/GPU dispatch, uncertainty propagation, result caching, and Prometheus metrics.

#### `backend/app/services/retraining_service.py`
Async continual-learning loop: receives datasets, schedules Celery fine-tuning jobs, validates metrics, hot-swaps checkpoints.

#### `backend/app/services/active_learning_service.py`
Query-strategy selection: entropy sampling flags highest-uncertainty samples for human review.

#### `backend/app/services/guardrails_service.py`
Prevents hallucination by bounding outputs, enforcing physical range constraints, and triggering OOD alerts.

#### `backend/app/services/postgis_service.py`
Stores detection events as PostGIS geometry, supports bounding-box queries, trajectory clustering, GeoJSON export.

#### `backend/app/services/mpa_service.py`
Loads MPA boundary shapefiles (MoEFCC / IUCN), cross-references acoustic events with protected zones, emits compliance alerts.

---

## 5. AI/ML Model Suite & Channels

Each model operates on its own input channel and can be run independently or fused inside OCEAN-PHYSNet.

---

### 5.1 OCEAN-PHYSNet (Master Physics-Constrained Model)
**File**: `backend/app/models/ocean_physnet.py`

#### Architecture

```
           +---------------------------------------------------------+
           |                  OCEAN-PHYSNet                          |
           +-------------+------------------+------------------------+
                         |                  |                    |
                   CHANNEL A          CHANNEL B            CHANNEL C
              +--------------+  +----------------+  +------------------+
              |  Hydrophone  |  |  AVS 4-channel |  |  Ocean State     |
              |  Waveform    |  |  P, Ux, Uy, Uz |  |  T, S, D, f      |
              +--------------+  +----------------+  +------------------+
                     |                  |                     |
              +--------------+  +----------------+  +------------------+
              | Complex STFT |  | Active Intensity|  | Mackenzie SSP    |
              | Spectral Enc |  | Covariance Mat  |  | Absorption Tensor|
              +--------------+  +----------------+  +------------------+
                     |                  |                     |
                     +------------------+---------------------+
                                        |
                             +---------------------+
                             |  Multipath Tokenizer |
                             | (direct/surf/bottom) |
                             +---------------------+
                                        |
                             +---------------------+
                             | Physics-Biased       |
                             | Cross-Attention       |
                             | (travel-time bias Bp) |
                             +---------------------+
                                        |
                             +---------------------+
                             | FNO Propagation Blk  |
                             | (Helmholtz residual) |
                             +---------------------+
                                        |
               +------------------------+---------------------+
               |                        |                      |
        +------+------+        +--------+------+      +--------+------+
        |Classification|        |DOA Head       |      |Range Head     |
        |Head          |        |Periodic sin/cos|      |Heteroscedastic|
        |[Biophonic,   |        |azimuth,elev   |      |mu_R, sigma_R  |
        | Anthropogenic|        +---------------+      +---------------+
        | Geophonic,   |
        | Tactical]    |        +---------------+
        +--------------+        | OOD Head      |
                                | Mahalanobis D |
                                +---------------+
```

#### Input Channels
| Channel | Tensor Shape | Description |
|---------|-------------|-------------|
| `x_hydro` | `(B, 1, T)` | Raw waveform at sampling rate f_s |
| `x_avs` | `(B, 4, T)` | Pressure P + velocity Ux, Uy, Uz |
| `x_ocean` | `(B, d_ocean)` | Ocean state: T, S, D, SSP, absorption, season |

#### Outputs
| Output | Shape | Description |
|--------|-------|-------------|
| `class_probs` | `(B, 4)` | Multi-class softmax probabilities |
| `doa_az` | `(B, 2)` | [sin θ, cos θ] — azimuth periodic form |
| `doa_el` | `(B, 2)` | [sin φ, cos φ] — elevation periodic form |
| `range_mu` | `(B, 1)` | Mean range estimate (metres) |
| `range_sigma` | `(B, 1)` | Range uncertainty (std. dev.) |
| `ood_score` | `(B,)` | Mahalanobis distance from training centroid |

---

### 5.2 EchoPhys-Lite (Lightweight Edge Model)
**File**: `backend/app/models/echophys_lite.py`

**Channel**:
```
Hydrophone Waveform --> Mel Spectrogram --> 2-Layer CNN --> 4-class output
```

| Property | Value |
|----------|-------|
| Parameters | ~3 M |
| Quantisation | INT8 |
| Inference latency | < 15 ms on NPU |
| AVS channel | No (single hydrophone) |
| Ocean conditioning | Default North Indian Ocean preset |
| Target hardware | Jetson Nano, Raspberry Pi 5 + Hailo-8, sonobuoys |

---

### 5.3 EchoPhys-Omni-3D (Full 3-D Acoustic Scene Model)
**File**: `backend/app/models/echophys_omni_3d.py`

**Channels**:
```
CHANNEL A: Hydrophone Array (N sensors) --> Beamforming --> Spatial Grid
CHANNEL B: AVS Vector Field             --> 3-D Intensity Map
CHANNEL C: Bathymetry + SSP Profile     --> Acoustic Propagation Volume
```

**Unique Capabilities**:
- 3-D sound-field reconstruction — outputs a 3-D voxel grid of acoustic intensity
- Multi-target tracking — simultaneous localisation of up to 5 independent sources
- Reverberation map — identifies echo-generating sea-bottom features
- Rendered live in the **Digital Twin** page

---

### 5.4 EchoPhys-OmniNet (Full Multimodal Fusion Model)

**All Channels**:
| Channel | Input | Encoder |
|---------|-------|---------|
| **Acoustic** | Hydrophone array | Complex STFT encoder |
| **Spatial** | AVS 4-channel | Active intensity encoder |
| **Visual** | Webcam frame | YOLO + ResNet backbone |
| **Ocean** | T, S, D, SSP | Physics state encoder |
| **Historical** | Last-N events | Temporal LSTM |

All channel embeddings are concatenated through a **Physics-Biased Cross-Attention Transformer** for joint classification, tracking, and threat-level scoring.

---

### 5.5 HydroPhys-OmniNet (Hydrophone-Specialised Biodiversity Classifier)
**File**: `backend/app/models/hydrophys_omninet.py`

**Channel**:
```
Hydrophone WAV --> Mel Spectrogram --> Audio Spectrogram Transformer (AST)
                                   --> 32-class bioacoustic taxonomy
```

**Species Classification (Indian Ocean Focus)**:
- Cetaceans: Blue whale, Humpback whale, Risso's dolphin, Indian Ocean bottlenose dolphin
- Fish: Indian mackerel, Croaker, Toadfish
- Crustaceans: Snapping shrimp
- Geophonic: Rain, Microseism, Submarine earthquakes (Indian tectonic zone)
- Anthropogenic: Container ships, Fishing trawlers, Oil rigs, Illegal AUV

---

### 5.6 Acoustic Classifier (Quick Triage)
**File**: `backend/app/sonar/acoustic_classifier.py`

4-layer Transformer encoder on MFCC + Mel spectrograms. 4-class output: Biophonic, Anthropogenic, Geophonic, Tactical Intruder.

---

### 5.7 AVS Locator (Physics-Based DOA)
**File**: `backend/app/sonar/avs_locator.py`

Pure physics — no deep learning:
- `compute_active_intensity(P, U)` — I = 0.5 * Re{P * U*}
- `estimate_doa(I_vector)` — θ = atan2(Iy, Ix), φ = atan2(Iz, |I_xy|)
- `estimate_range(TL, f, c)` — Transmission-loss inversion
- `to_gps(doa, range, sensor_gps)` — WGS-84 coordinate conversion

---

### 5.8 AI Models Summary Table

| Model | Channels | Parameters | Latency | Primary Use |
|-------|----------|-----------|---------|-------------|
| **OCEAN-PHYSNet** | Hydro + AVS + Ocean | ~45 M | ~80 ms | Full physics-aware inference |
| **EchoPhys-Lite** | Hydro only | ~3 M | ~15 ms | Edge / buoy deployment |
| **EchoPhys-Omni-3D** | Hydro Array + AVS + Bathymetry | ~120 M | ~200 ms | 3-D scene reconstruction |
| **EchoPhys-OmniNet** | All channels + Visual | ~200 M | ~350 ms | Fleet-scale fusion intelligence |
| **HydroPhys-OmniNet** | Hydro only | ~28 M | ~60 ms | Biodiversity / bioacoustics |
| **Acoustic Classifier** | Hydro (Mel) | ~8 M | ~20 ms | Quick triage (4 classes) |
| **AVS Locator** | AVS 4-channel | DSP (no DL) | ~5 ms | Physics-based DOA + range |

---

## 6. Core Mathematical Formulas

### 6.1 Mackenzie Sound-Speed Profile (SSP)

```
c(T, S, D) = 1448.96 + 4.591T - 0.05304T^2 + 0.0002374T^3
           + 1.340(S - 35) + 0.0163D + 1.675e-7 * D^2
           - 0.01025 * T * (S - 35) - 7.139e-13 * T * D^3
```

| Variable | Unit | Typical Indian Ocean Range |
|----------|------|--------------------------|
| T — Temperature | °C | 2 – 31 |
| S — Salinity | PSU | 32 – 37 |
| D — Depth | m | 0 – 5500 |
| c — Sound Speed | m/s | 1470 – 1540 |

**Why it matters**: Every bearing and range estimate depends on knowing how fast sound travels. The SSP varies dramatically between the warm Arabian Sea surface (1538 m/s) and the cold deep Indian Ocean (1480 m/s). Getting this wrong by even 5 m/s causes a 0.3% range error — potentially hundreds of metres at tactical distances.

---

### 6.2 Francois-Garrison Seawater Absorption

```
alpha(f) = (A1*P1*f1*f^2) / (f1^2 + f^2)
         + (A2*P2*f2*f^2) / (f2^2 + f^2)
         + A3*P3*f^2          [dB/km]
```

- **Term 1**: Boric acid relaxation (dominant at f < 10 kHz)
- **Term 2**: MgSO4 relaxation (dominant at 10–100 kHz)
- **Term 3**: Pure water viscosity (dominant at f > 100 kHz)

**Why it matters**: A 100 kHz AUV pinger is attenuated ~40 dB/km in warm shallow Indian coastal water. Without this correction, the model would grossly overestimate target range. The absorption coefficient directly biases the attention matrix B_phys in the cross-attention layer.

---

### 6.3 Active Intensity Vector (AVS DOA)

```
I(f) = 0.5 * Re{ P(f) * conj(U(f)) }

theta_az = atan2(Iy, Ix)
phi_el   = atan2(Iz, sqrt(Ix^2 + Iy^2))
```

**Why it matters**: The active intensity vector directly points toward the acoustic source without any array ambiguity. This physics-grounded bearing estimate requires only **one sensor node** (vs. a hydrophone array), making it ideal for single-buoy coastal stations.

---

### 6.4 Physics-Biased Cross-Attention

```
Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) + B_phys ) * V

B_phys[i,j] = -alpha(f) * r_ij / 8.686    [neper/m to dB]
```

where `r_ij` is the estimated path length from token i to j.

**Why it matters**: Without B_phys, the transformer treats all time-frequency tokens equally. By subtracting the physical attenuation, tokens from distant multipath arrivals are down-weighted, and the model automatically focuses on the primary acoustic path — exactly what a sonar operator would do manually.

---

### 6.5 Helmholtz Wave Equation Residual (FNO Block)

```
R_wave = laplacian(p_hat) + k^2 * p_hat

k = 2 * pi * f / c(z)

L_phys = lambda_w * || R_wave ||^2
```

**Why it matters**: Standard neural networks can learn to classify correctly on training data while violating physical wave mechanics. The Helmholtz constraint forces the latent acoustic field representation to be **physically consistent** — so the model remains reliable even when ocean conditions shift dramatically (e.g. monsoon thermocline restructuring in the Bay of Bengal).

---

### 6.6 Periodic DOA Loss (Azimuth & Elevation)

```
L_DOA = (sin(theta_hat) - sin(theta))^2 + (cos(theta_hat) - cos(theta))^2
      + (sin(phi_hat)   - sin(phi))^2   + (cos(phi_hat)   - cos(phi))^2
```

**Why it matters**: A naïve regression loss for bearing suffers from **wrap-around discontinuity** — a prediction of 359° vs ground-truth 1° incurs a huge loss of 358^2, even though the angular error is only 2°. The periodic sine/cosine formulation makes the loss smooth at all angles, enabling stable gradient descent.

---

### 6.7 Heteroscedastic Range Estimation

```
L_R = (R - R_hat)^2 / (2 * sigma_R^2) + 0.5 * log(sigma_R^2)
```

The model simultaneously outputs mean range `mu_R` and predicted uncertainty `sigma_R^2`.

**Why it matters**: In a complex multi-path environment (shallow coastal shelf near Chennai), `sigma_R` will be large — the operator knows to treat the estimate cautiously. In open-ocean deep-water conditions, `sigma_R` will be small — the operator can act with confidence. This replaces a single-point estimate with a **risk-aware probability distribution**.

---

### 6.8 Mahalanobis OOD Score

```
D_M(z) = sqrt( (z - mu_train)^T * inv(Sigma) * (z - mu_train) )
```

| Score Range | Interpretation | Operator Action |
|------------|---------------|----------------|
| D_M < 3 | In-distribution (familiar) | Proceed with classification |
| 3 <= D_M < 6 | Borderline — verify | Flag for analyst review |
| D_M >= 6 | Out-of-distribution | Do NOT trust output; escalate |

**Why it matters**: If a hostile AUV with a never-before-heard acoustic signature enters the detection zone, a conventional classifier will force it into one of the known classes. The Mahalanobis gate ensures the operator is warned: *"this sounds like nothing in our training set"* — preventing false-negative under novel threats.

---

### 6.9 Acoustic Complexity Index (ACI)

```
ACI = sum_t sum_f | A(t,f) - A(t-1,f) | / A(t,f)
```

**Why it matters**: ACI is a biodiversity proxy — complex biophonic soundscapes produce high ACI. Used by the MPA Service to generate coral reef health reports and detect anthropogenic disturbance in Indian Marine Protected Areas.

---

### 6.10 Normalised Difference Soundscape Index (NDSI)

```
NDSI = (Bio(1-2 kHz) - Anthro(2-8 kHz)) / (Bio(1-2 kHz) + Anthro(2-8 kHz))
     in [-1, +1]
```

- **+1**: Pure biophony (pristine marine ecosystem)
- **-1**: Pure technophony (industrial / vessel noise)

**Why it matters**: Used by the Indian Navy and CMFRI to assess whether a coastal area is ecologically healthy or degraded, enabling evidence-based conservation policy.

---

## 7. Data Processing Pipeline

### 7.1 End-to-End Flow

```
+------------------------------------------+
|      USER UPLOADS AUDIO (UI)             |
+------------------+-----------------------+
                   | WAV / FLAC / SDF
          +--------v---------+
          |  FastAPI Ingest  |
          |  /upload-audio   |
          +--------+---------+
                   |
+------------------v-------------------+
|         Audio Processor (DSP)        |
|  Resample to 22050 Hz               |
|  Trim silence (dB gate)             |
|  Extract MFCC (40 coefficients)     |
|  Extract Mel Spectrogram (128 bands)|
|  Compute Complex STFT               |
|  Compute NDSI, ACI, ADI, ZCR, RMS  |
+------------------+-------------------+
                   |
+------------------v-------------------+
|       Ocean State Engine             |
|  T, S, D sliders                    |
|  --> Mackenzie SSP                  |
|  --> Absorption tensor              |
|  --> Normalised Eo vector           |
+--+-----------+--+--+----------------+
   |           |     |
+--v--+   +---v--+  +v-----------+  +----+
|OCEAN|   |Acoust|  |HydroPhys-  |  |AVS |
|PHYS |   |Class |  |OmniNet     |  |Loc.|
|Net  |   |ifier |  |(bioacoust.)|  |    |
+--+--+   +---+--+  +-----------+  +--+-+
   |           |                      |
   +-----------+----------------------+
                        |
          +-------------v-----------+
          |   Inference Service     |
          |  Ensemble / Select      |
          |  Apply Guardrails       |
          |  OOD check              |
          |  Write to PostGIS DB    |
          +-------------+-----------+
                        |
          +-------------v-----------+
          |   React UI — Results    |
          |  Classification + Conf  |
          |  DOA radar + Range      |
          |  SSP profile canvas     |
          |  OOD gauge              |
          |  Audio playback (WAV)   |
          |  Map pin on GIS overlay |
          +-------------------------+
```

### 7.2 Retraining Loop

```
New Labelled Samples (UI Upload)
         |
         v
  Active Learning Service
  (entropy sampling -> select hardest samples)
         |
         v
  Celery Task Queue -> GPU Worker
  (incremental fine-tuning, EWC regularisation)
         |
         v
  Validation Metrics (F1, Balanced Accuracy, AUC)
         |
         v
  Guardrails Service
  (reject if new checkpoint degrades > 2% on held-out set)
         |
         v
  Hot-Swap Production Checkpoint
  (no server restart required)
```

---

## 8. Key Advantages Over Conventional Systems

| Feature | Conventional Passive Sonar | EchoPulseNet |
|---------|--------------------------|-----------------|
| **Physics integration** | Operator applies corrections manually | Built-in Mackenzie SSP + Absorption conditioning |
| **Phase information** | Usually discarded (magnitude-only) | Preserved via Complex STFT encoder |
| **Uncertainty** | Single-point output | Heteroscedastic range + angular variance |
| **Novel threat detection** | Silent failure (wrong class) | Mahalanobis OOD gate + analyst alert |
| **Multipath handling** | Manual post-processing | Multipath Tokenizer in latent space |
| **Edge deployment** | Requires shore-based server | INT8 NPU model, < 15 ms on Jetson/Intel NPU |
| **Retraining** | Months-long offline process | Async continual learning, < 30 min fine-tune |
| **Biodiversity monitoring** | Separate specialised system | Integrated HydroPhys-OmniNet channel |
| **Multi-modal fusion** | Separate sensor systems | Single unified inference pipeline |
| **Explainability** | Black box | Physics-biased attention maps, OOD scores |

---

## 9. India's Ocean, Environment & National Security

### 9.1 India's Strategic Maritime Domain

India possesses the **world's 7th largest Exclusive Economic Zone** — 2.37 million km2 encompassing:
- **Arabian Sea** (west coast — Mumbai, Gujarat, Kerala)
- **Bay of Bengal** (east coast — Tamil Nadu, Andhra Pradesh, Odisha, West Bengal)
- **Andaman & Nicobar Islands** (strategically located at the entrance of the Malacca Strait)
- **Lakshadweep Islands** (guardian of Arabian Sea shipping lanes)

EchoPulseNet is pre-configured for all four regions with region-specific SSP presets, monsoon-aware ocean parameters, and bathymetry data from INCOIS.

### 9.2 Seasonal Ocean Physics Presets

| Season | Region | Acoustic Effect | EchoPulseNet Adaptation |
|--------|--------|-----------------|------------------------|
| **Pre-Monsoon** (Mar–May) | Arabian Sea | Shallow thermocline, high SSP gradient | Thermocline preset: c drops sharply at 50 m |
| **SW Monsoon** (Jun–Sep) | Bay of Bengal | Freshwater runoff -> salinity drop -> lower SSP | Salinity reduced by 3–5 PSU |
| **Post-Monsoon** (Oct–Dec) | All regions | Mixing layer deepens, SSP uniform to 200 m | Mixed-layer preset, lower absorption |
| **Winter** (Jan–Feb) | North Indian Ocean | Coldest SST, deepest thermocline | Deep thermocline preset |

### 9.3 Marine Biodiversity & Environmental Monitoring

EchoPulseNet contributes directly to India's **Marine Protected Area (MPA) network** (106+ MPAs including Gulf of Mannar Marine National Park, Malvan Marine Sanctuary, Rani Jhansi Marine National Park):

- **Passive acoustic monitoring** of whale, dolphin, and dugong populations — no disturbance to animals
- **ACI / NDSI indices** computed for every recording → daily soundscape health report per MPA
- **Noise pollution alerts** when shipping noise exceeds 120 dB re 1 µPa thresholds
- **Coral reef bioacoustics** — healthy reefs are noisy; the system flags silent (bleached) reef zones
- **Real-time debris hotspot mapping** supporting Swachh Sagar Suraksha Kavacha

### 9.4 National Security Applications

| Threat | Detection Method | EchoPulseNet Feature |
|--------|-----------------|---------------------|
| AUV/UUV Intrusion | Broadband pulse signature | Tactical Intruder class + OOD alert |
| Submarine | Low-frequency blade-rate harmonics | HydroPhys-OmniNet low-freq band |
| IUU Trawler | Engine cavitation fingerprint | Anthropogenic class + GPS track |
| Unmanned Surface Drone | Combined acoustic + Webcam fusion | EchoPhys-OmniNet visual channel |

### 9.5 Alignment with Indian Government Initiatives

| Initiative | How EchoPulseNet Supports It |
|-----------|------------------------------|
| **Sagarmala** | Port-based hydrophone networks for vessel monitoring |
| **Blue Economy Policy 2020** | Enables sustainable fishery monitoring via acoustic biomass estimates |
| **Make in India** | Entire AI stack is India-developed; NPU deployment on approved hardware |
| **Deep Ocean Mission** | Provides acoustic pre-screening data for deep-sea research deployments |
| **INCOIS Data Integration** | Direct API integration for live T, S, SSP profile ingestion from Indian Ocean observing system |

### 9.6 Academic Partnerships (Roadmap)

| Institution | Collaboration |
|------------|--------------|
| **IIT Madras** — Ocean Engineering | Acoustics research, validation datasets |
| **CMFRI Kochi** | Marine fauna bioacoustics database |
| **NPOL Kochi** (Naval Physical and Oceanographic Lab) | Technology transfer |
| **NIO Goa** (National Institute of Oceanography) | Open-ocean monitoring buoy integration |

---

## 10. Deployment & Edge Hardware

### 10.1 Deployment Configurations

| Config | Hardware | Model | Latency | Use Case |
|--------|---------|-------|---------|---------|
| **Cloud Server** | GPU A100 / H100 | EchoPhys-OmniNet (full) | ~350 ms | Fleet command HQ |
| **Edge Server** | Intel Core Ultra + NPU | OCEAN-PHYSNet | ~80 ms | Patrol vessel |
| **Mini Edge** | NVIDIA Jetson AGX Orin | EchoPhys-Omni-3D | ~200 ms | Shore station |
| **Nano Edge** | Jetson Nano / RPi 5 + Hailo-8 | EchoPhys-Lite | ~15 ms | Buoy / sonobuoy |
| **Desktop App** | Windows PC (Tauri) | Acoustic Classifier | ~20 ms | Research workstation |

### 10.2 Continuous NPU Coprocessor
**File**: `scripts/run_continuous_npu_coprocessor.py`

```
Watch folder for new WAV files
        |
        v
Read + Preprocess audio
        |
        v
Dispatch to Intel NPU via OpenVINO async API
        |
        v
Receive predictions (< 100 ms)
        |
        v
Write JSON telemetry to output folder
        |
        v
FastAPI picks up telemetry -> pushes to UI via WebSocket
```

### 10.3 Jetson AGX Orin
**File**: `launch_jetson_agx.sh`

TensorRT-optimised model, CUDA 12.x, runs with `install_jetson_baremetal.sh` for bare-metal configuration — ship-board deployment without Docker overhead.

---

## 11. Future Roadmap

| Priority | Feature | Timeline |
|---------|---------|---------|
| High | **Federated Learning** — Aggregate model updates from 50+ buoys without sharing raw audio | Q4 2026 |
| High | **AIS Fusion** — Cross-reference acoustic detections with AIS vessel tracks | Q4 2026 |
| Medium | **3-D Ocean Acoustic Tomography** — Full 3-D sound-field reconstruction at basin scale | Q1 2027 |
| Medium | **Satellite SAR Integration** — Fuse SAR imagery with acoustic contacts for USV confirmation | Q1 2027 |
| Medium | **Swarm UAV Coordination** — Acoustic cue -> dispatch aerial drone for visual confirmation | Q2 2027 |
| Low | **Open-Source Research Release** — Apache 2.0 stripped version for Indian universities | Q3 2027 |
| Low | **INCOIS Live Data Feed** — Automated real-time T/S/SSP ingest from INCOIS ARGO floats | Q2 2027 |
| Low | **Marine Mammal Alert API** — Public REST API for CMFRI and MoEFCC whale sighting alerts | Q3 2027 |

---

## Appendix A — File Structure

```
EchoPulseNet/
+-- src/                                  # React Frontend
|   +-- pages/                            # 15 application pages
|   +-- components/                       # Reusable UI components
|   +-- context/                          # Global state providers
|   +-- services/                         # API client functions
|   +-- echophys/                         # Frontend physics utilities
|   +-- index.css                         # Design system tokens
+-- backend/
|   +-- app/
|       +-- api/routes.py                 # All API endpoints
|       +-- models/                       # AI/ML model definitions
|       |   +-- ocean_physnet.py          # OCEAN-PHYSNet master model
|       |   +-- echophys_lite.py          # Lightweight edge model
|       |   +-- echophys_omni_3d.py       # 3-D scene model
|       |   +-- hydrophys_omninet.py      # Biodiversity classifier
|       |   +-- ai_models.py              # Model registry & loader
|       +-- sonar/                        # DSP & physics modules
|       |   +-- audio_processor.py        # Feature extraction
|       |   +-- ocean_state.py            # Mackenzie SSP + Absorption
|       |   +-- ocean_phys_encoders.py    # Complex STFT + AVS encoder
|       |   +-- physics_attention.py      # FNO + biased cross-attention
|       |   +-- acoustic_classifier.py    # Quick-triage model
|       |   +-- avs_locator.py            # Physics-based DOA
|       +-- services/                     # Business logic
|           +-- inference_service.py      # Central inference orchestrator
|           +-- retraining_service.py     # Continual learning
|           +-- active_learning_service.py# Query strategy
|           +-- guardrails_service.py     # Output safety
|           +-- postgis_service.py        # Spatial data
|           +-- mpa_service.py            # Marine Protected Areas
|           +-- gpu_worker.py             # NPU/GPU dispatch
|           +-- report_service.py         # PDF/JSON report export
+-- scripts/
|   +-- run_continuous_npu_coprocessor.py # Always-on edge inference
+-- configs/                              # Model & deployment configs
+-- data/                                 # Training datasets
+-- models_checkpoints/                   # Saved model weights
+-- tests/                                # Unit + integration tests
```

---

## Appendix B — Glossary

| Term | Definition |
|------|-----------|
| **ACI** | Acoustic Complexity Index — biodiversity proxy |
| **AUV** | Autonomous Underwater Vehicle |
| **AVS** | Acoustic Vector Sensor — measures pressure P and particle velocity Ux, Uy, Uz |
| **DOA** | Direction of Arrival |
| **EEZ** | Exclusive Economic Zone — 200 nautical miles from India's coast |
| **FNO** | Fourier Neural Operator — physics-constrained deep learning block |
| **INCOIS** | Indian National Centre for Ocean Information Services |
| **MPA** | Marine Protected Area |
| **NDSI** | Normalized Difference Soundscape Index |
| **NPU** | Neural Processing Unit — dedicated AI accelerator |
| **OOD** | Out-of-Distribution |
| **SSP** | Sound-Speed Profile — variation of c with depth |
| **STFT** | Short-Time Fourier Transform |
| **TL** | Transmission Loss — sound energy lost over distance |
| **UUV** | Unmanned Underwater Vehicle |

---

*Document compiled 2026-09-01 · EchoPulseNet Marine Sonar Intelligence Platform · India*

*Developed with pride for India's maritime security, ocean science, and environmental stewardship.*

**Jai Hind. Jai Samudra.**

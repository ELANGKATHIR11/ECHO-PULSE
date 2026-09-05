# EchoPulseNet Model Registry & Lifecycle Management

**Last Updated**: September 4, 2026  
**Status Policy**: `ACTIVE | EXPERIMENTAL | DEPRECATED | ARCHIVED`

---

## Target Model Family (Unified 7-Model Suite)

| ID | Model Identifier | Role / Description | Status | Architecture | Input / Output Schema | Checkpoint |
|---|---|---|---|---|---|---|
| **1** | `OCEAN-PHYSNet-X` | Physics-aware multimodal fusion | `ACTIVE` | FNO Helmholtz + Cross-Attention + Fourier Encoders | In: Hydro (1,L), AVS (4,L), Ocean State (16)<br>Out: Class (4), Subclass (17), DOA (Az/El), Range, OOD | `models_checkpoints/ocean_physnet_best.pt` |
| **2** | `EchoPhys-Lite-X` | Adaptive low-latency edge inference | `ACTIVE` | 3-Ch Specular/Shadow Decomposition + BiMamba-Lite (780K params) | In: Sonar Image (3,H,W)<br>Out: 8-Class Bounding Boxes, Segmentation, Height Profile | `models_checkpoints/echophys_lite_best.pt` |
| **3** | `EchoPhys-OmniNet-X` | Reliability / physics-gated multimodal fusion | `ACTIVE` | Bilateral Wave-Equation State-Space (CAW-SSM) | In: 8-Ch Acoustic Physics Tensor (8,H,W)<br>Out: 8-Class Segmentation, Volumetric Strata, Anomaly | `models_checkpoints/hydrophys_omninet_extreme_best.pt` |
| **4** | `EchoPhys-Omni-3D-X` | 4D underwater state / localization | `ACTIVE` | 1D Strata Wavelet + 2D ProtoMask + 3D Volumetric Voxel Projector | In: Sub-bottom Ping (1024) + SSS (1,H,W)<br>Out: 4D Spatiotemporal Geotag, Volumetric Voxels, Height | `models_checkpoints/echophys_x_v3_unified_best.pt` |
| **5** | `HydroPhys-OmniNet-X` | Propagation-aware acoustic classification | `ACTIVE` | Continuous Acoustic Waveform SSM with FNO Propagation | In: Raw Waveform (1,L) + Ambient Sound Speed Profile<br>Out: Marine Soundscape Taxonomy, Source Level, Attenuation | `models_checkpoints/hydrophys_omninet_extreme_best.pt` |
| **6** | `Acoustic-Triage-Transformer-X` | Fast hierarchical acoustic classification | `ACTIVE` | Hierarchical Multi-Scale Transformer with Gated Attention | In: Complex STFT Spectrogram (B,F,T)<br>Out: Fast Triage (Bio/Anthropogenic/Geo/Tactical) + Severity | `models_checkpoints/acoustic_triage_transformer_best.pt` |
| **7** | `AVS-GeoPhysics-X` | Probabilistic spherical DOA + range + geolocation | `ACTIVE` | Cross-Spectral Active Intensity + Heteroscedastic Geodesic Kalman Filter | In: 4-Ch AVS (P, Ux, Uy, Uz) + Platform GPS + SSP<br>Out: Spherical DOA [$\cos\phi\cos\theta, \cos\phi\sin\theta, \sin\phi$], Range, Target GPS, Uncertainty Ellipse | `models_checkpoints/avs_geophysics_best.pt` |

---

## Legacy & Archived Models

| Legacy Model Name | Version | Archive Path | Status | Notes |
|---|---|---|---|---|
| `LightweightSonarUNet` | v1.0 | `archive/models/unet_shadow/v1.0/` | `ARCHIVED` | Early acoustic shadow segmentation baseline before physics Mamba. |
| `SonarAutoencoder` | v1.0 | `archive/models/seabed_autoencoder/v1.0/` | `ARCHIVED` | Simple convolutional autoencoder for seabed reconstruction anomaly detection. |
| `YOLOv12-Marine-Baseline` | v1.0 | `archive/models/yolov12_marine/v1.0/` | `DEPRECATED` | Conventional computer vision baseline lacking acoustic shadow physics and wave propagation constraints. |
| `EchoPhys-X-v1` | v1.0 | `archive/models/echophys_x/v1.0/` | `ARCHIVED` | First generation physics-informed acoustic prototype. |
| `EchoPhys-Unified-v2` | v2.0 | `archive/models/echophys_unified/v2.0/` | `ARCHIVED` | Precursor to v3 multi-silicon unified architecture. |

---

## Model Evaluation & Real Benchmarks Baseline

| Model Identifier | Primary Metric | Latency (Nominal) | Parameters | Model Size | Hardware Target |
|---|---|---|---|---|---|
| `OCEAN-PHYSNet-X` | Multimodal F1: 94.8% | 14.2 ms | 18.4M | 72.5 MB | dGPU RTX 5060 / Server |
| `EchoPhys-Lite-X` | mAP@0.5: 88.4%, Binary Debris F1: 91.2% | 2.6 ms | 0.78M | 2.2 MB | Edge NPU / Embedded AUV |
| `EchoPhys-OmniNet-X` | 8-Class mIoU: 81.6% | 5.8 ms | 1.61M | 19.2 MB | Jetson AGX / RTX 5060 |
| `EchoPhys-Omni-3D-X` | 3D Depth Error: <0.18m | 7.2 ms | 1.56M | 19.0 MB | Jetson AGX / RTX 5060 |
| `HydroPhys-OmniNet-X` | Soundscape Top-1 Acc: 93.1% | 5.8 ms | 1.61M | 19.2 MB | Hydrophone Subsea Node |
| `Acoustic-Triage-Transformer-X` | Triage Latency: <1.8 ms, Top-1 Acc: 96.4% | 1.8 ms | 0.45M | 1.8 MB | Low-Power Buoy / Towfish |
| `AVS-GeoPhysics-X` | Angular Error: <2.4°, Geolocation Error: <18m | 2.1 ms | 0.62M | 2.5 MB | AVS Sonar Array Node |

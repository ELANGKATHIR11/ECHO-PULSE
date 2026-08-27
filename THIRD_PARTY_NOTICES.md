# Third-Party Notices & Open Source Licensing

EchoPulseNet utilizes and acknowledges the following open-source software libraries, frameworks, and scientific datasets.

---

## 1. Third-Party Software Libraries

| Package | Source / Repository | License | Purpose | Modifications |
|---|---|---|---|---|
| **pyxtf** | [github.com/oysstu/pyxtf](https://github.com/oysstu/pyxtf) | MIT License | Triton eXtended Sonar (.XTF) binary packet decoding | Incorporated as standalone parser module |
| **OpenCV** | [opencv/opencv](https://github.com/opencv/opencv) | Apache 2.0 | Sonar 2D despeckling, CLAHE, bilateral filtering, contour analysis | Standard API usage |
| **PyTorch** | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | Deep learning model definition, tensor physics computation, and GPU acceleration | Standard framework usage |
| **Ultralytics YOLO** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | Comparative edge baseline object detection | Fine-tuned weights on marine sonar datasets |
| **FastAPI / Pydantic** | [tiangolo/fastapi](https://github.com/tiangolo/fastapi) | MIT License | Offline-first high-performance REST API backend | Standard API architecture |
| **Three.js** | [mrdoob/three.js](https://github.com/mrdoob/three.js) | MIT License | 3D target geospatial visualization and benthic digital twin rendering | Custom shaders & bathymetry terrain geometry |

---

## 2. Scientific Datasets & Public Registries

| Dataset | Source / Institution | Citation / License | Role in EchoPulseNet |
|---|---|---|---|
| **AI4Shipwrecks** | University of Michigan / DeepBlue Repository | CC BY 4.0 | Benchmark high-resolution side-scan shipwreck sonar imagery |
| **PING Crab Pot Dataset** | Hugging Face PING Ecosystem | Open Access / Research | Derelict ghost fishing gear and pot detection |
| **SeabedObjects** | Open Source Marine Benchmark Collection | Research License | Seabed anomaly, airplane wreckage, and mine-like objects |
| **Indian MPA Registry** | MoEFCC / NCCR / INCOIS Public Hydrographic Records | Public Domain (Government of India) | Geo-tagged marine protected zone polygons & EEZ boundaries |

---

## 3. Custom / Proposed AI Architecture
- **HydroPhys-OmniNet**: Proposed continuous wave-equation state-space (CAW-SSM) neural architecture combining 1D strata wavelet inversion, 8-channel physics tensor, and 2D/3D benthic target localization.
- **EchoPhys-X v3**: Proposed physics-informed bidirectional Mamba architecture for multi-channel acoustic backscatter segmentation.
- **Multi-Factor Confidence Fusion**: Proposed empirical fusion model combining detector confidence, shadow physics validation, morphological geometry, and autoencoder anomaly scores.

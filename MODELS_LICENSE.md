# PROPRIETARY DEEP LEARNING MODEL LICENSE & INTELLECTUAL PROPERTY NOTICE
**Copyright (c) 2026 Elangkathir & EchoPulseNet Marine AI Research Team. All Rights Reserved.**

---

### 🛡️ Proprietary Model Notice
The two proprietary deep learning model architectures, trained weight tensors, and custom mathematical neural operators contained within this repository:

1. **HydroPhys-OmniNet v4 (Multi-Modal Physical-Acoustic State-Space Sonar Foundation Model)**
   - *Architecture*: Homoscedastic uncertainty-weighted multi-task attention neural core (`HomoscedasticMultiTaskLoss`, multi-head grazing angle modulation, $A2C2f$ area attention heads).
   - *Weights & Checkpoints*: `models_checkpoints/yolov12_echopulse_marine.pt`, `runs/detect/echopulse_yolov12/weights/best.pt`, and associated fine-tuned LoRA adapter tensors.

2. **EchoPhys-X Sonar Neural Engine (Acoustic Shadow & Seabed Autoencoder Segmentation)**
   - *Architecture*: Dual-Head `LightweightSonarUNet` & `SonarAutoencoder` with Physics-Informed Slant-to-Ground range geometric kernels and `MultiFactorFusion` matrix.
   - *Weights & Checkpoints*: PyTorch neural state dictionaries and latent feature embeddings for benthic anomaly discovery.

---

### ⚖️ Terms of Private License

1. **PROPRIETARY & CONFIDENTIAL INTELLECTUAL PROPERTY**:
   The model architectures, source implementations in `backend/app/models/ai_models.py`, mathematical loss formulations, and neural weights (`.pt`, `.onnx`, `.engine`) are the exclusive intellectual property of the author (**Elangkathir**).

2. **RESTRICTIONS ON USE, DISTRIBUTION, & MODIFICATION**:
   - **No Unauthorized Commercialization**: You may not sell, lease, sublicense, distribute, or commercially exploit these 2 neural models or their distilled variants without explicit written permission from the copyright holder.
   - **No Reverse Engineering / Model Extraction**: Decompilation, weight extraction for dataset distillation, or unauthorized retraining of these model checkpoints is strictly prohibited.
   - **Evaluation & Hackathon Demonstration Only**: Permission is granted solely for non-commercial academic evaluation, research review, and evaluation during **Smart India Hackathon (SIH 2026)**.

3. **EXCLUSION FROM GPL v3.0**:
   While the surrounding web interface, dashboard, DSP processors, and API utilities in this repository are open under the **GNU General Public License v3.0 (GPLv3)**, the aforementioned 2 Deep Learning models and their neural weights are explicitly **EXEMPT** from GPLv3 and remain under this Private Proprietary License.

---
For commercial licensing, enterprise defense deployment, or NIOT integration queries:
**Contact**: `elangkathir@echopulse.net` / GitHub: `@ELANGKATHIR11`

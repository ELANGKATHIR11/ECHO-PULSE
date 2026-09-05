# Legacy Model Archive: YOLOv12 Marine Baseline v1.0

- **Original Architecture**: Standard YOLOv12 Object Detection Baseline (Ultralytics)
- **Input**: 3-Channel RGB/Grayscale Image (640x640)
- **Output**: 8-Class Bounding Boxes + Confidence
- **Status**: DEPRECATED
- **Replaced By**: `EchoPhys-Lite-X` (Acoustic BiMamba with physical specular highlight & shadow profiling)
- **Deficiency**: Lacks acoustic grazing angle physics, sediment penetration, and transmission loss compensation.
- **Original Checkpoint Reference**: `models_checkpoints/yolov12_echopulse_marine.pt`, `models_checkpoints/yolov12_echopulse_marine.onnx`
- **Parameters**: 1.12M

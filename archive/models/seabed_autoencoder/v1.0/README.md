# Legacy Model Archive: SonarAutoencoder v1.0

- **Original Architecture**: 2D Convolutional Autoencoder for normal seabed background reconstruction
- **Input**: 1-Channel Raw Sonar Image (128x128)
- **Output**: 1-Channel Reconstructed Image + L2 Reconstruction Error Anomaly Score
- **Status**: ARCHIVED / DEPRECATED
- **Replaced By**: Mahalanobis Out-of-Distribution and Physics Residual in `OCEAN-PHYSNet-X`
- **Original Checkpoint Reference**: `models_checkpoints/seabed_autoencoder.pt`, `models_checkpoints/seabed_autoencoder.onnx`
- **Parameters**: 0.18M

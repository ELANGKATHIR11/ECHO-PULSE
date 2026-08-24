import sys
import os
import io

# Force UTF-8 for windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from backend.app.models.ai_models import LightweightSonarUNet, SonarAutoencoder

def export_models():
    print("=== EXPORTING MODELS TO ONNX ===")
    os.makedirs("models_checkpoints", exist_ok=True)
    
    # 1. Export UNet
    unet = LightweightSonarUNet(in_channels=1, out_channels=2)
    unet.eval()
    dummy_input_unet = torch.randn(1, 1, 256, 256)
    unet_path = "models_checkpoints/unet_shadow.onnx"
    torch.onnx.export(
        unet,
        dummy_input_unet,
        unet_path,
        input_names=["sonar_image"],
        output_names=["shadow_mask"],
        opset_version=18
    )
    print(f"[PASS] Exported UNet Shadow Segmenter to {unet_path}")
    
    # 2. Export Autoencoder
    ae = SonarAutoencoder()
    ae.eval()
    dummy_input_ae = torch.randn(1, 1, 128, 128)
    ae_path = "models_checkpoints/seabed_autoencoder.onnx"
    torch.onnx.export(
        ae,
        dummy_input_ae,
        ae_path,
        input_names=["patch"],
        output_names=["reconstructed_patch"],
        opset_version=18
    )
    print(f"[PASS] Exported Conv Autoencoder to {ae_path}")
    print(f"[PASS] Exported UNet Shadow Segmenter to {unet_path}")
    
    # 2. Export Autoencoder
    ae = SonarAutoencoder()
    ae.eval()
    dummy_input_ae = torch.randn(1, 1, 128, 128)
    ae_path = "models_checkpoints/seabed_autoencoder.onnx"
    torch.onnx.export(
        ae,
        dummy_input_ae,
        ae_path,
        input_names=["patch"],
        output_names=["reconstructed_patch"],
        dynamic_axes={"patch": {0: "batch_size"}},
        opset_version=14
    )
    print(f"[PASS] Exported Conv Autoencoder to {ae_path}")

if __name__ == "__main__":
    export_models()

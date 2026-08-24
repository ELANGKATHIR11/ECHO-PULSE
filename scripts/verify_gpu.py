# GPU Verification Script for EchoPulseNet
import sys
import subprocess

def check_nvidia_smi():
    print("=== NVIDIA-SMI OUTPUT ===")
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(res.stdout if res.returncode == 0 else res.stderr)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

def check_torch_cuda():
    print("\n=== PYTORCH CUDA PROBE ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if hasattr(torch.cuda, "is_initialized"):
            print(f"torch.cuda.is_initialized(): {torch.cuda.is_initialized()}")
        if torch.cuda.is_available():
            print(f"Device Count: {torch.cuda.device_count()}")
            print(f"Device Name 0: {torch.cuda.get_device_name(0)}")
            # Test tensor allocation
            x = torch.randn(100, 100, device="cuda")
            print(f"CUDA Tensor test successful: sum={x.sum().item():.4f}")
        else:
            print("Notice: CUDA is not available to torch. Probing possible reasons (e.g. driver initialization, architecture support)...")
            try:
                # Attempt to initialize cuda directly to see actual driver error
                torch.cuda.init()
            except Exception as e:
                print(f"Direct torch.cuda.init() exception: {e}")
    except Exception as e:
        print(f"Error checking torch cuda: {e}")

if __name__ == "__main__":
    check_nvidia_smi()
    check_torch_cuda()

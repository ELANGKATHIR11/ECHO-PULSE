# System Diagnostics Script for EchoPulseNet
import sys
import os
import platform
import json

def run_diagnostics():
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": os.name,
        "cuda_available": False,
        "torch_version": None,
        "cuda_version": None,
        "device_count": 0,
        "device_names": [],
        "packages": {}
    }
    
    # Check PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
            info["current_device_properties"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                "capability": torch.cuda.get_device_capability(0)
            }
    except ImportError:
        info["torch_error"] = "PyTorch not installed in this environment"
    except Exception as e:
        info["torch_error"] = str(e)
        
    # Check other critical packages
    pkgs = [
        "numpy", "scipy", "cv2", "fastapi", "uvicorn", "pydantic", 
        "geopandas", "shapely", "pyproj", "onnxruntime", "pyxtf"
    ]
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            info["packages"][pkg] = "NOT_INSTALLED"
        except Exception as e:
            info["packages"][pkg] = f"ERROR: {str(e)}"

    print("=" * 60)
    print("ECHOPULSENET SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print(f"Python: {info['python_version'].split()[0]} ({platform.system()} {platform.release()})")
    if info.get("torch_version"):
        print(f"PyTorch: {info['torch_version']} | CUDA: {info['cuda_version']} | CUDA Available: {info['cuda_available']}")
        if info['cuda_available']:
            for idx, name in enumerate(info['device_names']):
                print(f"  [GPU {idx}] {name}")
                props = info.get("current_device_properties", {})
                print(f"         VRAM: {props.get('total_memory_gb', 'N/A')} GB, Compute Capability: {props.get('capability', 'N/A')}")
        else:
            print("  [WARN] CUDA is NOT available in this PyTorch build / environment.")
    else:
        print("PyTorch: NOT INSTALLED")
    
    print("-" * 60)
    print("Package Check:")
    for k, v in info["packages"].items():
        status = "[OK]" if v != "NOT_INSTALLED" and not v.startswith("ERROR") else "[MISSING]"
        print(f"  {status} {k:<15}: {v}")
    print("=" * 60)
    
    # Save diagnostics report
    os.makedirs("reports", exist_ok=True)
    with open("reports/system_diagnostics.json", "w") as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    run_diagnostics()

"""
================================================================================
EchoPulseNet Hardware Acceleration Engine: Intel(R) AI Boost NPU Dispatcher
Default Hardware Acceleration for All Deep Learning & Computer Vision Models
================================================================================

Dispatches ML/DL workloads to the hardware processors in order of priority:
  1. PRIMARY   : Intel(R) AI Boost NPU (OpenVINO 'NPU' Native Hardware Execution)
  2. SECONDARY : NVIDIA GeForce RTX 5060 Laptop GPU ('GPU.1' / CUDA:0)
  3. TERTIARY  : Intel(R) Graphics iGPU ('GPU.0' / OpenCL)
  4. FALLBACK  : Intel Core Ultra 9 275HX ('CPU')
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


# ==============================================================================
# Hardware NPU Prober & Manager
# ==============================================================================
class NPUHardwareManager:
    _instance = None
    _core = None
    _npu_available = False
    _npu_device_name = "None"
    _available_devices = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NPUHardwareManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if HAS_OPENVINO:
            try:
                self._core = ov.Core()
                self._available_devices = self._core.available_devices
                if "NPU" in self._available_devices:
                    self._npu_available = True
                    try:
                        self._npu_device_name = str(self._core.get_property("NPU", "FULL_DEVICE_NAME"))
                    except Exception:
                        self._npu_device_name = "Intel(R) AI Boost NPU"
                    print(f"[*] NPUHardwareManager: [ACTIVE] Hardware Accelerator Selected: {self._npu_device_name} (NPU)")
                else:
                    print(f"[!] NPUHardwareManager: NPU not found in OpenVINO devices: {self._available_devices}")
            except Exception as e:
                print(f"[!] NPUHardwareManager: Initialization error: {e}")
        else:
            print("[!] NPUHardwareManager: OpenVINO runtime not installed, falling back to Torch.")

    @property
    def is_npu_available(self) -> bool:
        return self._npu_available

    @property
    def npu_name(self) -> str:
        return self._npu_device_name

    @property
    def device_name(self) -> str:
        return self._npu_device_name

    @property
    def core(self) -> Optional[Any]:
        return self._core

    def get_preferred_device(self, user_override: Optional[str] = None) -> str:
        if user_override and user_override.lower() not in ["auto", "default"]:
            return user_override

        # Default to Intel AI Boost NPU
        if self._npu_available:
            return "NPU"
        elif torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def compile_for_npu(
        self,
        model_or_path: Union[str, Path, nn.Module],
        example_input: Optional[torch.Tensor] = None,
        device_target: str = "NPU"
    ) -> Optional[Any]:
        """
        Compiles an ONNX model or PyTorch nn.Module directly onto the Intel(R) AI Boost NPU.
        """
        if not HAS_OPENVINO or self._core is None:
            return None

        # Determine target device
        target = device_target if device_target in self._available_devices else ("NPU" if self._npu_available else "CPU")

        try:
            if isinstance(model_or_path, (str, Path)):
                model_file = Path(model_or_path)
                if model_file.exists():
                    ov_model = self._core.read_model(str(model_file))
                    compiled = self._core.compile_model(ov_model, target)
                    print(f"[PASS] Successfully compiled {model_file.name} to {self._npu_device_name} ({target})")
                    return compiled
            elif isinstance(model_or_path, nn.Module) and example_input is not None:
                model_or_path.eval()
                ov_model = ov.convert_model(model_or_path, example_input=example_input)
                compiled = self._core.compile_model(ov_model, target)
                print(f"[PASS] Successfully compiled PyTorch {type(model_or_path).__name__} to {self._npu_device_name} ({target})")
                return compiled
        except Exception as e:
            print(f"[!] Warning: NPU compilation failed for {model_or_path}: {e}")
            # Fallback to CPU/GPU if NPU compile failed for dynamic ops
            if target != "CPU":
                try:
                    ov_model = self._core.read_model(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else ov.convert_model(model_or_path, example_input=example_input)
                    compiled = self._core.compile_model(ov_model, "CPU")
                    print(f"[*] Fallback: Compiled to OpenVINO CPU engine.")
                    return compiled
                except Exception:
                    pass

        return None


# Global Singleton Instance
npu_manager = NPUHardwareManager()


def get_default_device() -> str:
    """Returns the default hardware device string (Defaults to NPU Intel AI Boost)."""
    return npu_manager.get_preferred_device()

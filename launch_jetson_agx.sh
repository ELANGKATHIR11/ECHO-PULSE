#!/usr/bin/env bash
# ==============================================================================
# EchoPulseNet - Jetson AGX Orin Native Launch Script
# Starts Local PostGIS and Unified Edge Server (Port 8000)
# ==============================================================================

echo "--------------------------------------------------------"
echo " Launching EchoPulseNet on NVIDIA Jetson AGX Orin..."
echo " Platform: JetPack ARM64 (Air-Gapped Edge Native Mode)"
echo "--------------------------------------------------------"

# 1. Start Local PostgreSQL if not running
if command -v systemctl &> /dev/null; then
    sudo systemctl start postgresql || true
fi

# 2. Set Jetson Performance Clocks (Max-N mode if nvpmodel available)
if command -v nvpmodel &> /dev/null; then
    echo "[JETSON AGX] Setting Performance Mode to MAXN..."
    sudo nvpmodel -m 0 || true
    sudo jetson_clocks || true
fi

# 3. Source Secure Environment Variables if available
if [ -f "/etc/echopulse/echopulse.env" ]; then
    set -a
    source /etc/echopulse/echopulse.env
    set +a
else
    export ECHOPULSENET_ENV="JETSON_AGX_BAREMETAL"
    export CUDA_VISIBLE_DEVICES=0
    export TENSORRT_CACHE_DIR="./cache/tensorrt"
    export ALLOWED_ORIGINS="http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000"
fi

# 4. Start Unified Fast Server (Single GPU Inference Worker Architecture)
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1

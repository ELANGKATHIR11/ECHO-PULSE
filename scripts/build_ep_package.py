#!/usr/bin/env python3
"""
EchoPulseNet .ep (Edge Package) Compiler & Self-Extracting Single-Click Bundle Generator
========================================================================================
Target: NVIDIA Jetson AGX (Orin / Xavier) Linux aarch64 (Bare-Metal, Zero Docker, Air-Gapped)
Generates: echopulsenet-jetson-agx-v2.6.0.ep

A single-click, self-extracting, self-installing binary bundle containing:
- Pre-compiled React + Vite + Leaflet + Three.js Digital Twin UI
- Python Unified AI Core & EchoPhys-Lite PINN + YOLOv12 Engines
- Native PostGIS Spatial Database auto-provisioner
- Jetson Hardware Clock / MAXN optimizer daemon
- Systemd service generator and auto-launcher
"""

import os
import sys
import tarfile
import io
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "dist"
OUTPUT_EP_FILE = ROOT_DIR / "echopulsenet-jetson-agx-v2.6.0.ep"

RUNNER_HEADER = """#!/usr/bin/env bash
# ==============================================================================
# EchoPulseNet NVIDIA JETSON AGX BARE-METAL SELF-EXTRACTING RUNTIME (.ep)
# Format: EchoPulse Edge Package (.ep)
# Single-Click Air-Gapped Zero-Cloud Marine Intelligence Platform
# ==============================================================================
set -e

# Terminal Colors
CYAN='\\033[0;36m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m'

echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}   _____      _            _____       _          _   _      _   ${NC}"
echo -e "${CYAN}  | ____| ___| |__   ___  |  __ \\ _   _| |___  ___| \\ | | ___| |_ ${NC}"
echo -e "${CYAN}  |  _|  / __| '_ \\ / _ \\ | |__) | | | | / __|/ _ \\  \\| |/ _ \\ __|${NC}"
echo -e "${CYAN}  | |___| (__| | | | (_) ||  ___/| |_| | \\__ \\  __/ |\\  |  __/ |_ ${NC}"
echo -e "${CYAN}  |_____|\\___|_| |_|\\___/ |_|     \\__,_|_|___/\\___|_| \\_|\\___|\\__|${NC}"
echo -e "${GREEN}  NVIDIA JETSON AGX EDGE-NATIVE RUNTIME (.ep Single-Click Bundle)${NC}"
echo -e "${CYAN}================================================================================${NC}"

# Check for root / sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}[ERROR] Please execute this single-click package with sudo privileges:${NC}"
  echo -e "  sudo ./${0##*/}"
  exit 1
fi

INSTALL_PATH="/opt/echopulsenet"
echo -e "${YELLOW}[1/6] Extracting Edge Package Payload to ${INSTALL_PATH}...${NC}"

mkdir -p ${INSTALL_PATH}
mkdir -p ${INSTALL_PATH}/uploads
mkdir -p ${INSTALL_PATH}/models_checkpoints
mkdir -p ${INSTALL_PATH}/cache/tensorrt

# Extract binary tar stream payload embedded at the end of this script
ARCHIVE_START_LINE=$(awk '/^__ECHOPULSE_PAYLOAD_START__/ {print NR + 1; exit 0; }' "$0")
tail -n +${ARCHIVE_START_LINE} "$0" | tar -xz -C ${INSTALL_PATH}

echo -e "${YELLOW}[2/6] Verifying & Installing Linux Jetson Dependencies (PostGIS, Python, Node)...${NC}"
apt-get update -y -qq
apt-get install -y -qq --no-install-recommends \
    python3-pip python3-dev build-essential libpq-dev \
    postgresql postgresql-contrib postgis \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx v4l-utils supervisor curl

# 3. Setup Local Air-Gapped PostGIS DB with Unique Runtime Secret
echo -e "${YELLOW}[3/6] Initializing Local PostGIS Spatial Database & System User...${NC}"
systemctl enable postgresql
systemctl start postgresql

# Create dedicated non-root application system user
if ! id -u echopulse &>/dev/null; then
    useradd -r -s /bin/false -d ${INSTALL_PATH} -M echopulse || true
fi

# Generate strong random credentials
DB_PASS=$(openssl rand -hex 24)
CONFIG_DIR="/etc/echopulse"
mkdir -p ${CONFIG_DIR}
chmod 700 ${CONFIG_DIR}

sudo -u postgres psql -c "CREATE USER echopulse WITH PASSWORD '${DB_PASS}';" || sudo -u postgres psql -c "ALTER USER echopulse WITH PASSWORD '${DB_PASS}';"
sudo -u postgres psql -c "CREATE DATABASE echopulse_gis OWNER echopulse;" || true
sudo -u postgres psql -d echopulse_gis -c "CREATE EXTENSION IF NOT EXISTS postgis;" || true
sudo -u postgres psql -d echopulse_gis -c "GRANT ALL PRIVILEGES ON DATABASE echopulse_gis TO echopulse;" || true

# Generate Master Encryption Key
MASTER_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || openssl rand -base64 32)
cat << ENV_EOF > ${CONFIG_DIR}/echopulse.env
DATABASE_URL=postgresql://echopulse:${DB_PASS}@127.0.0.1:5432/echopulse_gis
POSTGIS_DATABASE_URL=postgresql://echopulse:${DB_PASS}@127.0.0.1:5432/echopulse_gis
ECHOPULSENET_SECRET_KEY=${MASTER_KEY}
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000
PYTHONUNBUFFERED=1
ECHOPULSENET_ENV=JETSON_AGX_NATIVE_EP
CUDA_VISIBLE_DEVICES=0
TENSORRT_CACHE_DIR=/opt/echopulsenet/cache/tensorrt
ENV_EOF

chown -R echopulse:echopulse ${CONFIG_DIR}
chmod 600 ${CONFIG_DIR}/echopulse.env
chown -R echopulse:echopulse ${INSTALL_PATH}

# 4. Install Python Inference Dependencies
echo -e "${YELLOW}[4/6] Initializing Fast AI Backend and TensorRT Acceleration...${NC}"
python3 -m pip install --upgrade pip -q
pip3 install -q fastapi "uvicorn[standard]" ultralytics opencv-python-headless pydantic asyncpg psutil torch torchvision cryptography || true

# 5. Maximize Jetson Performance Clocks (MAX-N)
if command -v nvpmodel &> /dev/null; then
    echo -e "${GREEN}[5/6] Tuning Jetson AGX GPU/CPU Clocks (MAX-N Power Profile)...${NC}"
    nvpmodel -m 0 || true
    jetson_clocks || true
else
    echo -e "${YELLOW}[5/6] Non-Jetson Hardware detected. Skipping nvpmodel tuning.${NC}"
fi

# 6. Setup and Launch Systemd Daemon (Non-Root User 'echopulse', 1 Worker)
echo -e "${YELLOW}[6/6] Launching EchoPulseNet Edge Platform Engine (Non-Root User)...${NC}"

cat << 'SERVICE_EOF' > /etc/systemd/system/echopulsenet.service
[Unit]
Description=EchoPulseNet NVIDIA Jetson AGX Edge-Native Marine AI Platform
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=echopulse
Group=echopulse
WorkingDirectory=/opt/echopulsenet
EnvironmentFile=/etc/echopulse/echopulse.env
ExecStart=/usr/bin/python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE_EOF

systemctl daemon-reload
systemctl enable echopulsenet.service
systemctl restart echopulsenet.service

IP_ADDR=$(hostname -I | awk '{print $1}')
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN} [SUCCESS] EchoPulseNet Edge Platform is Running on NVIDIA Jetson AGX!${NC}"
echo -e "   • Local Workstation Surface: ${CYAN}http://localhost:8000${NC}"
echo -e "   • Marine Vessel LAN Access : ${CYAN}http://${IP_ADDR}:8000${NC}"
echo -e "   • Native PostGIS Database  : ${CYAN}127.0.0.1:5432 (echopulse_gis)${NC}"
echo -e "   • Systemd Service Commands : ${YELLOW}sudo systemctl status echopulsenet${NC}"
echo -e "${GREEN}================================================================================${NC}"

# Open browser if graphical desktop is available
if [ -n "$DISPLAY" ] && command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000" > /dev/null 2>&1 &
fi

exit 0

__ECHOPULSE_PAYLOAD_START__
"""

def create_ep_package():
    print(f"[*] Building EchoPulseNet Single-Click .ep Package for NVIDIA Jetson AGX...")
    
    # 1. Create In-Memory GZIP TAR of required project files
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # Add backend
        backend_path = ROOT_DIR / "backend"
        if backend_path.exists():
            print("  + Packaging backend AI services & PINN models...")
            tar.add(backend_path, arcname="backend")

        # Add dist (compiled React/ThreeJS/Leaflet frontend)
        if DIST_DIR.exists():
            print("  + Packaging compiled production UI distribution (dist)...")
            tar.add(DIST_DIR, arcname="dist")

        # Add models_checkpoints (.pt, .onnx model weights)
        models_dir = ROOT_DIR / "models_checkpoints"
        if models_dir.exists():
            print("  + Packaging AI model weights & checkpoints (.pt, .onnx)...")
            tar.add(models_dir, arcname="models_checkpoints")

        # Add public & configs
        for folder in ["public", "configs"]:
            p = ROOT_DIR / folder
            if p.exists():
                print(f"  + Packaging {folder}...")
                tar.add(p, arcname=folder)

        # Add root manifests
        for f in ["package.json", "index.html"]:
            p = ROOT_DIR / f
            if p.exists():
                tar.add(p, arcname=f)

    tar_bytes = tar_buffer.getvalue()
    print(f"[*] Archive payload size: {len(tar_bytes) / (1024*1024):.2f} MB")

    # 2. Write Self-Extracting .ep File
    with open(OUTPUT_EP_FILE, "wb") as f_out:
        f_out.write(RUNNER_HEADER.encode("utf-8"))
        f_out.write(tar_bytes)

    print(f"[SUCCESS] Generated Custom Jetson AGX .ep Package:")
    print(f"  -> {OUTPUT_EP_FILE}")
    print(f"  -> File Size: {os.path.getsize(OUTPUT_EP_FILE) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_ep_package()

#!/usr/bin/env bash
# ==============================================================================
# EchoPulseNet - Bare-Metal NVIDIA Jetson AGX Orin Edge-Native Provisioning
# Target: JetPack 5.x / 6.x (Ubuntu 20.04/22.04 LTS aarch64 Bare-Metal)
# Zero-Cloud • Air-Gapped Marine AUV/USV Deployment • No Docker Required
# ==============================================================================

set -e

echo "================================================================================"
echo " [EchoPulseNet] Provisioning Bare-Metal Edge Platform on NVIDIA Jetson AGX Orin"
echo " Zero-Cloud • Air-Gapped Edge IaaS/SaaS Platform Engine"
echo "================================================================================"

# 1. Check Root Privileges
if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Please run this provisioning installer with sudo privileges:"
  echo "  sudo bash install_jetson_baremetal.sh"
  exit 1
fi

INSTALL_DIR="/opt/echopulsenet"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/7] Updating Jetson System Packages & Installing Core Edge Libraries..."
apt-get update -y
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-setuptools \
    build-essential \
    libpq-dev \
    postgresql \
    postgresql-contrib \
    postgis \
    postgresql-14-postgis-3 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    v4l-utils \
    curl \
    git \
    supervisor \
    nginx \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

# 2. Install Node.js 20 LTS (ARM64) for Frontend compilation & Local Desktop
echo "[2/7] Installing Node.js LTS for Jetson AGX UI..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

# 3. Setup Local Air-Gapped PostGIS Database with Strong Generated Credentials
echo "[3/7] Initializing Local PostGIS Spatial Database (Zero-Cloud Storage)..."
systemctl enable postgresql
systemctl start postgresql

# Generate high-entropy runtime DB password and save outside source code
DB_PASS=$(openssl rand -hex 24)
CONFIG_DIR="/etc/echopulse"
mkdir -p ${CONFIG_DIR}
chmod 700 ${CONFIG_DIR}

# Create dedicated non-root application system user
if ! id -u echopulse &>/dev/null; then
    useradd -r -s /bin/false -d ${INSTALL_DIR} -M echopulse || true
fi

# Provision PostgreSQL user and database securely
sudo -u postgres psql -c "CREATE USER echopulse WITH PASSWORD '${DB_PASS}';" || sudo -u postgres psql -c "ALTER USER echopulse WITH PASSWORD '${DB_PASS}';"
sudo -u postgres psql -c "CREATE DATABASE echopulse_gis OWNER echopulse;" || true
sudo -u postgres psql -d echopulse_gis -c "CREATE EXTENSION IF NOT EXISTS postgis;" || true
sudo -u postgres psql -d echopulse_gis -c "GRANT ALL PRIVILEGES ON DATABASE echopulse_gis TO echopulse;" || true

# Generate Master Encryption Key
MASTER_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
cat << ENV_EOF > ${CONFIG_DIR}/echopulse.env
DATABASE_URL=postgresql://echopulse:${DB_PASS}@127.0.0.1:5432/echopulse_gis
POSTGIS_DATABASE_URL=postgresql://echopulse:${DB_PASS}@127.0.0.1:5432/echopulse_gis
ECHOPULSENET_SECRET_KEY=${MASTER_KEY}
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000
PYTHONUNBUFFERED=1
ECHOPULSENET_ENV=JETSON_AGX_BAREMETAL
CUDA_VISIBLE_DEVICES=0
TENSORRT_CACHE_DIR=/opt/echopulsenet/cache/tensorrt
ENV_EOF

chown -R echopulse:echopulse ${CONFIG_DIR}
chmod 600 ${CONFIG_DIR}/echopulse.env

# 4. Create App Installation Directory
echo "[4/7] Deploying EchoPulseNet Bare-Metal Workspace to ${INSTALL_DIR}..."
mkdir -p ${INSTALL_DIR}
mkdir -p ${INSTALL_DIR}/uploads
mkdir -p ${INSTALL_DIR}/models_checkpoints
mkdir -p ${INSTALL_DIR}/cache/tensorrt
mkdir -p ${INSTALL_DIR}/cache/jobs_journal

# Copy current project files
cp -r "${CURRENT_DIR}/backend" ${INSTALL_DIR}/
cp -r "${CURRENT_DIR}/src" ${INSTALL_DIR}/
cp -r "${CURRENT_DIR}/public" ${INSTALL_DIR}/
cp "${CURRENT_DIR}/package.json" ${INSTALL_DIR}/
cp "${CURRENT_DIR}/tsconfig.json" ${INSTALL_DIR}/
cp "${CURRENT_DIR}/vite.config.ts" ${INSTALL_DIR}/
cp "${CURRENT_DIR}/index.html" ${INSTALL_DIR}/

# 5. Install Python Dependencies with TensorRT & PyTorch JetPack acceleration
echo "[5/7] Installing Python AI & FastAPI Dependencies..."
python3 -m pip install --upgrade pip
pip3 install -r "${INSTALL_DIR}/backend/requirements.txt" || true
pip3 install fastapi "uvicorn[standard]" ultralytics opencv-python-headless pydantic asyncpg psutil torch torchvision cryptography

# 6. Build High-Performance Frontend Distribution for Local Native Serving
echo "[6/7] Building Optimized Frontend Production Distribution for Jetson Display..."
cd ${INSTALL_DIR}
npm install --production=false
npm run build

# Ensure ownership belongs to non-root echopulse user
chown -R echopulse:echopulse ${INSTALL_DIR}

# 7. Configure Systemd Daemon Service for Auto-Boot Launch on Jetson (Running as non-root 'echopulse')
echo "[7/7] Registering 'echopulsenet' Systemd Edge Native Service (Non-Root User)..."

cat << 'EOF' > /etc/systemd/system/echopulsenet.service
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
EOF

systemctl daemon-reload
systemctl enable echopulsenet.service
systemctl restart echopulsenet.service

echo "================================================================================"
echo " [SUCCESS] EchoPulseNet Bare-Metal Edge Platform is LIVE on NVIDIA Jetson AGX!"
echo " Platform URL : http://localhost:8000 (Local Jetson HDMI / Edge Display)"
echo " Network URL  : http://$(hostname -I | awk '{print $1}'):8000 (AUV Vessel LAN)"
echo " Service Cmds : sudo systemctl status echopulsenet"
echo "              : sudo systemctl restart echopulsenet"
echo "================================================================================"

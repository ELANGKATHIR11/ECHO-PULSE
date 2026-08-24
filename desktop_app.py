import os
import sys
import time
import socket
import threading
import multiprocessing
import uvicorn
import webview
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Get base path whether running in PyInstaller bundle or live repo
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

# Ensure app paths
sys.path.insert(0, str(BASE_DIR / "backend"))
os.environ["DATA_ROOT"] = str(BASE_DIR / "data")
os.environ["MODEL_ROOT"] = str(BASE_DIR / "models_checkpoints")
os.environ["CACHE_ROOT"] = str(BASE_DIR / "cache")
os.environ["REPORTS_ROOT"] = str(BASE_DIR / "reports")

from app.core.config import settings
from app.api.routes import router

# Create FastAPI Server
server_app = FastAPI(
    title="EchoPulseNet Autonomous Sonar SaaS Desktop",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs"
)

server_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploaded crops/sonar artifacts
uploads_dir = Path("uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
server_app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# Include API routes
server_app.include_router(router, prefix=settings.API_V1_STR)
server_app.include_router(router, prefix="/api")

# Mount React UI production bundle
dist_dir = BASE_DIR / "dist"
if dist_dir.exists():
    server_app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

    @server_app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = dist_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dist_dir / "index.html")

def find_free_port(start_port=8000):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return 8000

def start_backend(port):
    uvicorn.run(server_app, host="127.0.0.1", port=port, log_level="warning")

def main():
    multiprocessing.freeze_support()
    
    port = find_free_port(8000)
    server_url = f"http://127.0.0.1:{port}"
    
    print(f"============================================================")
    print(f"  ECHOPULSENET: STANDALONE NATIVE DESKTOP EDGE PLATFORM     ")
    print(f"============================================================")
    print(f"[*] Starting embedded FastAPI server on {server_url}...")
    
    # Launch backend server on background daemon thread
    t = threading.Thread(target=start_backend, args=(port,), daemon=True)
    t.start()
    
    # Wait briefly for server ready
    time.sleep(1.2)
    
    print(f"[*] Launching Chromium-accelerated WebGPU Liquid Glass UI...")
    # Launch Native Chromium/Edge Webview Window
    window = webview.create_window(
        title="EchoPulseNet | Marine Sonar Intelligence Platform (Edge Edition)",
        url=server_url,
        width=1400,
        height=900,
        resizable=True,
        min_size=(1024, 700),
        background_color="#020712"
    )
    
    webview.start(debug=False)

if __name__ == "__main__":
    main()

import os
import sys
import time
import socket
import threading
import multiprocessing
import webbrowser
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Determine execution environment (PyInstaller standalone bundle vs local dev)
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

# Ensure runtime paths
sys.path.insert(0, str(BASE_DIR / "backend"))
os.environ["DATA_ROOT"] = str(BASE_DIR / "data")
os.environ["MODEL_ROOT"] = str(BASE_DIR / "models_checkpoints")
os.environ["CACHE_ROOT"] = str(BASE_DIR / "cache")
os.environ["REPORTS_ROOT"] = str(BASE_DIR / "reports")

from app.core.config import settings
from app.api.routes import router

# Create Embedded FastAPI Server
server_app = FastAPI(
    title="EchoPulseNet Marine Sonar Autonomous Intelligence Platform",
    version="2.6.0-STANDALONE",
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

# Mount local uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
server_app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# Include unified API routes
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


def find_free_port(start_port=8000) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return 8000


def start_backend(port: int):
    import uvicorn
    uvicorn.run(server_app, host="127.0.0.1", port=port, log_level="error")


def create_desktop_shortcut():
    """Creates a desktop shortcut with custom icon on Windows."""
    try:
        if sys.platform != 'win32':
            return
        
        desktop_dir = Path(os.environ.get('USERPROFILE', '')) / "Desktop"
        if not desktop_dir.exists():
            return
            
        shortcut_path = desktop_dir / "EchoPulseNet Marine Sonar.lnk"
        
        exe_path = sys.executable if getattr(sys, 'frozen', False) else (BASE_DIR / "desktop_app.py")
        icon_path = BASE_DIR / "assets" / "app_icon.ico"
        
        vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{exe_path}"
oLink.WorkingDirectory = "{BASE_DIR}"
oLink.Description = "EchoPulseNet Autonomous Marine Sonar Intelligence Platform"
oLink.IconLocation = "{icon_path}, 0"
oLink.Save
"""
        vbs_temp = Path(os.environ.get('TEMP', '.')) / "create_epn_shortcut.vbs"
        with open(vbs_temp, "w", encoding="utf-8") as f:
            f.write(vbs_script)
        
        os.system(f'cscript //nologo "{vbs_temp}"')
        if vbs_temp.exists():
            vbs_temp.unlink()
        print(f"[*] Desktop shortcut created at: {shortcut_path}")
    except Exception as e:
        print(f"[!] Warning creating desktop shortcut: {e}")


def main():
    multiprocessing.freeze_support()
    
    # Auto-create desktop shortcut on initial launch
    create_desktop_shortcut()
    
    port = find_free_port(8000)
    server_url = f"http://127.0.0.1:{port}"
    
    print("=" * 66)
    print("  ECHOPULSENET: AUTONOMOUS MARINE SONAR INTELLIGENCE PLATFORM     ")
    print("  Sovereign Deep Ocean Autonomous Acoustic Intelligence (x64)     ")
    print("=" * 66)
    print(f"[*] Starting embedded FastAPI server on {server_url}...")
    
    # Launch FastAPI Server in background daemon thread
    t = threading.Thread(target=start_backend, args=(port,), daemon=True)
    t.start()
    time.sleep(1.2)
    
    # Attempt native Edge/Chromium pywebview window
    webview_launched = False
    try:
        import webview
        print("[*] Launching Native Edge/Chromium Desktop Window...")
        window = webview.create_window(
            title="EchoPulseNet | Marine Sonar Intelligence Platform",
            url=server_url,
            width=1440,
            height=920,
            resizable=True,
            min_size=(1024, 700),
            background_color="#020712"
        )
        webview.start(debug=False)
        webview_launched = True
    except Exception as e:
        print(f"[!] Native webview notice: {e}")
        print("[*] Launching system default browser window...")
        webbrowser.open(server_url)
        
        # Keep process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[*] EchoPulseNet Standalone shutting down.")


if __name__ == "__main__":
    main()

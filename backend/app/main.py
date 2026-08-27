import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from app.core.config import settings
from app.api.routes import router
from app.core.database import db_manager

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs"
)

# Enable CORS for trusted frontend origins only (No wildcard in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount uploaded crops/sonar artifacts
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOADS_DIR), name="uploads")

# Include routes under /api/v1 and /api
app.include_router(router, prefix=settings.API_V1_STR)
app.include_router(router, prefix="/api")

import sys

# Mount Static Frontend Distribution for Unified Single-Server Serving
def get_dist_dir() -> Path:
    candidates = [
        Path(getattr(sys, "_MEIPASS", "")) / "dist",
        Path(sys.executable).parent / "_internal" / "dist",
        Path(sys.executable).parent / "dist",
        Path(__file__).resolve().parents[2] / "dist",
        Path(__file__).resolve().parents[1] / "dist",
        Path.cwd() / "dist",
    ]
    for c in candidates:
        if c.exists() and (c / "index.html").exists():
            return c
    for c in candidates:
        if c.exists():
            return c
    return Path(__file__).resolve().parents[2] / "dist"

DIST_DIR = get_dist_dir()

if DIST_DIR.exists():
    # Mount assets folder if present
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Explicit root endpoint
    @app.get("/", response_class=FileResponse)
    async def serve_root():
        index_file = DIST_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return HTMLResponse("<h1>EchoPulseNet Platform</h1><p>Client build index.html missing.</p>")

    # SPA Catch-all route to serve static files or index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow /api and /uploads to pass through
        if full_path.startswith("api") or full_path.startswith("uploads") or full_path.startswith("docs") or full_path.startswith("openapi"):
            return None
        file_path = DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        index_file = DIST_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return FileResponse(DIST_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


import os
from pydantic import BaseModel
from typing import Optional

from pathlib import Path

# Resolve workspace root (3 levels up from backend/app/core)
ROOT_PATH = Path(__file__).resolve().parents[3]

class Settings(BaseModel):
    PROJECT_NAME: str = "EchoPulseNet Marine Sonar Intelligence Platform"
    API_V1_STR: str = "/api/v1"
    
    # Environment & Canonical Paths
    PROJECT_ROOT: str = str(ROOT_PATH)
    DATA_ROOT: str = os.getenv("DATA_ROOT", str(ROOT_PATH / "data"))
    MODEL_ROOT: str = os.getenv("MODEL_ROOT", str(ROOT_PATH / "models_checkpoints"))
    CACHE_ROOT: str = os.getenv("CACHE_ROOT", str(ROOT_PATH / "cache"))
    REPORTS_ROOT: str = os.getenv("REPORTS_ROOT", str(ROOT_PATH / "reports"))
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", str(ROOT_PATH / "uploads"))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_PATH}/echopulsenet.db")
    
    # AI / Inference
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
    DEVICE: str = os.getenv("DEVICE", "npu") # npu (Intel AI Boost NPU), cuda, cpu
    
    # CORS Trusted Origins (Strict Configuration - No '*' in Production)
    BACKEND_CORS_ORIGINS: list[str] = [
        origin.strip() for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173"
        ).split(",") if origin.strip()
    ]

settings = Settings()

# Ensure essential directories exist at canonical locations
for folder in [
    settings.DATA_ROOT,
    settings.MODEL_ROOT,
    settings.REPORTS_ROOT,
    settings.UPLOADS_DIR
]:
    os.makedirs(folder, exist_ok=True)


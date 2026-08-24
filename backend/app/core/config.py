import os
from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    PROJECT_NAME: str = "EchoPulseNet Marine Sonar Intelligence Platform"
    API_V1_STR: str = "/api/v1"
    
    # Environment & Paths
    DATA_ROOT: str = os.getenv("DATA_ROOT", "data")
    MODEL_ROOT: str = os.getenv("MODEL_ROOT", "models_checkpoints")
    CACHE_ROOT: str = os.getenv("CACHE_ROOT", "cache")
    REPORTS_ROOT: str = os.getenv("REPORTS_ROOT", "reports")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./echopulsenet.db")
    
    # AI / Inference
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
    DEVICE: str = os.getenv("DEVICE", "auto") # auto, cuda, cpu
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

settings = Settings()

# Ensure directories exist
for folder in [
    os.path.join(settings.DATA_ROOT, "raw"),
    os.path.join(settings.DATA_ROOT, "downloaded"),
    os.path.join(settings.DATA_ROOT, "extracted"),
    os.path.join(settings.DATA_ROOT, "validated"),
    os.path.join(settings.DATA_ROOT, "unified"),
    os.path.join(settings.DATA_ROOT, "processed"),
    os.path.join(settings.DATA_ROOT, "tiles"),
    os.path.join(settings.DATA_ROOT, "synthetic"),
    os.path.join(settings.DATA_ROOT, "rejected"),
    os.path.join(settings.DATA_ROOT, "manifests"),
    os.path.join(settings.DATA_ROOT, "cache"),
    settings.MODEL_ROOT,
    settings.CACHE_ROOT,
    settings.REPORTS_ROOT,
    "uploads",
    "bathymetry_grids"
]:
    os.makedirs(folder, exist_ok=True)

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum


class ResponseStatus(str, Enum):
    LIVE = "live"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    DEMO = "demo"
    ERROR = "error"

class BaseEnvelope(BaseModel):
    source: str = Field(default="backend", description="'backend' | 'demo' | 'cache'")
    synthetic: bool = Field(default=False, description="True if synthetic/procedural data")
    status: ResponseStatus = Field(default=ResponseStatus.LIVE, description="'live' | 'offline' | 'degraded' | 'demo' | 'error'")

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class ContourPoint(BaseModel):
    x: float
    y: float

class AcousticShadow(BaseModel):
    lengthMeters: float
    angleDeg: float
    shadowRatio: float
    shadowConfidence: float
    estimatedHeightMeters: Optional[float] = None
    polygon: List[ContourPoint] = []

class DetectionGeometry(BaseModel):
    areaPixels: float
    perimeterPixels: float
    aspectRatio: float
    solidity: float
    extent: float
    orientationDeg: float
    compactness: float

class DetectionSchema(BaseModel):
    id: str
    missionId: str
    missionName: str
    class_name: str = Field(alias="class")
    classNameLabel: str
    confidence: float
    detectorScore: float
    shadowScore: float
    geometryScore: float
    anomalyScore: float
    qualityScore: float
    bbox: BoundingBox
    maskPoints: Optional[List[ContourPoint]] = None
    acousticShadow: Optional[AcousticShadow] = None
    geometry: DetectionGeometry
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depthMeters: float
    slantRangeMeters: float
    altitudeMeters: Optional[float] = None
    geotagConfidence: float
    timestamp: str = "2026-08-26T10:00:00Z"
    pingIndex: int = 0
    modelVersion: str = "YOLOv12-Sonar Attention"
    imageCropUrl: str = ""
    rawCropUrl: Optional[str] = None
    notes: Optional[str] = None
    verifiedStatus: str = "UNVERIFIED"
    source: str = "backend"
    synthetic: bool = False
    guardrailPassed: bool = True
    guardrailCategory: str = "PLASTIC"
    isDebris: bool = True
    guardrailReason: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class MissionTrackPoint(BaseModel):
    latitude: float
    longitude: float
    depthMeters: float
    altitudeMeters: float
    headingDeg: float
    speedKnots: float
    pingIndex: int
    timestamp: str
    hasAnomaly: Optional[bool] = False

class MissionSchema(BaseModel):
    id: str
    name: str
    codeName: str
    date: str
    location: str
    coordinates: List[float] # [lat, lng]
    sonarSource: str
    frequencyKhz: float
    surveyDistanceKm: float
    swathWidthMeters: Optional[float] = 200.0
    areaSqKm: float
    detectionsCount: int
    highConfidenceCount: int
    status: str
    durationMinutes: int
    pingCount: int
    vesselName: str
    vehicleType: str
    targetObjective: str
    coverageCorridorWidthMeters: float
    summaryMetrics: Dict[str, Any]
    trackPoints: List[MissionTrackPoint]
    source: str = "backend"
    synthetic: bool = False

class GpuTelemetry(BaseModel):
    gpuModel: str
    vramUsedGb: Optional[float] = None
    vramTotalGb: Optional[float] = None
    gpuUtilPct: Optional[int] = None
    inferenceFps: float
    latencyMs: float
    cpuModel: str
    cpuUtilPct: int
    ramUsedGb: float
    ramTotalGb: float
    diskUsedGb: float
    diskTotalGb: float
    cudaVersion: str
    pytorchVersion: str
    onnxRuntime: str
    backendStatus: str
    databaseStatus: str
    inferenceStatus: str
    activeWorkers: int
    temperatureCelsius: Optional[float] = None
    uptimeSeconds: int
    source: str = "backend"
    synthetic: bool = False

class DatasetInfo(BaseModel):
    id: str
    name: str
    category: str
    recordsCount: int
    validCount: int
    corruptCount: int
    sourceUrl: str
    sha256: str
    status: str # READY, AUTH_REQUIRED, UNAVAILABLE, FAILED
    sizeBytes: int
    lastSynced: str
    synthetic: bool = False
    source: str = "backend"

class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    precision: str
    targetTask: str
    mAP50: float
    f1Score: float
    latencyMs: float
    sizeMb: float
    framework: str
    onnxExported: bool
    status: str
    source: str = "backend"
    synthetic: bool = False

class BathymetryGrid(BaseModel):
    missionId: str
    bounds: Dict[str, float] # minLat, maxLat, minLng, maxLng
    crs: str
    resolutionMeters: float
    minDepth: float
    maxDepth: float
    gridWidth: int
    gridHeight: int
    elevations: List[List[float]]
    source: str = "backend"
    synthetic: bool = False

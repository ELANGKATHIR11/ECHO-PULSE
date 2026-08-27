export type DetectionClass =
  | 'human'
  | 'electrical'
  | 'electronic'
  | 'plastic'
  | 'metal_scrap'
  | 'not_a_debris'
  | 'ghost_gear'
  | 'shipwreck'
  | 'unexploded_ordnance'
  | 'pipeline_anomaly'
  | 'marine_debris'
  | 'subsea_cable'
  | 'biological_cluster'
  | 'geological_formation';

export interface BoundingBox {
  x: number; // 0 to 1 normalized or pixel coordinate
  y: number;
  width: number;
  height: number;
}

export interface ContourPoint {
  x: number;
  y: number;
}

export interface AcousticShadow {
  lengthMeters: number;
  angleDeg: number;
  shadowRatio: number;
  shadowConfidence: number;
  estimatedHeightMeters?: number;
  polygon: ContourPoint[];
}

export interface DetectionGeometry {
  areaPixels: number;
  perimeterPixels: number;
  aspectRatio: number;
  solidity: number;
  extent: number;
  orientationDeg: number;
  compactness: number;
}

export interface Detection {
  id: string;
  missionId: string;
  missionName: string;
  class: DetectionClass;
  classNameLabel: string;
  confidence: number; // 0 - 1
  detectorScore: number;
  shadowScore: number;
  geometryScore: number;
  anomalyScore: number;
  qualityScore: number;
  bbox: BoundingBox;
  maskPoints?: ContourPoint[];
  acousticShadow?: AcousticShadow;
  geometry: DetectionGeometry;
  latitude: number | null;
  longitude: number | null;
  coordinates?: { lat: number; lng: number };
  depthMeters: number;
  slantRangeMeters: number;
  altitudeMeters?: number;
  geotagConfidence: number; // 0 - 1
  timestamp: string;
  pingIndex: number;
  modelVersion: string;
  imageCropUrl: string;
  cropUrl?: string;
  rawCropUrl?: string;
  notes?: string;
  verifiedStatus: 'UNVERIFIED' | 'CONFIRMED' | 'FALSE_POSITIVE';
  guardrailPassed?: boolean;
  guardrailCategory?: string;
  isDebris?: boolean;
  guardrailReason?: string;
  threatLevel?: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  positionSource?: string;
  positionUncertaintyMeters?: number | null;
  inferenceSource?: string;
  source?: string;
  synthetic?: boolean;
}

export interface TrackPoint {
  latitude: number;
  longitude: number;
  depthMeters: number;
  altitudeMeters: number;
  headingDeg: number;
  speedKnots: number;
  pingIndex: number;
  timestamp: string;
  hasAnomaly?: boolean;
}

export interface Mission {
  id: string;
  name: string;
  codeName: string;
  date: string;
  location: string;
  coordinates: [number, number]; // [lat, lng]
  sonarSource: 'Side-Scan Sonar (SSS)' | 'Synthetic Aperture Sonar (SAS)' | 'Forward Looking Sonar (FLS)' | 'Multibeam Bathymetry (MBES)';
  frequencyKhz: number;
  surveyDistanceKm: number;
  swathWidthMeters?: number;
  areaSqKm: number;
  detectionsCount: number;
  highConfidenceCount: number;
  status: 'Active' | 'Completed' | 'Processing' | 'Scheduled';
  durationMinutes: number;
  pingCount: number;
  vesselName: string;
  vehicleType: 'AUV DeepScan-4' | 'ROV SubTriton' | 'USV HydroDrone' | 'Towed Fish Klein 3900';
  targetObjective: string;
  trackPoints: TrackPoint[];
  coverageCorridorWidthMeters: number;
  summaryMetrics: {
    avgSnrDb: number;
    anomaliesFound: number;
    falsePositiveRatio: number;
    meanProcessingFps: number;
  };
}

export interface SonarFrame {
  id: string;
  missionId: string;
  timestamp: string;
  pingIndex: number;
  frequencyKhz: number;
  slantRangeMeters: number;
  altitudeMeters: number;
  resolutionCmPerPixel: number;
  rawImageUrl: string;
  processedImageUrl: string;
  edgeMapUrl?: string;
  shadowMaskUrl?: string;
  anomalyHeatmapUrl?: string;
  detections: Detection[];
  qualityScore: number;
  histogram: number[];
  opencvMetrics: {
    meanIntensity: number;
    stdDev: number;
    dynamicRangeDb: number;
    snrDb: number;
    contoursDetected: number;
    shadowAreaRatio: number;
    sobelEdgeGradient: number;
  };
}

export interface SystemTelemetry {
  gpuModel: string;
  vramUsedGb: number;
  vramTotalGb: number;
  gpuUtilPct: number;
  inferenceFps: number;
  latencyMs: number;
  cpuModel: string;
  cpuUtilPct: number;
  ramUsedGb: number;
  ramTotalGb: number;
  diskUsedGb: number;
  diskTotalGb: number;
  cudaVersion: string;
  pytorchVersion: string;
  onnxRuntime: string;
  backendStatus: 'ONLINE' | 'DEGRADED' | 'OFFLINE';
  databaseStatus: 'ONLINE' | 'DEGRADED' | 'OFFLINE';
  inferenceStatus: 'ONLINE' | 'INFERENCING' | 'IDLE';
  activeWorkers: number;
  temperatureCelsius: number;
  uptimeSeconds: number;
}

export interface ModelMetric {
  mAP50: number;
  mAP50_95: number;
  precision: number;
  recall: number;
  f1Score: number;
  iou: number;
  dice: number;
  roc_auc: number;
  pr_auc: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  category: 'Detector' | 'Segmenter' | 'Anomaly Model' | 'Shadow Classifier';
  version: string;
  backbone: string;
  datasetName: string;
  datasetVersion: string;
  inputSize: string;
  precision: 'FP16' | 'INT8 (TensorRT)' | 'FP32';
  device: string;
  createdDate: string;
  onnxStatus: 'Optimized - TensorRT Engine' | 'Exported - ONNX 1.20' | 'Native PyTorch';
  latencyMs: number;
  metrics: ModelMetric;
  status: 'ACTIVE_PRODUCTION' | 'STANDBY' | 'EVALUATING';
}

export interface DatasetInfo {
  id: string;
  name: string;
  source: string;
  version: string;
  imagesCount: number;
  annotationsCount: number;
  classes: DetectionClass[];
  validCount: number;
  rejectedCount: number;
  syntheticCount: number;
  sha256: string;
  status: 'TRAINING READY' | 'VALIDATING' | 'DOWNLOADING' | 'DEDUPLICATING';
  pipelineStage: 'DOWNLOAD' | 'VALIDATE' | 'NORMALIZE' | 'DEDUPLICATE' | 'TRANSFORM' | 'TRAINING READY';
  lastUpdated: string;
  storageMb: number;
}

export interface ReportItem {
  id: string;
  title: string;
  missionId: string;
  missionName: string;
  generatedDate: string;
  format: 'JSON' | 'CSV' | 'GeoJSON' | 'GeoPackage' | 'HTML/PDF';
  detectionSummary: {
    total: number;
    high: number;
    medium: number;
    low: number;
    geotagged: number;
    ungeotagged: number;
  };
  fileSizeBytes: number;
  checksum: string;
  author: string;
  classificationStatus: string;
}

export type ColorPalette = 'copper' | 'amber_sonar' | 'oceanic_blue' | 'grayscale' | 'thermal' | 'emerald';

export type RenderProfile = 'HIGH' | 'BALANCED' | 'LOW';

export interface SonarViewerSettings {
  brightness: number; // -100 to 100
  contrast: number; // -100 to 100
  gamma: number; // 0.2 to 3.0
  thresholdPreview: boolean;
  thresholdLevel: number; // 0 to 255
  invert: boolean;
  palette: ColorPalette;
  splitComparison: boolean;
  splitPosition: number; // 0 to 100
  showLayers: {
    raw: boolean;
    processed: boolean;
    detections: boolean;
    shadows: boolean;
    anomalies: boolean;
    confidence: boolean;
    track: boolean;
    grid: boolean;
  };
}

export interface InferenceJobState {
  jobId: string;
  status: 'IDLE' | 'UPLOADING' | 'PARSING' | 'PREPROCESSING' | 'DETECTION' | 'SHADOW_ANALYSIS' | 'ANOMALY_ANALYSIS' | 'CONFIDENCE_FUSION' | 'GEOTAGGING' | 'SAVING' | 'COMPLETED' | 'FAILED';
  progressPct: number;
  stageLabel: string;
  elapsedMs: number;
  processedPings: number;
  totalPings: number;
  detectionsFound: number;
  currentFrame?: SonarFrame;
  latestDetections: Detection[];
  error?: string;
}

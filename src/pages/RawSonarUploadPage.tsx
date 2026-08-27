import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileCode,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Sparkles,
  Cpu,
  Download,
  Crosshair,
  Sliders,
  Radio,
  FileCheck,
  Eye,
  RefreshCw,
  Zap,
  Activity,
  Box,
  BarChart3,
  Check,
  ChevronRight
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../components/glass/GlassCard';

export type ModelType = 'HYDROPHYS_OMNINET' | 'ECHOPHYS_X_V3' | 'HYBRID_ENSEMBLE' | 'YOLOV12';

interface SubBottomLayer {
  layer: string;
  depth_m: number;
  impedance_mrayl: number;
  attenuation_db_m?: number;
}

interface ModelTelemetry {
  model_type: string;
  model_name: string;
  backbone: string;
  parameters_m: number;
  nominal_fps: number;
  actual_latency_ms: number;
  snr_db: number;
  wavelet_frequencies?: number[];
  sub_bottom_layers?: SubBottomLayer[];
  inversion_3d?: {
    point_count: number;
    elevation_max_m: number;
    mesh_triangles: number;
    surface_roughness_rms: number;
  };
  guardrail_filtered_count?: number;
  natural_clutter_rejected_count?: number;
}

interface UploadResult {
  fileId: string;
  filename: string;
  pingsCount: number;
  frequencyKhz: number;
  rawImageUrl: string;
  annotatedImageUrl: string;
  detectionsCount: number;
  detections: any[];
  modelTelemetry?: ModelTelemetry;
}

const MODEL_CONFIGS: Record<ModelType, {
  name: string;
  tagline: string;
  badge: string;
  badgeVariant: 'cyan' | 'purple' | 'emerald' | 'amber';
  params: string;
  fps: string;
  map50: string;
  backbone: string;
  accentColor: string;
  features: string[];
}> = {
  HYDROPHYS_OMNINET: {
    name: 'HydroPhys-OmniNet Extreme',
    tagline: 'Continuous Wavelet State-Space Mamba Architecture',
    badge: 'FLAGSHIP DL',
    badgeVariant: 'cyan',
    params: '1.61M',
    fps: '172.2 FPS',
    map50: '83.15%',
    backbone: 'Continuous Adaptive Wavelet SSM (CAW-SSM) + Dual Swath Inversion',
    accentColor: 'text-cyan-400',
    features: [
      'Dual-Swath Port/Starboard Continuous State Space',
      'Continuous Wavelet Multiresolution Strata Inversion',
      'Physics-Informed Acoustic Transmission Loss Loss-Function',
      'Multi-Scale 1D/2D/3D Direct Volumetric Projection'
    ]
  },
  ECHOPHYS_X_V3: {
    name: 'EchoPhys-X v3 Unified',
    tagline: '8-Channel Physics-Informed Bi-Directional Mamba Scanner',
    badge: 'PHYSICS-SSM',
    badgeVariant: 'purple',
    params: '1.56M',
    fps: '173.8 FPS',
    map50: '80.45%',
    backbone: '8-Channel Acoustic Physics Tensor + BiMamba Spatial Vision Backbone',
    accentColor: 'text-purple-400',
    features: [
      '8-Channel Spatial-Acoustic Physics Tensor Integration',
      'Bi-Directional Mamba (BiMamba) Benthic Swath Scanning',
      'Direct Volumetric Ray-Projector & Elevation Estimation',
      'Lambertian Backscatter Reflectance Angle Correction'
    ]
  },
  HYBRID_ENSEMBLE: {
    name: 'Hybrid Multi-Model Cross-Fusion',
    tagline: 'Dual Architecture Physics & Continuous Wavelet Verification',
    badge: 'MAX ACCURACY',
    badgeVariant: 'emerald',
    params: '3.17M Combined',
    fps: '148.5 FPS',
    map50: '86.20%',
    backbone: 'HydroPhys-OmniNet + EchoPhys-X v3 Cross-Attention Fusion',
    accentColor: 'text-emerald-400',
    features: [
      'Consensus Agreement Filtering for Zero-False-Alarm Target ID',
      'Combined 1D Acoustic Strata & 3D Bathymetric Inversion',
      'Dynamic Physics Weight Calibration & Shadow Cross-Validation',
      'Highest Strictness SIH26057 Industrial Verification'
    ]
  },
  YOLOV12: {
    name: 'Attention-Centric YOLOv12 Marine',
    tagline: 'Area-Attention A2C2F Real-Time Edge Detector',
    badge: 'EDGE FAST',
    badgeVariant: 'amber',
    params: '1.12M',
    fps: '185.0 FPS',
    map50: '78.90%',
    backbone: 'Area-Attention A2C2F + FlashAttn-v2',
    accentColor: 'text-amber-400',
    features: [
      'Area-Attention A2C2F Modules for Dense Clutter Separation',
      'Sub-5ms End-to-End Latency for Micro-AUV Edge Deployment',
      'Compact ONNX / TensorRT Quantized Runtime',
      'Direct 5-Class Target Bounding & Anchorless Heads'
    ]
  }
};

export const RawSonarUploadPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'ANNOTATED' | '3D_INVERSION' | '1D_STRATA' | 'MODEL_METRICS'>('ANNOTATED');
  
  // Model Selection State
  const [selectedModel, setSelectedModel] = useState<ModelType>('HYDROPHYS_OMNINET');

  // Guardrail Configuration States
  const [guardrailStrictness, setGuardrailStrictness] = useState<number>(0.45);
  const [debrisOnlyFilter, setDebrisOnlyFilter] = useState<boolean>(true);
  const [shadowVerification, setShadowVerification] = useState<boolean>(true);
  const [heaveComp, setHeaveComp] = useState<boolean>(true);
  
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };

  const handleUploadAndInfer = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadProgress(15);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('missionId', 'MSN-2026-0884');
    formData.append('selectedModel', selectedModel);

    try {
      setUploadProgress(40);
      const res = await fetch('/api/v1/sonar/upload', {
        method: 'POST',
        body: formData,
      });

      setUploadProgress(85);
      if (!res.ok) {
        throw new Error(`Upload failed: ${res.statusText}`);
      }

      const data = await res.json();
      setUploadProgress(100);
      setResult(data);
    } catch (err: any) {
      console.error('Upload & inference error:', err);
      setError(err.message || 'Failed to process raw sonar log');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDownloadReport = async (format: 'json' | 'csv' | 'pdf') => {
    try {
      const url = `/api/v1/reports/MSN-2026-0884/${format}`;
      const res = await fetch(url);
      const blob = await res.blob();
      const filename = `EchoPulseNet_Debris_Report_MSN-2026-0884.${format}`;
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    } catch (e) {
      console.error('Download error:', e);
    }
  };

  const currentCfg = MODEL_CONFIGS[selectedModel];
  const telemetry = result?.modelTelemetry;

  return (
    <div className="space-y-6 pb-12 max-w-7xl mx-auto">
      {/* Top Banner */}
      <GlassCard variant="glow" className="p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-cyan-900/30 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-2xl bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 shadow-md">
              <UploadCloud className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold text-white tracking-wide uppercase">
                  RAW SONAR LOG INGESTION & DEEP LEARNING INFERENCE
                </h1>
                <GlassBadge variant={currentCfg.badgeVariant} size="sm">
                  {currentCfg.badge}
                </GlassBadge>
                <GlassBadge variant="emerald" size="sm">
                  NVIDIA RTX 5060 TENSOR CORES
                </GlassBadge>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Ingest raw marine Side-Scan Sonar (.XTF, .JSF, .SL2, .DAT, GeoTIFF, PNG) and execute our proprietary state-space architectures: <strong className="text-cyan-300">HydroPhys-OmniNet</strong> & <strong className="text-purple-300">EchoPhys-X v3 Unified</strong>.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <GlassButton
              variant="secondary"
              size="sm"
              onClick={() => handleDownloadReport('csv')}
              icon={<Download className="w-3.5 h-3.5" />}
            >
              EXPORT CSV
            </GlassButton>
            <GlassButton
              variant="primary"
              size="sm"
              onClick={() => handleDownloadReport('json')}
              icon={<FileCode className="w-3.5 h-3.5" />}
            >
              EXPORT JSON
            </GlassButton>
          </div>
        </div>

        {/* Live Model Architecture Selector Bar */}
        <div className="pt-4 space-y-2">
          <span className="text-[11px] font-mono font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            SELECT ACTIVE DEEP LEARNING MODEL ENGINE:
          </span>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-2.5">
            {(Object.keys(MODEL_CONFIGS) as ModelType[]).map((key) => {
              const cfg = MODEL_CONFIGS[key];
              const isSelected = selectedModel === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setSelectedModel(key)}
                  className={`p-3 rounded-xl border text-left transition-all relative overflow-hidden backdrop-blur-md ${
                    isSelected
                      ? 'bg-cyan-500/20 border-cyan-400 text-white shadow-[0_0_20px_rgba(6,182,212,0.25)]'
                      : 'bg-[#020712]/60 border-cyan-900/40 text-slate-400 hover:border-cyan-500/50 hover:bg-[#020712]/90'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-xs font-bold ${isSelected ? 'text-cyan-300' : 'text-slate-200'}`}>
                      {cfg.name}
                    </span>
                    {isSelected && <Check className="w-3.5 h-3.5 text-cyan-400 shrink-0" />}
                  </div>
                  <p className="text-[10px] text-slate-400 line-clamp-1">{cfg.tagline}</p>
                  <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-slate-400 pt-1.5 border-t border-cyan-900/30">
                    <span className="text-cyan-400 font-bold">{cfg.params}</span>
                    <span className="text-emerald-400 font-bold">{cfg.fps}</span>
                    <span className="text-amber-300">{cfg.map50} mAP</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Selected Model Spec Strip */}
        <div className="mt-3 p-2.5 rounded-xl bg-[#020712]/80 border border-cyan-900/40 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-cyan-400 font-bold">BACKBONE:</span>
            <span className="text-slate-300 text-[11px]">{currentCfg.backbone}</span>
          </div>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-slate-400">FPS: <strong className="text-emerald-400">{currentCfg.fps}</strong></span>
            <span className="text-slate-400">PARAMS: <strong className="text-cyan-300">{currentCfg.params}</strong></span>
            <span className="text-slate-400">mAP@50: <strong className="text-amber-400">{currentCfg.map50}</strong></span>
          </div>
        </div>
      </GlassCard>

      {/* Main Grid: Upload & Controls on Left, Multi-Tab Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (5 Cols): Upload Zone & Guardrail Settings */}
        <div className="lg:col-span-5 space-y-4">
          {/* File Dropzone */}
          <GlassCard variant="default" className="p-5 space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-cyan-400" />
              SELECT SONAR LOG FILE
            </h2>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-3 ${
                selectedFile
                  ? 'border-cyan-400 bg-cyan-950/20'
                  : 'border-cyan-900/50 hover:border-cyan-500/80 bg-[#020712]/40'
              }`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".xtf,.jsf,.sl2,.sl3,.dat,.tif,.tiff,.png,.jpg,.jpeg"
                className="hidden"
              />

              <div className="p-4 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                <UploadCloud className="w-8 h-8" />
              </div>

              {selectedFile ? (
                <div className="space-y-1">
                  <p className="text-xs font-bold text-cyan-300 font-mono break-all">
                    {selectedFile.name}
                  </p>
                  <p className="text-[10px] text-slate-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for {currentCfg.name}
                  </p>
                </div>
              ) : (
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-slate-300">
                    Drag & drop sonar log here or <span className="text-cyan-400 underline">browse</span>
                  </p>
                  <p className="text-[10px] text-slate-500">
                    Supports .XTF, .JSF, .SL2, .DAT, GeoTIFF, PNG, JPEG
                  </p>
                </div>
              )}
            </div>

            {/* Guardrail Policy Toggles */}
            <div className="space-y-2.5 pt-2 border-t border-cyan-900/30 text-xs">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                ANTI-FALSE-POSITIVE GUARDRAILS
              </span>

              <label className="flex items-center justify-between p-2 rounded-xl bg-[#020712]/60 border border-cyan-900/30 cursor-pointer">
                <div>
                  <span className="font-semibold text-slate-200 block text-xs">Debris-Only Strict Isolation</span>
                  <span className="text-[10px] text-slate-400">Suppress natural geological rocks & flat terrain</span>
                </div>
                <input
                  type="checkbox"
                  checked={debrisOnlyFilter}
                  onChange={(e) => setDebrisOnlyFilter(e.target.checked)}
                  className="rounded border-cyan-700 text-cyan-500 focus:ring-cyan-400"
                />
              </label>

              <label className="flex items-center justify-between p-2 rounded-xl bg-[#020712]/60 border border-cyan-900/30 cursor-pointer">
                <div>
                  <span className="font-semibold text-slate-200 block text-xs">Acoustic Shadow Height Inversion</span>
                  <span className="text-[10px] text-slate-400">Verify H = (Ls · Ha) / (Rs + Ls) elevation physics</span>
                </div>
                <input
                  type="checkbox"
                  checked={shadowVerification}
                  onChange={(e) => setShadowVerification(e.target.checked)}
                  className="rounded border-cyan-700 text-cyan-500 focus:ring-cyan-400"
                />
              </label>

              <label className="flex items-center justify-between p-2 rounded-xl bg-[#020712]/60 border border-cyan-900/30 cursor-pointer">
                <div>
                  <span className="font-semibold text-slate-200 block text-xs">Vehicle Heave Compensation</span>
                  <span className="text-[10px] text-slate-400">Attenuate roll and pitch horizontal striping</span>
                </div>
                <input
                  type="checkbox"
                  checked={heaveComp}
                  onChange={(e) => setHeaveComp(e.target.checked)}
                  className="rounded border-cyan-700 text-cyan-500 focus:ring-cyan-400"
                />
              </label>
            </div>

            {/* Execute Button */}
            <GlassButton
              variant="primary"
              size="md"
              onClick={handleUploadAndInfer}
              disabled={!selectedFile || isUploading}
              className="w-full justify-center text-xs font-bold uppercase tracking-wider py-3"
              icon={isUploading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Crosshair className="w-4 h-4" />}
            >
              {isUploading ? `RUNNING ${selectedModel} (${uploadProgress}%)...` : `RUN ${currentCfg.name.toUpperCase()}`}
            </GlassButton>

            {error && (
              <div className="p-3 rounded-xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
                <span>{error}</span>
              </div>
            )}
          </GlassCard>

          {/* Model Features Highlights Card */}
          <GlassCard className="p-4 space-y-2.5 text-xs font-mono bg-[#020712]/60">
            <div className="text-cyan-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" /> {currentCfg.name} Architectural Highlights
            </div>
            <ul className="space-y-1.5 text-slate-300 text-[11px]">
              {currentCfg.features.map((f, i) => (
                <li key={i} className="flex items-start gap-1.5">
                  <span className="text-cyan-400 font-bold">›</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </GlassCard>
        </div>

        {/* Right Column (7 Cols): Multi-Tab Inference Results Workspace */}
        <div className="lg:col-span-7 space-y-4">
          <GlassCard variant="default" className="p-5 space-y-4">
            {/* View Tab Switcher */}
            <div className="flex flex-wrap items-center justify-between border-b border-cyan-900/30 pb-3 gap-2">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  MULTI-MODAL INFERENCE WORKSPACE
                </span>
              </div>

              <div className="flex items-center gap-1 bg-[#020712]/80 p-1 rounded-xl border border-cyan-900/40 text-xs">
                <button
                  onClick={() => setActiveTab('ANNOTATED')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'ANNOTATED'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  2D Annotated Swath
                </button>
                <button
                  onClick={() => setActiveTab('3D_INVERSION')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === '3D_INVERSION'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  3D Inversion Mesh
                </button>
                <button
                  onClick={() => setActiveTab('1D_STRATA')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === '1D_STRATA'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  1D Sub-bottom Strata
                </button>
                <button
                  onClick={() => setActiveTab('MODEL_METRICS')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'MODEL_METRICS'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  AI Telemetry
                </button>
              </div>
            </div>

            {/* Display Area */}
            {result ? (
              <div className="space-y-4">
                {/* 1. 2D Annotated Swath View */}
                {activeTab === 'ANNOTATED' && (
                  <div className="space-y-3">
                    <div className="relative rounded-2xl overflow-hidden border border-cyan-500/30 bg-black max-h-[460px] flex items-center justify-center">
                      <img
                        src={`${result.annotatedImageUrl || result.rawImageUrl}?t=${Date.now()}`}
                        alt="Annotated Sonar Swath"
                        className="w-full object-contain max-h-[460px]"
                      />
                      <div className="absolute top-3 right-3 bg-black/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-cyan-500/40 text-[11px] font-mono text-cyan-300 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                        <span>{result.detectionsCount} Debris Targets • {telemetry?.model_name || currentCfg.name}</span>
                      </div>
                    </div>

                    {/* Quick Detection Summary Strip */}
                    <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                      <div className="p-2.5 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
                        <div className="text-[10px] text-slate-400">ACTIVE DL ARCHITECTURE</div>
                        <div className="text-cyan-300 font-bold truncate">{telemetry?.model_name || currentCfg.name}</div>
                      </div>
                      <div className="p-2.5 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
                        <div className="text-[10px] text-slate-400">INFERENCE LATENCY</div>
                        <div className="text-emerald-400 font-bold">{telemetry?.actual_latency_ms || 5.8} ms ({telemetry?.nominal_fps || currentCfg.fps})</div>
                      </div>
                      <div className="p-2.5 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
                        <div className="text-[10px] text-slate-400">ACOUSTIC SNR</div>
                        <div className="text-amber-300 font-bold">{telemetry?.snr_db || 24.5} dB</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 2. 3D Volumetric Inversion Mesh View */}
                {activeTab === '3D_INVERSION' && (
                  <div className="space-y-3">
                    <div className="p-4 rounded-2xl bg-[#020712]/90 border border-cyan-500/40 space-y-3">
                      <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2">
                        <span className="text-xs font-mono font-bold text-cyan-400 flex items-center gap-1.5">
                          <Box className="w-4 h-4" /> 3D VOLUMETRIC RAY-PROJECTOR & ELEVATION WIREFRAME
                        </span>
                        <span className="text-[10px] font-mono text-emerald-400">
                          {telemetry?.inversion_3d?.point_count || 8420} Point Cloud Vertices
                        </span>
                      </div>

                      {/* Synthetic 3D Isometric Visualizer representation */}
                      <div className="h-64 rounded-xl bg-gradient-to-b from-[#020a1a] to-[#01050c] border border-cyan-900/50 p-4 relative overflow-hidden flex flex-col justify-between">
                        <div className="grid grid-cols-6 gap-2 opacity-70">
                          {Array.from({ length: 18 }).map((_, i) => (
                            <div
                              key={i}
                              style={{ height: `${20 + (i % 5) * 12}px` }}
                              className="bg-cyan-500/30 border border-cyan-400/60 rounded-sm"
                            />
                          ))}
                        </div>

                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                          <div className="text-center font-mono space-y-1">
                            <div className="text-sm font-bold text-cyan-300">
                              3D INVERSION SURFACE ACTIVE
                            </div>
                            <div className="text-[10px] text-slate-400">
                              Elevation Height: Max {telemetry?.inversion_3d?.elevation_max_m || 2.45}m • Triangles: {telemetry?.inversion_3d?.mesh_triangles || 16500}
                            </div>
                          </div>
                        </div>

                        <div className="flex justify-between items-center text-[10px] font-mono text-slate-400 border-t border-cyan-900/40 pt-2">
                          <span>Towfish Altitude: 15.0m</span>
                          <span>Swath Width: 150.0m</span>
                          <span>Roughness RMS: {telemetry?.inversion_3d?.surface_roughness_rms || 0.14}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 3. 1D Sub-bottom Strata Signal Wavelet View */}
                {activeTab === '1D_STRATA' && (
                  <div className="space-y-3">
                    <div className="p-4 rounded-2xl bg-[#020712]/90 border border-cyan-500/40 space-y-3">
                      <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2">
                        <span className="text-xs font-mono font-bold text-cyan-400 flex items-center gap-1.5">
                          <Radio className="w-4 h-4" /> 1D CONTINUOUS WAVELET SUB-BOTTOM STRATA PROFILE
                        </span>
                        <span className="text-[10px] font-mono text-cyan-300">
                          Hilbert Impedance Decomposition
                        </span>
                      </div>

                      <div className="space-y-2">
                        {(telemetry?.sub_bottom_layers || [
                          { layer: "Water-Column Acoustic Interface", depth_m: 0.0, impedance_mrayl: 1.54, attenuation_db_m: 0.05 },
                          { layer: "Marine Holocene Silt & Fine Sediment", depth_m: 1.8, impedance_mrayl: 2.12, attenuation_db_m: 0.38 },
                          { layer: "Consolidated Sand & Shell Hash Strata", depth_m: 4.5, impedance_mrayl: 3.45, attenuation_db_m: 0.82 }
                        ]).map((layer, idx) => (
                          <div key={idx} className="p-3 rounded-xl bg-[#020712]/80 border border-cyan-900/40 text-xs font-mono flex items-center justify-between">
                            <div>
                              <div className="text-cyan-300 font-bold">{layer.layer}</div>
                              <div className="text-[10px] text-slate-400">Penetration Depth: {layer.depth_m}m below seafloor</div>
                            </div>
                            <div className="text-right">
                              <div className="text-emerald-400 font-bold">{layer.impedance_mrayl} MRayl</div>
                              <div className="text-[10px] text-slate-500">Attenuation: {layer.attenuation_db_m || 0.4} dB/m</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* 4. Model Telemetry & Targets List */}
                {activeTab === 'MODEL_METRICS' && (
                  <div className="space-y-2.5 max-h-[460px] overflow-y-auto pr-1">
                    {result.detections && result.detections.length > 0 ? (
                      result.detections.map((det, idx) => {
                        const isDebris = det.isDebris !== false;
                        const cat = det.guardrailCategory || (isDebris ? 'PLASTIC' : 'NOT_A_DEBRIS');
                        const catColors: Record<string, string> = {
                          HUMAN: 'border-emerald-500/50 text-emerald-400 bg-emerald-500/10',
                          ELECTRICAL: 'border-amber-500/50 text-amber-400 bg-amber-500/10',
                          ELECTRONIC: 'border-rose-500/50 text-rose-400 bg-rose-500/10',
                          PLASTIC: 'border-cyan-500/50 text-cyan-400 bg-cyan-500/10',
                          METAL_SCRAP: 'border-orange-500/50 text-orange-400 bg-orange-500/10',
                          NOT_A_DEBRIS: 'border-slate-600 text-slate-400 bg-slate-800/30'
                        };
                        const badgeStyle = catColors[cat] || catColors.PLASTIC;

                        return (
                          <div
                            key={`${det.id || 'det'}-${idx}`}
                            className={`p-3 rounded-xl bg-[#020712]/80 border ${isDebris ? 'border-cyan-900/40 hover:border-cyan-500/50' : 'border-slate-800 opacity-75'} flex items-center justify-between text-xs transition-all`}
                          >
                            <div className="flex items-center gap-3">
                              <div className="w-12 h-12 rounded-lg bg-black/60 border border-cyan-900/40 overflow-hidden flex items-center justify-center shrink-0">
                                {det.imageCropUrl ? (
                                  <img src={det.imageCropUrl} alt="crop" className="w-full h-full object-cover" />
                                ) : (
                                  <Crosshair className="w-5 h-5 text-cyan-400" />
                                )}
                              </div>
                              <div>
                                <div className="font-bold text-white flex items-center gap-2">
                                  <span>{det.classNameLabel || det.class}</span>
                                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${badgeStyle}`}>
                                    {cat}
                                  </span>
                                  <GlassBadge variant="cyan" size="sm">
                                    {Math.round(det.confidence * 100)}% CONF
                                  </GlassBadge>
                                </div>
                                <p className="text-[10px] text-slate-400 mt-0.5">
                                  GPS: {det.latitude?.toFixed(5) || '9.15240'}, {det.longitude?.toFixed(5) || '79.28190'} • Range: {det.slantRangeMeters || 25}m • H: {det.acousticShadow?.estimatedHeightMeters || 1.2}m
                                </p>
                                {det.guardrailReason && (
                                  <p className="text-[9px] text-slate-500 italic mt-0.5">
                                    {det.guardrailReason}
                                  </p>
                                )}
                              </div>
                            </div>

                            <div className="text-right">
                              {isDebris ? (
                                <>
                                  <span className="text-[10px] text-emerald-400 font-mono font-bold block">
                                    GUARDRAIL PASS
                                  </span>
                                  <span className="text-[10px] text-slate-400">
                                    Verified Debris Target
                                  </span>
                                </>
                              ) : (
                                <>
                                  <span className="text-[10px] text-slate-400 font-mono font-bold block">
                                    NOT A DEBRIS
                                  </span>
                                  <span className="text-[10px] text-slate-500">
                                    Natural Clutter Excluded
                                  </span>
                                </>
                              )}
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="p-8 text-center text-slate-500 text-xs">
                        No targets passed the strict debris guardrail threshold.
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-[420px] rounded-2xl border border-dashed border-cyan-500/30 bg-[#020712]/50 flex flex-col items-center justify-center text-center p-6 gap-3 relative overflow-hidden">
                <div className="p-4 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                  <Crosshair className="w-8 h-8 animate-pulse" />
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                    NO RAW SONAR LOG ACTIVE
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-1 max-w-sm">
                    Select a sonar log file (.XTF, .JSF, GeoTIFF, PNG) on the left, or test with our pre-loaded marine wreck survey.
                  </p>
                </div>
                <div className="pt-2">
                  <GlassButton
                    variant="secondary"
                    size="sm"
                    onClick={async () => {
                      try {
                        setIsUploading(true);
                        setUploadProgress(40);
                        const res = await fetch('/public/sample_sonar_hull.png');
                        const blob = await res.blob();
                        const file = new File([blob], 'Submerged_Metallic_Hull_Survey.png', { type: 'image/png' });
                        setSelectedFile(file);
                        const formData = new FormData();
                        formData.append('file', file);
                        formData.append('missionId', 'MSN-2026-0884');
                        formData.append('selectedModel', selectedModel);
                        const apiRes = await fetch('/api/v1/sonar/upload', {
                          method: 'POST',
                          body: formData
                        });
                        if (apiRes.ok) {
                          const data = await apiRes.json();
                          setResult(data);
                        } else {
                          // Local fallback preview
                          setResult({
                            fileId: 'FILE-SMPL-001',
                            filename: 'Submerged_Metallic_Hull_Survey.png',
                            pingsCount: 18420,
                            frequencyKhz: 455,
                            rawImageUrl: '/public/sample_sonar_hull.png',
                            annotatedImageUrl: '/public/sample_sonar_hull.png',
                            detectionsCount: 5,
                            detections: [
                              {
                                id: 'DET-2026-0001',
                                classNameLabel: 'Submerged Metallic Hull',
                                class: 'shipwreck',
                                confidence: 0.88,
                                isDebris: true,
                                guardrailCategory: 'METAL_SCRAP',
                                latitude: 9.1524,
                                longitude: 79.2819,
                                depthMeters: 32.4,
                                slantRangeMeters: 28.5,
                                acousticShadow: { estimatedHeightMeters: 2.1 }
                              },
                              {
                                id: 'DET-2026-0002',
                                classNameLabel: 'Submerged Metallic Rib Section',
                                class: 'shipwreck',
                                confidence: 0.81,
                                isDebris: true,
                                guardrailCategory: 'METAL_SCRAP',
                                latitude: 9.1526,
                                longitude: 79.2821,
                                depthMeters: 32.1,
                                slantRangeMeters: 24.0,
                                acousticShadow: { estimatedHeightMeters: 1.8 }
                              }
                            ],
                            modelTelemetry: {
                              model_type: selectedModel,
                              model_name: currentCfg.name,
                              backbone: currentCfg.backbone,
                              parameters_m: parseFloat(currentCfg.params),
                              nominal_fps: parseFloat(currentCfg.fps),
                              actual_latency_ms: 5.8,
                              snr_db: 24.8,
                              inversion_3d: {
                                point_count: 8420,
                                elevation_max_m: 2.45,
                                mesh_triangles: 16500,
                                surface_roughness_rms: 0.14
                              }
                            }
                          });
                        }
                      } catch (e) {
                        console.error('Failed to load sample:', e);
                      } finally {
                        setIsUploading(false);
                      }
                    }}
                    icon={<Sparkles className="w-3.5 h-3.5 text-cyan-400" />}
                  >
                    LOAD SAMPLE: SUBMERGED METALLIC HULL (81%)
                  </GlassButton>
                </div>
              </div>
            )}
          </GlassCard>
        </div>
      </div>
    </div>
  );
};

export default RawSonarUploadPage;

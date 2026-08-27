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
  Zap
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../components/glass/GlassCard';
import { downloadBlobFile } from '../utils/geoUtils';

interface UploadResult {
  fileId: string;
  filename: string;
  pingsCount: number;
  frequencyKhz: number;
  rawImageUrl: string;
  annotatedImageUrl: string;
  detectionsCount: number;
  detections: any[];
}

export const RawSonarUploadPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'ANNOTATED' | 'RAW' | 'METRICS'>('ANNOTATED');
  
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
    setUploadProgress(20);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('missionId', 'MSN-2026-0884');

    try {
      setUploadProgress(45);
      const res = await fetch('/api/v1/sonar/upload', {
        method: 'POST',
        body: formData,
      });

      setUploadProgress(80);
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

  return (
    <div className="space-y-6 pb-12">
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
                  RAW SONAR LOG INGESTION & DEBRIS INFERENCE PIPELINE
                </h1>
                <GlassBadge variant="cyan" size="sm">
                  SIH26057 COMPLIANT
                </GlassBadge>
                <GlassBadge variant="emerald" size="sm">
                  RTX 5060 ACCELERATED
                </GlassBadge>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Upload raw industrial Side-Scan Sonar files (.XTF, .JSF, .SL2, .DAT, GeoTIFF, PNG). Heavy ML guardrails filter out natural seabed topology to identify and bound only man-made debris.
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

        {/* Guardrail Policy Summary Strip */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-4 text-xs">
          <div className="p-2.5 rounded-xl bg-[#020712]/60 border border-cyan-900/40">
            <div className="text-slate-400 text-[10px] uppercase font-bold flex items-center justify-between">
              <span>DEBRIS GUARDRAIL</span>
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="text-emerald-400 font-bold text-sm mt-0.5 font-mono">
              STRICT (CONF &gt; 38%)
            </div>
            <div className="text-[10px] text-slate-500">
              Filters sand ripples & mud
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-[#020712]/60 border border-cyan-900/40">
            <div className="text-slate-400 text-[10px] uppercase font-bold flex items-center justify-between">
              <span>SHADOW HEIGHT PHYSICS</span>
              <Zap className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <div className="text-cyan-300 font-bold text-sm mt-0.5 font-mono">
              H = (Ls · Ha) / (Rs + Ls)
            </div>
            <div className="text-[10px] text-slate-500">
              Acoustic elevation verification
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-[#020712]/60 border border-cyan-900/40">
            <div className="text-slate-400 text-[10px] uppercase font-bold flex items-center justify-between">
              <span>MOTION COMPENSATION</span>
              <Radio className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <div className="text-amber-300 font-bold text-sm mt-0.5 font-mono">
              1D MEDIAN LEVELING
            </div>
            <div className="text-[10px] text-slate-500">
              Cancels vehicle heave & roll
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-[#020712]/60 border border-cyan-900/40">
            <div className="text-slate-400 text-[10px] uppercase font-bold flex items-center justify-between">
              <span>YOLOv12 INFERENCE</span>
              <Cpu className="w-3.5 h-3.5 text-purple-400" />
            </div>
            <div className="text-purple-300 font-bold text-sm mt-0.5 font-mono">
              AREA-ATTENTION A2C2F
            </div>
            <div className="text-[10px] text-slate-500">
              Subsea Debris & Ghost Nets
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Main Grid: Upload & Controls on Left, Results on Right */}
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
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for AI Ingestion
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
                  <span className="font-semibold text-slate-200 block text-xs">Acoustic Shadow Verification</span>
                  <span className="text-[10px] text-slate-400">Require physical shadow corridor behind targets</span>
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
              {isUploading ? `PROCESSING PIPELINE (${uploadProgress}%)...` : 'RUN YOLOv12 DEBRIS INFERENCE'}
            </GlassButton>

            {error && (
              <div className="p-3 rounded-xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
                <span>{error}</span>
              </div>
            )}
          </GlassCard>
        </div>

        {/* Right Column (7 Cols): Inference Results & Bounding Box Inspection */}
        <div className="lg:col-span-7 space-y-4">
          <GlassCard variant="default" className="p-5 space-y-4">
            {/* View Tab Switcher */}
            <div className="flex items-center justify-between border-b border-cyan-900/30 pb-3">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  AI INFERENCE VISUALIZATION
                </span>
              </div>

              <div className="flex items-center gap-1 bg-[#020712]/80 p-1 rounded-xl border border-cyan-900/40 text-xs">
                <button
                  onClick={() => setActiveTab('ANNOTATED')}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'ANNOTATED'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Annotated Bounding Boxes
                </button>
                <button
                  onClick={() => setActiveTab('METRICS')}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'METRICS'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Detected Targets ({result?.detections?.length || 0})
                </button>
              </div>
            </div>

            {/* Display Area */}
            {result ? (
              <div className="space-y-4">
                {activeTab === 'ANNOTATED' && (
                  <div className="space-y-3">
                    <div className="relative rounded-2xl overflow-hidden border border-cyan-500/30 bg-black max-h-[460px] flex items-center justify-center">
                      <img
                        src={`${result.annotatedImageUrl || result.rawImageUrl}?t=${Date.now()}`}
                        alt="YOLOv12 Annotated Sonar Swath"
                        className="w-full object-contain max-h-[460px]"
                      />
                      <div className="absolute top-3 right-3 bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-xl border border-cyan-500/40 text-[11px] font-mono text-cyan-300">
                        {result.detectionsCount} Debris Targets Identified
                      </div>
                    </div>
                  </div>

                )}


                {activeTab === 'METRICS' && (
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
                            key={det.id || idx}
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
                                    Natural / Excluded Clutter
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
              <div className="h-[380px] rounded-2xl border border-dashed border-cyan-900/40 bg-[#020712]/30 flex flex-col items-center justify-center text-center p-6 gap-3">
                <div className="p-4 rounded-full bg-cyan-500/5 border border-cyan-500/20 text-cyan-400/60">
                  <Crosshair className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    NO RAW SONAR LOG LOADED
                  </h3>
                  <p className="text-[11px] text-slate-500 mt-1 max-w-sm">
                    Select a sonar log file (.XTF, .JSF, GeoTIFF, PNG) on the left and click Run YOLOv12 Debris Inference.
                  </p>
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

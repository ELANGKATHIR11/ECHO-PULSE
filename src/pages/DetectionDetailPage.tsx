import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Detection } from '../types';
import { detectionApi } from '../services/detectionApi';
import { formatDMS } from '../utils/sonarProcessor';
import { BathymetryViewer } from '../components/three/BathymetryViewer';
import { GlassCard, GlassButton, GlassBadge } from '../components/glass/GlassCard';
import {
  Crosshair,
  ArrowLeft,
  ArrowRight,
  Shield,
  CheckCircle2,
  XCircle,
  Layers,
  Activity,
  MapPin,
  Radio,
  Box,
  Ruler,
} from 'lucide-react';

export const DetectionDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [detection, setDetection] = useState<Detection | null>(null);
  const [allDetections, setAllDetections] = useState<Detection[]>([]);
  const [activeNotes, setActiveNotes] = useState<string>('');
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'3D_RECON' | 'ACOUSTIC_CROP'>('3D_RECON');

  useEffect(() => {
    if (!id) return;
    detectionApi.getDetections().then((list) => {
      setAllDetections(list);
      const found = list.find((d) => d.id === id) || list[0];
      setDetection(found);
      setActiveNotes(found.notes || '');
    });
  }, [id]);

  if (!detection) {
    return (
      <div className="p-8 text-cyan-400 font-mono flex items-center justify-center">
        <Crosshair className="w-5 h-5 animate-spin mr-2" /> Loading subsea detection analysis...
      </div>
    );
  }

  const currentIndex = allDetections.findIndex((d) => d.id === detection.id);
  const prevDetection = currentIndex > 0 ? allDetections[currentIndex - 1] : null;
  const nextDetection = currentIndex < allDetections.length - 1 ? allDetections[currentIndex + 1] : null;

  const handleUpdateStatus = async (status: 'CONFIRMED' | 'FALSE_POSITIVE' | 'UNVERIFIED') => {
    const updated = await detectionApi.updateDetectionVerification(detection.id, status, activeNotes);
    setDetection(updated);
    setSaveStatus(`Target status updated to: ${status}`);
    setTimeout(() => setSaveStatus(null), 3000);
  };

  return (
    <div className="p-4 md:p-6 max-w-[1700px] mx-auto w-full font-mono space-y-4">
      {/* Header & Navigation */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
        <div className="flex items-center gap-3">
          <GlassButton
            variant="secondary"
            size="sm"
            onClick={() => navigate('/detections')}
            icon={<ArrowLeft className="w-4 h-4" />}
          >
            BACK
          </GlassButton>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white dark:text-white light:text-slate-900 uppercase">{detection.id}</h1>
              <GlassBadge variant="cyan" size="sm">
                {detection.classNameLabel}
              </GlassBadge>
              <GlassBadge
                variant={
                  detection.verifiedStatus === 'CONFIRMED'
                    ? 'emerald'
                    : detection.verifiedStatus === 'FALSE_POSITIVE'
                    ? 'rose'
                    : 'amber'
                }
                size="sm"
              >
                {detection.verifiedStatus}
              </GlassBadge>
            </div>
            <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-0.5">
              Mission Survey: <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">{detection.missionName}</span> ({detection.missionId})
            </p>
          </div>
        </div>

        {/* Previous / Next Navigation */}
        <div className="flex items-center gap-2">
          {prevDetection && (
            <GlassButton
              variant="secondary"
              size="sm"
              onClick={() => navigate(`/detections/${prevDetection.id}`)}
              icon={<ArrowLeft className="w-3.5 h-3.5" />}
            >
              PREV
            </GlassButton>
          )}
          {nextDetection && (
            <GlassButton
              variant="secondary"
              size="sm"
              onClick={() => navigate(`/detections/${nextDetection.id}`)}
            >
              <span>NEXT</span>
              <ArrowRight className="w-3.5 h-3.5 ml-1" />
            </GlassButton>
          )}
          <GlassButton
            variant="primary"
            size="sm"
            onClick={() => navigate(`/sonar?detectionId=${detection.id}`)}
            icon={<Radio className="w-3.5 h-3.5" />}
          >
            SONAR WORKSTATION
          </GlassButton>
        </div>
      </div>

      {saveStatus && (
        <div className="bg-emerald-950/80 dark:bg-emerald-950/80 light:bg-emerald-50 border border-emerald-500/40 text-emerald-300 dark:text-emerald-300 light:text-emerald-800 px-3 py-2 rounded-xl text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{saveStatus}</span>
        </div>
      )}

      {/* Main 3-Column Forensic Inspection Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column (6 cols): 3D Reconstructed Seabed & Acoustic Crop */}
        <div className="lg:col-span-6 space-y-4">
          <GlassCard variant="default" className="p-4 space-y-3">
            {/* View Switcher Header */}
            <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2 text-xs">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveTab('3D_RECON')}
                  className={`px-2.5 py-1 rounded-xl text-[10px] uppercase font-bold tracking-wider transition-all flex items-center gap-1.5 ${
                    activeTab === '3D_RECON'
                      ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300 shadow-sm'
                      : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
                  }`}
                >
                  <Box className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                  <span>3D Seabed Reconstruction</span>
                </button>
                <button
                  onClick={() => setActiveTab('ACOUSTIC_CROP')}
                  className={`px-2.5 py-1 rounded-xl text-[10px] uppercase font-bold tracking-wider transition-all flex items-center gap-1.5 ${
                    activeTab === 'ACOUSTIC_CROP'
                      ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300 shadow-sm'
                      : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                  <span>Raw Sonar Backscatter</span>
                </button>
              </div>

              <span className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">Ping #{detection.pingIndex}</span>
            </div>

            {/* Viewport View */}
            {activeTab === '3D_RECON' ? (
              <div className="h-[360px] w-full rounded-xl overflow-hidden border border-cyan-500/20 dark:border-cyan-500/20 light:border-slate-200">
                <BathymetryViewer
                  detection={detection}
                  depthMeters={detection.depthMeters}
                  areaSizeMeters={30}
                  bathymetryAvailable={true}
                  className="h-full w-full"
                />
              </div>
            ) : (
              <div className="relative h-[360px] bg-[#020712] dark:bg-[#020712] light:bg-slate-900 border border-cyan-500/20 dark:border-cyan-500/20 light:border-slate-300 rounded-xl overflow-hidden flex items-center justify-center p-4">
                {detection.cropUrl ? (
                  <img
                    src={detection.cropUrl}
                    alt={detection.classNameLabel}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full relative flex items-center justify-center bg-gradient-to-br from-[#061426] to-[#020712] rounded-lg border border-cyan-500/30">
                    <div className="w-28 h-20 rounded-md border-2 border-cyan-400 bg-cyan-400/20 shadow-lg shadow-cyan-500/30 flex items-center justify-center text-xs font-bold text-cyan-200">
                      Specular Peak
                    </div>
                    {detection.acousticShadow && (
                      <div className="w-36 h-20 ml-2 bg-red-950/60 border border-red-500/70 border-dashed rounded-md flex items-center justify-center text-[10px] text-red-300">
                        Shadow ({detection.acousticShadow.lengthMeters}m)
                      </div>
                    )}
                  </div>
                )}
                <div className="absolute top-2 left-2 bg-[#020712]/90 px-2 py-0.5 rounded text-[10px] text-cyan-300 border border-cyan-900/40">
                  Raw Backscatter 455 kHz
                </div>
              </div>
            )}

            {/* Surveyor Verification & Action Notes */}
            <div className="space-y-2 pt-2 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 text-xs">
              <label className="block text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold text-[11px] uppercase">
                Surveyor Forensic Notes & Marine Findings:
              </label>
              <textarea
                rows={3}
                value={activeNotes}
                onChange={(e) => setActiveNotes(e.target.value)}
                placeholder="Add verification notes, entanglement risk, or salvage coordinates..."
                className="w-full bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 rounded-xl p-2.5 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900 focus:outline-none focus:border-cyan-400"
              />

              <div className="flex items-center gap-2 pt-1">
                <GlassButton
                  variant="primary"
                  size="sm"
                  onClick={() => handleUpdateStatus('CONFIRMED')}
                  className="flex-1 text-xs"
                  icon={<CheckCircle2 className="w-3.5 h-3.5" />}
                >
                  CONFIRM TARGET
                </GlassButton>
                <GlassButton
                  variant="danger"
                  size="sm"
                  onClick={() => handleUpdateStatus('FALSE_POSITIVE')}
                  className="flex-1 text-xs"
                  icon={<XCircle className="w-3.5 h-3.5" />}
                >
                  FALSE POSITIVE
                </GlassButton>
              </div>
            </div>
          </GlassCard>
        </div>

        {/* Right Column (6 cols): Multi-Score AI Confidence & Shadow Ray-Tracing */}
        <div className="lg:col-span-6 space-y-4 text-xs">
          {/* Multi-Stage Neural Matrix */}
          <GlassCard variant="default" className="p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2">
              <span className="font-bold text-white dark:text-white light:text-slate-900 text-[11px] uppercase tracking-wider flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                MULTI-STAGE NEURAL CONFIDENCE MATRIX
              </span>
              <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold text-sm font-mono">
                {(detection.confidence * 100).toFixed(1)}% OVERALL
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-center">
              <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold">YOLOv11</div>
                <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold text-base mt-0.5 font-mono">
                  {(detection.detectorScore * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500 dark:text-slate-500 light:text-slate-500 mt-0.5">Backbone</div>
              </div>
              <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold">Shadow Score</div>
                <div className="text-amber-400 dark:text-amber-400 light:text-amber-700 font-bold text-base mt-0.5 font-mono">
                  {(detection.shadowScore * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500 dark:text-slate-500 light:text-slate-500 mt-0.5">Grazing Ray</div>
              </div>
              <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold">Geometry</div>
                <div className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold text-base mt-0.5 font-mono">
                  {(detection.geometryScore * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500 dark:text-slate-500 light:text-slate-500 mt-0.5">SAM2 Contour</div>
              </div>
              <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold">PatchCore</div>
                <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold text-base mt-0.5 font-mono">
                  {(detection.anomalyScore * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500 dark:text-slate-500 light:text-slate-500 mt-0.5">Anomaly Dev</div>
              </div>
            </div>
          </GlassCard>

          {/* Morphological Geometry & Acoustic Shadow Ray-Tracing */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Morphological Geometry */}
            <GlassCard variant="default" className="p-3.5 space-y-2">
              <div className="font-bold text-white dark:text-white light:text-slate-900 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-1.5 flex items-center gap-1.5 text-[11px] uppercase">
                <Shield className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                Morphological Geometry
              </div>
              <div className="space-y-1.5 text-[11px] pt-1">
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Area (Pixels):</span>
                  <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-bold font-mono">{detection.geometry.areaPixels} px²</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Perimeter:</span>
                  <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono">{detection.geometry.perimeterPixels} px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Aspect Ratio:</span>
                  <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono">{detection.geometry.aspectRatio.toFixed(2)} : 1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Solidity / Extent:</span>
                  <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono">
                    {detection.geometry.solidity.toFixed(2)} / {detection.geometry.extent.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Orientation:</span>
                  <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">{detection.geometry.orientationDeg.toFixed(1)}°</span>
                </div>
              </div>
            </GlassCard>

            {/* Acoustic Shadow Ray-Tracing Physics */}
            <GlassCard variant="default" className="p-3.5 space-y-2">
              <div className="font-bold text-white dark:text-white light:text-slate-900 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-1.5 flex items-center gap-1.5 text-[11px] uppercase">
                <Ruler className="w-3.5 h-3.5 text-amber-400" />
                Acoustic Shadow Physics
              </div>
              <div className="space-y-1.5 text-[11px] pt-1">
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Shadow Length (Ls):</span>
                  <span className="text-amber-300 dark:text-amber-300 light:text-amber-700 font-bold font-mono">
                    {detection.acousticShadow?.lengthMeters || 3.4} m
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Est. Target Height (Ht):</span>
                  <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">
                    {detection.acousticShadow?.estimatedHeightMeters || 1.2} m
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Slant Range (Rs):</span>
                  <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono">{detection.slantRangeMeters?.toFixed(1)} m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Tow Altitude (H):</span>
                  <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono">{detection.altitudeMeters?.toFixed(1)} m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Calculated Volume:</span>
                  <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">
                    {((detection.geometry.areaPixels * (detection.acousticShadow?.estimatedHeightMeters || 1)) / 100).toFixed(1)} m³
                  </span>
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Geospatial Coordinates */}
          <GlassCard variant="default" className="p-3.5 space-y-2">
            <div className="font-bold text-white dark:text-white light:text-slate-900 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-1.5 flex items-center gap-1.5 text-[11px] uppercase">
              <MapPin className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
              Geospatial Navigation Datum (WGS84)
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
              <div>
                <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Latitude:</span>
                <div className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold font-mono">
                  {detection.latitude !== null ? formatDMS(detection.latitude, true) : 'N/A'}
                </div>
              </div>
              <div>
                <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Longitude:</span>
                <div className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold font-mono">
                  {detection.longitude !== null ? formatDMS(detection.longitude, false) : 'N/A'}
                </div>
              </div>
              <div>
                <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Target Depth:</span>
                <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">{detection.depthMeters} m</div>
              </div>
              <div>
                <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">RTK GPS Status:</span>
                <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">
                  {(detection.geotagConfidence * 100).toFixed(0)}% RTK LOCK
                </div>
              </div>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
};

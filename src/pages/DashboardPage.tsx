import React, { useState, useEffect } from 'react';
import { useNavigate, useOutletContext } from 'react-router-dom';
import { Mission, Detection, SystemTelemetry, RenderProfile } from '../types';
import { missionApi } from '../services/missionApi';
import { detectionApi } from '../services/detectionApi';
import { systemApi } from '../services/systemApi';
import { SonarViewer } from '../components/sonar/SonarViewer';
import { MissionMap } from '../components/gis/MissionMap';
import { ThreeOceanScene } from '../components/three/ThreeOceanScene';
import { OpenCvAnalysisPanel } from '../components/sonar/OpenCvAnalysisPanel';
import { GlassCard, GlassPanel, GlassButton, GlassStat, GlassBadge, KpiCard } from '../components/glass/GlassCard';
import {
  Compass,
  Cpu,
  Zap,
  Radio,
  Layers,
  Sparkles,
  ExternalLink,
  Shield,
  Activity,
  Box,
  MapPin,
  Flame,
  Play,
  RotateCcw,
  Sliders,
  ChevronRight,
  Maximize2,
  Anchor,
  Camera,
} from 'lucide-react';

interface ContextType {
  renderProfile: RenderProfile;
  activeMissionName: string;
  setActiveMissionName: (name: string) => void;
}

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { renderProfile } = useOutletContext<ContextType>();

  const [missions, setMissions] = useState<Mission[]>([]);
  const [selectedMission, setSelectedMission] = useState<Mission | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);
  const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(null);
  const [viewMode, setViewMode] = useState<'3D_DIGITAL_TWIN' | 'GIS' | 'OPENCV_INSPECT'>('3D_DIGITAL_TWIN');
  const [activeHistogram, setActiveHistogram] = useState<number[]>([]);
  const [pingIndex, setPingIndex] = useState<number>(3200);

  useEffect(() => {
    let isMounted = true;
    const loadInitialData = async () => {
      try {
        const [msns, dets, telem] = await Promise.all([
          missionApi.getMissions(),
          detectionApi.getDetections(),
          systemApi.getTelemetry(),
        ]);
        if (isMounted) {
          setMissions(msns);
          const active = msns.find((m) => m.status === 'Active') || msns[0];
          setSelectedMission(active);
          setDetections(dets);
          if (dets.length > 0) setSelectedDetection(dets[0]);
          setTelemetry(telem);
        }
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
      }
    };

    loadInitialData();
    const interval = setInterval(async () => {
      try {
        const telem = await systemApi.getTelemetry();
        if (isMounted) setTelemetry(telem);
      } catch {}
    }, 2500);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleMissionChange = async (missionId: string) => {
    const found = missions.find((m) => m.id === missionId);
    if (found) {
      setSelectedMission(found);
      const dets = await detectionApi.getDetections({ missionId: found.id });
      setDetections(dets);
      if (dets.length > 0) setSelectedDetection(dets[0]);
    }
  };

  if (!selectedMission) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 text-cyan-400 font-mono">
        <Radio className="w-6 h-6 animate-spin mr-2" /> Initializing Marine Sonar Command Center...
      </div>
    );
  }

  // Aggregate KPIs
  const ghostGearCount = detections.filter((d) => d.class === 'ghost_gear').length;
  const shipwreckCount = detections.filter((d) => d.class === 'shipwreck').length;
  const uxoCount = detections.filter((d) => d.class === 'unexploded_ordnance').length;
  const avgConfidence = (
    (detections.reduce((acc, d) => acc + d.confidence, 0) / Math.max(1, detections.length)) *
    100
  ).toFixed(1);

  return (
    <div className="flex-1 p-3 md:p-5 flex flex-col gap-4 max-w-[1920px] mx-auto w-full font-mono">
      {/* Row 1: Hero Standardized KPI Metrics Strip (12-Column Grid) */}
      <div className="grid grid-cols-12 gap-4">
        {/* Active Survey Mission */}
        <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
          <KpiCard className="kpi-card-interactive">
            <div className="kpi-header">
              <span className="flex items-center gap-1.5 text-white dark:text-white light:text-slate-900 text-[10px] tracking-widest uppercase">
                <Compass className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                ACTIVE SURVEY
              </span>
              <GlassBadge variant="emerald" size="sm" pulse>
                {selectedMission.status}
              </GlassBadge>
            </div>

            <div className="kpi-body">
              <select
                value={selectedMission.id}
                onChange={(e) => handleMissionChange(e.target.value)}
                className="w-full bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-white border border-cyan-500/30 dark:border-cyan-500/30 light:border-slate-300 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold focus:outline-none focus:border-cyan-400 cursor-pointer"
              >
                {missions.map((m) => (
                  <option key={m.id} value={m.id} className="bg-[#040D1B] text-slate-200">
                    {m.name} ({m.id})
                  </option>
                ))}
              </select>
              <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 mt-1.5 flex items-center justify-between">
                <span>{selectedMission.sonarSource}</span>
                <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-600 font-bold">{selectedMission.frequencyKhz} kHz</span>
              </div>
            </div>

            <div className="kpi-footer">
              <span>Swath: {selectedMission.surveyDistanceKm} km</span>
              <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">{selectedMission.swathWidthMeters}m Port/Stbd</span>
            </div>
          </KpiCard>
        </div>

        {/* Neural Target Detections */}
        <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
          <KpiCard className="kpi-card-interactive">
            <div className="kpi-header">
              <span className="flex items-center gap-1.5 text-white dark:text-white light:text-slate-900 text-[10px] tracking-widest uppercase">
                <Shield className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                IDENTIFIED TARGETS
              </span>
              <GlassBadge variant="cyan" size="sm">
                {detections.length} PINNED
              </GlassBadge>
            </div>

            <div className="kpi-body">
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-1.5 rounded-lg border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                  <div className="text-[9px] uppercase tracking-wider text-slate-500 dark:text-slate-500 light:text-slate-600 font-bold">Ghost Gear</div>
                  <div className="text-amber-400 dark:text-amber-400 light:text-amber-700 font-bold text-sm">{ghostGearCount}</div>
                </div>
                <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-1.5 rounded-lg border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                  <div className="text-[9px] uppercase tracking-wider text-slate-500 dark:text-slate-500 light:text-slate-600 font-bold">Wrecks</div>
                  <div className="text-pink-400 dark:text-pink-400 light:text-pink-700 font-bold text-sm">{shipwreckCount}</div>
                </div>
                <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-1.5 rounded-lg border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                  <div className="text-[9px] uppercase tracking-wider text-slate-500 dark:text-slate-500 light:text-slate-600 font-bold">Mean Conf</div>
                  <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold text-sm">{avgConfidence}%</div>
                </div>
              </div>
            </div>

            <div className="kpi-footer">
              <span>False-Pos: {(selectedMission.summaryMetrics.falsePositiveRatio * 100).toFixed(1)}%</span>
              <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">YOLOv11+SAM2</span>
            </div>
          </KpiCard>
        </div>

        {/* Real-time Hardware Telemetry */}
        <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
          <KpiCard className="kpi-card-interactive">
            <div className="kpi-header">
              <span className="flex items-center gap-1.5 text-white dark:text-white light:text-slate-900 text-[10px] tracking-widest uppercase">
                <Cpu className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" />
                NVIDIA RTX 5060
              </span>
              <span className="text-[10px] text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">
                {telemetry && telemetry.temperatureCelsius !== null && telemetry.temperatureCelsius !== undefined
                  ? `${telemetry.temperatureCelsius}°C`
                  : 'ACTIVE'}
              </span>
            </div>

            <div className="kpi-body">
              <div className="flex items-baseline justify-between">
                <div>
                  <div className="text-2xl font-black text-white dark:text-white light:text-slate-900">
                    {telemetry && telemetry.gpuUtilPct !== null && telemetry.gpuUtilPct !== undefined
                      ? `${telemetry.gpuUtilPct}%`
                      : 'ONLINE'}
                  </div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 font-bold">CUDA Compute</div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800">
                    {telemetry?.inferenceFps ? `${telemetry.inferenceFps} FPS` : '58.4 FPS'}
                  </div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600">
                    {telemetry?.latencyMs ?? 17.1}ms Latency
                  </div>
                </div>
              </div>
            </div>

            <div className="kpi-footer">
              <span>
                VRAM:{' '}
                {telemetry?.vramUsedGb !== null && telemetry?.vramUsedGb !== undefined
                  ? `${telemetry.vramUsedGb} / ${telemetry.vramTotalGb || 8} GB`
                  : '3.6 / 8 GB'}
              </span>
              <span className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold">FP16 TensorRT</span>
            </div>
          </KpiCard>
        </div>

        {/* 3D Digital Twin Quick Launcher */}
        <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
          <KpiCard className="kpi-card-interactive">
            <div className="kpi-header">
              <span className="flex items-center gap-1.5 text-white dark:text-white light:text-slate-900 text-[10px] tracking-widest uppercase">
                <Box className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600 animate-pulse" />
                DIGITAL TWIN
              </span>
              <GlassBadge variant="cyan" size="sm">
                3D WebGL
              </GlassBadge>
            </div>

            <div className="kpi-body">
              <p className="text-[11px] text-slate-300 dark:text-slate-300 light:text-slate-700 leading-relaxed font-sans line-clamp-2">
                Procedural bathymetry, volumetric sonar beam pulse, and acoustic shadow optics.
              </p>
            </div>

            <div className="kpi-footer flex items-center gap-1.5 w-full">
              <GlassButton
                variant="primary"
                size="sm"
                className="flex-1 text-[10px] tracking-wider"
                icon={<Box className="w-3 h-3" />}
                onClick={() => navigate(`/digital-twin?missionId=${selectedMission.id}`)}
              >
                3D TWIN
              </GlassButton>
              <GlassButton
                variant="secondary"
                size="sm"
                className="flex-1 text-[10px] tracking-wider text-cyan-300 dark:text-cyan-300 light:text-[#00639b]"
                icon={<Camera className="w-3 h-3" />}
                onClick={() => navigate('/webcam-tracker')}
              >
                LIVE WEBCAM
              </GlassButton>
            </div>
          </KpiCard>
        </div>
      </div>

      {/* Row 2: Main Workspace Split (Left: Waterfall Sonar Viewer, Right: 3D Twin Centerpiece / GIS / OpenCV) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-[570px]">
        {/* Left 7 Columns: Sonar Waterfall Workstation Preview */}
        <div className="lg:col-span-7 flex flex-col h-[570px]">
          <SonarViewer
            detections={detections}
            selectedDetectionId={selectedDetection?.id}
            onSelectDetection={(d) => setSelectedDetection(d)}
            onHistogramUpdate={setActiveHistogram}
            missionName={selectedMission.name}
            pingIndex={pingIndex}
            onPingChange={setPingIndex}
          />
        </div>

        {/* Right 5 Columns: 3D Digital Twin Centerpiece / GIS Command Map / OpenCV Telemetry */}
        <div className="lg:col-span-5 flex flex-col h-[570px] rounded-2xl overflow-hidden bg-[#040E1E]/75 dark:bg-[#040E1E]/75 light:bg-white/85 backdrop-blur-2xl border border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 shadow-[0_16px_40px_-10px_rgba(0,0,0,0.75),inset_0_1px_1.5px_0_rgba(255,255,255,0.25)] dark:shadow-[0_16px_40px_-10px_rgba(0,0,0,0.75),inset_0_1px_1.5px_0_rgba(255,255,255,0.25)] light:shadow-[0_10px_30px_-6px_rgba(15,23,42,0.08),inset_0_1px_1.5px_0_rgba(255,255,255,0.95)] relative">
          {/* Top Liquid Specular Reflection */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/50 to-transparent pointer-events-none z-20" />

          {/* Sub-tab Switcher Header */}
          <div className="h-12 bg-[#020814]/70 dark:bg-[#020814]/70 light:bg-slate-50/80 backdrop-blur-md border-b border-cyan-900/35 dark:border-cyan-900/35 light:border-slate-200/80 px-3.5 flex items-center justify-between text-xs shrink-0 relative z-10">
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setViewMode('3D_DIGITAL_TWIN')}
                className={`px-3 py-1.5 rounded-xl text-[10px] uppercase font-mono font-bold tracking-wider transition-all flex items-center gap-1.5 relative overflow-hidden backdrop-blur-xl ${
                  viewMode === '3D_DIGITAL_TWIN'
                    ? 'bg-gradient-to-b from-cyan-500/30 to-cyan-600/15 dark:from-cyan-500/30 dark:to-cyan-600/15 light:from-sky-100 light:to-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/60 dark:border-cyan-400/60 light:border-sky-400 shadow-[0_0_16px_rgba(34,211,238,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)]'
                    : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-transparent'
                }`}
              >
                <Box className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                <span>3D Ocean Floor</span>
              </button>

              <button
                onClick={() => setViewMode('GIS')}
                className={`px-3 py-1.5 rounded-xl text-[10px] uppercase font-mono font-bold tracking-wider transition-all flex items-center gap-1.5 relative overflow-hidden backdrop-blur-xl ${
                  viewMode === 'GIS'
                    ? 'bg-gradient-to-b from-cyan-500/30 to-cyan-600/15 dark:from-cyan-500/30 dark:to-cyan-600/15 light:from-sky-100 light:to-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/60 dark:border-cyan-400/60 light:border-sky-400 shadow-[0_0_16px_rgba(34,211,238,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)]'
                    : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-transparent'
                }`}
              >
                <Compass className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                <span>GIS Swath Map</span>
              </button>

              <button
                onClick={() => setViewMode('OPENCV_INSPECT')}
                className={`px-3 py-1.5 rounded-xl text-[10px] uppercase font-mono font-bold tracking-wider transition-all flex items-center gap-1.5 relative overflow-hidden backdrop-blur-xl ${
                  viewMode === 'OPENCV_INSPECT'
                    ? 'bg-gradient-to-b from-cyan-500/30 to-cyan-600/15 dark:from-cyan-500/30 dark:to-cyan-600/15 light:from-sky-100 light:to-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/60 dark:border-cyan-400/60 light:border-sky-400 shadow-[0_0_16px_rgba(34,211,238,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)]'
                    : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-transparent'
                }`}
              >
                <Activity className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                <span>OpenCV Filter</span>
              </button>
            </div>

            {viewMode === '3D_DIGITAL_TWIN' && (
              <GlassButton
                variant="ghost"
                size="sm"
                icon={<Maximize2 className="w-3 h-3" />}
                onClick={() => navigate(`/digital-twin?missionId=${selectedMission.id}`)}
                className="text-[10px] text-cyan-400 dark:text-cyan-400 light:text-sky-700 hover:text-cyan-300"
              >
                EXPAND 3D
              </GlassButton>
            )}
          </div>

          {/* Sub-view Viewport Content */}
          <div className="flex-1 relative overflow-hidden z-10">
            {viewMode === '3D_DIGITAL_TWIN' && (
              <ThreeOceanScene
                mission={selectedMission}
                detections={detections}
                selectedDetectionId={selectedDetection?.id}
                onSelectDetection={(d) => setSelectedDetection(d)}
                renderProfile={renderProfile}
              />
            )}

            {viewMode === 'GIS' && (
              <MissionMap
                mission={selectedMission}
                allMissions={missions}
                detections={detections}
                selectedDetectionId={selectedDetection?.id}
                onSelectDetection={(d) => setSelectedDetection(d)}
                className="h-full w-full"
              />
            )}

            {viewMode === 'OPENCV_INSPECT' && (
              <div className="h-full overflow-y-auto p-4 bg-[#030914] dark:bg-[#030914] light:bg-slate-50">
                <OpenCvAnalysisPanel
                  histogram={activeHistogram}
                  detection={selectedDetection}
                  altitudeMeters={selectedMission.trackPoints[0]?.altitudeMeters ?? 8.5}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Row 3: Active Target Detection Cards Feed */}
      <GlassCard variant="default" className="p-4">
        <div className="flex items-center justify-between pb-3 border-b border-cyan-900/30 text-xs">
          <span className="font-bold text-white text-[11px] uppercase tracking-widest flex items-center gap-2">
            <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
            VERIFIED ACOUSTIC TARGET FEED ({detections.length} DETECTIONS)
          </span>
          <button
            onClick={() => navigate('/detections')}
            className="text-[11px] uppercase tracking-widest text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-bold transition-colors"
          >
            <span>View All Targets ({detections.length})</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        <div className="grid grid-cols-12 gap-4 mt-3.5">
          {detections.slice(0, 4).map((det, idx) => (
            <div
              key={`${det.id}-${idx}`}
              onClick={() => {
                setSelectedDetection(det);
                setPingIndex(det.pingIndex);
              }}
              className={`col-span-12 sm:col-span-6 lg:col-span-3 p-3.5 rounded-2xl border transition-all duration-200 cursor-pointer flex flex-col justify-between ${
                selectedDetection?.id === det.id
                  ? 'bg-cyan-500/15 dark:bg-cyan-500/15 light:bg-sky-100/90 border-cyan-400 dark:border-cyan-400 light:border-sky-400 shadow-[0_0_16px_rgba(34,211,238,0.25)]'
                  : 'bg-[#020712]/70 dark:bg-[#020712]/70 light:bg-white/80 border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 hover:border-cyan-500/40 hover:bg-[#040E1E] dark:hover:bg-[#040E1E] light:hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5 pb-1.5 border-b border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200">
                <span className="font-bold text-white dark:text-white light:text-slate-900 text-xs font-mono">{det.id}</span>
                <GlassBadge variant={det.confidence > 0.85 ? 'emerald' : 'cyan'} size="sm">
                  {(det.confidence * 100).toFixed(0)}% CONF
                </GlassBadge>
              </div>

              <div className="text-xs font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 uppercase tracking-wide truncate my-1">
                {det.classNameLabel}
              </div>

              <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 mt-2 flex items-center justify-between pt-1.5 border-t border-cyan-900/25 dark:border-cyan-900/25 light:border-slate-200 font-mono">
                <span>Depth: {det.depthMeters}m</span>
                <span className={det.acousticShadow ? 'text-amber-400 dark:text-amber-400 light:text-amber-700 font-bold' : 'text-slate-500'}>
                  {det.acousticShadow ? `Shadow: ${det.acousticShadow.lengthMeters}m` : 'No Shadow'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
};

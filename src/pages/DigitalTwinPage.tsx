import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  DigitalTwinCanvas,
  DigitalTwinLayers,
  DigitalTwinCameraMode,
  DigitalTwinColorScheme,
  SonarShaderConfig,
  SonarPulseMode,
} from '../components/three/DigitalTwinCanvas';
import { GlassCard, GlassButton, GlassBadge } from '../components/glass/GlassCard';
import { Mission, Detection } from '../types';
import { missionApi } from '../services/missionApi';
import { detectionApi } from '../services/detectionApi';
import { formatDMS } from '../utils/sonarProcessor';
import {
  Compass,
  Layers,
  Camera,
  Play,
  Pause,
  RotateCcw,
  Maximize2,
  Minimize2,
  ChevronRight,
  Eye,
  Radio,
  Sliders,
  Crosshair,
  Activity,
  Shield,
  Sparkles,
  ExternalLink,
  MapPin,
  Anchor,
  Cpu,
  Zap,
  Waves,
  Volume2,
} from 'lucide-react';

export const DigitalTwinPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [missions, setMissions] = useState<Mission[]>([]);
  const [selectedMissionId, setSelectedMissionId] = useState<string>('MSN-2025-08-01');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);

  // Digital Twin state
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackProgress, setPlaybackProgress] = useState<number>(0.35);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const [cameraMode, setCameraMode] = useState<DigitalTwinCameraMode>('FREE_ORBIT');
  const [colorScheme, setColorScheme] = useState<DigitalTwinColorScheme>('OCEANIC');
  const [hoveredPoint, setHoveredPoint] = useState<{ depthMeters: number; lat: number; lng: number } | null>(
    null
  );

  // Shader-based Acoustic Sonar Pulse Configuration
  const [sonarConfig, setSonarConfig] = useState<SonarShaderConfig>({
    pulseMode: 'DUAL_COMBINED',
    pulseSpeed: 1.2,
    pulseFrequency: 2.4, // 455 kHz representation
    pulseIntensity: 1.4,
    swathWidth: 24.0,
    lastPingTimestamp: 0,
  });

  const [isManualPingActive, setIsManualPingActive] = useState<boolean>(false);
  const [isBeamEngineCollapsed, setIsBeamEngineCollapsed] = useState<boolean>(false);

  const [layers, setLayers] = useState<DigitalTwinLayers>({
    bathymetry: true,
    sonarBeam: true,
    sonarPulse: true,
    detections: true,
    shadows: true,
    heatmap: false,
    contours: true,
    grid: true,
    vessel: true,
    particles: true,
  });

  // Load missions & detections
  useEffect(() => {
    missionApi.getMissions().then((data) => {
      setMissions(data);
      if (data.length > 0) {
        const queryMissionId = searchParams.get('missionId');
        if (queryMissionId && data.some((m) => m.id === queryMissionId)) {
          setSelectedMissionId(queryMissionId);
        } else {
          setSelectedMissionId(data[0].id);
        }
      }
    });

    detectionApi.getDetections().then((data) => {
      setDetections(data);
      const queryDetId = searchParams.get('detectionId');
      if (queryDetId) {
        const found = data.find((d) => d.id === queryDetId);
        if (found) setSelectedDetection(found);
      }
    });
  }, [searchParams]);

  // Active mission
  const activeMission = missions.find((m) => m.id === selectedMissionId) || missions[0];

  // Filter detections for active mission
  const missionDetections = detections.filter(
    (d) => d.missionId === activeMission?.id
  );

  // Playback timer loop
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setPlaybackProgress((prev) => {
        const next = prev + 0.003 * playbackSpeed;
        return next > 1 ? 0 : next;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed]);

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  const handleSelectDetection = (d: Detection) => {
    setSelectedDetection(d);
    setSearchParams({ missionId: selectedMissionId, detectionId: d.id });
  };

  const handleTriggerManualPing = () => {
    setSonarConfig((prev) => ({
      ...prev,
      lastPingTimestamp: Date.now(),
    }));
    setIsManualPingActive(true);
    setTimeout(() => setIsManualPingActive(false), 2400);
  };

  if (!activeMission) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-4rem)] text-cyan-400 font-mono">
        <span className="animate-pulse">INITIALIZING DIGITAL TWIN ENVIRONMENT...</span>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[calc(100vh-3.5rem)] flex flex-col overflow-hidden font-sans select-none"
    >
      {/* TOP INTEGRATED HUD BAR - Strict Stacking Context (z-index: 100) on top of all viewport elements */}
      <div 
        className="digital-twin-hud relative z-[100] isolate h-14 bg-[#030A17]/90 dark:bg-[#030A17]/90 light:bg-white/90 backdrop-blur-2xl border-b border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 px-4 flex items-center justify-between shrink-0 select-none shadow-[0_4px_20px_rgba(0,0,0,0.4)]"
        style={{ zIndex: 100 }}
      >
        {/* Top Liquid Specular Reflection Edge */}
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/60 to-transparent pointer-events-none" />

        {/* Left: Mission & Subsea Telemetry HUD */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping shrink-0" />
            <span className="font-bold text-white dark:text-white light:text-slate-900 uppercase text-xs tracking-wider whitespace-nowrap hidden sm:inline">
              DIGITAL TWIN HUD
            </span>
          </div>

          <div className="h-4 w-[1px] bg-cyan-500/30 dark:bg-cyan-500/30 light:bg-slate-300 shrink-0" />

          <select
            value={selectedMissionId}
            onChange={(e) => {
              setSelectedMissionId(e.target.value);
              setSearchParams({ missionId: e.target.value });
            }}
            className="bg-[#020712]/90 dark:bg-[#020712]/90 light:bg-white border border-cyan-500/40 dark:border-cyan-500/40 light:border-sky-300 rounded-lg px-2.5 py-1.5 text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-semibold text-xs max-w-[180px] sm:max-w-[280px] md:max-w-[340px] truncate focus:outline-none focus:border-cyan-300 shadow-inner"
          >
            {missions.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.id})
              </option>
            ))}
          </select>

          {/* Real-time Subsea Telemetry Strip */}
          <div className="hidden lg:flex items-center gap-3 px-3 py-1 text-xs rounded-xl bg-[#020814]/60 dark:bg-[#020814]/60 light:bg-slate-100/90 backdrop-blur-md border border-cyan-900/35 dark:border-cyan-900/35 light:border-slate-300 shrink-0 shadow-inner">
            <div className="whitespace-nowrap">
              <span className="text-slate-400 dark:text-slate-400 light:text-slate-500 text-[10px] uppercase font-bold">DEPTH:</span>{' '}
              <strong className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono">
                {(42.8 + Math.sin(playbackProgress * 10) * 1.5).toFixed(1)}m
              </strong>
            </div>
            <div className="h-3 w-[1px] bg-cyan-900/40 dark:bg-cyan-900/40 light:bg-slate-300" />
            <div className="whitespace-nowrap">
              <span className="text-slate-400 dark:text-slate-400 light:text-slate-500 text-[10px] uppercase font-bold">ALT:</span>{' '}
              <strong className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-mono">8.4m</strong>
            </div>
            <div className="h-3 w-[1px] bg-cyan-900/40 dark:bg-cyan-900/40 light:bg-slate-300" />
            <div className="whitespace-nowrap hidden xl:block">
              <span className="text-slate-400 dark:text-slate-400 light:text-slate-500 text-[10px] uppercase font-bold">C:</span>{' '}
              <strong className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-mono">1,500 m/s</strong>
            </div>
            <div className="h-3 w-[1px] bg-cyan-900/40 dark:bg-cyan-900/40 light:bg-slate-300 hidden xl:block" />
            <div className="whitespace-nowrap">
              <span className="text-slate-400 dark:text-slate-400 light:text-slate-500 text-[10px] uppercase font-bold">SPD:</span>{' '}
              <strong className="text-amber-300 dark:text-amber-300 light:text-amber-700 font-mono">3.2 kts</strong>
            </div>
          </div>
        </div>

        {/* Right: Sounder Coordinates & Action Tools */}
        <div className="flex items-center gap-2 shrink-0">
          {hoveredPoint && (
            <div className="hidden sm:flex px-2.5 py-1 text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 items-center gap-2 rounded-xl bg-[#020814]/60 dark:bg-[#020814]/60 light:bg-slate-100/90 border border-cyan-900/35 dark:border-cyan-900/35 light:border-slate-300">
              <Crosshair className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
              <span className="font-mono text-[11px]">
                {formatDMS(hoveredPoint.lat, true)}, {formatDMS(hoveredPoint.lng, false)}
              </span>
              <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-bold font-mono">(-{hoveredPoint.depthMeters}m)</span>
            </div>
          )}

          <GlassButton
            variant="secondary"
            size="sm"
            onClick={toggleFullscreen}
            icon={isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          >
            {isFullscreen ? 'EXIT' : 'FULLSCREEN'}
          </GlassButton>
        </div>
      </div>

      {/* 3D WebGL Digital Twin Canvas Viewport with Overlaid Floating Controls */}
      <div className="digital-twin-viewport relative flex-1 overflow-hidden z-0 isolate">
        {/* WebGL Canvas */}
        <div className="absolute inset-0 z-0">
          <DigitalTwinCanvas
            mission={activeMission}
            allMissions={missions}
            detections={missionDetections}
            selectedDetectionId={selectedDetection?.id}
            onSelectDetection={handleSelectDetection}
            layers={layers}
            cameraMode={cameraMode}
            colorScheme={colorScheme}
            playbackProgress={playbackProgress}
            onHoverPoint={setHoveredPoint}
            sonarConfig={sonarConfig}
          />
        </div>

        {/* FLOATING LEFT TOOLBAR: Strict Stacking Context (z-index: 10) in background relative to HUD */}
        <div 
          className="digital-twin-overlay-left absolute top-4 left-4 z-[10] isolate w-72 max-h-[calc(100%-6rem)] overflow-y-auto space-y-3 pointer-events-auto pr-1"
          style={{ zIndex: 10 }}
        >
          {/* SHADER-BASED ACOUSTIC SONAR PULSE ENGINE CARD */}
          <GlassCard 
            variant="glow" 
            className="sonar-beam-engine relative z-[10] isolate p-3 space-y-2.5 border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300 shadow-[0_0_24px_rgba(34,211,238,0.15)]"
            style={{ zIndex: 10 }}
          >
            <div className="flex items-center justify-between border-b border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200 pb-1.5">
              <button
                onClick={() => setIsBeamEngineCollapsed(!isBeamEngineCollapsed)}
                className="flex items-center gap-1.5 text-left group"
              >
                <Waves className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600 animate-pulse shrink-0" />
                <span className="text-xs font-bold text-white dark:text-white light:text-slate-900 uppercase tracking-wide group-hover:text-cyan-300 dark:group-hover:text-cyan-300 light:group-hover:text-sky-700 transition-colors">
                  Acoustic Sonar Beam Engine
                </span>
              </button>
              <div className="flex items-center gap-1.5">
                <GlassBadge variant={layers.sonarPulse ? 'emerald' : 'slate'} size="sm">
                  {layers.sonarPulse ? 'GLSL' : 'OFF'}
                </GlassBadge>
                <button
                  onClick={() => setIsBeamEngineCollapsed(!isBeamEngineCollapsed)}
                  className="p-1 rounded-md text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 hover:bg-cyan-950/30 transition-all text-[11px]"
                  title={isBeamEngineCollapsed ? 'Expand Beam Engine' : 'Collapse Beam Engine'}
                >
                  {isBeamEngineCollapsed ? '▼' : '▲'}
                </button>
              </div>
            </div>

            {!isBeamEngineCollapsed && (
              <div className="space-y-2.5 pt-0.5 animate-in fade-in duration-200">
                {/* Sonar Beam Scanning Mode Buttons */}
                <div className="space-y-1">
                  <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold flex justify-between">
                    <span>Beam Mode</span>
                    <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-700 font-semibold">{sonarConfig.pulseMode}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                    {[
                      { id: 'DUAL_COMBINED', label: 'DUAL CHIRP' },
                      { id: 'SWATH_SWEEP', label: 'SWATH FAN' },
                      { id: 'CHIRP_RADIAL', label: 'RADIAL PING' },
                      { id: 'SECTOR_RADAR', label: '360° SECTOR' },
                    ].map((mode) => (
                      <button
                        key={mode.id}
                        onClick={() =>
                          setSonarConfig((prev) => ({
                            ...prev,
                            pulseMode: mode.id as SonarPulseMode,
                          }))
                        }
                        className={`px-2 py-1.5 rounded-lg transition-all text-center font-bold truncate ${
                          sonarConfig.pulseMode === mode.id
                            ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-100 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400 dark:border-cyan-400 light:border-sky-400 shadow-[0_0_12px_rgba(34,211,238,0.3)]'
                            : 'bg-[#020712]/70 dark:bg-[#020712]/70 light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-slate-200 dark:hover:text-slate-200 light:hover:text-slate-900 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200'
                        }`}
                      >
                        {mode.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Sonar Frequency Band Selection */}
                <div className="space-y-1">
                  <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold flex justify-between">
                    <span>Frequency Band</span>
                    <span className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-semibold">
                      {sonarConfig.pulseFrequency < 2.0
                        ? '100 kHz'
                        : sonarConfig.pulseFrequency < 3.2
                        ? '455 kHz CHIRP'
                        : '900 kHz HD'}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-1 text-[10px]">
                    {[
                      { label: '100 kHz', freq: 1.5 },
                      { label: '455 kHz', freq: 2.4 },
                      { label: '900 kHz', freq: 3.8 },
                    ].map((band) => (
                      <button
                        key={band.label}
                        onClick={() =>
                          setSonarConfig((prev) => ({ ...prev, pulseFrequency: band.freq }))
                        }
                        className={`py-1 rounded text-center transition-all font-semibold ${
                          Math.abs(sonarConfig.pulseFrequency - band.freq) < 0.3
                            ? 'bg-purple-600/30 dark:bg-purple-600/30 light:bg-purple-100 text-purple-200 dark:text-purple-200 light:text-purple-900 border border-purple-400'
                            : 'bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-slate-200 dark:hover:text-slate-200 light:hover:text-slate-900 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200'
                        }`}
                      >
                        {band.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Pulse Propagation Speed & Gain Controls */}
                <div className="grid grid-cols-2 gap-2 text-[10px]">
                  <div className="space-y-1">
                    <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold">Pulse Speed:</span>
                    <div className="flex gap-1">
                      {[0.6, 1.2, 2.4].map((spd) => (
                        <button
                          key={spd}
                          onClick={() => setSonarConfig((prev) => ({ ...prev, pulseSpeed: spd }))}
                          className={`flex-1 py-0.5 rounded text-center font-bold ${
                            Math.abs(sonarConfig.pulseSpeed - spd) < 0.2
                              ? 'bg-cyan-500/30 dark:bg-cyan-500/30 light:bg-sky-100 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400'
                              : 'bg-[#020712] dark:bg-[#020712] light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-600 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200'
                          }`}
                        >
                          {spd === 0.6 ? '0.5x' : spd === 1.2 ? '1x' : '2x'}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold">
                      <span>Gain:</span>
                      <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">{(sonarConfig.pulseIntensity).toFixed(1)}x</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2.5"
                      step="0.1"
                      value={sonarConfig.pulseIntensity}
                      onChange={(e) =>
                        setSonarConfig((prev) => ({
                          ...prev,
                          pulseIntensity: parseFloat(e.target.value),
                        }))
                      }
                      className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-[#020712] dark:bg-[#020712] light:bg-slate-200 rounded-lg"
                    />
                  </div>
                </div>

                {/* High Energy Manual Ping Shockwave Trigger Button */}
                <GlassButton
                  variant={isManualPingActive ? 'amber' : 'primary'}
                  size="sm"
                  onClick={handleTriggerManualPing}
                  icon={<Zap className={`w-3.5 h-3.5 ${isManualPingActive ? 'animate-bounce text-amber-300' : ''}`} />}
                  className="w-full text-xs font-bold"
                >
                  {isManualPingActive ? 'PROPAGATING SHOCKWAVE...' : 'TRIGGER ACOUSTIC PING'}
                </GlassButton>
              </div>
            )}
          </GlassCard>

        {/* Camera Director Preset Controls */}
        <GlassCard variant="default" className="p-3 space-y-2">
          <div className="text-[11px] font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-1.5 uppercase border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-1.5">
            <Camera className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
            Camera Perspectives
          </div>
          <div className="grid grid-cols-2 gap-1.5 text-[10px]">
            <button
              onClick={() => setCameraMode('FREE_ORBIT')}
              className={`px-2 py-1.5 rounded-lg transition-all font-bold ${
                cameraMode === 'FREE_ORBIT'
                  ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-100 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400 dark:border-cyan-400 light:border-sky-400 shadow-[0_0_12px_rgba(34,211,238,0.25)]'
                  : 'bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200'
              }`}
            >
              FREE ORBIT
            </button>
            <button
              onClick={() => setCameraMode('FOLLOW_AUV')}
              className={`px-2 py-1.5 rounded-lg transition-all font-bold ${
                cameraMode === 'FOLLOW_AUV'
                  ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-100 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400 dark:border-cyan-400 light:border-sky-400 shadow-[0_0_12px_rgba(34,211,238,0.25)]'
                  : 'bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200'
              }`}
            >
              FOLLOW AUV
            </button>
            <button
              onClick={() => setCameraMode('PLAN_VIEW')}
              className={`px-2 py-1.5 rounded-lg transition-all font-bold ${
                cameraMode === 'PLAN_VIEW'
                  ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-100 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400 dark:border-cyan-400 light:border-sky-400 shadow-[0_0_12px_rgba(34,211,238,0.25)]'
                  : 'bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200'
              }`}
            >
              PLAN VIEW (2D)
            </button>
            <button
              onClick={() => setCameraMode('SIDE_PROFILE')}
              className={`px-2 py-1.5 rounded-lg transition-all font-bold ${
                cameraMode === 'SIDE_PROFILE'
                  ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-100 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400 dark:border-cyan-400 light:border-sky-400 shadow-[0_0_12px_rgba(34,211,238,0.25)]'
                  : 'bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200'
              }`}
            >
              SIDE PROFILE
            </button>
          </div>
        </GlassCard>

        {/* Digital Twin Layer Toggles */}
        <GlassCard variant="default" className="p-3 space-y-2">
          <div className="text-[11px] font-bold text-white dark:text-white light:text-slate-900 flex items-center justify-between uppercase border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-1.5">
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
              Digital Twin Layers
            </span>
            <span className="text-[9px] text-cyan-400 dark:text-cyan-400 light:text-sky-600 font-bold">10 ACTIVE</span>
          </div>

          <div className="space-y-1.5 text-xs text-slate-300 dark:text-slate-300 light:text-slate-700">
            {[
              { id: 'sonarPulse', label: 'Shader Acoustic Sonar Pulse' },
              { id: 'bathymetry', label: 'Seabed Bathymetry DEM' },
              { id: 'sonarBeam', label: 'Volumetric Sonar Cone' },
              { id: 'detections', label: '3D Target Anomaly Pins' },
              { id: 'shadows', label: 'Acoustic Shadow Silhouettes' },
              { id: 'heatmap', label: 'Ping Density Heatmap' },
              { id: 'contours', label: 'Depth Contour Isolines' },
              { id: 'grid', label: 'Spatial Coordinate Grid' },
              { id: 'vessel', label: 'Surface Survey Vessel' },
              { id: 'particles', label: 'Suspended Marine Snow' },
            ].map((layer) => (
              <label
                key={layer.id}
                className="flex items-center justify-between hover:bg-cyan-950/30 dark:hover:bg-cyan-950/30 light:hover:bg-slate-100 px-1.5 py-0.5 rounded-lg cursor-pointer transition-colors"
              >
                <span className="text-[11px]">{layer.label}</span>
                <input
                  type="checkbox"
                  checked={(layers as any)[layer.id]}
                  onChange={(e) =>
                    setLayers((prev) => ({ ...prev, [layer.id]: e.target.checked }))
                  }
                  className="rounded border-cyan-500/40 text-cyan-400 focus:ring-0 bg-[#020712] dark:bg-[#020712] light:bg-white"
                />
              </label>
            ))}
          </div>
        </GlassCard>

        {/* Bathymetry Colormap Palette */}
        <GlassCard variant="default" className="p-3 space-y-2">
          <div className="text-[11px] font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-1.5 uppercase border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
            Elevation Colormap
          </div>
          <select
            value={colorScheme}
            onChange={(e) => setColorScheme(e.target.value as any)}
            className="w-full bg-[#020712]/90 dark:bg-[#020712]/90 light:bg-white border border-cyan-500/40 dark:border-cyan-500/40 light:border-sky-300 rounded-lg px-2 py-1.5 text-cyan-300 dark:text-cyan-300 light:text-sky-800 text-xs focus:outline-none"
          >
            <option value="OCEANIC">Oceanic Cyan Gradient</option>
            <option value="BATHYMETRIC_DEM">GEBCO Bathymetric Rainbow</option>
            <option value="THERMAL_SNR">Thermal Acoustic SNR</option>
            <option value="ABYSS">Deep Abyss Midnight</option>
          </select>
        </GlassCard>
      </div>

      {/* FLOATING RIGHT INSPECTOR: Selected Subsea Detection HUD */}
      {selectedDetection && (
        <div 
          className="digital-twin-overlay-right absolute top-4 right-4 z-[20] isolate w-80 max-h-[calc(100%-5.5rem)] overflow-y-auto space-y-3 pointer-events-auto animate-in fade-in slide-in-from-right-4 duration-300"
          style={{ zIndex: 20 }}
        >
          <GlassCard variant="glow" className="p-4 space-y-3">
            <div className="flex items-start justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2">
              <div>
                <GlassBadge variant="cyan" size="sm">
                  {selectedDetection.class.replace('_', ' ')}
                </GlassBadge>
                <h3 className="text-sm font-bold text-white dark:text-white light:text-slate-900 mt-1">{selectedDetection.id}</h3>
              </div>
              <button
                onClick={() => setSelectedDetection(null)}
                className="text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900 text-xs px-1"
              >
                ✕
              </button>
            </div>

            {/* Sonar Thumbnail Crop */}
            {selectedDetection.cropUrl && (
              <div className="relative rounded-lg overflow-hidden border border-cyan-500/30 h-28 bg-[#020712] dark:bg-[#020712] light:bg-slate-100">
                <img
                  src={selectedDetection.cropUrl}
                  alt={selectedDetection.classNameLabel}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-1 right-1 bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-white/90 px-1.5 py-0.5 rounded text-[9px] font-mono text-cyan-300 dark:text-cyan-300 light:text-sky-800">
                  ACOUSTIC CROP
                </div>
              </div>
            )}

            {/* Neural Confidence & Anomaly Score */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-slate-50 p-2 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold">YOLOv11 Conf</div>
                <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold text-base font-mono">
                  {(selectedDetection.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-slate-50 p-2 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold">PatchCore Score</div>
                <div className="text-amber-400 dark:text-amber-400 light:text-amber-700 font-bold text-base font-mono">
                  {(selectedDetection.anomalyScore * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Acoustic Shadow Dimensions */}
            <div className="bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 text-xs space-y-1">
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold flex justify-between">
                <span>Acoustic Shadow Ray-Tracing</span>
                <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-700">{selectedDetection.acousticShadow?.lengthMeters}m Shadow</span>
              </div>
              <div className="text-slate-300 dark:text-slate-300 light:text-slate-700 text-[11px]">
                Estimated Height:{' '}
                <strong className="text-cyan-300 dark:text-cyan-300 light:text-sky-800">
                  {selectedDetection.acousticShadow?.estimatedHeightMeters ?? '2.75'} meters
                </strong>
              </div>
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px]">
                Slant Range: {selectedDetection.slantRangeMeters?.toFixed(1)}m | Altitude: {selectedDetection.altitudeMeters?.toFixed(1) ?? '8.5'}m
              </div>
            </div>

            {/* Geolocation */}
            <div className="text-[11px] text-slate-300 dark:text-slate-300 light:text-slate-700 flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600 shrink-0" />
              <span className="font-mono">
                {selectedDetection.latitude !== null ? formatDMS(selectedDetection.latitude, true) : 'N/A'},{' '}
                {selectedDetection.longitude !== null ? formatDMS(selectedDetection.longitude, false) : 'N/A'}
              </span>
            </div>

            {/* Navigation Actions */}
            <div className="pt-2 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 flex items-center gap-2">
              <GlassButton
                variant="primary"
                size="sm"
                className="w-full text-[11px]"
                icon={<Radio className="w-3.5 h-3.5" />}
                onClick={() => navigate(`/sonar?detectionId=${selectedDetection.id}`)}
              >
                OPEN IN WORKSTATION
              </GlassButton>
              <GlassButton
                variant="secondary"
                size="sm"
                className="w-full text-[11px]"
                icon={<ExternalLink className="w-3.5 h-3.5" />}
                onClick={() => navigate(`/detections/${selectedDetection.id}`)}
              >
                DETAILS
              </GlassButton>
            </div>
          </GlassCard>
        </div>
      )}

      {/* FLOATING BOTTOM TIMELINE & MISSION SCRUBBER */}
      <div 
        className="digital-twin-overlay-bottom absolute bottom-4 left-4 right-4 z-[20] isolate flex items-center justify-center pointer-events-none"
        style={{ zIndex: 20 }}
      >
        <GlassCard variant="default" className="w-full max-w-4xl p-3 flex items-center gap-4 pointer-events-auto">
          {/* Play/Pause & Speed */}
          <div className="flex items-center gap-2">
            <GlassButton
              variant="primary"
              size="sm"
              icon={isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? 'PAUSE' : 'PLAY'}
            </GlassButton>
            <GlassButton
              variant="ghost"
              size="sm"
              icon={<RotateCcw className="w-3.5 h-3.5" />}
              onClick={() => setPlaybackProgress(0)}
              title="Reset Timeline"
            >
              RESET
            </GlassButton>
            <button
              onClick={() => setPlaybackSpeed(playbackSpeed === 1 ? 2 : playbackSpeed === 2 ? 4 : 1)}
              className="px-2 py-1 bg-[#020712] dark:bg-[#020712] light:bg-white border border-cyan-500/40 dark:border-cyan-500/40 light:border-sky-300 rounded-lg text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold text-[10px] hover:border-cyan-300"
            >
              {playbackSpeed}x
            </button>
          </div>

          {/* Timeline Slider */}
          <div className="flex-1 flex items-center gap-3">
            <span className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold whitespace-nowrap">
              AUV SURVEY TRACK:
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.001"
              value={playbackProgress}
              onChange={(e) => {
                setPlaybackProgress(parseFloat(e.target.value));
                setIsPlaying(false);
              }}
              className="flex-1 accent-cyan-400 cursor-pointer h-1.5 bg-[#020712] dark:bg-[#020712] light:bg-slate-200 rounded-lg"
            />
            <span className="text-xs font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono min-w-[45px]">
              {(playbackProgress * 100).toFixed(0)}%
            </span>
          </div>

          {/* Target Count Indicator */}
          <div className="hidden md:flex items-center gap-2 text-xs border-l border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pl-3">
            <Crosshair className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
            <span className="text-slate-300 dark:text-slate-300 light:text-slate-700">
              <strong className="text-cyan-300 dark:text-cyan-300 light:text-sky-800">{missionDetections.length}</strong> Target Pins
            </span>
          </div>
        </GlassCard>
      </div>
    </div>
  </div>
  );
};

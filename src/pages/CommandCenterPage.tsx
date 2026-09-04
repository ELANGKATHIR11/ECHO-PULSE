import React, { useState, useEffect, useRef } from 'react';
import { GlassCard, GlassButton, GlassBadge } from '../components/glass/GlassCard';
import { SonarWaterfallCanvas } from '../components/sonar/SonarWaterfallCanvas';
import { DigitalTwinCanvas } from '../components/three/DigitalTwinCanvas';
import { sonarAudio } from '../utils/audioSynthesizer';
import {
  Shield,
  Activity,
  Crosshair,
  Volume2,
  VolumeX,
  Maximize2,
  Minimize2,
  Compass,
  AlertTriangle,
  Radio,
  Layers,
  Zap,
  MapPin,
  RefreshCw,
  Sliders,
  ChevronRight
} from 'lucide-react';

interface HazardZone {
  hazard_id: string;
  hazard_type: string;
  threat_level: string;
  center: { lat: number; lng: number };
  target_count: number;
  polygon_wgs84: [number, number][];
  area_sq_meters: number;
  recommended_action: string;
}

export const CommandCenterPage: React.FC = () => {
  const [isAudioMuted, setIsAudioMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeColormap, setActiveColormap] = useState<'amber' | 'cobalt' | 'thermal' | 'monochrome'>('cobalt');
  const [gain, setGain] = useState(1.2);
  const [contrast, setContrast] = useState(1.4);
  const [hazardZones, setHazardZones] = useState<HazardZone[]>([]);
  const [threatAlert, setThreatAlert] = useState<{ label: string; confidence: number; location: string } | null>({
    label: 'Ghost Net Entangled Cluster',
    confidence: 0.94,
    location: '9.1524° N, 79.2819° E'
  });
  const [pingCount, setPingCount] = useState(14820);

  // Sound effects toggle
  const toggleAudio = () => {
    const nextState = !isAudioMuted;
    setIsAudioMuted(nextState);
    sonarAudio.setMuted(nextState);
    if (!nextState) {
      sonarAudio.playSonarPing();
    }
  };

  // Fullscreen container ref
  const containerRef = useRef<HTMLDivElement>(null);
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  // Periodic sonar ping audio heartbeat
  useEffect(() => {
    const interval = setInterval(() => {
      setPingCount((p) => p + 1);
      if (!isAudioMuted && Math.random() > 0.4) {
        sonarAudio.playSonarPing(820 + Math.random() * 80, 0.8);
      }
    }, 4500);
    return () => clearInterval(interval);
  }, [isAudioMuted]);

  // Load PostGIS Hazard Zones
  useEffect(() => {
    fetch('/api/v1/gis/hazard-zones')
      .then((res) => res.json())
      .then((data) => {
        if (data.zones) setHazardZones(data.zones);
      })
      .catch(() => {
        // Fallback default zones
        setHazardZones([
          {
            hazard_id: 'HAZARD-GULF-MANNAR-01',
            hazard_type: 'GHOST_NET_ENTANGLEMENT_ZONE',
            threat_level: 'CRITICAL',
            center: { lat: 9.1524, lng: 79.2819 },
            target_count: 4,
            polygon_wgs84: [],
            area_sq_meters: 18400,
            recommended_action: 'Immediate ROV mechanical shear recovery.'
          }
        ]);
      });
  }, []);

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-[calc(100vh-4.5rem)] bg-[#01040a] text-slate-100 overflow-hidden font-sans select-none"
    >
      {/* TOP TACTICAL HUD STATUS STRIP */}
      <header className="h-12 border-b border-cyan-900/40 bg-[#020712]/90 backdrop-blur-md px-4 flex items-center justify-between shrink-0 z-30">
        <div className="flex items-center gap-3">
          <div className="p-1.5 rounded-lg bg-cyan-500/20 border border-cyan-400/50 text-cyan-400">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <h1 className="text-xs font-bold tracking-wider uppercase text-white flex items-center gap-2">
              <span>DEFENSE COMMAND CENTER HUD</span>
              <GlassBadge variant="cyan" size="sm">
                NIOT-MoES DEFENSE MODE
              </GlassBadge>
            </h1>
          </div>
        </div>

        {/* Global Live Telemetry */}
        <div className="hidden md:flex items-center gap-4 text-[11px] font-mono">
          <div className="flex items-center gap-1.5 text-cyan-300">
            <Radio className="w-3.5 h-3.5 animate-pulse text-cyan-400" />
            <span>PING: #{pingCount.toLocaleString()}</span>
          </div>
          <div className="h-3 w-[1px] bg-cyan-900/40" />
          <div className="text-emerald-400">AUV DEPTH: 44.2m</div>
          <div className="h-3 w-[1px] bg-cyan-900/40" />
          <div className="text-amber-400">SPEED: 3.4 kts</div>
          <div className="h-3 w-[1px] bg-cyan-900/40" />
          <div className="text-purple-300">SWATH: 150m (455 kHz)</div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {/* Audio Synthesizer */}
          <button
            onClick={toggleAudio}
            className={`p-2 rounded-xl border text-xs flex items-center gap-1.5 transition-all ${
              isAudioMuted
                ? 'bg-slate-800/40 border-slate-700 text-slate-400'
                : 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_12px_rgba(6,182,212,0.4)]'
            }`}
            title="Acoustic Sonar Sound Feed"
          >
            {isAudioMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
            <span className="text-[10px] font-bold uppercase">{isAudioMuted ? 'MUTE' : 'AUDIO LIVE'}</span>
          </button>

          {/* Fullscreen */}
          <GlassButton variant="secondary" size="sm" onClick={toggleFullscreen}>
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </GlassButton>
        </div>
      </header>

      {/* 4-QUADRANT COMMAND CENTER MATRIX */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 grid-rows-2 gap-2 p-2 overflow-hidden">
        {/* QUADRANT 1 (TOP-LEFT): 60 FPS CASCADING SONAR WATERFALL */}
        <div className="relative rounded-xl border border-cyan-900/40 bg-[#020712] overflow-hidden flex flex-col shadow-inner">
          <div className="absolute top-2 left-2 z-20 flex items-center gap-2 bg-[#020712]/80 backdrop-blur-md px-2.5 py-1 rounded-lg border border-cyan-500/30 text-[10px] font-mono text-cyan-300">
            <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span>Q1 // 60 FPS ACOUSTIC SIDESCAN WATERFALL</span>
          </div>

          {/* Colormap Switcher */}
          <div className="absolute top-2 right-2 z-20 flex items-center gap-1 bg-[#020712]/80 backdrop-blur-md p-1 rounded-lg border border-cyan-900/40 text-[9px]">
            {(['cobalt', 'amber', 'thermal', 'monochrome'] as const).map((map) => (
              <button
                key={map}
                onClick={() => setActiveColormap(map)}
                className={`px-2 py-0.5 rounded font-bold uppercase transition-all ${
                  activeColormap === map
                    ? 'bg-cyan-500 text-black'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {map}
              </button>
            ))}
          </div>

          <div className="flex-1 w-full h-full pt-4">
            <SonarWaterfallCanvas />
          </div>
        </div>

        {/* QUADRANT 2 (TOP-RIGHT): 3D SUBSEA DIGITAL TWIN & ACOUSTIC CONE */}
        <div className="relative rounded-xl border border-cyan-900/40 bg-[#020712] overflow-hidden flex flex-col shadow-inner">
          <div className="absolute top-2 left-2 z-20 flex items-center gap-2 bg-[#020712]/80 backdrop-blur-md px-2.5 py-1 rounded-lg border border-cyan-500/30 text-[10px] font-mono text-cyan-300">
            <Layers className="w-3.5 h-3.5 text-purple-400" />
            <span>Q2 // 3D BATHYMETRIC DIGITAL TWIN & FAN-BEAM</span>
          </div>

          <div className="flex-1 w-full h-full">
            <DigitalTwinCanvas
              mission={{
                id: 'MSN-2026-COMMAND',
                name: 'MoES/NIOT Autonomous Gulf of Mannar Survey',
                codeName: 'OPERATION-OCEAN-CLEAN',
                date: '2026-08-26',
                location: 'Gulf of Mannar Biosphere Reserve',
                coordinates: [9.1524, 79.2819],
                sonarSource: 'Side-Scan Sonar (SSS)',
                frequencyKhz: 455,
                surveyDistanceKm: 18.5,
                swathWidthMeters: 150,
                areaSqKm: 24.5,
                detectionsCount: 5,
                highConfidenceCount: 4,
                status: 'Active',
                durationMinutes: 180,
                pingCount: 18420,
                vesselName: 'SAGAR NIDHI (AUV-04)',
                vehicleType: 'AUV DeepScan-4',
                targetObjective: 'Real-time multi-quadrant acoustic telemetry feed.',
                trackPoints: [],
                coverageCorridorWidthMeters: 150,
                summaryMetrics: {
                  avgSnrDb: 24.2,
                  anomaliesFound: 5,
                  falsePositiveRatio: 0.02,
                  meanProcessingFps: 277
                }
              }}
              colorScheme="OCEANIC"
              cameraMode="FREE_ORBIT"
              layers={{
                bathymetry: true,
                sonarBeam: true,
                sonarPulse: true,
                detections: true,
                shadows: true,
                heatmap: true,
                contours: true,
                grid: true,
                vessel: true,
                particles: true
              }}
              playbackProgress={0.45}
              sonarConfig={{
                pulseMode: 'DUAL_COMBINED',
                pulseSpeed: 1.4,
                pulseFrequency: 2.5,
                pulseIntensity: 1.5,
                swathWidth: 24.0,
                lastPingTimestamp: Date.now()
              }}
            />
          </div>
        </div>



        {/* QUADRANT 3 (BOTTOM-LEFT): GEOSPATIAL POSTGIS HAZARD MATRIX */}
        <div className="relative rounded-xl border border-cyan-900/40 bg-[#020712] overflow-hidden flex flex-col shadow-inner p-3 space-y-3">
          <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2">
            <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
              <MapPin className="w-4 h-4 text-emerald-400" />
              <span>Q3 // POSTGIS SPATIAL HAZARD CORRIDORS</span>
            </div>
            <GlassBadge variant="emerald" size="sm">
              EPSG:4326 WGS84
            </GlassBadge>
          </div>

          {/* Hazard Cards */}
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {hazardZones.map((hz) => (
              <div
                key={hz.hazard_id}
                className="p-3 rounded-xl bg-[#020817]/90 border border-cyan-900/40 hover:border-cyan-500/50 transition-all text-xs space-y-1.5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white font-mono">{hz.hazard_id}</span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      hz.threat_level === 'CRITICAL'
                        ? 'bg-red-950/70 border border-red-500/50 text-red-300 animate-pulse'
                        : 'bg-amber-950/70 border border-amber-500/50 text-amber-300'
                    }`}
                  >
                    {hz.threat_level} THREAT
                  </span>
                </div>
                <div className="text-[11px] text-slate-300 font-semibold">{hz.hazard_type.replace(/_/g, ' ')}</div>
                <p className="text-[10px] text-slate-400">
                  Center: {hz.center.lat.toFixed(4)}° N, {hz.center.lng.toFixed(4)}° E • Area: {hz.area_sq_meters.toLocaleString()} m² • Targets: {hz.target_count}
                </p>
                <div className="text-[10px] text-emerald-300 bg-emerald-950/30 p-1.5 rounded border border-emerald-500/20 font-mono">
                  ACTION: {hz.recommended_action}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* QUADRANT 4 (BOTTOM-RIGHT): REAL-TIME THREAT PRIORITY MATRIX */}
        <div className="relative rounded-xl border border-cyan-900/40 bg-[#020712] overflow-hidden flex flex-col shadow-inner p-3 space-y-3">
          <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2">
            <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
              <Crosshair className="w-4 h-4 text-cyan-400" />
              <span>Q4 // REAL-TIME THREAT CLASSIFICATION HUD</span>
            </div>
            <GlassBadge variant="cyan" size="sm">
              HYDOPHYS-OMNINET v4
            </GlassBadge>
          </div>

          {/* Critical Lock Banner */}
          {threatAlert && (
            <div className="p-3 rounded-xl bg-cyan-950/40 border border-cyan-500/40 flex items-center justify-between text-xs">
              <div className="space-y-0.5">
                <div className="text-white font-bold flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                  <span>{threatAlert.label}</span>
                </div>
                <div className="text-[10px] text-slate-400">GPS: {threatAlert.location}</div>
              </div>
              <div className="text-right">
                <div className="text-emerald-400 font-bold font-mono text-sm">
                  {(threatAlert.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500 uppercase">FUSED CONFIDENCE</div>
              </div>
            </div>
          )}

          {/* Quick Sliders */}
          <div className="p-3 rounded-xl bg-[#020817]/90 border border-cyan-900/40 space-y-2 text-xs">
            <div className="flex items-center justify-between text-[11px] font-bold text-slate-300">
              <span className="flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                <span>DSP GAIN & CONTRAST EQUALIZATION</span>
              </span>
              <span className="text-cyan-400 font-mono">{gain.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.5"
              step="0.1"
              value={gain}
              onChange={(e) => setGain(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
          </div>

          {/* System Certifications */}
          <div className="mt-auto p-2.5 rounded-xl bg-[#020712] border border-cyan-900/30 flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>EDGE COMPUTE: NVIDIA RTX 5060 (CUDA 12.8)</span>
            <span className="text-emerald-400 font-bold">IHO S-44 VERIFIED</span>
          </div>
        </div>
      </div>
    </div>
  );
};

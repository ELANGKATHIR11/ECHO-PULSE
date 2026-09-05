import React, { useState, useEffect, useRef } from 'react';
import {
  Compass, ShieldAlert, Crosshair, Radar, Navigation, MapPin,
  AlertOctagon, Volume2, VolumeX, Eye, Activity, Play, Pause,
  Layers, Lock, Wifi, RefreshCw, Radio, Anchor, Server
} from 'lucide-react';

interface TacticalTarget {
  id: string;
  callsign: string;
  classification: string;
  subclass: string;
  lat: number;
  lng: number;
  depth_m: number;
  speed_knots: number;
  heading_deg: number;
  range_m: number;
  relative_bearing_deg: number;
  threat_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  confidence: number;
  signal_to_noise_db: number;
  geofence_status: string;
  acoustic_signature: string;
  track_history: { lat: number; lng: number; depth_m: number }[];
}

export const AvsSurveillancePage: React.FC = () => {
  const [platform, setPlatform] = useState({
    name: 'AVS Ocean Sentinel Buoy #01',
    lat: 12.9822,
    lng: 80.2544,
    heading_deg: 45.0,
    depth_m: 12.0
  });

  const [targets, setTargets] = useState<TacticalTarget[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<TacticalTarget | null>(null);
  const [isSimulating, setIsSimulating] = useState(true);
  const [alarmActive, setAlarmActive] = useState(true);
  const [audioMuted, setAudioMuted] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [avsTelemetry, setAvsTelemetry] = useState({
    pressure_rms_pa: 14.8,
    velocity_ux_mps: 0.0034,
    velocity_uy_mps: 0.0058,
    velocity_uz_mps: -0.0012,
    doa_azimuth_deg: 58.4,
    doa_elevation_deg: -8.5,
    doa_confidence: 0.94,
    sound_speed_mps: 1512.0
  });

  const mapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Fetch initial tactical targets from API or initialize state
  useEffect(() => {
    fetchTargets();
    const interval = setInterval(() => {
      if (isSimulating) {
        fetchTargets();
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [isSimulating]);

  const fetchTargets = () => {
    fetch('http://127.0.0.1:8000/api/hydrophone/tactical-targets')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'SUCCESS') {
          setTargets(data.targets);
          if (!selectedTarget && data.targets.length > 0) {
            setSelectedTarget(data.targets[0]);
          }
        }
      })
      .catch(() => {
        // Fallback simulation
        const now = Date.now() / 1000;
        const mockTargets: TacticalTarget[] = [
          {
            id: 'TGT-AUV-089',
            callsign: 'Intruder Stealth AUV (Echo-9)',
            classification: 'Tactical Intruder',
            subclass: 'Autonomous Underwater Vehicle (AUV) Electric Propulsion',
            lat: 12.9915 + 0.002 * Math.sin(now * 0.1),
            lng: 80.2710 + 0.002 * Math.cos(now * 0.1),
            depth_m: 24.5,
            speed_knots: 6.8,
            heading_deg: 142.0,
            range_m: 1280.0,
            relative_bearing_deg: 58.4,
            threat_level: 'CRITICAL',
            confidence: 0.94,
            signal_to_noise_db: 18.2,
            geofence_status: 'BREACHED_HARBOR_DEFENSE_ZONE',
            acoustic_signature: '400Hz 3-Blade Harmonic Hum',
            track_history: [
              { lat: 12.9880, lng: 80.2680, depth_m: 22.0 },
              { lat: 12.9895, lng: 80.2692, depth_m: 23.5 },
              { lat: 12.9915, lng: 80.2710, depth_m: 24.5 }
            ]
          },
          {
            id: 'TGT-USV-041',
            callsign: 'Unmanned Surface Intruder (Vector-X)',
            classification: 'Tactical Intruder',
            subclass: 'Unmanned Surface Vehicle (USV) High-Speed Jet',
            lat: 13.0120 - 0.001 * Math.sin(now * 0.15),
            lng: 80.2840 - 0.001 * Math.cos(now * 0.15),
            depth_m: 0.5,
            speed_knots: 28.4,
            heading_deg: 215.0,
            range_m: 2850.0,
            relative_bearing_deg: 112.6,
            threat_level: 'HIGH',
            confidence: 0.89,
            signal_to_noise_db: 24.5,
            geofence_status: 'PERIMETER_WARNING',
            acoustic_signature: 'High-RPM Waterjet Impeller',
            track_history: [
              { lat: 13.0180, lng: 80.2910, depth_m: 0.5 },
              { lat: 13.0150, lng: 80.2875, depth_m: 0.5 },
              { lat: 13.0120, lng: 80.2840, depth_m: 0.5 }
            ]
          },
          {
            id: 'BIO-MAMMAL-012',
            callsign: 'Humpback Whale Pod Alpha',
            classification: 'Biophonic',
            subclass: 'Humpback Whale Song / Vocalization',
            lat: 12.9650,
            lng: 80.2980,
            depth_m: 45.0,
            speed_knots: 3.2,
            heading_deg: 80.0,
            range_m: 3950.0,
            relative_bearing_deg: 135.0,
            threat_level: 'LOW',
            confidence: 0.98,
            signal_to_noise_db: 22.0,
            geofence_status: 'OUTSIDE_PERIMETER',
            acoustic_signature: 'Low Frequency FM Whistle (350Hz-2.5kHz)',
            track_history: [
              { lat: 12.9620, lng: 80.2920, depth_m: 42.0 },
              { lat: 12.9650, lng: 80.2980, depth_m: 45.0 }
            ]
          }
        ];
        setTargets(mockTargets);
        if (!selectedTarget) setSelectedTarget(mockTargets[0]);
      });
  };

  // Render Real-Time Tactical Marine Map on Canvas
  useEffect(() => {
    if (!mapCanvasRef.current) return;
    const canvas = mapCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.parentElement?.clientWidth || 800;
    canvas.height = 540;

    const centerLat = platform.lat;
    const centerLng = platform.lng;
    const scale = 11000 * zoomLevel; // pixels per degree

    const toScreenX = (lng: number) => canvas.width / 2 + (lng - centerLng) * scale;
    const toScreenY = (lat: number) => canvas.height / 2 - (lat - centerLat) * scale;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Dark Nautical Sea Background
      const bgGrad = ctx.createRadialGradient(
        canvas.width / 2, canvas.height / 2, 50,
        canvas.width / 2, canvas.height / 2, canvas.width / 1.5
      );
      bgGrad.addColorStop(0, '#041026');
      bgGrad.addColorStop(1, '#020712');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // 2. Nautical Grid Lines
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.08)';
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // 3. Range Rings (1000m, 2000m, 3000m, 4000m)
      const platX = toScreenX(platform.lng);
      const platY = toScreenY(platform.lat);
      const metersToPixels = scale / 111320; // Approx meters per degree

      [1000, 2000, 3000, 4000].forEach(radiusMeters => {
        const rPx = radiusMeters * metersToPixels;
        ctx.beginPath();
        ctx.arc(platX, platY, rPx, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.15)';
        ctx.stroke();

        ctx.fillStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.font = '9px monospace';
        ctx.fillText(`${radiusMeters}m`, platX + rPx + 4, platY - 2);
      });

      // 4. Geofence Perimeter Zone (Harbor Defense)
      const fenceRadius = 1800 * metersToPixels;
      ctx.beginPath();
      ctx.arc(platX, platY, fenceRadius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(244, 63, 94, 0.03)';
      ctx.fill();

      // 5. DOA Acoustic Bearing Line & Beam Cone for Target 1
      if (selectedTarget) {
        const bearingRad = (selectedTarget.relative_bearing_deg - 90) * (Math.PI / 180);
        const tgtX = toScreenX(selectedTarget.lng);
        const tgtY = toScreenY(selectedTarget.lat);

        // Acoustic Beam Cone
        const spreadRad = 6 * (Math.PI / 180);
        ctx.beginPath();
        ctx.moveTo(platX, platY);
        ctx.arc(platX, platY, 4500 * metersToPixels, bearingRad - spreadRad, bearingRad + spreadRad);
        ctx.closePath();
        ctx.fillStyle = 'rgba(6, 182, 212, 0.08)';
        ctx.fill();

        // Line of Bearing (LOB)
        ctx.beginPath();
        ctx.moveTo(platX, platY);
        ctx.lineTo(tgtX, tgtY);
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 6. Draw Targets & Tracks
      targets.forEach(tgt => {
        const x = toScreenX(tgt.lng);
        const y = toScreenY(tgt.lat);

        // Breadcrumb track
        if (tgt.track_history && tgt.track_history.length > 1) {
          ctx.beginPath();
          tgt.track_history.forEach((pt, i) => {
            const hx = toScreenX(pt.lng);
            const hy = toScreenY(pt.lat);
            if (i === 0) ctx.moveTo(hx, hy);
            else ctx.lineTo(hx, hy);
          });
          ctx.strokeStyle = tgt.threat_level === 'CRITICAL' ? 'rgba(244, 63, 94, 0.5)' : 'rgba(6, 182, 212, 0.5)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Target Marker
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = tgt.threat_level === 'CRITICAL' ? '#f43f5e' : (tgt.threat_level === 'HIGH' ? '#f59e0b' : '#10b981');
        ctx.fill();

        // Pulse Ring for Critical Drone Target
        if (tgt.threat_level === 'CRITICAL') {
          ctx.beginPath();
          ctx.arc(x, y, 14, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.6)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Target Heading Vector
        const headRad = (tgt.heading_deg - 90) * (Math.PI / 180);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(headRad) * 22, y + Math.sin(headRad) * 22);
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#f8fafc';
        ctx.font = '10px monospace font-bold';
        ctx.fillText(tgt.callsign, x + 10, y - 8);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px monospace';
        ctx.fillText(`${tgt.speed_knots} kts • Depth: ${tgt.depth_m}m`, x + 10, y + 4);
      });

      // 7. Draw Platform Node (AVS Array)
      ctx.beginPath();
      ctx.arc(platX, platY, 9, 0, Math.PI * 2);
      ctx.fillStyle = '#06b6d4';
      ctx.fill();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.fillStyle = '#38bdf8';
      ctx.font = '11px monospace font-bold';
      ctx.fillText(`[AVS SENSOR PLATFORM] ${platform.name}`, platX + 14, platY - 10);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px monospace';
      ctx.fillText(`${platform.lat.toFixed(4)}°N, ${platform.lng.toFixed(4)}°E`, platX + 14, platY + 4);
    };

    render();
  }, [platform, targets, selectedTarget, zoomLevel]);

  return (
    <div className="min-h-screen bg-[#020712] text-slate-100 p-6 space-y-6">
      {/* Top Threat Banner */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-[#17050d] via-[#1f0b18] to-[#0b1329] border border-rose-500/30 shadow-2xl backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-rose-500/20 border border-rose-400/40 text-rose-400 animate-pulse">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-black tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-rose-400 via-amber-200 to-cyan-300">
                AVS-GeoPhysics-X SURVEILLANCE & GEO-LOCALIZATION
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-black bg-rose-500/20 text-rose-300 border border-rose-500/50">
                RETRAINED BEST (avs_geophysics_best.pt)
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono flex items-center gap-2">
              <span>PROBABILISTIC SPHERICAL 3D DOA</span>
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span>HETEROSCEDASTIC RANGE UNCERTAINTY</span>
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span className="text-cyan-400">WGS-84 GEODETIC CONVERGENCE (Loss: 1.3973)</span>
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsSimulating(!isSimulating)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-mono text-xs font-semibold transition-all ${
              isSimulating
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                : 'bg-slate-800 text-slate-400 border border-slate-700'
            }`}
          >
            {isSimulating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isSimulating ? 'LIVE FEED ACTIVE' : 'FEED PAUSED'}</span>
          </button>

          <button
            onClick={() => setAudioMuted(!audioMuted)}
            className="p-2.5 rounded-xl bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
          >
            {audioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4 text-cyan-400" />}
          </button>
        </div>
      </div>

      {/* Main Grid: Tactical Map (8 cols) & Telemetry Panel (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Tactical Marine Map */}
        <div className="lg:col-span-8 space-y-4">
          <div className="relative rounded-2xl bg-[#040e24] border border-cyan-900/40 shadow-2xl p-3 overflow-hidden">
            {/* Map Header HUD */}
            <div className="absolute top-6 left-6 z-10 flex items-center gap-4 bg-[#020712]/80 backdrop-blur-md px-4 py-2 rounded-xl border border-cyan-900/50">
              <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
                <Radar className="w-4 h-4 animate-spin" />
                <span>ARRAY: 4-CH AVS VECTOR SENSOR</span>
              </div>
              <span className="text-slate-600">|</span>
              <div className="text-xs font-mono text-slate-300">
                TARGETS TRACKED: <span className="text-cyan-400 font-bold">{targets.length}</span>
              </div>
            </div>

            {/* Zoom Controls */}
            <div className="absolute top-6 right-6 z-10 flex flex-col gap-1.5 bg-[#020712]/80 backdrop-blur-md p-1.5 rounded-xl border border-cyan-900/50">
              <button
                onClick={() => setZoomLevel(prev => Math.min(prev + 0.25, 2.5))}
                className="w-7 h-7 rounded bg-slate-800/80 hover:bg-cyan-500/20 text-cyan-300 flex items-center justify-center font-bold text-sm"
              >
                +
              </button>
              <button
                onClick={() => setZoomLevel(prev => Math.max(prev - 0.25, 0.5))}
                className="w-7 h-7 rounded bg-slate-800/80 hover:bg-cyan-500/20 text-cyan-300 flex items-center justify-center font-bold text-sm"
              >
                -
              </button>
            </div>

            {/* Tactical Canvas */}
            <canvas ref={mapCanvasRef} className="w-full h-[540px] block rounded-xl" />

            {/* Map Footer Compass HUD */}
            <div className="absolute bottom-6 left-6 z-10 flex items-center gap-6 bg-[#020712]/90 backdrop-blur-md px-4 py-2.5 rounded-xl border border-cyan-900/50 text-xs font-mono">
              <div className="flex items-center gap-2">
                <Compass className="w-4 h-4 text-cyan-400" />
                <span className="text-slate-400">BEARING (DOA):</span>
                <span className="text-cyan-300 font-bold">{avsTelemetry.doa_azimuth_deg}°</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400">ELEVATION:</span>
                <span className="text-teal-300 font-bold">{avsTelemetry.doa_elevation_deg}°</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400">SOUND SPEED:</span>
                <span className="text-indigo-300 font-bold">{avsTelemetry.sound_speed_mps} m/s</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Intruder Telemetry & Kinematics (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Target List Tabs */}
          <div className="space-y-2">
            <span className="text-xs font-mono text-slate-400 tracking-wider">
              ACTIVE DETECTED CONTACTS:
            </span>
            <div className="space-y-2">
              {targets.map(tgt => (
                <div
                  key={tgt.id}
                  onClick={() => setSelectedTarget(tgt)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all flex items-center justify-between ${
                    selectedTarget?.id === tgt.id
                      ? (tgt.threat_level === 'CRITICAL' ? 'bg-rose-950/40 border-rose-500/80 shadow-lg shadow-rose-950/30' : 'bg-cyan-950/40 border-cyan-500/80 shadow-lg shadow-cyan-950/30')
                      : 'bg-[#040e24] hover:bg-[#07173b] border-slate-800/80'
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold text-slate-200">
                        {tgt.callsign}
                      </span>
                    </div>
                    <div className="text-[10px] font-mono text-slate-400">
                      {tgt.subclass}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                      tgt.threat_level === 'CRITICAL' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}>
                      {tgt.threat_level}
                    </span>
                    <div className="text-[10px] font-mono text-cyan-400 mt-1">
                      {tgt.range_m}m • {tgt.speed_knots}kts
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Selected Target Detailed Telemetry */}
          {selectedTarget && (
            <div className="p-5 rounded-2xl bg-[#040e24] border border-cyan-900/40 shadow-xl space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Crosshair className="w-5 h-5 text-rose-400" />
                  <h2 className="text-sm font-mono font-bold tracking-wider text-slate-100">
                    TARGET KINEMATICS & GEO-POSITION
                  </h2>
                </div>
                <span className="text-xs font-mono text-rose-400 font-bold">
                  {selectedTarget.id}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-[#020614] border border-slate-800">
                  <div className="text-[10px] font-mono text-slate-400">ESTIMATED RANGE</div>
                  <div className="text-lg font-mono font-bold text-cyan-300">
                    {selectedTarget.range_m} m
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    Bearing: {selectedTarget.relative_bearing_deg}°
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[#020614] border border-slate-800">
                  <div className="text-[10px] font-mono text-slate-400">SPEED / DEPTH</div>
                  <div className="text-lg font-mono font-bold text-teal-300">
                    {selectedTarget.speed_knots} kts
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    Depth: {selectedTarget.depth_m} m
                  </div>
                </div>

                <div className="col-span-2 p-3 rounded-xl bg-[#020614] border border-slate-800">
                  <div className="text-[10px] font-mono text-slate-400">WGS-84 TARGET GPS</div>
                  <div className="text-sm font-mono font-bold text-indigo-300">
                    {selectedTarget.lat.toFixed(6)}° N, {selectedTarget.lng.toFixed(6)}° E
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    Heading: {selectedTarget.heading_deg}° • SNR: {selectedTarget.signal_to_noise_db} dB
                  </div>
                </div>

                <div className="col-span-2 p-3 rounded-xl bg-[#020614] border border-slate-800">
                  <div className="text-[10px] font-mono text-slate-400">ACOUSTIC SIGNATURE</div>
                  <div className="text-xs font-mono font-bold text-amber-300">
                    {selectedTarget.acoustic_signature}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

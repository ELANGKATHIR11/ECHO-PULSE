import React, { useState, useEffect, useMemo } from 'react';
import {
  MapContainer,
  TileLayer,
  Polyline,
  Polygon,
  Marker,
  Popup,
  useMap,
  CircleMarker,
} from 'react-leaflet';
import L from 'leaflet';
import { Mission, Detection } from '../../types';
import { formatDMS } from '../../utils/sonarProcessor';
import {
  generateAggregatedMissionHeatmap,
  HeatmapMetricMode,
  HeatmapScope,
  PingDensityNode,
  HistoricalHeatmapStats,
} from '../../utils/heatmapUtils';
import {
  Layers,
  Compass,
  AlertCircle,
  Flame,
  Radio,
  Sliders,
  Maximize2,
  Minimize2,
  Calendar,
  Zap,
  Info,
  CheckSquare,
  Square,
} from 'lucide-react';

interface MissionMapProps {
  mission: Mission;
  allMissions?: Mission[];
  detections?: Detection[];
  selectedDetectionId?: string | null;
  onSelectDetection?: (detection: Detection) => void;
  className?: string;
  showLayersControl?: boolean;
  initialHeatmapActive?: boolean;
}

// Custom Leaflet Icons for Sonar Detections
function createCustomIcon(detClass: string, isSelected: boolean) {
  let color = '#22d3ee';
  let label = 'DET';

  if (detClass === 'ghost_gear') {
    color = '#f59e0b';
    label = 'NET';
  } else if (detClass === 'shipwreck') {
    color = '#ec4899';
    label = 'WRECK';
  } else if (detClass === 'unexploded_ordnance') {
    color = '#ef4444';
    label = 'UXO';
  } else if (detClass === 'pipeline_anomaly') {
    color = '#8b5cf6';
    label = 'PIPE';
  } else if (detClass === 'biological_cluster') {
    color = '#10b981';
    label = 'REEF';
  }

  const border = isSelected
    ? 'border-2 border-white scale-125 shadow-[0_0_12px_#22D3EE]'
    : 'border border-cyan-900/60 shadow-md';

  return L.divIcon({
    className: 'custom-sonar-marker',
    html: `
      <div style="background-color: ${color}; transform: translate(-50%, -50%);" 
           class="w-6 h-6 rounded-full flex items-center justify-center font-mono font-bold text-[9px] text-slate-950 ${border} transition-transform">
        ${label}
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
}

// Controller component to zoom to bounds or selected marker
function MapController({
  center,
  zoom,
  bounds,
}: {
  center?: [number, number];
  zoom?: number;
  bounds?: L.LatLngBoundsExpression;
}) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [40, 40] });
    } else if (center) {
      map.setView(center, zoom || 14);
    }
  }, [center, zoom, bounds, map]);

  return null;
}

export const MissionMap: React.FC<MissionMapProps> = ({
  mission,
  allMissions = [],
  detections = [],
  selectedDetectionId,
  onSelectDetection,
  className = 'h-full w-full min-h-[400px]',
  showLayersControl = true,
  initialHeatmapActive = true,
}) => {
  // Layer visibility toggles
  const [activeLayers, setActiveLayers] = useState({
    corridor: true,
    route: true,
    detections: true,
    historicalHeatmap: initialHeatmapActive,
    allHistoricalSwaths: false,
  });

  // Heatmap customization state
  const [heatmapScope, setHeatmapScope] = useState<HeatmapScope>('ALL_HISTORICAL');
  const [heatmapMetric, setHeatmapMetric] = useState<HeatmapMetricMode>('PING_DENSITY');
  const [kernelBandwidth, setKernelBandwidth] = useState<number>(80);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.65);
  const [showHeatmapSettings, setShowHeatmapSettings] = useState<boolean>(false);
  const [selectedMissionsForHeatmap, setSelectedMissionsForHeatmap] = useState<string[]>(
    allMissions.map((m) => m.id)
  );

  const [offlineMode, setOfflineMode] = useState(false);
  const [selectedHeatNode, setSelectedHeatNode] = useState<PingDensityNode | null>(null);

  // Filter missions by user selection
  const filteredMissionsForHeatmap = useMemo(() => {
    return allMissions.filter((m) => selectedMissionsForHeatmap.includes(m.id));
  }, [allMissions, selectedMissionsForHeatmap]);

  // Generate Aggregated Mission Heatmap data
  const { nodes: heatNodes, stats: heatmapStats } = useMemo(() => {
    return generateAggregatedMissionHeatmap(
      filteredMissionsForHeatmap,
      mission.id,
      heatmapScope,
      heatmapMetric,
      kernelBandwidth
    );
  }, [filteredMissionsForHeatmap, mission.id, heatmapScope, heatmapMetric, kernelBandwidth]);

  // Track coordinates for current active mission
  const routePositions = useMemo<[number, number][]>(() => {
    if (!mission.trackPoints || mission.trackPoints.length === 0) {
      return [mission.coordinates];
    }
    return mission.trackPoints.map((tp) => [tp.latitude, tp.longitude]);
  }, [mission]);

  // Compute Sonar Swath Coverage Polygon for current mission
  const coveragePolygon = useMemo<[number, number][]>(() => {
    if (routePositions.length < 2) return [];
    const offset = ((mission.coverageCorridorWidthMeters || 200) / 111320) * 0.5;

    const leftSide: [number, number][] = routePositions.map(([lat, lng]) => [
      lat + offset,
      lng - offset * 0.8,
    ]);
    const rightSide: [number, number][] = [...routePositions]
      .reverse()
      .map(([lat, lng]) => [lat - offset, lng + offset * 0.8]);

    return [...leftSide, ...rightSide];
  }, [routePositions, mission.coverageCorridorWidthMeters]);

  // Compute historical swaths for other missions
  const historicalSwaths = useMemo(() => {
    return allMissions
      .filter((m) => m.id !== mission.id)
      .map((m) => {
        if (!m.trackPoints || m.trackPoints.length < 2) return null;
        const pts: [number, number][] = m.trackPoints.map((tp) => [tp.latitude, tp.longitude]);
        const offset = ((m.coverageCorridorWidthMeters || 180) / 111320) * 0.5;
        const left: [number, number][] = pts.map(([lat, lng]) => [lat + offset, lng - offset * 0.8]);
        const right: [number, number][] = [...pts].reverse().map(([lat, lng]) => [lat - offset, lng + offset * 0.8]);
        return {
          mission: m,
          polygon: [...left, ...right],
          route: pts,
        };
      })
      .filter(Boolean);
  }, [allMissions, mission.id]);

  // Map Bounds calculation
  const mapBounds = useMemo<L.LatLngBoundsExpression>(() => {
    const latLngs: [number, number][] = [...routePositions];
    detections.forEach((d) => {
      if (d.latitude !== null && d.longitude !== null) {
        latLngs.push([d.latitude, d.longitude]);
      }
    });
    if (latLngs.length === 0) {
      return [
        [mission.coordinates[0] - 0.05, mission.coordinates[1] - 0.05],
        [mission.coordinates[0] + 0.05, mission.coordinates[1] + 0.05],
      ];
    }
    return L.latLngBounds(latLngs.map(([lat, lng]) => L.latLng(lat, lng))).pad(0.25);
  }, [routePositions, detections, mission.coordinates]);

  const toggleMissionSelection = (mId: string) => {
    if (selectedMissionsForHeatmap.includes(mId)) {
      if (selectedMissionsForHeatmap.length > 1) {
        setSelectedMissionsForHeatmap(selectedMissionsForHeatmap.filter((id) => id !== mId));
      }
    } else {
      setSelectedMissionsForHeatmap([...selectedMissionsForHeatmap, mId]);
    }
  };

  return (
    <div className={`relative overflow-hidden rounded-lg border border-cyan-900/30 bg-[#02060C] ${className}`}>
      {/* Top Map HUD Status Overlay */}
      <div className="absolute top-3 left-3 z-[1000] flex flex-wrap items-center gap-2 bg-[#050B14]/90 backdrop-blur-md border border-cyan-900/40 px-3 py-1.5 rounded-sm text-xs font-mono shadow-xl">
        <div className="flex items-center gap-2">
          <Compass className="w-3.5 h-3.5 text-cyan-400 animate-spin" style={{ animationDuration: '16s' }} />
          <span className="text-white font-bold">{mission.name}</span>
        </div>
        <span className="text-slate-600 hidden sm:inline">|</span>
        <span className="text-cyan-400 hidden sm:inline">
          {formatDMS(mission.coordinates[0], true)}, {formatDMS(mission.coordinates[1], false)}
        </span>
        <span className="text-slate-600">|</span>
        <span className="text-emerald-400 font-semibold">{detections.length} TARGETS</span>

        {activeLayers.historicalHeatmap && (
          <>
            <span className="text-slate-600">|</span>
            <span className="flex items-center gap-1 text-[11px] font-bold text-amber-400 bg-amber-950/40 px-1.5 py-0.5 rounded border border-amber-500/30">
              <Flame className="w-3 h-3 text-amber-400" />
              HEATMAP: {heatmapStats.totalAggregatedPings.toLocaleString()} PINGS
            </span>
          </>
        )}
      </div>

      {/* Layer Control & Heatmap Toggle Toolbar */}
      {showLayersControl && (
        <div className="absolute top-3 right-3 z-[1000] flex flex-col gap-2">
          {/* Main Layer Widget */}
          <div className="bg-[#050B14]/95 backdrop-blur-md border border-cyan-900/40 p-2.5 rounded-sm flex flex-col gap-1.5 text-[11px] font-mono shadow-2xl min-w-[200px]">
            <div className="flex items-center justify-between text-slate-400 font-bold px-1 pb-1 border-b border-cyan-900/30">
              <div className="flex items-center gap-1.5 text-cyan-300">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span>GIS LAYERS</span>
              </div>
              <button
                onClick={() => setShowHeatmapSettings(!showHeatmapSettings)}
                className={`p-1 rounded text-[10px] transition-colors ${
                  showHeatmapSettings
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-cyan-400'
                }`}
                title="Configure Mission Coverage Heatmap"
              >
                <Sliders className="w-3 h-3" />
              </button>
            </div>

            {/* Heatmap Layer Master Toggle */}
            <label className="flex items-center gap-2 px-1 py-1 text-cyan-300 bg-cyan-950/30 border border-cyan-500/30 rounded-sm hover:bg-cyan-950/50 cursor-pointer font-bold transition-all">
              <input
                type="checkbox"
                checked={activeLayers.historicalHeatmap}
                onChange={(e) =>
                  setActiveLayers({ ...activeLayers, historicalHeatmap: e.target.checked })
                }
                className="accent-cyan-400 rounded cursor-pointer"
              />
              <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider">
                <Flame className="w-3 h-3 text-amber-400" />
                Coverage Heatmap
              </span>
            </label>

            {/* Other Standard Layers */}
            <label className="flex items-center gap-2 px-1 py-0.5 text-slate-300 hover:text-white cursor-pointer">
              <input
                type="checkbox"
                checked={activeLayers.corridor}
                onChange={(e) => setActiveLayers({ ...activeLayers, corridor: e.target.checked })}
                className="accent-cyan-400 rounded cursor-pointer"
              />
              <span className="text-[10px]">Active Swath ({mission.coverageCorridorWidthMeters || 200}m)</span>
            </label>

            <label className="flex items-center gap-2 px-1 py-0.5 text-slate-300 hover:text-white cursor-pointer">
              <input
                type="checkbox"
                checked={activeLayers.route}
                onChange={(e) => setActiveLayers({ ...activeLayers, route: e.target.checked })}
                className="accent-cyan-400 rounded cursor-pointer"
              />
              <span className="text-[10px]">AUV Track & Waypoints</span>
            </label>

            <label className="flex items-center gap-2 px-1 py-0.5 text-slate-300 hover:text-white cursor-pointer">
              <input
                type="checkbox"
                checked={activeLayers.detections}
                onChange={(e) => setActiveLayers({ ...activeLayers, detections: e.target.checked })}
                className="accent-cyan-400 rounded cursor-pointer"
              />
              <span className="text-[10px]">Target Pins ({detections.length})</span>
            </label>

            <label className="flex items-center gap-2 px-1 py-0.5 text-slate-300 hover:text-white cursor-pointer">
              <input
                type="checkbox"
                checked={activeLayers.allHistoricalSwaths}
                onChange={(e) =>
                  setActiveLayers({ ...activeLayers, allHistoricalSwaths: e.target.checked })
                }
                className="accent-cyan-400 rounded cursor-pointer"
              />
              <span className="text-[10px]">Historical Swath Corridors</span>
            </label>
          </div>

          {/* Expanded Heatmap Advanced Settings Panel */}
          {showHeatmapSettings && activeLayers.historicalHeatmap && (
            <div className="bg-[#050B14]/95 backdrop-blur-md border border-cyan-500/30 p-3 rounded-sm flex flex-col gap-2.5 text-[11px] font-mono shadow-2xl min-w-[240px]">
              <div className="flex items-center justify-between border-b border-cyan-900/30 pb-1.5">
                <span className="font-bold text-white text-[10px] uppercase tracking-wider flex items-center gap-1.5">
                  <Flame className="w-3.5 h-3.5 text-amber-400" />
                  HEATMAP SETTINGS
                </span>
                <button
                  onClick={() => setShowHeatmapSettings(false)}
                  className="text-slate-400 hover:text-white text-[10px]"
                >
                  ✕
                </button>
              </div>

              {/* Aggregation Scope */}
              <div>
                <span className="text-[9px] uppercase tracking-wider text-slate-500 block mb-1">
                  Aggregation Scope
                </span>
                <div className="grid grid-cols-2 gap-1">
                  <button
                    onClick={() => setHeatmapScope('ALL_HISTORICAL')}
                    className={`px-2 py-1 rounded-sm text-[9px] uppercase font-bold transition-all ${
                      heatmapScope === 'ALL_HISTORICAL'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400'
                        : 'bg-[#02060C] text-slate-400 border border-cyan-900/30 hover:text-slate-200'
                    }`}
                  >
                    All Historical ({allMissions.length})
                  </button>
                  <button
                    onClick={() => setHeatmapScope('ACTIVE_ONLY')}
                    className={`px-2 py-1 rounded-sm text-[9px] uppercase font-bold transition-all ${
                      heatmapScope === 'ACTIVE_ONLY'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400'
                        : 'bg-[#02060C] text-slate-400 border border-cyan-900/30 hover:text-slate-200'
                    }`}
                  >
                    Active Only
                  </button>
                </div>
              </div>

              {/* Metric Type */}
              <div>
                <span className="text-[9px] uppercase tracking-wider text-slate-500 block mb-1">
                  Density Metric
                </span>
                <select
                  value={heatmapMetric}
                  onChange={(e) => setHeatmapMetric(e.target.value as HeatmapMetricMode)}
                  className="w-full bg-[#02060C] border border-cyan-900/40 rounded px-2 py-1 text-[10px] text-cyan-300 font-semibold focus:outline-none focus:border-cyan-400"
                >
                  <option value="PING_DENSITY">Ping Density (Pings/km²)</option>
                  <option value="SWATH_OVERLAP">Swath Overlap Multiplier</option>
                  <option value="ANOMALY_RATE">Anomaly Anomaly Rate</option>
                  <option value="FREQUENCY_DIST">Acoustic Frequency (kHz)</option>
                </select>
              </div>

              {/* Bandwidth / Radius */}
              <div>
                <div className="flex justify-between text-[9px] uppercase text-slate-400 mb-0.5">
                  <span>Kernel Bandwidth</span>
                  <span className="text-cyan-300 font-bold">{kernelBandwidth}m</span>
                </div>
                <input
                  type="range"
                  min="30"
                  max="150"
                  step="10"
                  value={kernelBandwidth}
                  onChange={(e) => setKernelBandwidth(Number(e.target.value))}
                  className="w-full accent-cyan-400 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                />
              </div>

              {/* Opacity */}
              <div>
                <div className="flex justify-between text-[9px] uppercase text-slate-400 mb-0.5">
                  <span>Layer Opacity</span>
                  <span className="text-cyan-300 font-bold">{Math.round(heatmapOpacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="1.0"
                  step="0.05"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                  className="w-full accent-cyan-400 h-1 bg-slate-800 rounded appearance-none cursor-pointer"
                />
              </div>

              {/* Historical Missions Checklist */}
              {heatmapScope === 'ALL_HISTORICAL' && (
                <div className="border-t border-cyan-900/30 pt-2">
                  <span className="text-[9px] uppercase tracking-wider text-slate-500 block mb-1.5">
                    Include Missions:
                  </span>
                  <div className="space-y-1 max-h-28 overflow-y-auto pr-1">
                    {allMissions.map((m) => {
                      const isChecked = selectedMissionsForHeatmap.includes(m.id);
                      return (
                        <div
                          key={m.id}
                          onClick={() => toggleMissionSelection(m.id)}
                          className="flex items-center justify-between px-1.5 py-1 bg-[#02060C] hover:bg-cyan-950/30 rounded border border-cyan-900/20 cursor-pointer text-[9px]"
                        >
                          <div className="flex items-center gap-1.5 truncate max-w-[170px]">
                            {isChecked ? (
                              <CheckSquare className="w-3 h-3 text-cyan-400 shrink-0" />
                            ) : (
                              <Square className="w-3 h-3 text-slate-600 shrink-0" />
                            )}
                            <span className={isChecked ? 'text-slate-200 font-semibold' : 'text-slate-500'}>
                              {m.id}
                            </span>
                          </div>
                          <span className="text-slate-500 font-mono">
                            {(m.pingCount / 1000).toFixed(0)}k
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Heatmap Color Scale Legend & Aggregate Summary HUD */}
      {activeLayers.historicalHeatmap && (
        <div className="absolute bottom-3 right-3 z-[1000] bg-[#050B14]/90 backdrop-blur-md border border-cyan-900/40 p-2.5 rounded-sm font-mono text-[10px] shadow-2xl flex flex-col gap-2 max-w-[320px]">
          <div className="flex items-center justify-between border-b border-cyan-900/30 pb-1.5">
            <span className="font-bold text-white text-[9px] uppercase tracking-wider flex items-center gap-1">
              <Flame className="w-3 h-3 text-amber-400" />
              COVERAGE PING DENSITY (PINGS / KM²)
            </span>
            <span className="text-[9px] text-cyan-400 font-bold">
              {heatmapStats.totalCoverageAreaSqKm} km²
            </span>
          </div>

          {/* Color Gradient Bar */}
          <div className="space-y-1">
            <div className="h-2 w-full rounded-sm bg-gradient-to-r from-[#06b6d4] via-[#10b981] via-[#f59e0b] to-[#f43f5e] shadow-inner" />
            <div className="flex justify-between text-[8px] text-slate-400 uppercase">
              <span>&lt;10k (Base)</span>
              <span>20k (Dense)</span>
              <span>35k (Multi)</span>
              <span className="text-rose-400 font-bold">&gt;45k (Ultra)</span>
            </div>
          </div>

          {/* Aggregated Quick Metrics */}
          <div className="grid grid-cols-3 gap-1.5 pt-1 border-t border-cyan-900/20 text-center text-[9px]">
            <div className="bg-[#02060C] p-1 rounded border border-cyan-900/30">
              <div className="text-slate-500 text-[8px] uppercase">Aggregated</div>
              <div className="text-cyan-400 font-bold">{heatmapStats.totalAggregatedPings.toLocaleString()}</div>
            </div>
            <div className="bg-[#02060C] p-1 rounded border border-cyan-900/30">
              <div className="text-slate-500 text-[8px] uppercase">Surveys</div>
              <div className="text-white font-bold">{heatmapStats.totalMissionsIncluded} Missions</div>
            </div>
            <div className="bg-[#02060C] p-1 rounded border border-cyan-900/30">
              <div className="text-slate-500 text-[8px] uppercase">Avg Overlap</div>
              <div className="text-amber-400 font-bold">{heatmapStats.avgOverlapIndex}x Pass</div>
            </div>
          </div>
        </div>
      )}

      {/* Offline Mode Alert Indicator */}
      {offlineMode && (
        <div className="absolute bottom-3 left-3 z-[1000] bg-amber-950/80 border border-amber-500/50 text-amber-300 px-3 py-1 rounded text-xs font-mono flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
          <span>OFFLINE GIS CACHE ACTIVE — Local Bathymetry Rendered</span>
        </div>
      )}

      {/* Leaflet Map Container */}
      <MapContainer
        center={mission.coordinates}
        zoom={14}
        bounds={mapBounds}
        scrollWheelZoom={true}
        className="h-full w-full bg-[#02060C]"
      >
        <MapController bounds={mapBounds} />

        {/* Watermark-Free Esri World Ocean Bathymetry Tiles */}
        <TileLayer
          attribution='Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, CSUMB'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean/MapServer/tile/{z}/{y}/{x}"
          maxZoom={18}
          eventHandlers={{
            tileerror: () => setOfflineMode(true),
          }}
        />

        {/* Historical Swaths Corridors Layer */}
        {activeLayers.allHistoricalSwaths &&
          historicalSwaths.map((hs, idx) => {
            if (!hs) return null;
            return (
              <React.Fragment key={`hist-swath-${hs.mission.id}`}>
                <Polygon
                  positions={hs.polygon}
                  pathOptions={{
                    color: '#0284c7',
                    weight: 1,
                    dashArray: '2, 4',
                    fillColor: '#0369a1',
                    fillOpacity: 0.1,
                  }}
                />
                <Polyline
                  positions={hs.route}
                  pathOptions={{
                    color: '#0284c7',
                    weight: 1.5,
                    dashArray: '4, 4',
                    opacity: 0.6,
                  }}
                />
              </React.Fragment>
            );
          })}

        {/* Acoustic Coverage Swath Corridor (Current Active Mission) */}
        {activeLayers.corridor && coveragePolygon.length > 2 && (
          <Polygon
            positions={coveragePolygon}
            pathOptions={{
              color: '#00f0ff',
              weight: 1.5,
              dashArray: '3, 6',
              fillColor: '#00f0ff',
              fillOpacity: 0.12,
            }}
          />
        )}

        {/* Survey Track Polyline (Current Mission) */}
        {activeLayers.route && routePositions.length > 1 && (
          <>
            <Polyline
              positions={routePositions}
              pathOptions={{
                color: '#38bdf8',
                weight: 3,
                opacity: 0.9,
              }}
            />
            {/* Track waypoints */}
            {mission.trackPoints.map((tp, idx) => (
              <CircleMarker
                key={idx}
                center={[tp.latitude, tp.longitude]}
                radius={tp.hasAnomaly ? 5 : 3}
                pathOptions={{
                  color: tp.hasAnomaly ? '#f59e0b' : '#38bdf8',
                  fillColor: tp.hasAnomaly ? '#f59e0b' : '#0284c7',
                  fillOpacity: 0.9,
                  weight: 1,
                }}
              >
                <Popup>
                  <div className="font-mono text-xs p-1 space-y-1">
                    <div className="font-bold text-cyan-300">Ping #{tp.pingIndex}</div>
                    <div className="text-slate-300">
                      Depth: {tp.depthMeters}m | Alt: {tp.altitudeMeters}m
                    </div>
                    <div className="text-slate-400">
                      Heading: {tp.headingDeg}° | Spd: {tp.speedKnots} kn
                    </div>
                    <div className="text-slate-400">{tp.timestamp}</div>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </>
        )}

        {/* MISSION COVERAGE HEATMAP LAYER (Aggregated Ping Density) */}
        {activeLayers.historicalHeatmap &&
          heatNodes.map((node) => {
            // Visual circle radius based on kernel bandwidth and density
            const pixelRadius = Math.max(16, Math.min(48, Math.round(kernelBandwidth * 0.35)));

            return (
              <CircleMarker
                key={node.id}
                center={[node.lat, node.lng]}
                radius={pixelRadius}
                pathOptions={{
                  color: node.densityLevel >= 3 ? node.color : 'transparent',
                  weight: node.densityLevel === 4 ? 2 : 1,
                  fillColor: node.color,
                  fillOpacity: node.fillOpacity * heatmapOpacity,
                }}
                eventHandlers={{
                  click: () => setSelectedHeatNode(node),
                }}
              >
                <Popup>
                  <div className="font-mono text-xs p-1.5 space-y-2 min-w-[240px]">
                    <div className="flex items-center justify-between border-b border-cyan-500/30 pb-1">
                      <span className="font-bold text-white flex items-center gap-1 text-[11px]">
                        <Flame className="w-3.5 h-3.5 text-amber-400" />
                        PING DENSITY CELL
                      </span>
                      <span
                        className="text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider"
                        style={{
                          backgroundColor: `${node.color}25`,
                          color: node.color,
                          border: `1px solid ${node.color}50`,
                        }}
                      >
                        LEVEL {node.densityLevel} ({(node.intensity * 100).toFixed(0)}%)
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                      <div className="bg-[#050B14] p-1.5 rounded border border-cyan-900/30">
                        <span className="text-slate-500 uppercase text-[8px] block">Aggregated Pings</span>
                        <span className="text-cyan-300 font-bold text-xs">
                          {node.pingCount.toLocaleString()}
                        </span>
                      </div>
                      <div className="bg-[#050B14] p-1.5 rounded border border-cyan-900/30">
                        <span className="text-slate-500 uppercase text-[8px] block">Swath Overlap</span>
                        <span className="text-amber-400 font-bold text-xs">
                          {node.overlapMultiplier}x Coverage
                        </span>
                      </div>
                      <div className="bg-[#050B14] p-1.5 rounded border border-cyan-900/30">
                        <span className="text-slate-500 uppercase text-[8px] block">Density Rate</span>
                        <span className="text-white font-bold">
                          {node.pingsPerSqKm.toLocaleString()}/km²
                        </span>
                      </div>
                      <div className="bg-[#050B14] p-1.5 rounded border border-cyan-900/30">
                        <span className="text-slate-500 uppercase text-[8px] block">Mean Acoustic SNR</span>
                        <span className="text-emerald-400 font-bold">
                          {node.estimatedSnrDb} dB
                        </span>
                      </div>
                    </div>

                    {/* Contributing Historical Missions */}
                    <div className="border-t border-cyan-900/30 pt-1.5">
                      <span className="text-[9px] uppercase tracking-wider text-slate-400 block mb-1">
                        Contributing Surveys ({node.contributingMissions.length}):
                      </span>
                      <div className="space-y-1">
                        {node.contributingMissions.map((cm) => (
                          <div
                            key={cm.id}
                            className="bg-[#050B14] p-1 rounded border border-cyan-900/20 flex items-center justify-between text-[9px]"
                          >
                            <div>
                              <div className="text-cyan-300 font-bold">{cm.id}</div>
                              <div className="text-slate-500 text-[8px]">{cm.sonarSource} ({cm.frequencyKhz} kHz)</div>
                            </div>
                            <span className="text-slate-300 font-mono font-bold">
                              {cm.pingsInCell.toLocaleString()} pings
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="text-[9px] text-slate-500 pt-1 border-t border-cyan-900/20">
                      Coordinates: {formatDMS(node.lat, true)}, {formatDMS(node.lng, false)}
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

        {/* Detections Target Markers */}
        {activeLayers.detections &&
          detections.map((det, idx) => {
            if (det.latitude === null || det.longitude === null) return null;
            const isSelected = det.id === selectedDetectionId;

            return (
              <Marker
                key={`${det.id}-${idx}`}
                position={[det.latitude, det.longitude]}
                icon={createCustomIcon(det.class, isSelected)}
                eventHandlers={{
                  click: () => onSelectDetection && onSelectDetection(det),
                }}
              >
                <Popup>
                  <div className="font-mono text-xs p-1 space-y-1.5 min-w-[200px]">
                    <div className="flex items-center justify-between border-b border-cyan-500/30 pb-1">
                      <span className="font-bold text-cyan-300">{det.id}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 font-semibold">
                        {(det.confidence * 100).toFixed(1)}% CONF
                      </span>
                    </div>

                    <div className="font-sans font-semibold text-slate-100 text-[13px]">
                      {det.classNameLabel}
                    </div>

                    <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-300 pt-1">
                      <div>
                        <span className="text-slate-400">Depth:</span> {det.depthMeters}m
                      </div>
                      <div>
                        <span className="text-slate-400">Slant:</span> {det.slantRangeMeters}m
                      </div>
                      <div>
                        <span className="text-slate-400">Shadow:</span>{' '}
                        {det.acousticShadow ? `${det.acousticShadow.lengthMeters}m` : 'None'}
                      </div>
                      <div>
                        <span className="text-slate-400">Anomaly:</span>{' '}
                        {(det.anomalyScore * 100).toFixed(0)}%
                      </div>
                    </div>

                    <div className="text-[10px] text-slate-400 pt-1 border-t border-[#1a314d]">
                      <div>
                        {formatDMS(det.latitude, true)}, {formatDMS(det.longitude, false)}
                      </div>
                      <div>Model: {det.modelVersion}</div>
                    </div>
                  </div>
                </Popup>
              </Marker>
            );
          })}
      </MapContainer>
    </div>
  );
};

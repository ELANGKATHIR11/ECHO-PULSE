import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  CircleMarker,
  Tooltip,
  ScaleControl,
  useMap
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  Database,
  Activity,
  Server,
  RefreshCw,
  MapPin,
  Compass,
  Layers,
  Radio,
  Search,
  CheckCircle2,
  AlertTriangle,
  Download,
  Crosshair,
  Shield,
  Clock,
  Sparkles,
  Zap,
  Globe,
  Table as TableIcon
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../components/glass/GlassCard';
import { sensorFusion, SystemGpsState } from '../services/sensorFusionService';
import { useTheme } from '../context/ThemeContext';

// Fix Leaflet Default Marker Icon
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

interface PostgresRecord {
  id: string;
  mission_id: string;
  mission_name?: string;
  target_class: string;
  class_name_label: string;
  confidence: number;
  detector_score?: number;
  latitude: number;
  longitude: number;
  depth_meters?: number;
  slant_range_meters?: number;
  verification_status?: string;
  operator_notes?: string;
  model_version?: string;
  created_at?: string;
}

interface PostgisStatus {
  postgis_enabled: boolean;
  connected: boolean;
  database_url: string;
  driver: string;
  spatial_ref_system: string;
  total_records_count: number;
  last_synced: string;
}

const CATEGORY_COLORS: Record<string, { hex: string; bg: string; border: string }> = {
  ghost_gear: { hex: '#2ecc71', bg: 'bg-emerald-500/20', border: 'border-emerald-500' },
  shipwreck: { hex: '#e67e22', bg: 'bg-amber-500/20', border: 'border-amber-500' },
  unexploded_ordnance: { hex: '#e74c3c', bg: 'bg-rose-500/20', border: 'border-rose-500' },
  pipeline_anomaly: { hex: '#3498db', bg: 'bg-blue-500/20', border: 'border-blue-500' },
  marine_debris: { hex: '#9b59b6', bg: 'bg-purple-500/20', border: 'border-purple-500' },
  subsea_cable: { hex: '#f1c40f', bg: 'bg-yellow-500/20', border: 'border-yellow-500' },
  plastic: { hex: '#06b6d4', bg: 'bg-cyan-500/20', border: 'border-cyan-500' },
  default: { hex: '#06b6d4', bg: 'bg-cyan-500/20', border: 'border-cyan-500' }
};

// Custom Marker Icon
const createMarkerIcon = (targetClass: string, isSelected: boolean) => {
  const norm = targetClass.toLowerCase();
  const color = CATEGORY_COLORS[norm]?.hex || CATEGORY_COLORS.default.hex;
  const size = isSelected ? 34 : 26;

  return L.divIcon({
    className: 'custom-postgres-marker-icon',
    html: `
      <div style="
        width: ${size}px;
        height: ${size}px;
        background: radial-gradient(circle, ${color} 45%, #020712 95%);
        border: 2px solid ${color};
        border-radius: 50%;
        box-shadow: 0 0 14px ${color}, inset 0 0 6px ${color};
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="width: 6px; height: 6px; border-radius: 50%; background: #ffffff;"></div>
      </div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
};

// Map Recenter Helper
const MapController: React.FC<{ center: [number, number]; zoom?: number }> = ({ center, zoom = 12 }) => {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, zoom, { animate: true, duration: 1.2 });
  }, [center, zoom, map]);
  return null;
};

export const PostgresSpatialDataPage: React.FC = () => {
  const { isDark } = useTheme();
  const [status, setStatus] = useState<PostgisStatus | null>(null);
  const [records, setRecords] = useState<PostgresRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [selectedRecord, setSelectedRecord] = useState<PostgresRecord | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'MAP_VIEW' | 'TABLE_VIEW' | 'DATABASE_SCHEMA'>('MAP_VIEW');
  const [liveGps, setLiveGps] = useState<SystemGpsState>(sensorFusion.getGpsState());

  // Real-time GPS poller
  useEffect(() => {
    const timer = setInterval(() => {
      setLiveGps(sensorFusion.getGpsState());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch PostgreSQL status and records
  const fetchData = async (silent = false) => {
    if (!silent) setIsRefreshing(true);
    try {
      const [statusRes, recordsRes] = await Promise.all([
        fetch('/api/v1/gis/postgis/status'),
        fetch('/api/v1/gis/postgis/detections?limit=200')
      ]);

      if (statusRes.ok) {
        const sData = await statusRes.json();
        setStatus(sData);
      }

      if (recordsRes.ok) {
        const rData = await recordsRes.json();
        if (Array.isArray(rData)) {
          setRecords(rData);
          if (!selectedRecord && rData.length > 0) {
            setSelectedRecord(rData[0]);
          }
        }
      }
    } catch (err) {
      console.debug('Postgres data fetch notice:', err);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  // Initial load + periodic polling for real-time live entries
  useEffect(() => {
    fetchData();
    const interval = setInterval(() => {
      fetchData(true);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Filter records based on search query
  const filteredRecords = useMemo(() => {
    return records.filter((r) => {
      const q = searchQuery.toLowerCase();
      return (
        (r.class_name_label && r.class_name_label.toLowerCase().includes(q)) ||
        (r.target_class && r.target_class.toLowerCase().includes(q)) ||
        (r.id && r.id.toLowerCase().includes(q)) ||
        (r.mission_id && r.mission_id.toLowerCase().includes(q))
      );
    });
  }, [records, searchQuery]);

  // Center coordinate for map view
  const mapCenter: [number, number] = useMemo(() => {
    if (selectedRecord && selectedRecord.latitude && selectedRecord.longitude) {
      return [selectedRecord.latitude, selectedRecord.longitude];
    }
    if (liveGps.isLiveGps && liveGps.latitude && liveGps.longitude) {
      return [liveGps.latitude, liveGps.longitude];
    }
    if (filteredRecords.length > 0 && filteredRecords[0].latitude) {
      return [filteredRecords[0].latitude, filteredRecords[0].longitude];
    }
    return [9.1524, 79.2819]; // Default Gulf of Mannar
  }, [selectedRecord, liveGps, filteredRecords]);

  return (
    <div className="space-y-6 pb-12 max-w-7xl mx-auto font-mono">
      {/* Top Banner: PostgreSQL Connection & Telemetry Status */}
      <GlassCard variant="glow" className="p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-cyan-900/30 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-2xl bg-cyan-500/20 border border-cyan-400/50 text-cyan-300 shadow-md">
              <Database className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold text-white tracking-wide uppercase">
                  POSTGRESQL / POSTGIS SPATIAL DATABASE ENGINE
                </h1>
                <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border flex items-center gap-1.5 ${
                  status?.connected
                    ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300'
                    : 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                }`}>
                  <span className={`w-2 h-2 rounded-full ${status?.connected ? 'bg-emerald-400' : 'bg-cyan-400'} animate-ping`} />
                  {status?.connected ? 'POSTGIS ONLINE' : 'DATABASE PERSISTENCE ACTIVE'}
                </span>
                <GlassBadge variant="cyan" size="sm">
                  EPSG:4326 WGS84
                </GlassBadge>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Real-time geospatial synchronization engine storing all detections streamed from <strong className="text-cyan-300">Live AI Camera</strong> and <strong className="text-purple-300">Raw Sonar Ingestion</strong> with system GPS triangulation.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <GlassButton
              variant="secondary"
              size="sm"
              onClick={() => fetchData()}
              disabled={isRefreshing}
              icon={<RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />}
            >
              {isRefreshing ? 'SYNCING...' : 'SYNC POSTGIS'}
            </GlassButton>
          </div>
        </div>

        {/* Telemetry Metric Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 text-xs">
          <div className="p-3 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
            <div className="text-[10px] text-slate-400 uppercase font-bold flex items-center justify-between">
              <span>POSTGIS TOTAL RECORDS</span>
              <Server className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <div className="text-cyan-300 font-bold text-base mt-1">
              {records.length} Detections Logged
            </div>
            <div className="text-[10px] text-slate-500">Live Auto-Sync Active</div>
          </div>

          <div className="p-3 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
            <div className="text-[10px] text-slate-400 uppercase font-bold flex items-center justify-between">
              <span>SYSTEM GPS TELEMETRY</span>
              <Compass className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <div className="text-emerald-400 font-bold text-base mt-1 truncate">
              {liveGps.latitude.toFixed(5)}°, {liveGps.longitude.toFixed(5)}°
            </div>
            <div className="text-[10px] text-slate-500">
              {liveGps.isLiveGps ? 'System Geolocation Active' : 'Coastal Coordinate Default'}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
            <div className="text-[10px] text-slate-400 uppercase font-bold flex items-center justify-between">
              <span>DATABASE DRIVER</span>
              <Zap className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <div className="text-amber-300 font-bold text-base mt-1">
              {status?.driver || 'SQLAlchemy+PostGIS'}
            </div>
            <div className="text-[10px] text-slate-500">Pooled Connection Engine</div>
          </div>

          <div className="p-3 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
            <div className="text-[10px] text-slate-400 uppercase font-bold flex items-center justify-between">
              <span>DATABASE URI ENDPOINT</span>
              <Activity className="w-3.5 h-3.5 text-purple-400" />
            </div>
            <div className="text-purple-300 font-bold text-sm mt-1 truncate font-mono">
              {status?.database_url || 'postgresql://localhost:5432'}
            </div>
            <div className="text-[10px] text-slate-500">SSL Encrypted Channel</div>
          </div>
        </div>
      </GlassCard>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (5 Cols): Real-Time Spatial Detections Feed */}
        <div className="lg:col-span-5 space-y-4">
          <GlassCard variant="default" className="p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2.5">
              <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-cyan-400" />
                POSTGIS STORED DETECTIONS ({filteredRecords.length})
              </span>
              <span className="text-[10px] text-emerald-400 flex items-center gap-1 font-bold">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                LIVE STREAM
              </span>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="w-4 h-4 text-cyan-400 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search by class, ID, mission, or coordinates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-2 bg-[#020712]/80 border border-cyan-900/50 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400"
              />
            </div>

            {/* Records List */}
            <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
              {filteredRecords.length === 0 ? (
                <div className="p-8 text-center text-slate-500 text-xs">
                  No spatial detections found. Ingest sonar or stream from AI Cam to populate PostgreSQL.
                </div>
              ) : (
                filteredRecords.map((rec) => {
                  const isSelected = selectedRecord?.id === rec.id;
                  const norm = rec.target_class.toLowerCase();
                  const cfg = CATEGORY_COLORS[norm] || CATEGORY_COLORS.default;

                  return (
                    <div
                      key={rec.id}
                      onClick={() => setSelectedRecord(rec)}
                      className={`p-3 rounded-xl border text-xs cursor-pointer transition-all ${
                        isSelected
                          ? 'bg-cyan-500/20 border-cyan-400 text-white shadow-[0_0_15px_rgba(6,182,212,0.25)]'
                          : 'bg-[#020712]/60 border-cyan-900/30 text-slate-300 hover:border-cyan-500/50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-cyan-300 flex items-center gap-1.5">
                          <MapPin className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                          {rec.class_name_label || rec.target_class}
                        </span>
                        <GlassBadge variant="cyan" size="sm">
                          {Math.round(rec.confidence * 100)}% CONF
                        </GlassBadge>
                      </div>

                      <p className="text-[10px] text-slate-400">
                        GPS: {Number(rec.latitude || 9.1524).toFixed(5)}°N, {Number(rec.longitude || 79.2819).toFixed(5)}°E
                      </p>

                      <div className="flex items-center justify-between text-[9px] text-slate-500 pt-1.5 mt-1.5 border-t border-cyan-900/30">
                        <span>ID: {rec.id}</span>
                        <span>{rec.mission_id || 'LIVE-SURVEY'}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </GlassCard>
        </div>

        {/* Right Column (7 Cols): Real-time Interactive Leaflet Map & Data Inspector */}
        <div className="lg:col-span-7 space-y-4">
          <GlassCard variant="default" className="p-4 space-y-3">
            {/* View Mode Switcher */}
            <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2.5">
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  POSTGIS GEOSPATIAL MAP & REAL-TIME CARTOGRAPHY
                </span>
              </div>

              <div className="flex items-center gap-1 bg-[#020712]/80 p-1 rounded-xl border border-cyan-900/40 text-xs">
                <button
                  onClick={() => setActiveTab('MAP_VIEW')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'MAP_VIEW'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Live Map
                </button>
                <button
                  onClick={() => setActiveTab('TABLE_VIEW')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'TABLE_VIEW'
                      ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Table Grid
                </button>
              </div>
            </div>

            {/* Map View */}
            {activeTab === 'MAP_VIEW' && (
              <div className="space-y-3">
                <div className="relative rounded-2xl overflow-hidden border border-cyan-500/30 h-[480px] bg-[#020712] shadow-inner">
                  <MapContainer
                    center={mapCenter}
                    zoom={11}
                    scrollWheelZoom={true}
                    className="w-full h-full z-10"
                  >
                    <MapController center={mapCenter} />
                    <TileLayer
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                      url={
                        isDark
                          ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                          : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
                      }
                    />
                    <ScaleControl position="bottomleft" />

                    {/* Live System Vessel/GPS Location Marker */}
                    <CircleMarker
                      center={[liveGps.latitude, liveGps.longitude]}
                      radius={9}
                      pathOptions={{ color: '#10b981', fillColor: '#10b981', fillOpacity: 0.8, weight: 3 }}
                    >
                      <Popup>
                        <div className="font-mono text-xs p-1 space-y-1">
                          <strong className="text-emerald-400 block">SYSTEM VESSEL GPS POSITION</strong>
                          <div>Latitude: {liveGps.latitude.toFixed(5)}°N</div>
                          <div>Longitude: {liveGps.longitude.toFixed(5)}°E</div>
                          <div>Heading: {liveGps.headingDeg}°</div>
                        </div>
                      </Popup>
                      <Tooltip permanent direction="top" className="font-mono text-[10px]">
                        VESSEL GPS
                      </Tooltip>
                    </CircleMarker>

                    {/* Stored PostGIS Detection Markers */}
                    {filteredRecords.map((rec) => {
                      if (!rec.latitude || !rec.longitude) return null;
                      const isSelected = selectedRecord?.id === rec.id;
                      return (
                        <Marker
                          key={rec.id}
                          position={[rec.latitude, rec.longitude]}
                          icon={createMarkerIcon(rec.target_class, isSelected)}
                          eventHandlers={{
                            click: () => setSelectedRecord(rec),
                          }}
                        >
                          <Popup>
                            <div className="font-mono text-xs p-1 space-y-1">
                              <strong className="text-cyan-400 block">{rec.class_name_label || rec.target_class}</strong>
                              <div>ID: {rec.id}</div>
                              <div>Confidence: {Math.round(rec.confidence * 100)}%</div>
                              <div>GPS: {rec.latitude.toFixed(5)}°N, {rec.longitude.toFixed(5)}°E</div>
                              <div>Depth: {rec.depth_meters || 18.5}m</div>
                              <div>Mission: {rec.mission_id}</div>
                            </div>
                          </Popup>
                        </Marker>
                      );
                    })}
                  </MapContainer>

                  {/* Floating Map Overlay Telemetry HUD */}
                  <div className="absolute top-3 left-3 z-20 pointer-events-none bg-[#020712]/90 backdrop-blur-md px-3 py-1.5 rounded-xl border border-cyan-500/40 text-[11px] font-mono text-cyan-300 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                    <span>POSTGIS EPSG:4326 • {filteredRecords.length} PINPOINTS RENDERED</span>
                  </div>
                </div>

                {/* Selected Detection Detail Card */}
                {selectedRecord && (
                  <div className="p-3.5 rounded-xl bg-[#020712]/80 border border-cyan-500/40 text-xs space-y-2">
                    <div className="flex items-center justify-between border-b border-cyan-900/30 pb-1.5">
                      <span className="font-bold text-cyan-300 flex items-center gap-1.5">
                        <Crosshair className="w-4 h-4 text-cyan-400" />
                        SELECTED TARGET: {selectedRecord.class_name_label || selectedRecord.target_class}
                      </span>
                      <GlassBadge variant="cyan" size="sm">
                        {Math.round(selectedRecord.confidence * 100)}% CONFIDENCE
                      </GlassBadge>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                      <div>
                        <span className="text-slate-500 block text-[9px]">LATITUDE</span>
                        <span className="text-white font-bold">{selectedRecord.latitude?.toFixed(5)}° N</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[9px]">LONGITUDE</span>
                        <span className="text-white font-bold">{selectedRecord.longitude?.toFixed(5)}° E</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[9px]">DEPTH</span>
                        <span className="text-emerald-400 font-bold">{selectedRecord.depth_meters || 18.5} m</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block text-[9px]">MISSION ID</span>
                        <span className="text-cyan-400 font-bold truncate block">{selectedRecord.mission_id}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Table Grid View */}
            {activeTab === 'TABLE_VIEW' && (
              <div className="space-y-3">
                <div className="rounded-xl overflow-hidden border border-cyan-900/40 max-h-[500px] overflow-y-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#020712]/90 border-b border-cyan-900/40 text-[10px] text-slate-400 uppercase">
                      <tr>
                        <th className="p-2.5">Target Class</th>
                        <th className="p-2.5">Confidence</th>
                        <th className="p-2.5">Latitude</th>
                        <th className="p-2.5">Longitude</th>
                        <th className="p-2.5">Mission</th>
                        <th className="p-2.5">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-cyan-900/20 text-[11px]">
                      {filteredRecords.map((r) => (
                        <tr
                          key={r.id}
                          onClick={() => {
                            setSelectedRecord(r);
                            setActiveTab('MAP_VIEW');
                          }}
                          className="hover:bg-cyan-500/10 cursor-pointer transition-colors"
                        >
                          <td className="p-2.5 font-bold text-cyan-300">{r.class_name_label || r.target_class}</td>
                          <td className="p-2.5 text-emerald-400 font-bold">{Math.round(r.confidence * 100)}%</td>
                          <td className="p-2.5 text-slate-300">{Number(r.latitude || 9.1524).toFixed(5)}°</td>
                          <td className="p-2.5 text-slate-300">{Number(r.longitude || 79.2819).toFixed(5)}°</td>
                          <td className="p-2.5 text-slate-400">{r.mission_id}</td>
                          <td className="p-2.5 text-cyan-400 font-bold">STORED</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </GlassCard>
        </div>
      </div>
    </div>
  );
};

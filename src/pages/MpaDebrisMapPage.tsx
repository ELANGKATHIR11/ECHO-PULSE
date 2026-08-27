import React, { useState, useEffect, useMemo } from 'react';
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polygon,
  Polyline,
  Tooltip,
  useMap
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  Shield,
  MapPin,
  AlertTriangle,
  Layers,
  Filter,
  Download,
  Info,
  Compass,
  Radio,
  Sparkles,
  Search,
  CheckCircle2,
  ExternalLink,
  ChevronRight,
  Maximize2,
  Anchor,
  FileText,
  Activity,
  Cpu
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../components/glass/GlassCard';
import { mpaApi, MpaZone, MpaDebrisGeoTag, MpaSummaryMetrics } from '../services/mpaApi';
import { useTheme } from '../context/ThemeContext';

// Fix Leaflet Default Marker Icon in React
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Category Colors & Styling
const CATEGORY_COLORS: Record<string, { bg: string; border: string; text: string; hex: string }> = {
  PLASTIC: { bg: 'bg-cyan-500/20', border: 'border-cyan-500', text: 'text-cyan-400', hex: '#06b6d4' },
  METAL_SCRAP: { bg: 'bg-amber-500/20', border: 'border-amber-500', text: 'text-amber-400', hex: '#f59e0b' },
  ELECTRICAL: { bg: 'bg-yellow-500/20', border: 'border-yellow-500', text: 'text-yellow-400', hex: '#eab308' },
  ELECTRONIC: { bg: 'bg-red-500/20', border: 'border-red-500', text: 'text-red-400', hex: '#ef4444' },
  HUMAN: { bg: 'bg-emerald-500/20', border: 'border-emerald-500', text: 'text-emerald-400', hex: '#10b981' }
};

const THREAT_HALOS: Record<string, string> = {
  CRITICAL: 'rgba(239, 68, 68, 0.75)',
  HIGH: 'rgba(245, 158, 11, 0.70)',
  MEDIUM: 'rgba(6, 182, 212, 0.65)',
  LOW: 'rgba(16, 185, 129, 0.50)'
};

// Custom Marker Pin Generator
const createGeoTagIcon = (targetClass: string, threatLevel: string, isSelected: boolean) => {
  const color = CATEGORY_COLORS[targetClass]?.hex || '#06b6d4';
  const haloColor = THREAT_HALOS[threatLevel] || 'rgba(6,182,212,0.6)';
  const size = isSelected ? 34 : 26;

  return L.divIcon({
    className: 'custom-debris-geotag-icon',
    html: `
      <div style="
        width: ${size}px;
        height: ${size}px;
        background: radial-gradient(circle, ${color} 30%, rgba(2,7,18,0.9) 90%);
        border: 2px solid ${color};
        border-radius: 50%;
        box-shadow: 0 0 16px ${haloColor}, inset 0 0 8px ${color};
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        position: relative;
        transition: transform 0.2s ease;
      ">
        <div style="width: 8px; height: 8px; background: #fff; border-radius: 50%; box-shadow: 0 0 4px #fff;"></div>
        ${threatLevel === 'CRITICAL' ? `<div style="position: absolute; inset: -4px; border: 1.5px dashed #ef4444; border-radius: 50%; animation: spin 4s linear infinite;"></div>` : ''}
      </div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
};

// Map Recenter Controller Helper
const MapController: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, zoom, { duration: 1.2 });
  }, [center, zoom, map]);
  return null;
};

export const MpaDebrisMapPage: React.FC = () => {
  const { isDark } = useTheme();
  const [mpaZones, setMpaZones] = useState<MpaZone[]>([]);
  const [debrisTags, setDebrisTags] = useState<MpaDebrisGeoTag[]>([]);
  const [eezCoords, setEezCoords] = useState<[number, number][]>([]);
  const [summary, setSummary] = useState<MpaSummaryMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Filters
  const [selectedMpaId, setSelectedMpaId] = useState<string>('ALL');
  const [selectedClass, setSelectedClass] = useState<string>('ALL');
  const [selectedThreat, setSelectedThreat] = useState<string>('ALL');
  const [selectedAgency, setSelectedAgency] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Active Tag Selection
  const [selectedTag, setSelectedTag] = useState<MpaDebrisGeoTag | null>(null);
  const [mapCenter, setMapCenter] = useState<[number, number]>([14.5, 78.5]);
  const [mapZoom, setMapZoom] = useState<number>(5);

  // Layers Toggles
  const [showMpaPolygons, setShowMpaPolygons] = useState<boolean>(true);
  const [showEezBoundary, setShowEezBoundary] = useState<boolean>(true);
  const [showDebrisPins, setShowDebrisPins] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [zonesData, tagsData, eezData, summaryData] = await Promise.all([
          mpaApi.getMpaZones(),
          mpaApi.getDebrisGeoTags(),
          mpaApi.getIndianEez(),
          mpaApi.getMpaSummary()
        ]);
        setMpaZones(zonesData);
        setDebrisTags(tagsData);
        setEezCoords(eezData);
        setSummary(summaryData);
        if (tagsData.length > 0) setSelectedTag(tagsData[0]);
      } catch (err) {
        console.error('Failed to fetch MPA geospatial dataset:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Filtered Debris Tags
  const filteredTags = useMemo(() => {
    return debrisTags.filter((tag) => {
      if (selectedMpaId !== 'ALL' && tag.mpa_id !== selectedMpaId) return false;
      if (selectedClass !== 'ALL' && tag.target_class !== selectedClass) return false;
      if (selectedThreat !== 'ALL' && tag.threat_level !== selectedThreat) return false;
      if (selectedAgency !== 'ALL' && !tag.certifying_agency.includes(selectedAgency)) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const matchesRef = tag.official_agency_ref.toLowerCase().includes(q);
        const matchesName = tag.marine_label.toLowerCase().includes(q);
        const matchesMpa = tag.mpa_name.toLowerCase().includes(q);
        const matchesCategory = tag.sub_category.toLowerCase().includes(q);
        if (!matchesRef && !matchesName && !matchesMpa && !matchesCategory) return false;
      }
      return true;
    });
  }, [debrisTags, selectedMpaId, selectedClass, selectedThreat, selectedAgency, searchQuery]);

  const handleSelectMpa = (mpa: MpaZone) => {
    setSelectedMpaId(mpa.id);
    setMapCenter(mpa.center_coords);
    setMapZoom(9);
  };

  const handleSelectTag = (tag: MpaDebrisGeoTag) => {
    setSelectedTag(tag);
    setMapCenter([tag.latitude, tag.longitude]);
    setMapZoom(11);
  };

  const exportGeoJson = () => {
    const geojson = {
      type: 'FeatureCollection',
      name: 'Indian_MPA_Official_Debris_Registry_EchoPulseNet',
      crs: { type: 'name', properties: { name: 'urn:ogc:def:crs:OGC:1.3:CRS84' } },
      features: filteredTags.map((t) => ({
        type: 'Feature',
        properties: {
          id: t.id,
          official_ref: t.official_agency_ref,
          agency: t.certifying_agency,
          mpa: t.mpa_name,
          category: t.target_class,
          label: t.marine_label,
          depth_m: t.depth_meters,
          threat: t.threat_level,
          clean_coast_index: t.clean_coast_index_score,
          vessel: t.survey_vessel,
          timestamp: t.tag_timestamp
        },
        geometry: {
          type: 'Point',
          coordinates: [t.longitude, t.latitude]
        }
      }))
    };

    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Indian_MPA_Debris_GeoTags_${new Date().toISOString().split('T')[0]}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top Header & Context Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-cyan-900/30 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Shield className="w-5 h-5" />
            </span>
            <span className="text-xs font-mono font-bold tracking-widest text-cyan-400 uppercase">
              Official Indian Maritime Protected Areas & GIS Registry
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white dark:text-white light:text-slate-900">
            Indian Sea Boundary & MPA Debris Geo-Tag Registry
          </h1>
          <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-0.5">
            Authoritative Geospatial Surveillance across Indian Territorial Waters, 200 NM EEZ, and Ecologically Sensitive MPAs (MoES / NCCR / INCOIS / CMFRI / CSIR-NIO).
          </p>
        </div>

        <div className="flex items-center gap-2">
          <GlassButton
            size="sm"
            variant="secondary"
            icon={<Download className="w-3.5 h-3.5 text-cyan-400" />}
            onClick={exportGeoJson}
          >
            EXPORT GEOJSON
          </GlassButton>
          <GlassButton
            size="sm"
            variant="primary"
            icon={<Radio className="w-3.5 h-3.5 animate-pulse text-cyan-400" />}
            onClick={() => {
              setMapCenter([14.5, 78.5]);
              setMapZoom(5);
              setSelectedMpaId('ALL');
            }}
          >
            RESET ALL-INDIA EXTENT
          </GlassButton>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <GlassCard className="p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>OFFICIAL MPAs</span>
            <Shield className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-cyan-400">
            {summary?.total_mpas || mpaZones.length || 10}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Covering {summary?.total_mpa_area_sq_km?.toLocaleString() || '30,000+'} sq.km Seafloor
          </div>
        </GlassCard>

        <GlassCard className="p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>TAGGED DEBRIS</span>
            <MapPin className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400">
            {filteredTags.length} <span className="text-xs text-slate-400 font-normal">/ {debrisTags.length} Verified</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Certified via MoES / NCCR Swachh Sagar
          </div>
        </GlassCard>

        <GlassCard className="p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>CRITICAL HAZARDS</span>
            <AlertTriangle className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-red-400">
            {debrisTags.filter((t) => t.threat_level === 'CRITICAL').length}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Direct Reef & Turtle Entanglement Risks
          </div>
        </GlassCard>

        <GlassCard className="p-3.5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>EEZ SURVEILLANCE</span>
            <Compass className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400">
            200 NM
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Arabian Sea, Bay of Bengal, Andaman Sea
          </div>
        </GlassCard>
      </div>

      {/* Main Map & Geospatial Operations Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Interactive Leaflet Map */}
        <div className="lg:col-span-2 space-y-3">
          {/* Map Controls Header Bar */}
          <GlassCard className="p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-slate-400 font-mono font-bold">LAYERS:</span>
              <button
                onClick={() => setShowMpaPolygons(!showMpaPolygons)}
                className={`px-2.5 py-1 rounded-lg border text-[11px] font-mono transition-all ${
                  showMpaPolygons
                    ? 'bg-cyan-500/20 border-cyan-500/60 text-cyan-300'
                    : 'bg-slate-800/40 border-slate-700/50 text-slate-400'
                }`}
              >
                MPA Boundaries ({mpaZones.length})
              </button>
              <button
                onClick={() => setShowEezBoundary(!showEezBoundary)}
                className={`px-2.5 py-1 rounded-lg border text-[11px] font-mono transition-all ${
                  showEezBoundary
                    ? 'bg-amber-500/20 border-amber-500/60 text-amber-300'
                    : 'bg-slate-800/40 border-slate-700/50 text-slate-400'
                }`}
              >
                Indian 200NM EEZ
              </button>
              <button
                onClick={() => setShowDebrisPins(!showDebrisPins)}
                className={`px-2.5 py-1 rounded-lg border text-[11px] font-mono transition-all ${
                  showDebrisPins
                    ? 'bg-emerald-500/20 border-emerald-500/60 text-emerald-300'
                    : 'bg-slate-800/40 border-slate-700/50 text-slate-400'
                }`}
              >
                Geo-Tags ({filteredTags.length})
              </button>
            </div>

            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              <span className="font-mono text-[11px] text-slate-300">WGS84 EPSG:4326</span>
            </div>
          </GlassCard>

          {/* Leaflet Map Canvas Container */}
          <GlassCard className="p-1 h-[540px] relative overflow-hidden rounded-2xl border border-cyan-900/40">
            <MapContainer
              center={mapCenter}
              zoom={mapZoom}
              scrollWheelZoom={true}
              className="w-full h-full rounded-xl z-0"
              style={{ background: '#020814' }}
            >
              <MapController center={mapCenter} zoom={mapZoom} />

              {/* High-Resolution Bathymetric / Ocean CartoDB Tiles */}
              <TileLayer
                attribution='&copy; <a href="https://carto.com/">CARTO</a> | NCCR MoES India'
                url={
                  isDark
                    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                    : 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
                }
              />

              {/* Indian 200 NM Exclusive Economic Zone (EEZ) Polyline */}
              {showEezBoundary && eezCoords.length > 0 && (
                <Polyline
                  positions={eezCoords}
                  pathOptions={{
                    color: '#f59e0b',
                    weight: 2,
                    dashArray: '6, 8',
                    opacity: 0.85
                  }}
                >
                  <Tooltip sticky>
                    <div className="font-mono text-xs">
                      <strong>Indian 200 NM EEZ Maritime Boundary</strong>
                      <div>Territorial Sea & Contiguous Fishery Zone</div>
                    </div>
                  </Tooltip>
                </Polyline>
              )}

              {/* Official Indian MPA Polygons */}
              {showMpaPolygons &&
                mpaZones.map((mpa) => (
                  <Polygon
                    key={mpa.id}
                    positions={mpa.boundary_polygon}
                    pathOptions={{
                      color: selectedMpaId === mpa.id ? '#06b6d4' : '#38bdf8',
                      fillColor: selectedMpaId === mpa.id ? '#06b6d4' : '#0284c7',
                      fillOpacity: selectedMpaId === mpa.id ? 0.35 : 0.15,
                      weight: selectedMpaId === mpa.id ? 3 : 1.5
                    }}
                    eventHandlers={{
                      click: () => handleSelectMpa(mpa)
                    }}
                  >
                    <Tooltip sticky>
                      <div className="font-mono text-xs p-1">
                        <div className="font-bold text-cyan-400">{mpa.name}</div>
                        <div className="text-slate-300">State: {mpa.state} ({mpa.sea_sector})</div>
                        <div className="text-slate-400">Area: {mpa.area_sq_km} sq.km | Est: {mpa.established_year}</div>
                        <div className="text-amber-300 text-[10px] mt-0.5">{mpa.threat_status}</div>
                      </div>
                    </Tooltip>
                  </Polygon>
                ))}

              {/* Official Debris GeoTag Pins */}
              {showDebrisPins &&
                filteredTags.map((tag) => (
                  <Marker
                    key={tag.id}
                    position={[tag.latitude, tag.longitude]}
                    icon={createGeoTagIcon(tag.target_class, tag.threat_level, selectedTag?.id === tag.id)}
                    eventHandlers={{
                      click: () => handleSelectTag(tag)
                    }}
                  >
                    <Popup>
                      <div className="p-1 font-mono text-xs space-y-1">
                        <div className="font-bold text-cyan-400">{tag.official_agency_ref}</div>
                        <div className="text-white font-semibold">{tag.marine_label}</div>
                        <div className="text-slate-400">MPA: {tag.mpa_name}</div>
                        <div className="text-slate-400">Depth: {tag.depth_meters}m | Conf: {(tag.detection_confidence * 100).toFixed(1)}%</div>
                        <div className="flex items-center gap-1.5 mt-1">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            tag.threat_level === 'CRITICAL' ? 'bg-red-500 text-white' : 'bg-amber-500 text-black'
                          }`}>
                            {tag.threat_level}
                          </span>
                          <span className="text-[10px] text-slate-300">Agency: {tag.certifying_agency}</span>
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                ))}
            </MapContainer>

            {/* Map Legend Overlay */}
            <div className="absolute bottom-3 left-3 z-[1000] p-2.5 rounded-xl bg-[#020712]/85 backdrop-blur-md border border-cyan-900/50 text-[10px] font-mono space-y-1.5 shadow-xl max-w-xs">
              <div className="font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1">
                <Layers className="w-3 h-3" /> MAP CLASSIFICATION LEGEND
              </div>
              <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-slate-300">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 border border-cyan-300" />
                  <span>PLASTIC (Nets/ALDFG)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400 border border-amber-300" />
                  <span>METAL (Wreck/UXO)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 border border-yellow-300" />
                  <span>ELECTRICAL (Cables)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-400 border border-red-300" />
                  <span>ELECTRONIC (E-Waste)</span>
                </div>
              </div>
              <div className="pt-1 border-t border-cyan-900/40 text-[9px] text-slate-400 flex items-center justify-between">
                <span>--- 200NM Indian EEZ</span>
                <span>■ Protected MPA Zone</span>
              </div>
            </div>
          </GlassCard>
        </div>

        {/* Right Col: Filters, Official Agency Accreditation & Tag Inspector */}
        <div className="space-y-4">
          {/* Search & Filter Toolbar */}
          <GlassCard className="p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2">
              <span className="text-xs font-mono font-bold text-cyan-400 flex items-center gap-1.5">
                <Filter className="w-3.5 h-3.5" /> GEOSPATIAL FILTERS
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                {filteredTags.length} Matched
              </span>
            </div>

            {/* Text Search */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search Agency ID, Label, MPA..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 text-xs font-mono rounded-lg bg-[#020712]/80 border border-cyan-900/50 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400"
              />
            </div>

            {/* Select MPA */}
            <div>
              <label className="text-[10px] font-mono text-slate-400 uppercase">Marine Protected Area</label>
              <select
                value={selectedMpaId}
                onChange={(e) => {
                  const id = e.target.value;
                  setSelectedMpaId(id);
                  if (id !== 'ALL') {
                    const mpa = mpaZones.find((m) => m.id === id);
                    if (mpa) {
                      setMapCenter(mpa.center_coords);
                      setMapZoom(9);
                    }
                  } else {
                    setMapCenter([14.5, 78.5]);
                    setMapZoom(5);
                  }
                }}
                className="w-full mt-1 p-1.5 text-xs font-mono rounded-lg bg-[#020712]/80 border border-cyan-900/50 text-slate-200 focus:outline-none focus:border-cyan-400"
              >
                <option value="ALL">All Indian MPAs ({mpaZones.length} Sectors)</option>
                {mpaZones.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.state})
                  </option>
                ))}
              </select>
            </div>

            {/* Category & Threat Selectors */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] font-mono text-slate-400 uppercase">Category</label>
                <select
                  value={selectedClass}
                  onChange={(e) => setSelectedClass(e.target.value)}
                  className="w-full mt-1 p-1.5 text-xs font-mono rounded-lg bg-[#020712]/80 border border-cyan-900/50 text-slate-200 focus:outline-none focus:border-cyan-400"
                >
                  <option value="ALL">All Classes</option>
                  <option value="PLASTIC">Plastic (Nets)</option>
                  <option value="METAL_SCRAP">Metal Scrap</option>
                  <option value="ELECTRICAL">Electrical</option>
                  <option value="ELECTRONIC">Electronic</option>
                  <option value="HUMAN">Human / Diver</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] font-mono text-slate-400 uppercase">Threat Level</label>
                <select
                  value={selectedThreat}
                  onChange={(e) => setSelectedThreat(e.target.value)}
                  className="w-full mt-1 p-1.5 text-xs font-mono rounded-lg bg-[#020712]/80 border border-cyan-900/50 text-slate-200 focus:outline-none focus:border-cyan-400"
                >
                  <option value="ALL">All Threats</option>
                  <option value="CRITICAL">Critical</option>
                  <option value="HIGH">High</option>
                  <option value="MEDIUM">Medium</option>
                </select>
              </div>
            </div>

            {/* Agency Selector */}
            <div>
              <label className="text-[10px] font-mono text-slate-400 uppercase">Certifying Agency</label>
              <select
                value={selectedAgency}
                onChange={(e) => setSelectedAgency(e.target.value)}
                className="w-full mt-1 p-1.5 text-xs font-mono rounded-lg bg-[#020712]/80 border border-cyan-900/50 text-slate-200 focus:outline-none focus:border-cyan-400"
              >
                <option value="ALL">All Official Agencies</option>
                <option value="NCCR">NCCR (MoES)</option>
                <option value="INCOIS">INCOIS (MoES)</option>
                <option value="CMFRI">CMFRI (ICAR)</option>
                <option value="CSIR-NIO">CSIR-NIO</option>
                <option value="ICG">Indian Coast Guard (ICG)</option>
              </select>
            </div>
          </GlassCard>

          {/* Selected Geo-Tag Inspector Card */}
          {selectedTag ? (
            <GlassCard className="p-4 space-y-3 border-cyan-500/40">
              <div className="flex items-start justify-between gap-2 border-b border-cyan-900/40 pb-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 font-bold border border-cyan-500/30">
                      {selectedTag.official_agency_ref}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {selectedTag.certifying_agency}
                    </span>
                  </div>
                  <h3 className="text-sm font-bold text-white mt-1 leading-snug">
                    {selectedTag.marine_label}
                  </h3>
                </div>

                <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded ${
                  selectedTag.threat_level === 'CRITICAL' ? 'bg-red-500/30 text-red-300 border border-red-500/50' : 'bg-amber-500/30 text-amber-300 border border-amber-500/50'
                }`}>
                  {selectedTag.threat_level}
                </span>
              </div>

              {/* Exact Coordinate Metrics */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-[#020712]/60 p-2.5 rounded-xl border border-cyan-900/40">
                <div>
                  <div className="text-[9px] text-slate-400">DECIMAL COORDS</div>
                  <div className="text-cyan-300 font-bold">{selectedTag.latitude.toFixed(4)}°N, {selectedTag.longitude.toFixed(4)}°E</div>
                </div>
                <div>
                  <div className="text-[9px] text-slate-400">DMS FORMAT</div>
                  <div className="text-slate-200 text-[11px] truncate">{selectedTag.coordinates_dms}</div>
                </div>
                <div>
                  <div className="text-[9px] text-slate-400">BATHYMETRIC DEPTH</div>
                  <div className="text-slate-200">{selectedTag.depth_meters} meters</div>
                </div>
                <div>
                  <div className="text-[9px] text-slate-400">SHADOW HEIGHT (H)</div>
                  <div className="text-emerald-400 font-bold">{selectedTag.estimated_height_meters}m</div>
                </div>
              </div>

              {/* Acoustic & Survey Details */}
              <div className="space-y-1 text-[11px] text-slate-300 font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-400">MPA Sector:</span>
                  <span className="text-cyan-300">{selectedTag.mpa_name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Clean Coast Index:</span>
                  <span className="text-amber-400 font-bold">{selectedTag.clean_coast_index_score} / 20</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Survey Platform:</span>
                  <span>{selectedTag.survey_vessel}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Acoustic Frequency:</span>
                  <span>{selectedTag.sonar_frequency_khz} kHz SSS</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Confidence Score:</span>
                  <span className="text-cyan-400 font-bold">{(selectedTag.detection_confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Notes Box */}
              <div className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-[11px] text-slate-400 italic">
                "{selectedTag.notes}"
              </div>

              <div className="pt-2 flex items-center justify-between border-t border-cyan-900/30">
                <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> {selectedTag.verification_status}
                </span>
                <span className="text-[9px] text-slate-500 font-mono">
                  {new Date(selectedTag.tag_timestamp).toLocaleDateString()}
                </span>
              </div>
            </GlassCard>
          ) : (
            <GlassCard className="p-6 text-center text-slate-400 text-xs font-mono">
              <MapPin className="w-8 h-8 mx-auto mb-2 text-cyan-400/50" />
              <div>Click any Geo-Tag Pin on the map to inspect official agency survey metadata.</div>
            </GlassCard>
          )}

          {/* Official Agency Accreditation Footer */}
          <GlassCard className="p-3 text-[10px] font-mono text-slate-400 space-y-1 bg-[#020712]/50">
            <div className="text-cyan-400 font-bold flex items-center gap-1 uppercase">
              <FileText className="w-3 h-3" /> Official Indian Marine Data Compliance
            </div>
            <div>Data integrated in accordance with MoES NCCR Marine Litter Assessment Framework & MoEFCC Wildlife Protection Guidelines.</div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
};

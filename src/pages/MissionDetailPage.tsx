import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Mission, Detection } from '../types';
import { missionApi } from '../services/missionApi';
import { detectionApi } from '../services/detectionApi';
import { MissionMap } from '../components/gis/MissionMap';
import { ThreeOceanScene } from '../components/three/ThreeOceanScene';
import { DetectionTable } from '../components/detections/DetectionTable';
import { exportDetectionsToGeoJSON, exportDetectionsToCSV, downloadBlobFile } from '../utils/geoUtils';
import {
  Compass,
  ArrowLeft,
  Radio,
  Download,
  Box,
  MapPin,
  Activity,
  Layers,
} from 'lucide-react';
import { formatDMS } from '../utils/sonarProcessor';
import { GlassCard, GlassBadge, GlassButton, KpiCard } from '../components/glass/GlassCard';

export const MissionDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [allMissions, setAllMissions] = useState<Mission[]>([]);
  const [mission, setMission] = useState<Mission | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);
  const [activeTab, setActiveTab] = useState<'GIS' | '3D' | 'DETECTIONS'>('GIS');

  useEffect(() => {
    missionApi.getMissions().then(setAllMissions);
  }, []);

  useEffect(() => {
    if (!id) return;
    missionApi.getMissionById(id).then((m) => {
      if (m) {
        setMission(m);
        detectionApi.getDetections({ missionId: m.id }).then((dets) => {
          setDetections(dets);
          if (dets.length > 0) setSelectedDetection(dets[0]);
        });
      }
    });
  }, [id]);

  if (!mission) {
    return (
      <div className="p-8 text-cyan-400 font-mono flex items-center justify-center">
        <Radio className="w-5 h-5 animate-spin mr-2" /> Loading mission data...
      </div>
    );
  }

  const handleExportGeoJSON = () => {
    const geojson = exportDetectionsToGeoJSON(detections, mission);
    downloadBlobFile(geojson, `${mission.id}_detections.geojson`, 'application/geo+json');
  };

  const handleExportCSV = () => {
    const csv = exportDetectionsToCSV(detections);
    downloadBlobFile(csv, `${mission.id}_detections.csv`, 'text/csv');
  };

  return (
    <div className="p-4 md:p-6 max-w-[1700px] mx-auto w-full font-mono space-y-4">
      {/* Back Button & Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/missions')}
            className="p-2 rounded-xl bg-[#0b1624]/80 dark:bg-[#0b1624]/80 light:bg-white hover:bg-cyan-950/40 text-slate-300 dark:text-slate-300 light:text-slate-700 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-white dark:text-white light:text-slate-900">{mission.name}</h1>
              <GlassBadge variant="cyan" size="sm">
                {mission.id}
              </GlassBadge>
            </div>
            <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-0.5">{mission.location}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <GlassButton
            variant="primary"
            size="sm"
            onClick={() => navigate('/sonar')}
            icon={<Radio className="w-4 h-4" />}
          >
            Open Sonar Workstation
          </GlassButton>
          <GlassButton
            variant="secondary"
            size="sm"
            onClick={handleExportGeoJSON}
            icon={<Download className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />}
          >
            Export GeoJSON
          </GlassButton>
        </div>
      </div>

      {/* Mission Quick Stats Grid (12-Column Grid Alignment) */}
      <div className="grid grid-cols-12 gap-4 text-xs">
        <div className="col-span-12 sm:col-span-6 md:col-span-4 lg:col-span-2 flex">
          <KpiCard>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] font-bold uppercase tracking-wider">
              Sonar Transducer
            </div>
            <div className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold my-1 truncate">
              {mission.sonarSource}
            </div>
            <div className="text-cyan-400 dark:text-cyan-400 light:text-sky-600 text-[10px] font-mono font-bold">
              {mission.frequencyKhz} kHz
            </div>
          </KpiCard>
        </div>

        <div className="col-span-12 sm:col-span-6 md:col-span-4 lg:col-span-2 flex">
          <KpiCard>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] font-bold uppercase tracking-wider">
              Survey Corridor
            </div>
            <div className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold my-1 font-mono">
              {mission.surveyDistanceKm} km
            </div>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] font-mono">
              {mission.areaSqKm} km² swath
            </div>
          </KpiCard>
        </div>

        <div className="col-span-12 sm:col-span-6 md:col-span-4 lg:col-span-2 flex">
          <KpiCard>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] font-bold uppercase tracking-wider">
              Total Pings
            </div>
            <div className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold my-1 font-mono">
              {mission.pingCount.toLocaleString()}
            </div>
            <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-600 text-[10px] font-mono font-bold">
              100% Parsed
            </div>
          </KpiCard>
        </div>

        <div className="col-span-12 sm:col-span-6 md:col-span-4 lg:col-span-2 flex">
          <KpiCard>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] font-bold uppercase tracking-wider">
              AI Detections
            </div>
            <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold my-1 font-mono">
              {detections.length} Targets
            </div>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] font-mono">
              {detections.filter((d) => d.confidence > 0.9).length} High-Conf
            </div>
          </KpiCard>
        </div>

        <div className="col-span-12 sm:col-span-6 md:col-span-4 lg:col-span-2 flex">
          <KpiCard>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] font-bold uppercase tracking-wider">
              Vessel / Vehicle
            </div>
            <div className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold my-1 truncate">
              {mission.vesselName}
            </div>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px]">
              {mission.vehicleType}
            </div>
          </KpiCard>
        </div>

        <div className="col-span-12 sm:col-span-6 md:col-span-4 lg:col-span-2 flex">
          <KpiCard>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] font-bold uppercase tracking-wider">
              Datum Position
            </div>
            <div className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold my-1 text-[11px] font-mono truncate">
              {formatDMS(mission.coordinates[0], true)}
            </div>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] font-mono truncate">
              {formatDMS(mission.coordinates[1], false)}
            </div>
          </KpiCard>
        </div>
      </div>

      {/* Main View Container with Switcher */}
      <GlassCard variant="default" className="overflow-hidden flex flex-col p-0">
        {/* Navigation Tabs */}
        <div className="h-12 bg-[#08121e]/60 dark:bg-[#08121e]/60 light:bg-slate-100/90 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 px-4 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab('GIS')}
              className={`px-3 py-1.5 rounded-xl transition-all font-bold text-xs flex items-center gap-1.5 ${
                activeTab === 'GIS'
                  ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-200 text-cyan-300 dark:text-cyan-300 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-400'
                  : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              <span>GIS Tactical Map</span>
            </button>
            <button
              onClick={() => setActiveTab('3D')}
              className={`px-3 py-1.5 rounded-xl transition-all font-bold text-xs flex items-center gap-1.5 ${
                activeTab === '3D'
                  ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-200 text-cyan-300 dark:text-cyan-300 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-400'
                  : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
              }`}
            >
              <Box className="w-3.5 h-3.5" />
              <span>3D Bathymetry Overview</span>
            </button>
            <button
              onClick={() => setActiveTab('DETECTIONS')}
              className={`px-3 py-1.5 rounded-xl transition-all font-bold text-xs flex items-center gap-1.5 ${
                activeTab === 'DETECTIONS'
                  ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-200 text-cyan-300 dark:text-cyan-300 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-400'
                  : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Target Table ({detections.length})</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] text-slate-400 dark:text-slate-400 light:text-slate-600">Mean SNR:</span>
            <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold text-xs">{mission.summaryMetrics.avgSnrDb} dB</span>
          </div>
        </div>

        {/* Viewport Content */}
        <div className="p-3 min-h-[550px]">
          {activeTab === 'GIS' && (
            <MissionMap
              mission={mission}
              allMissions={allMissions.length > 0 ? allMissions : undefined}
              detections={detections}
              selectedDetectionId={selectedDetection?.id}
              onSelectDetection={setSelectedDetection}
              className="h-[550px] w-full rounded-xl overflow-hidden"
            />
          )}

          {activeTab === '3D' && (
            <div className="h-[550px] w-full rounded-xl overflow-hidden">
              <ThreeOceanScene
                mission={mission}
                detections={detections}
                selectedDetectionId={selectedDetection?.id}
                onSelectDetection={setSelectedDetection}
                renderProfile="HIGH"
              />
            </div>
          )}

          {activeTab === 'DETECTIONS' && (
            <DetectionTable
              detections={detections}
              onSelectDetection={setSelectedDetection}
              selectedId={selectedDetection?.id}
              onExportCsv={handleExportCSV}
            />
          )}
        </div>
      </GlassCard>
    </div>
  );
};

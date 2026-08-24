import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mission } from '../types';
import { missionApi } from '../services/missionApi';
import { MissionMap } from '../components/gis/MissionMap';
import {
  Compass,
  Plus,
  Search,
  MapPin,
  ArrowRight,
  Flame,
  LayoutGrid,
} from 'lucide-react';
import { GlassCard, GlassButton, GlassBadge } from '../components/glass/GlassCard';

export const MissionsPage: React.FC = () => {
  const navigate = useNavigate();
  const [missions, setMissions] = useState<Mission[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [viewMode, setViewMode] = useState<'GRID' | 'HEATMAP_GIS'>('GRID');

  // New Mission form state
  const [newMissionName, setNewMissionName] = useState('');
  const [newSonarSource, setNewSonarSource] = useState<'Side-Scan Sonar (SSS)' | 'Synthetic Aperture Sonar (SAS)'>('Side-Scan Sonar (SSS)');
  const [newFrequency, setNewFrequency] = useState<number>(455);
  const [newLocation, setNewLocation] = useState('Coastal Survey Zone');

  useEffect(() => {
    missionApi.getMissions().then(setMissions);
  }, []);

  const handleCreateMission = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMissionName) return;

    const created = await missionApi.createMission({
      name: newMissionName,
      sonarSource: newSonarSource,
      frequencyKhz: newFrequency,
      location: newLocation,
      coordinates: [9.1524, 79.2819],
    });

    setMissions([created, ...missions]);
    setShowCreateModal(false);
    setNewMissionName('');
  };

  const filteredMissions = missions.filter(
    (m) =>
      m.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.codeName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="p-4 md:p-6 max-w-[1600px] mx-auto w-full font-sans space-y-6">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
        <div>
          <h1 className="text-xl font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-2.5">
            <Compass className="w-6 h-6 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
            MARINE SURVEY MISSIONS
          </h1>
          <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-1">
            Active acoustic survey logs, autonomous AUV tracks, and target density records
          </p>
        </div>

        <div className="flex items-center gap-2">
          <GlassButton
            variant="primary"
            size="md"
            icon={<Plus className="w-4 h-4" />}
            onClick={() => setShowCreateModal(true)}
          >
            New Sonar Mission
          </GlassButton>
        </div>
      </div>

      {/* Search & View Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search missions by name, ID, sector, or vessel..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900 placeholder-slate-500 focus:outline-none focus:border-cyan-400"
          />
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-1 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-100 border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 p-1 rounded-xl text-xs">
          <button
            onClick={() => setViewMode('GRID')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide flex items-center gap-1.5 transition-all whitespace-nowrap ${
              viewMode === 'GRID'
                ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300'
                : 'text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900'
            }`}
          >
            <LayoutGrid className="w-3.5 h-3.5" />
            <span>Mission Cards</span>
          </button>
          <button
            onClick={() => setViewMode('HEATMAP_GIS')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide flex items-center gap-1.5 transition-all whitespace-nowrap ${
              viewMode === 'HEATMAP_GIS'
                ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300 shadow-sm'
                : 'text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900'
            }`}
          >
            <Flame className="w-3.5 h-3.5 text-amber-400" />
            <span>Coverage Heatmap</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {viewMode === 'HEATMAP_GIS' ? (
        <div className="space-y-3">
          <GlassCard variant="glow" className="p-3.5 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-amber-400 animate-pulse" />
              <span className="font-bold text-white dark:text-white light:text-slate-900 uppercase tracking-wide">
                GLOBAL ACOUSTIC PING DENSITY COMMAND CENTER
              </span>
            </div>
            <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-xs">
              Aggregated across <strong className="text-cyan-300 dark:text-cyan-300 light:text-sky-800">{missions.length}</strong> Historical Surveys
            </span>
          </GlassCard>

          <div className="h-[640px] rounded-2xl overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 shadow-lg">
            {missions.length > 0 && (
              <MissionMap
                mission={missions[0]}
                allMissions={missions}
                initialHeatmapActive={true}
                className="h-full w-full"
              />
            )}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-12 gap-4">
          {filteredMissions.map((mission) => (
            <GlassCard
              key={mission.id}
              variant="default"
              className="col-span-12 lg:col-span-6 p-5 flex flex-col justify-between group cursor-pointer transition-all hover:scale-[1.01]"
              onClick={() => navigate(`/missions/${mission.id}`)}
            >
              <div>
                <div className="flex items-center justify-between pb-2.5 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 text-sm">{mission.id}</span>
                    <GlassBadge variant="cyan" size="sm">
                      {mission.codeName}
                    </GlassBadge>
                  </div>
                  <GlassBadge
                    variant={
                      mission.status === 'Active'
                        ? 'emerald'
                        : mission.status === 'Processing'
                        ? 'amber'
                        : 'slate'
                    }
                    size="sm"
                  >
                    {mission.status.toUpperCase()}
                  </GlassBadge>
                </div>

                <h2 className="text-base font-bold text-white dark:text-white light:text-slate-900 mt-3 group-hover:text-cyan-300 dark:group-hover:text-cyan-300 light:group-hover:text-sky-700 transition-colors">
                  {mission.name}
                </h2>
                <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-1 line-clamp-2">
                  {mission.targetObjective}
                </p>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 my-3 text-xs">
                  <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[10px] uppercase font-bold">Sonar Source</div>
                    <div className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-semibold truncate mt-0.5">{mission.sonarSource}</div>
                  </div>
                  <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[10px] uppercase font-bold">Frequency</div>
                    <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-semibold mt-0.5 font-mono">{mission.frequencyKhz} kHz</div>
                  </div>
                  <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[10px] uppercase font-bold">Detections</div>
                    <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-semibold mt-0.5 font-mono">{mission.detectionsCount} Targets</div>
                  </div>
                  <div className="bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[10px] uppercase font-bold">Survey Area</div>
                    <div className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-semibold mt-0.5 font-mono">{mission.areaSqKm} km²</div>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 flex items-center justify-between text-xs text-slate-400 dark:text-slate-400 light:text-slate-600">
                <div className="flex items-center gap-1.5">
                  <MapPin className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                  <span className="truncate max-w-[220px]">{mission.location}</span>
                </div>
                <div className="flex items-center gap-1 text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-bold group-hover:translate-x-1 transition-transform">
                  <span>Mission Workspace</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              </div>
            </GlassCard>
          ))}
        </div>
      )}

      {/* Modal: New Sonar Mission */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md flex items-center justify-center p-4">
          <GlassCard variant="glow" className="w-full max-w-md p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2">
              <h3 className="text-sm font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 flex items-center gap-2">
                <Compass className="w-4 h-4" />
                NEW SURVEY MISSION CONFIGURATION
              </h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateMission} className="space-y-3.5">
              <div>
                <label className="block text-slate-300 dark:text-slate-300 light:text-slate-700 text-xs font-bold mb-1">Mission Name:</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Lakshadweep Subsea Habitat Survey"
                  value={newMissionName}
                  onChange={(e) => setNewMissionName(e.target.value)}
                  className="w-full bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded-xl px-3 py-2 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900 focus:outline-none focus:border-cyan-400"
                />
              </div>

              <div>
                <label className="block text-slate-300 dark:text-slate-300 light:text-slate-700 text-xs font-bold mb-1">Location / Sector:</label>
                <input
                  type="text"
                  value={newLocation}
                  onChange={(e) => setNewLocation(e.target.value)}
                  className="w-full bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded-xl px-3 py-2 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900 focus:outline-none focus:border-cyan-400"
                />
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="block text-slate-300 dark:text-slate-300 light:text-slate-700 text-xs font-bold mb-1">Sonar Source:</label>
                  <select
                    value={newSonarSource}
                    onChange={(e: any) => setNewSonarSource(e.target.value)}
                    className="w-full bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded-xl px-2.5 py-2 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900"
                  >
                    <option value="Side-Scan Sonar (SSS)">Side-Scan Sonar (SSS)</option>
                    <option value="Synthetic Aperture Sonar (SAS)">Synthetic Aperture (SAS)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-slate-300 dark:text-slate-300 light:text-slate-700 text-xs font-bold mb-1">Frequency (kHz):</label>
                  <input
                    type="number"
                    value={newFrequency}
                    onChange={(e) => setNewFrequency(Number(e.target.value))}
                    className="w-full bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded-xl px-3 py-2 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900"
                  />
                </div>
              </div>

              <div className="pt-3 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 flex justify-end gap-2">
                <GlassButton
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowCreateModal(false)}
                >
                  Cancel
                </GlassButton>
                <GlassButton
                  type="submit"
                  variant="primary"
                  size="sm"
                >
                  Create Mission
                </GlassButton>
              </div>
            </form>
          </GlassCard>
        </div>
      )}
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Detection, Mission, ModelInfo, DatasetInfo, SystemTelemetry } from '../types';
import { missionApi } from '../services/missionApi';
import { detectionApi } from '../services/detectionApi';
import { systemApi } from '../services/systemApi';
import {
  BarChart3,
  Box,
  Database,
  Activity,
  PieChart,
  TrendingUp,
  Cpu,
  Zap,
  Shield,
  RefreshCw,
  CheckCircle2,
  Server,
  Filter,
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton, KpiCard } from '../components/glass/GlassCard';

export const AnalyticsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTab = searchParams.get('tab') || 'analytics';
  const [activeTab, setActiveTab] = useState<'analytics' | 'models' | 'datasets' | 'system'>(
    (initialTab as any) || 'analytics'
  );

  const [missions, setMissions] = useState<Mission[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(null);

  const [selectedMissionId, setSelectedMissionId] = useState<string>('ALL');
  const [validatingDatasetId, setValidatingDatasetId] = useState<string | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);

  useEffect(() => {
    missionApi.getMissions().then(setMissions);
    detectionApi.getDetections().then(setDetections);
    systemApi.getModels().then(setModels);
    systemApi.getDatasets().then(setDatasets);

    const fetchTelem = async () => {
      try {
        const data = await systemApi.getTelemetry();
        setTelemetry(data);
      } catch {}
    };
    fetchTelem();
    const interval = setInterval(fetchTelem, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (tab: 'analytics' | 'models' | 'datasets' | 'system') => {
    setActiveTab(tab);
    setSearchParams({ tab });
  };

  const handleValidateDataset = async (datasetId: string) => {
    setValidatingDatasetId(datasetId);
    setValidationMessage(null);
    const res = await systemApi.triggerDatasetValidation(datasetId);
    setValidatingDatasetId(null);
    setValidationMessage(res.message);
    setTimeout(() => setValidationMessage(null), 4000);
  };

  const filteredDetections =
    selectedMissionId === 'ALL'
      ? detections
      : detections.filter((d) => d.missionId === selectedMissionId);

  // Class breakdown
  const classCounts: Record<string, number> = {};
  filteredDetections.forEach((d) => {
    classCounts[d.classNameLabel] = (classCounts[d.classNameLabel] || 0) + 1;
  });

  const confBins = [
    { label: '< 60%', count: filteredDetections.filter((d) => d.confidence < 0.6).length },
    { label: '60 - 75%', count: filteredDetections.filter((d) => d.confidence >= 0.6 && d.confidence < 0.75).length },
    { label: '75 - 85%', count: filteredDetections.filter((d) => d.confidence >= 0.75 && d.confidence < 0.85).length },
    { label: '85 - 95%', count: filteredDetections.filter((d) => d.confidence >= 0.85 && d.confidence < 0.95).length },
    { label: '95 - 100%', count: filteredDetections.filter((d) => d.confidence >= 0.95).length },
  ];

  const anomalyBins = [
    { label: 'Low (<0.5)', count: filteredDetections.filter((d) => d.anomalyScore < 0.5).length },
    { label: 'Medium (0.5-0.8)', count: filteredDetections.filter((d) => d.anomalyScore >= 0.5 && d.anomalyScore < 0.8).length },
    { label: 'High (0.8-0.95)', count: filteredDetections.filter((d) => d.anomalyScore >= 0.8 && d.anomalyScore < 0.95).length },
    { label: 'Critical (>0.95)', count: filteredDetections.filter((d) => d.anomalyScore >= 0.95).length },
  ];

  const maxClassCount = Math.max(...Object.values(classCounts), 1);
  const maxConfCount = Math.max(...confBins.map((b) => b.count), 1);
  const maxAnomalyCount = Math.max(...anomalyBins.map((b) => b.count), 1);

  return (
    <div className="p-4 md:p-6 max-w-[1700px] mx-auto w-full font-sans space-y-6">
      {/* Top Header & Integrated Tab Navigation */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
        <div>
          <h1 className="text-xl font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-2.5 uppercase tracking-wide">
            <BarChart3 className="w-6 h-6 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
            INTELLIGENCE & MODEL REGISTRY
          </h1>
          <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-1">
            Operational analytics, TensorRT neural benchmarks, training datasets, and telemetry
          </p>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-100 border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 p-1 rounded-xl text-xs gap-1">
          <button
            onClick={() => handleTabChange('analytics')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-bold tracking-wide transition-all whitespace-nowrap ${
              activeTab === 'analytics'
                ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300'
                : 'text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Analytics</span>
          </button>

          <button
            onClick={() => handleTabChange('models')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-bold tracking-wide transition-all whitespace-nowrap ${
              activeTab === 'models'
                ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300'
                : 'text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900'
            }`}
          >
            <Box className="w-3.5 h-3.5" />
            <span>Models & Benchmarks</span>
          </button>

          <button
            onClick={() => handleTabChange('datasets')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-bold tracking-wide transition-all whitespace-nowrap ${
              activeTab === 'datasets'
                ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300'
                : 'text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Datasets & ETL</span>
          </button>

          <button
            onClick={() => handleTabChange('system')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-bold tracking-wide transition-all whitespace-nowrap ${
              activeTab === 'system'
                ? 'bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300'
                : 'text-slate-400 hover:text-white dark:hover:text-white light:hover:text-slate-900'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Hardware Telemetry</span>
          </button>
        </div>
      </div>

      {/* TAB 1: ACOUSTIC ANALYTICS */}
      {activeTab === 'analytics' && (
        <div className="space-y-4">
          {/* Filter Bar & KPI Strip */}
          <GlassCard variant="glow" className="p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
              <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase text-[10px] font-bold">Survey Dataset Filter:</span>
              <select
                value={selectedMissionId}
                onChange={(e) => setSelectedMissionId(e.target.value)}
                className="bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 rounded-lg px-2.5 py-1 text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-semibold focus:outline-none focus:border-cyan-400 text-xs"
              >
                <option value="ALL">All Combined Missions ({missions.length})</option>
                {missions.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.id})
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-4 text-slate-300 dark:text-slate-300 light:text-slate-700 text-xs">
              <div>
                Analyzed Targets: <strong className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono font-bold">{filteredDetections.length}</strong>
              </div>
              <div className="h-4 w-[1px] bg-cyan-900/40 dark:bg-cyan-900/40 light:bg-slate-300" />
              <div>
                Mean Confidence:{' '}
                <strong className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-mono font-bold">
                  {(
                    (filteredDetections.reduce((acc, d) => acc + d.confidence, 0) /
                      Math.max(1, filteredDetections.length)) *
                    100
                  ).toFixed(1)}
                  %
                </strong>
              </div>
            </div>
          </GlassCard>

          {/* Charts Grid (12-Column Grid Alignment) */}
          <div className="grid grid-cols-12 gap-4">
            {/* Classification Spectrum */}
            <GlassCard variant="default" className="col-span-12 lg:col-span-6 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2.5 text-xs">
                <span className="font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-2 uppercase tracking-wide">
                  <PieChart className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                  Target Classification Spectrum
                </span>
                <GlassBadge variant="cyan" size="sm">
                  {Object.keys(classCounts).length} Acoustic Classes
                </GlassBadge>
              </div>

              <div className="space-y-3 pt-1">
                {Object.entries(classCounts).map(([label, count]) => {
                  const pct = ((count / filteredDetections.length) * 100).toFixed(0);
                  const barPct = ((count / maxClassCount) * 100).toFixed(0);

                  return (
                    <div key={label} className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold">{label}</span>
                        <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">
                          {count} ({pct}%)
                        </span>
                      </div>
                      <div className="w-full h-2 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-200 rounded-full overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300">
                        <div
                          style={{ width: `${barPct}%` }}
                          className="h-full bg-gradient-to-r from-cyan-600 to-teal-400 rounded-full transition-all duration-300"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </GlassCard>

            {/* Neural Confidence Distribution */}
            <GlassCard variant="default" className="col-span-12 lg:col-span-6 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2.5 text-xs">
                <span className="font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-2 uppercase tracking-wide">
                  <Activity className="w-4 h-4 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" />
                  Neural Confidence Density
                </span>
                <GlassBadge variant="emerald" size="sm">
                  YOLOv11 + SAM2 Fusion
                </GlassBadge>
              </div>

              <div className="space-y-3 pt-1">
                {confBins.map((bin) => {
                  const barPct = ((bin.count / maxConfCount) * 100).toFixed(0);
                  return (
                    <div key={bin.label} className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold">{bin.label}</span>
                        <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">{bin.count} Targets</span>
                      </div>
                      <div className="w-full h-2 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-200 rounded-full overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300">
                        <div
                          style={{ width: `${barPct}%` }}
                          className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </GlassCard>

            {/* Anomaly Deviation Index */}
            <GlassCard variant="default" className="col-span-12 lg:col-span-6 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2.5 text-xs">
                <span className="font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-2 uppercase tracking-wide">
                  <Shield className="w-4 h-4 text-amber-400" />
                  Seabed Anomaly Deviation Index
                </span>
                <GlassBadge variant="amber" size="sm">
                  PatchCore Memory Bank
                </GlassBadge>
              </div>

              <div className="space-y-3 pt-1">
                {anomalyBins.map((bin) => {
                  const barPct = ((bin.count / maxAnomalyCount) * 100).toFixed(0);
                  return (
                    <div key={bin.label} className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold">{bin.label}</span>
                        <span className="text-amber-300 dark:text-amber-300 light:text-amber-700 font-bold font-mono">{bin.count} Targets</span>
                      </div>
                      <div className="w-full h-2 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-200 rounded-full overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300">
                        <div
                          style={{ width: `${barPct}%` }}
                          className="h-full bg-amber-500 rounded-full transition-all duration-300"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </GlassCard>

            {/* Survey Comparison Benchmark */}
            <GlassCard variant="default" className="col-span-12 lg:col-span-6 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2.5 text-xs">
                <span className="font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-2 uppercase tracking-wide">
                  <TrendingUp className="w-4 h-4 text-purple-400" />
                  Cross-Survey Benchmark
                </span>
                <GlassBadge variant="purple" size="sm">
                  {missions.length} Missions
                </GlassBadge>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 uppercase font-bold">
                    <tr>
                      <th className="pb-2.5">Mission ID</th>
                      <th className="pb-2.5">Sonar Type</th>
                      <th className="pb-2.5">Targets</th>
                      <th className="pb-2.5">Avg SNR</th>
                      <th className="pb-2.5">FPS</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-cyan-900/20 dark:divide-cyan-900/20 light:divide-slate-200">
                    {missions.map((m) => (
                      <tr key={m.id} className="hover:bg-cyan-950/20 dark:hover:bg-cyan-950/20 light:hover:bg-sky-50 transition-colors">
                        <td className="py-2.5 font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono">{m.id}</td>
                        <td className="py-2.5 text-slate-300 dark:text-slate-300 light:text-slate-700 truncate max-w-[130px] font-semibold">{m.sonarSource}</td>
                        <td className="py-2.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-semibold font-mono">{m.detectionsCount}</td>
                        <td className="py-2.5 text-slate-300 dark:text-slate-300 light:text-slate-700 font-mono">{m.summaryMetrics.avgSnrDb} dB</td>
                        <td className="py-2.5 text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold font-mono">{m.summaryMetrics.meanProcessingFps}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </GlassCard>
          </div>
        </div>
      )}

      {/* TAB 2: AI MODELS & BENCHMARKS */}
      {activeTab === 'models' && (
        <div className="space-y-4">
          <div className="grid grid-cols-12 gap-4">
            {models.map((model) => (
              <GlassCard
                key={model.id}
                variant="default"
                className="col-span-12 lg:col-span-6 p-5 space-y-4"
              >
                <div className="flex items-center justify-between pb-2.5 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                  <div>
                    <span className="font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 text-sm">{model.name}</span>
                    <span className="text-xs ml-2 text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">({model.version})</span>
                  </div>
                  <GlassBadge
                    variant={model.status === 'ACTIVE_PRODUCTION' ? 'emerald' : 'amber'}
                    size="sm"
                  >
                    {model.status.replace('_', ' ')}
                  </GlassBadge>
                </div>

                <div className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 space-y-1">
                  <div>Category: <strong className="text-slate-200 dark:text-slate-200 light:text-slate-800">{model.category}</strong></div>
                  <div className="truncate">Backbone: <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-mono">{model.backbone}</span></div>
                </div>

                {/* Specs */}
                <div className="grid grid-cols-4 gap-2 text-center text-xs">
                  <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] uppercase font-bold">Precision</div>
                    <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">{model.precision}</div>
                  </div>
                  <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] uppercase font-bold">Tensor</div>
                    <div className="text-white dark:text-white light:text-slate-900 font-bold font-mono">{model.inputSize}</div>
                  </div>
                  <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] uppercase font-bold">Latency</div>
                    <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">{model.latencyMs} ms</div>
                  </div>
                  <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] uppercase font-bold">Engine</div>
                    <div className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold truncate text-xs font-mono">{model.onnxStatus.split(' ')[0]}</div>
                  </div>
                </div>

                {/* Benchmarks */}
                <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 rounded-xl p-3.5 space-y-2.5">
                  <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold flex justify-between">
                    <span>Validation Benchmarks</span>
                    <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-700">Trained on {model.datasetName}</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-center text-xs">
                    <div className="bg-[#0A121E]/80 dark:bg-[#0A121E]/80 light:bg-white p-2 rounded-lg border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200">
                      <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] font-bold">mAP@50</div>
                      <div className="text-white dark:text-white light:text-slate-900 font-bold font-mono">{(model.metrics.mAP50 * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-[#0A121E]/80 dark:bg-[#0A121E]/80 light:bg-white p-2 rounded-lg border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200">
                      <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] font-bold">Precision</div>
                      <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">{(model.metrics.precision * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-[#0A121E]/80 dark:bg-[#0A121E]/80 light:bg-white p-2 rounded-lg border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200">
                      <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] font-bold">Recall</div>
                      <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">{(model.metrics.recall * 100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-[#0A121E]/80 dark:bg-[#0A121E]/80 light:bg-white p-2 rounded-lg border border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200">
                      <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[9px] font-bold">F1-Score</div>
                      <div className="text-amber-300 dark:text-amber-300 light:text-amber-700 font-bold font-mono">{(model.metrics.f1Score * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}

      {/* TAB 3: DATASETS & ETL */}
      {activeTab === 'datasets' && (
        <div className="space-y-4">
          {validationMessage && (
            <div className="bg-emerald-950/80 dark:bg-emerald-950/80 light:bg-emerald-50 border border-emerald-500/50 text-emerald-300 dark:text-emerald-300 light:text-emerald-800 px-4 py-2.5 rounded-xl text-xs flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>{validationMessage}</span>
            </div>
          )}

          <div className="grid grid-cols-12 gap-4">
            {datasets.map((dataset) => (
              <GlassCard
                key={dataset.id}
                variant="default"
                className="col-span-12 lg:col-span-6 p-5 flex flex-col justify-between space-y-4"
              >
                <div>
                  <div className="flex items-center justify-between pb-2.5 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                    <div>
                      <span className="font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 text-sm">{dataset.name}</span>
                      <span className="text-xs ml-2 text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">({dataset.id})</span>
                    </div>
                    <GlassBadge variant="emerald" size="sm">
                      {dataset.status}
                    </GlassBadge>
                  </div>

                  <div className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-2">
                    Source: <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-semibold">{dataset.source}</span>
                  </div>

                  <div className="grid grid-cols-3 gap-2.5 my-3 text-xs">
                    <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                      <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[10px] font-bold uppercase">Images</div>
                      <div className="text-white dark:text-white light:text-slate-900 font-bold font-mono mt-0.5">{dataset.imagesCount.toLocaleString()}</div>
                    </div>
                    <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                      <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[10px] font-bold uppercase">Annotations</div>
                      <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono mt-0.5">{dataset.annotationsCount.toLocaleString()}</div>
                    </div>
                    <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
                      <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 text-[10px] font-bold uppercase">Storage Size</div>
                      <div className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-bold font-mono mt-0.5">{(dataset.storageMb / 1024).toFixed(1)} GB</div>
                    </div>
                  </div>

                  {/* SHA256 */}
                  <div className="bg-[#050B14]/60 dark:bg-[#050B14]/60 light:bg-slate-50 p-2.5 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 text-xs text-slate-400 dark:text-slate-400 light:text-slate-600">
                    <div className="text-slate-500 dark:text-slate-500 light:text-slate-600 flex items-center gap-1.5 mb-1 text-[10px] font-bold uppercase">
                      <Shield className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                      <span>SHA-256 Checksum:</span>
                    </div>
                    <div className="font-mono text-slate-300 dark:text-slate-300 light:text-slate-800 text-[11px] truncate">{dataset.sha256}</div>
                  </div>
                </div>

                <div className="pt-3 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 flex items-center justify-between text-xs">
                  <span className="text-[11px] text-slate-500 dark:text-slate-500 light:text-slate-600">Updated: {dataset.lastUpdated}</span>
                  <GlassButton
                    variant="secondary"
                    size="sm"
                    onClick={() => handleValidateDataset(dataset.id)}
                    disabled={validatingDatasetId === dataset.id}
                    icon={
                      <RefreshCw
                        className={`w-3.5 h-3.5 ${validatingDatasetId === dataset.id ? 'animate-spin' : ''}`}
                      />
                    }
                  >
                    {validatingDatasetId === dataset.id ? 'Validating...' : 'Validate SHA-256'}
                  </GlassButton>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}

      {/* TAB 4: HARDWARE TELEMETRY (Standardized KPI Cards in 12-Column Grid) */}
      {activeTab === 'system' && telemetry && (
        <div className="space-y-4">
          <div className="grid grid-cols-12 gap-4">
            {/* GPU */}
            <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
              <KpiCard className="kpi-card-interactive">
                <div className="kpi-header">
                  <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold flex items-center gap-1.5">
                    <Cpu className="w-4 h-4" />
                    NVIDIA RTX 5060 GPU
                  </span>
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">{telemetry.temperatureCelsius}°C</span>
                </div>
                <div className="kpi-body">
                  <div className="text-3xl font-extrabold text-white dark:text-white light:text-slate-900 font-mono my-1">{telemetry.gpuUtilPct}%</div>
                  <div className="w-full h-2 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-200 rounded-full overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300">
                    <div
                      style={{ width: `${telemetry.gpuUtilPct}%` }}
                      className="h-full bg-emerald-500 rounded-full"
                    />
                  </div>
                </div>
                <div className="kpi-footer font-mono">
                  <span>CUDA 12.8 • TensorRT</span>
                  <span>3,840 Cores</span>
                </div>
              </KpiCard>
            </div>

            {/* VRAM */}
            <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
              <KpiCard className="kpi-card-interactive">
                <div className="kpi-header">
                  <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold flex items-center gap-1.5">
                    <Zap className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                    VRAM ALLOCATION
                  </span>
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">
                    {Math.round((telemetry.vramUsedGb / telemetry.vramTotalGb) * 100)}%
                  </span>
                </div>
                <div className="kpi-body">
                  <div className="text-3xl font-extrabold text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono my-1">
                    {telemetry.vramUsedGb} GB
                    <span className="text-xs text-slate-500 dark:text-slate-500 light:text-slate-600 font-normal ml-1">/ {telemetry.vramTotalGb} GB</span>
                  </div>
                  <div className="w-full h-2 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-200 rounded-full overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300">
                    <div
                      style={{
                        width: `${Math.round((telemetry.vramUsedGb / telemetry.vramTotalGb) * 100)}%`,
                      }}
                      className="h-full bg-cyan-500 rounded-full"
                    />
                  </div>
                </div>
                <div className="kpi-footer font-mono">
                  <span>Weights: 2.18 GB</span>
                  <span>Buffers: 1.42 GB</span>
                </div>
              </KpiCard>
            </div>

            {/* CPU */}
            <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
              <KpiCard className="kpi-card-interactive">
                <div className="kpi-header">
                  <span className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold flex items-center gap-1.5">
                    <Server className="w-4 h-4 text-purple-400" />
                    HOST CPU & RAM
                  </span>
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">16 Threads</span>
                </div>
                <div className="kpi-body">
                  <div className="text-3xl font-extrabold text-white dark:text-white light:text-slate-900 font-mono my-1">{telemetry.cpuUtilPct}%</div>
                  <div className="w-full h-2 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-200 rounded-full overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300">
                    <div
                      style={{ width: `${telemetry.cpuUtilPct}%` }}
                      className="h-full bg-purple-500 rounded-full"
                    />
                  </div>
                </div>
                <div className="kpi-footer font-mono">
                  <span>RAM: {telemetry.ramUsedGb} GB</span>
                  <span>{telemetry.activeWorkers} OpenCV Workers</span>
                </div>
              </KpiCard>
            </div>

            {/* Throughput */}
            <div className="col-span-12 sm:col-span-6 lg:col-span-3 flex">
              <KpiCard className="kpi-card-interactive">
                <div className="kpi-header">
                  <span className="text-amber-300 dark:text-amber-300 light:text-amber-700 font-bold flex items-center gap-1.5">
                    <Zap className="w-4 h-4 text-amber-400" />
                    INFERENCE SPEED
                  </span>
                  <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">{telemetry.latencyMs} ms</span>
                </div>
                <div className="kpi-body">
                  <div className="text-3xl font-extrabold text-amber-300 dark:text-amber-300 light:text-amber-700 font-mono my-1">{telemetry.inferenceFps} <span className="text-sm font-normal">FPS</span></div>
                  <div className="w-full h-2 bg-[#050B14]/80 dark:bg-[#050B14]/80 light:bg-slate-200 rounded-full overflow-hidden border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300">
                    <div
                      style={{ width: `${Math.min(100, (telemetry.inferenceFps / 60) * 100)}%` }}
                      className="h-full bg-amber-400 rounded-full"
                    />
                  </div>
                </div>
                <div className="kpi-footer font-mono">
                  <span>PyTorch {telemetry.pytorchVersion}</span>
                  <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">FP16 TensorRT</span>
                </div>
              </KpiCard>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

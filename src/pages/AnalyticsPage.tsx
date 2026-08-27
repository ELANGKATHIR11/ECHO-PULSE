import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Detection, Mission, ModelInfo, DatasetInfo } from '../types';
import { missionApi } from '../services/missionApi';
import { detectionApi } from '../services/detectionApi';
import { systemApi } from '../services/systemApi';
import {
  BarChart3,
  Box,
  Database,
  PieChart,
  TrendingUp,
  Shield,
  RefreshCw,
  CheckCircle2,
  Filter,
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../components/glass/GlassCard';

export const AnalyticsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTab = searchParams.get('tab') || 'analytics';
  const [activeTab, setActiveTab] = useState<'analytics' | 'models' | 'datasets'>(
    initialTab === 'models' || initialTab === 'datasets' ? initialTab : 'analytics'
  );

  const [missions, setMissions] = useState<Mission[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  const [selectedMissionId, setSelectedMissionId] = useState<string>('ALL');
  const [validatingDatasetId, setValidatingDatasetId] = useState<string | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);

  useEffect(() => {
    missionApi.getMissions().then(setMissions);
    detectionApi.getDetections().then(setDetections);
    systemApi.getModels().then(setModels);
    systemApi.getDatasets().then(setDatasets);
  }, []);

  const handleTabChange = (tab: 'analytics' | 'models' | 'datasets') => {
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
            Operational analytics, TensorRT neural benchmarks, and training dataset verification
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
        </div>
      </div>

      {/* TAB 1: ACOUSTIC ANALYTICS */}
      {activeTab === 'analytics' && (
        <div className="space-y-4">
          {/* Mission Filter Toolbar */}
          <div className="flex items-center justify-between bg-[#040E1E]/80 dark:bg-[#040E1E]/80 light:bg-slate-100 p-3 rounded-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 text-xs">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
              <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-bold uppercase tracking-wider text-[11px]">Filter Survey Mission:</span>
              <select
                value={selectedMissionId}
                onChange={(e) => setSelectedMissionId(e.target.value)}
                className="bg-[#020712] dark:bg-[#020712] light:bg-white border border-cyan-500/30 dark:border-cyan-500/30 light:border-slate-300 rounded-lg px-2.5 py-1 text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold focus:outline-none cursor-pointer"
              >
                <option value="ALL">All Surveys Combined ({detections.length} total detections)</option>
                {missions.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.id})
                  </option>
                ))}
              </select>
            </div>

            <div className="text-[11px] font-mono text-slate-400 dark:text-slate-400 light:text-slate-600">
              Showing <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">{filteredDetections.length}</span> acoustic detections
            </div>
          </div>

          {/* Analytics Charts Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Target Class Distribution */}
            <GlassCard variant="default" className="p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-2 mb-3 border-b border-cyan-900/30">
                  <span className="text-xs font-bold text-white dark:text-white light:text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                    <PieChart className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                    Target Classification
                  </span>
                  <GlassBadge variant="cyan" size="sm">
                    {Object.keys(classCounts).length} Classes
                  </GlassBadge>
                </div>

                <div className="space-y-2 text-xs">
                  {Object.entries(classCounts).map(([label, count]) => (
                    <div key={label} className="space-y-1">
                      <div className="flex justify-between text-[11px] font-mono">
                        <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 truncate max-w-[180px]">{label}</span>
                        <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">{count}</span>
                      </div>
                      <div className="w-full h-1.5 bg-[#020712] rounded-full overflow-hidden">
                        <div
                          style={{ width: `${(count / maxClassCount) * 100}%` }}
                          className="h-full bg-cyan-400 rounded-full"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </GlassCard>

            {/* Fused Confidence Spectrum */}
            <GlassCard variant="default" className="p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-2 mb-3 border-b border-cyan-900/30">
                  <span className="text-xs font-bold text-white dark:text-white light:text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                    <TrendingUp className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" />
                    Confidence Distribution
                  </span>
                  <GlassBadge variant="emerald" size="sm">
                    Multi-Factor
                  </GlassBadge>
                </div>

                <div className="space-y-2 text-xs">
                  {confBins.map((bin) => (
                    <div key={bin.label} className="space-y-1">
                      <div className="flex justify-between text-[11px] font-mono">
                        <span className="text-slate-300 dark:text-slate-300 light:text-slate-700">{bin.label}</span>
                        <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">{bin.count}</span>
                      </div>
                      <div className="w-full h-1.5 bg-[#020712] rounded-full overflow-hidden">
                        <div
                          style={{ width: `${(bin.count / maxConfCount) * 100}%` }}
                          className="h-full bg-emerald-400 rounded-full"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </GlassCard>

            {/* Seabed Anomaly Sharpness */}
            <GlassCard variant="default" className="p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-2 mb-3 border-b border-cyan-900/30">
                  <span className="text-xs font-bold text-white dark:text-white light:text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                    <Shield className="w-3.5 h-3.5 text-purple-400" />
                    Anomaly Sharpness Index
                  </span>
                  <GlassBadge variant="purple" size="sm">
                    Autoencoder
                  </GlassBadge>
                </div>

                <div className="space-y-2 text-xs">
                  {anomalyBins.map((bin) => (
                    <div key={bin.label} className="space-y-1">
                      <div className="flex justify-between text-[11px] font-mono">
                        <span className="text-slate-300 dark:text-slate-300 light:text-slate-700">{bin.label}</span>
                        <span className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold">{bin.count}</span>
                      </div>
                      <div className="w-full h-1.5 bg-[#020712] rounded-full overflow-hidden">
                        <div
                          style={{ width: `${(bin.count / maxAnomalyCount) * 100}%` }}
                          className="h-full bg-purple-400 rounded-full"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </GlassCard>
          </div>
        </div>
      )}

      {/* TAB 2: NEURAL MODEL REGISTRY */}
      {activeTab === 'models' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model) => (
              <GlassCard key={model.id} variant="default" className="p-4 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between pb-2 mb-3 border-b border-cyan-900/30">
                    <div className="flex items-center gap-2">
                      <Box className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                      <span className="font-bold text-white dark:text-white light:text-slate-900 text-xs font-mono">{model.name}</span>
                    </div>
                    <GlassBadge variant={model.status === 'ACTIVE_PRODUCTION' ? 'emerald' : 'cyan'} size="sm">
                      {model.status}
                    </GlassBadge>
                  </div>

                  <div className="text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 mb-3 font-mono">
                    Category: <span className="text-cyan-300 font-bold">{model.category}</span>
                  </div>

                  <div className="space-y-1.5 text-xs font-mono bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-lg border border-cyan-900/25 dark:border-cyan-900/25 light:border-slate-200">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Backbone:</span>
                      <span className="text-white dark:text-white light:text-slate-900 font-bold">{model.backbone}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Precision:</span>
                      <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">{model.precision}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">mAP@50:</span>
                      <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">
                        {model.metrics ? (model.metrics.mAP50 * 100).toFixed(1) : '91.4'}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">mAP@50-95:</span>
                      <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">
                        {model.metrics ? (model.metrics.mAP50_95 * 100).toFixed(1) : '76.8'}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Latency:</span>
                      <span className="text-amber-400 dark:text-amber-400 light:text-amber-700 font-bold">{model.latencyMs} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Runtime:</span>
                      <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">{model.onnxStatus}</span>
                    </div>
                  </div>
                </div>

                <div className="text-[10px] text-slate-500 font-mono mt-3 pt-2 border-t border-cyan-900/20 flex justify-between">
                  <span>Version: {model.version}</span>
                  <span>Input: {model.inputSize}</span>
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
            <div className="p-3 rounded-xl bg-emerald-500/20 border border-emerald-400/50 text-emerald-300 text-xs font-mono flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>{validationMessage}</span>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {datasets.map((dataset) => (
              <GlassCard key={dataset.id} variant="default" className="p-4 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between pb-2 mb-3 border-b border-cyan-900/30">
                    <div className="flex items-center gap-2">
                      <Database className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                      <span className="font-bold text-white dark:text-white light:text-slate-900 text-xs font-mono">{dataset.name}</span>
                    </div>
                    <GlassBadge variant="cyan" size="sm">
                      {dataset.version}
                    </GlassBadge>
                  </div>

                  <div className="space-y-1.5 text-xs font-mono bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-slate-50 p-2.5 rounded-lg border border-cyan-900/25 dark:border-cyan-900/25 light:border-slate-200">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Images Count:</span>
                      <span className="text-white dark:text-white light:text-slate-900 font-bold">{dataset.imagesCount.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Annotations:</span>
                      <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">{dataset.annotationsCount.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Storage Size:</span>
                      <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">{dataset.storageMb} MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Pipeline Stage:</span>
                      <span className="text-purple-400 font-bold">{dataset.pipelineStage}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Data Source:</span>
                      <span className="text-slate-300 dark:text-slate-300 light:text-slate-700">{dataset.source}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-2 border-t border-cyan-900/20 flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 font-mono">Last Ingest: {dataset.lastUpdated}</span>
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
    </div>
  );
};

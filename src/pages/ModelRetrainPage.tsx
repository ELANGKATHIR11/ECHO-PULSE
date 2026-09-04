import React, { useState, useEffect } from 'react';
import {
  Cpu, Play, RefreshCw, CheckCircle, Database, BarChart2,
  Settings, Layers, TrendingUp, ShieldCheck, AlertCircle, FileText, Zap
} from 'lucide-react';

interface DatasetSummary {
  total_annotated_samples: number;
  categories: Record<string, number>;
  classes_supported: string[];
  active_model_checkpoint: string;
  last_retrained: string;
  model_architecture: string;
}

interface EpochStat {
  epoch: number;
  train_loss: number;
  val_accuracy: number;
  f1_score: number;
  learning_rate: number;
}

interface JobStatus {
  status: 'IDLE' | 'RUNNING' | 'COMPLETED';
  job_id?: string;
  current_epoch?: number;
  epochs_total?: number;
  current_loss?: number;
  current_val_acc?: number;
  f1_score?: number;
  epoch_history?: EpochStat[];
  duration_sec?: number;
  final_checkpoint?: string;
  backbone?: string;
}

export const ModelRetrainPage: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus>({ status: 'IDLE' });
  const [epochs, setEpochs] = useState(15);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [backbone, setBackbone] = useState('EchoPhys-X Marine Audio Spectrogram Transformer');
  const [isStarting, setIsStarting] = useState(false);

  // Poll status
  useEffect(() => {
    fetchSummary();
    const interval = setInterval(() => {
      fetchJobStatus();
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  const fetchSummary = () => {
    fetch('http://127.0.0.1:8000/api/retrain/datasets')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'SUCCESS') {
          setSummary(data.summary);
        }
      })
      .catch(() => {
        setSummary({
          total_annotated_samples: 1420,
          categories: {
            'Biophonic': 480,
            'Anthropogenic': 410,
            'Tactical Intruder': 310,
            'Geophonic': 220
          },
          classes_supported: [
            'Biophonic (Whales, Dolphins, Snapping Shrimp)',
            'Anthropogenic (Cargo Cavitation, Piling, Seismic Airguns)',
            'Geophonic (Hydrothermal, Seismic, Sea Rain)',
            'Tactical Intruder (AUVs, UUVs, USVs, DPV Divers)'
          ],
          active_model_checkpoint: 'echophys_x_marine_v3.pt',
          last_retrained: '2026-08-31T18:30:00Z',
          model_architecture: 'EchoPhys-X Marine Audio Spectrogram Transformer (AST)'
        });
      });
  };

  const fetchJobStatus = () => {
    fetch('http://127.0.0.1:8000/api/retrain/status')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'SUCCESS' && data.job) {
          setJobStatus(data.job);
        }
      })
      .catch(() => {});
  };

  const handleStartRetrain = () => {
    setIsStarting(true);
    const formData = new FormData();
    formData.append('epochs', epochs.toString());
    formData.append('batch_size', batchSize.toString());
    formData.append('learning_rate', learningRate.toString());
    formData.append('backbone', backbone);

    fetch('http://127.0.0.1:8000/api/retrain/start', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(() => {
        fetchJobStatus();
      })
      .finally(() => setIsStarting(false));
  };

  const history = jobStatus.epoch_history || [];

  return (
    <div className="min-h-screen bg-[#020712] text-slate-100 p-6 space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-[#0d1633] via-[#0b1b42] to-[#040e24] border border-cyan-500/30 shadow-2xl backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/20 border border-cyan-400/40 text-cyan-400">
            <Cpu className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 via-teal-200 to-indigo-300">
              ACOUSTIC MODEL RETRAINING & ACTIVE LEARNING STUDIO
            </h1>
            <p className="text-xs text-slate-400 font-mono flex items-center gap-2">
              <span>CONTINUOUS FINE-TUNING & ADAPTATION PIPELINE</span>
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span className="text-cyan-400">ON-DEMAND EDGE DEPLOYMENT</span>
            </p>
          </div>
        </div>

        {/* Current Active Checkpoint Badge */}
        <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-[#020614] border border-cyan-900/50 text-xs font-mono">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span className="text-slate-400">DEPLOYED CHECKPOINT:</span>
          <span className="text-cyan-300 font-bold">{summary?.active_model_checkpoint}</span>
        </div>
      </div>

      {/* Main Grid: Config (4 cols) & Live Telemetry (8 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Hyperparameters & Dataset Pool (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Training Hyperparameters Card */}
          <div className="p-5 rounded-2xl bg-[#040e24] border border-cyan-900/40 shadow-xl space-y-4">
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <Settings className="w-5 h-5 text-cyan-400" />
              <h2 className="text-sm font-mono font-bold tracking-wider text-slate-100">
                RETRAINING CONFIGURATION
              </h2>
            </div>

            <div className="space-y-3 font-mono text-xs">
              <div className="space-y-1">
                <label className="text-slate-400">BACKBONE ARCHITECTURE</label>
                <select
                  value={backbone}
                  onChange={e => setBackbone(e.target.value)}
                  className="w-full bg-[#020614] border border-slate-700 rounded-lg p-2 text-cyan-300 focus:outline-none focus:border-cyan-400"
                >
                  <option>EchoPhys-X Marine Audio Spectrogram Transformer</option>
                  <option>CNN-BiMamba Continuous Wave Classifier</option>
                  <option>ResNet50-Acoustic Deep Feature Head</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-slate-400">TRAINING EPOCHS</label>
                  <input
                    type="number"
                    min="5"
                    max="50"
                    value={epochs}
                    onChange={e => setEpochs(parseInt(e.target.value))}
                    className="w-full bg-[#020614] border border-slate-700 rounded-lg p-2 text-slate-200"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-slate-400">BATCH SIZE</label>
                  <input
                    type="number"
                    min="4"
                    max="64"
                    step="4"
                    value={batchSize}
                    onChange={e => setBatchSize(parseInt(e.target.value))}
                    className="w-full bg-[#020614] border border-slate-700 rounded-lg p-2 text-slate-200"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-slate-400">INITIAL LEARNING RATE</label>
                <input
                  type="number"
                  step="0.0001"
                  value={learningRate}
                  onChange={e => setLearningRate(parseFloat(e.target.value))}
                  className="w-full bg-[#020614] border border-slate-700 rounded-lg p-2 text-slate-200"
                />
              </div>

              <button
                onClick={handleStartRetrain}
                disabled={jobStatus.status === 'RUNNING' || isStarting}
                className={`w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all mt-4 ${
                  jobStatus.status === 'RUNNING'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 cursor-not-allowed animate-pulse'
                    : 'bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-[#020712] shadow-lg shadow-cyan-500/20'
                }`}
              >
                {jobStatus.status === 'RUNNING' ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                <span>{jobStatus.status === 'RUNNING' ? 'TRAINING IN PROGRESS...' : 'START CONTINUOUS RETRAINING'}</span>
              </button>
            </div>
          </div>

          {/* Dataset Pool Stats */}
          {summary && (
            <div className="p-5 rounded-2xl bg-[#040e24] border border-cyan-900/40 shadow-xl space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                <Database className="w-5 h-5 text-indigo-400" />
                <h2 className="text-sm font-mono font-bold tracking-wider text-slate-100">
                  ANNOTATED SAMPLES IN POOL
                </h2>
              </div>

              <div className="space-y-2">
                {Object.entries(summary.categories).map(([cat, count]) => (
                  <div key={cat} className="flex justify-between items-center text-xs font-mono p-2 rounded-lg bg-[#020614] border border-slate-800">
                    <span className="text-slate-300">{cat}</span>
                    <span className="text-cyan-400 font-bold">{count} recordings</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Live Training Telemetry & Curves (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Progress / Live Epoch Card */}
          <div className="p-5 rounded-2xl bg-[#040e24] border border-cyan-900/40 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-cyan-400" />
                <h2 className="text-sm font-mono font-bold tracking-wider text-slate-100">
                  LIVE TRAINING TELEMETRY & CONVERGENCE
                </h2>
              </div>
              <span className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold ${
                jobStatus.status === 'RUNNING'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse'
                  : (jobStatus.status === 'COMPLETED' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-slate-800 text-slate-400')
              }`}>
                STATUS: {jobStatus.status}
              </span>
            </div>

            {/* Metrics HUD */}
            <div className="grid grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-[#020614] border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">CURRENT EPOCH</div>
                <div className="text-xl font-mono font-black text-cyan-300">
                  {jobStatus.current_epoch || 0} / {jobStatus.epochs_total || epochs}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-[#020614] border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">TRAINING LOSS</div>
                <div className="text-xl font-mono font-black text-rose-300">
                  {jobStatus.current_loss !== undefined ? jobStatus.current_loss.toFixed(4) : '--'}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-[#020614] border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">VAL ACCURACY</div>
                <div className="text-xl font-mono font-black text-emerald-300">
                  {jobStatus.current_val_acc !== undefined ? `${jobStatus.current_val_acc.toFixed(1)}%` : '--'}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-[#020614] border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">F1-SCORE</div>
                <div className="text-xl font-mono font-black text-indigo-300">
                  {jobStatus.f1_score !== undefined ? jobStatus.f1_score.toFixed(3) : '--'}
                </div>
              </div>
            </div>

            {/* Epoch History Table */}
            <div className="space-y-2 pt-2">
              <div className="text-xs font-mono text-slate-400">EPOCH LOGS:</div>
              <div className="max-h-[220px] overflow-y-auto space-y-1.5 pr-1">
                {history.length === 0 ? (
                  <div className="p-4 rounded-xl bg-[#020614] text-center text-xs font-mono text-slate-500">
                    No active epoch telemetry yet. Click "Start Continuous Retraining" to launch.
                  </div>
                ) : (
                  history.slice().reverse().map(ep => (
                    <div key={ep.epoch} className="flex justify-between items-center p-2 rounded-lg bg-[#020614] border border-slate-800/80 text-xs font-mono">
                      <span className="text-cyan-400 font-bold">Epoch {ep.epoch}</span>
                      <span className="text-slate-400">Loss: <strong className="text-rose-300">{ep.train_loss}</strong></span>
                      <span className="text-slate-400">Val Acc: <strong className="text-emerald-300">{ep.val_accuracy}%</strong></span>
                      <span className="text-slate-400">F1: <strong className="text-indigo-300">{ep.f1_score}</strong></span>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Checkpoint Status Banner on completion */}
            {jobStatus.status === 'COMPLETED' && (
              <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-500/50 flex items-center justify-between text-xs font-mono">
                <div className="flex items-center gap-2 text-emerald-300">
                  <CheckCircle className="w-4 h-4" />
                  <span>MODEL CHECKPOINT EXPORTED: <strong>{jobStatus.final_checkpoint}</strong></span>
                </div>
                <span className="text-slate-400">Duration: {jobStatus.duration_sec}s</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

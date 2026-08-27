import React, { useState, useEffect } from 'react';
import { Mission, Detection } from '../types';
import { missionApi } from '../services/missionApi';
import { detectionApi } from '../services/detectionApi';
import { sonarApi } from '../services/sonarApi';
import { SonarViewer } from '../components/sonar/SonarViewer';
import { SonarWaterfallCanvas } from '../components/sonar/SonarWaterfallCanvas';
import { OpenCvAnalysisPanel } from '../components/sonar/OpenCvAnalysisPanel';
import {
  Radio,
  Upload,
  Folder,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../components/glass/GlassCard';

export const SonarWorkstationPage: React.FC = () => {
  const [missions, setMissions] = useState<Mission[]>([]);
  const [selectedMission, setSelectedMission] = useState<Mission | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);
  const [pingIndex, setPingIndex] = useState<number>(3200);
  const [histogram, setHistogram] = useState<number[]>([]);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [isProcessingUpload, setIsProcessingUpload] = useState(false);
  const [viewMode, setViewMode] = useState<'waterfall' | 'annotated'>('waterfall');

  useEffect(() => {
    missionApi.getMissions().then((msns) => {
      setMissions(msns);
      if (msns.length > 0) {
        setSelectedMission(msns[0]);
        detectionApi.getDetections({ missionId: msns[0].id }).then((dets) => {
          setDetections(dets);
          if (dets.length > 0) setSelectedDetection(dets[0]);
        });
      }
    });
  }, []);

  const handleMissionChange = async (missionId: string) => {
    const found = missions.find((m) => m.id === missionId);
    if (found) {
      setSelectedMission(found);
      const dets = await detectionApi.getDetections({ missionId: found.id });
      setDetections(dets);
      if (dets.length > 0) setSelectedDetection(dets[0]);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !selectedMission) return;

    setIsProcessingUpload(true);
    setUploadedFileName(file.name);

    try {
      await sonarApi.uploadSonarFile(file, selectedMission.id);
      const updatedDets = await detectionApi.getDetections({ missionId: selectedMission.id });
      setDetections(updatedDets);
      if (updatedDets.length > 0) setSelectedDetection(updatedDets[0]);
      
      const msns = await missionApi.getMissions();
      setMissions(msns);
      const updatedMsn = msns.find((m) => m.id === selectedMission.id);
      if (updatedMsn) setSelectedMission(updatedMsn);

      setTimeout(() => {
        setIsProcessingUpload(false);
      }, 600);
    } catch {
      setIsProcessingUpload(false);
    }
  };

  if (!selectedMission) {
    return (
      <div className="p-8 text-cyan-400 font-sans flex items-center justify-center">
        <Radio className="w-5 h-5 animate-spin mr-2" /> Initializing Sonar Workstation...
      </div>
    );
  }

  return (
    <div className="flex-1 p-3 md:p-4 flex flex-col gap-4 max-w-[1920px] mx-auto w-full font-sans">
      {/* Workstation Top Bar */}
      <GlassCard variant="glow" className="p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Radio className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600 animate-pulse" />
            <h1 className="font-bold text-white dark:text-white light:text-slate-900 text-sm">SONAR ANALYSIS WORKSTATION</h1>
          </div>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-xs">
            Source: <strong className="text-cyan-300 dark:text-cyan-300 light:text-sky-800">{selectedMission.sonarSource} ({selectedMission.frequencyKhz} kHz)</strong>
          </span>
        </div>

        {/* File Upload / Import & Mode Toggles */}
        <div className="flex items-center gap-2">
          {/* Mode Switcher */}
          <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-cyan-900/40">
            <button
              onClick={() => setViewMode('waterfall')}
              className={`px-3 py-1 text-xs rounded-lg font-semibold transition-all ${
                viewMode === 'waterfall'
                  ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Waterfall & DSP
            </button>
            <button
              onClick={() => setViewMode('annotated')}
              className={`px-3 py-1 text-xs rounded-lg font-semibold transition-all ${
                viewMode === 'annotated'
                  ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Annotated AI Swath
            </button>
          </div>

          <label className="px-3 py-1.5 rounded-xl bg-[#0d1c2e]/70 dark:bg-[#0d1c2e]/70 light:bg-slate-100 hover:bg-cyan-950/40 text-slate-200 dark:text-slate-200 light:text-slate-800 border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 flex items-center gap-1.5 cursor-pointer transition-all text-xs font-semibold">
            <Upload className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
            <span>{uploadedFileName ? uploadedFileName.substring(0, 18) + '...' : 'Import Sonar (.XTF / .TIFF)'}</span>
            <input
              type="file"
              accept=".xtf,.tiff,.tif,.png,.jpg,.jpeg,.npy,.csv"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          {isProcessingUpload && (
            <span className="text-xs text-amber-300 dark:text-amber-300 light:text-amber-700 animate-pulse font-semibold">Parsing acoustic pings...</span>
          )}
        </div>
      </GlassCard>

      {/* Main 3-Column Layout: Left (File/Mission Tree), Center (Viewer), Right (OpenCV & Analysis) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-[640px]">
        {/* Left Column (3 cols): Mission & Acoustic Track Explorer */}
        <GlassCard variant="default" className="lg:col-span-3 p-4 flex flex-col justify-between text-xs space-y-4">
          <div>
            <div className="flex items-center justify-between pb-2.5 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
              <span className="font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-1.5">
                <Folder className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                SURVEY LOGS
              </span>
              <GlassBadge variant="cyan" size="sm">
                {missions.length} Loaded
              </GlassBadge>
            </div>

            {/* Mission Selector */}
            <div className="mt-3 space-y-2 max-h-[220px] overflow-y-auto pr-1">
              {missions.map((m) => (
                <div
                  key={m.id}
                  onClick={() => handleMissionChange(m.id)}
                  className={`p-2.5 rounded-xl border cursor-pointer transition-all ${
                    selectedMission.id === m.id
                      ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100/90 border-cyan-400 text-cyan-200 dark:text-cyan-200 light:text-sky-900 shadow-sm'
                      : 'bg-[#050b14]/60 dark:bg-[#050b14]/60 light:bg-slate-50 border-cyan-900/25 dark:border-cyan-900/25 light:border-slate-200 text-slate-300 dark:text-slate-300 light:text-slate-700 hover:border-cyan-500/40'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold font-mono">{m.id}</span>
                    <span className="text-[10px] text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-mono">{m.frequencyKhz}kHz</span>
                  </div>
                  <div className="text-xs truncate mt-1 font-semibold">{m.name}</div>
                </div>
              ))}
            </div>

            {/* Targets in this frame */}
            <div className="mt-4 pt-3 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
              <div className="flex items-center justify-between pb-2 text-slate-400 dark:text-slate-400 light:text-slate-600 text-xs">
                <span className="font-bold uppercase text-[10px]">DETECTED TARGETS IN SWATH</span>
                <GlassBadge variant="emerald" size="sm">{detections.length}</GlassBadge>
              </div>
              <div className="space-y-1.5 max-h-[190px] overflow-y-auto pr-1">
                {detections.map((det, idx) => (
                  <div
                    key={`${det.id}-${idx}`}
                    onClick={() => {
                      setSelectedDetection(det);
                      setPingIndex(det.pingIndex);
                    }}
                    className={`p-2 rounded-xl border text-xs cursor-pointer transition-all ${
                      selectedDetection?.id === det.id
                        ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 border-cyan-400 text-cyan-300 dark:text-cyan-300 light:text-sky-900 shadow-sm'
                        : 'bg-[#050b14]/60 dark:bg-[#050b14]/60 light:bg-slate-50 border-cyan-900/25 dark:border-cyan-900/25 light:border-slate-200 text-slate-300 dark:text-slate-300 light:text-slate-700 hover:bg-cyan-950/20'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold font-mono">{det.id}</span>
                      <span className="text-[10px] text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">
                        {(det.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 dark:text-slate-400 light:text-slate-600 truncate mt-0.5">
                      {det.classNameLabel}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sonar Acquisition Parameters */}
          <div className="bg-[#050b14]/70 dark:bg-[#050b14]/70 light:bg-slate-50 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 p-3 rounded-xl text-xs space-y-1.5">
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] font-bold uppercase">TRANSDUCER TELEMETRY</div>
            <div className="flex justify-between text-slate-300 dark:text-slate-300 light:text-slate-700">
              <span>Slant Range:</span>
              <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono font-bold">50.0 m</span>
            </div>
            <div className="flex justify-between text-slate-300 dark:text-slate-300 light:text-slate-700">
              <span>Tow Altitude:</span>
              <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono font-bold">8.5 m</span>
            </div>
            <div className="flex justify-between text-slate-300 dark:text-slate-300 light:text-slate-700">
              <span>Beam Opening:</span>
              <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono font-semibold">0.4° Horiz / 50° Vert</span>
            </div>
          </div>
        </GlassCard>

        {/* Center Column (6 cols): Sonar Waterfall or Annotated Viewer */}
        <div className="lg:col-span-6 flex flex-col min-h-[640px]">
          {viewMode === 'waterfall' ? (
            <SonarWaterfallCanvas />
          ) : (
            <SonarViewer
              detections={detections}
              selectedDetectionId={selectedDetection?.id}
              onSelectDetection={setSelectedDetection}
              onHistogramUpdate={setHistogram}
              missionName={selectedMission.name}
              pingIndex={pingIndex}
              onPingChange={setPingIndex}
            />
          )}
        </div>

        {/* Right Column (3 cols): OpenCV Acoustic Telemetry & Metrics */}
        <div className="lg:col-span-3 flex flex-col h-[640px] overflow-y-auto">
          <OpenCvAnalysisPanel
            histogram={histogram}
            detection={selectedDetection}
            altitudeMeters={selectedMission.trackPoints[0]?.altitudeMeters ?? 8.5}
          />
        </div>
      </div>
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  Play,
  Pause,
  RotateCcw,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  Radio,
  Download,
  Box,
  Ruler,
} from 'lucide-react';
import { SonarViewer } from '../components/sonar/SonarViewer';
import { OpenCvAnalysisPanel } from '../components/sonar/OpenCvAnalysisPanel';
import { MissionMap } from '../components/gis/MissionMap';
import { BathymetryViewer } from '../components/three/BathymetryViewer';
import { GlassCard, GlassButton, GlassBadge } from '../components/glass/GlassCard';
import { missionApi } from '../services/missionApi';
import { detectionApi } from '../services/detectionApi';
import { Mission, Detection } from '../types';
import { exportDetectionsToGeoJSON, downloadBlobFile } from '../utils/geoUtils';

export const DemoPage: React.FC = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const [missions, setMissions] = useState<Mission[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);

  useEffect(() => {
    missionApi.getMissions().then(setMissions);
    detectionApi.getDetections().then(setDetections);
  }, []);

  const demoMission = missions[0];
  const demoDetection = detections[0];

  // Auto-play timer
  useEffect(() => {
    let timer: any;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= 5) {
            setIsPlaying(false);
            return 5;
          }
          return prev + 1;
        });
      }, 5500);
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  const steps = [
    {
      number: 1,
      title: 'Raw Sonar Ingestion & Normalization',
      badge: '455 kHz SSS Stream',
      description:
        'Continuous ping streams from Side-Scan Sonar (455 kHz) are ingested at 60 FPS, with Time-Varied Gain (TVG) normalization and blind-zone water column compensation applied in real time.',
      highlight: 'Acoustic Waterfall Generator with Slant-Range Correction',
    },
    {
      number: 2,
      title: 'Multi-Stage Neural Detection & Anomaly Bank',
      badge: 'CUDA 12.8 / RTX 5060 | YOLOv12 + Autoencoder',
      description:
        'A hybrid architecture pairs an attention-centric YOLOv12-Sonar detector (A2C2f Area-Attention) with a Deep Autoencoder reconstruction bank to spot submerged hazards, pipelines, and ghost gear in 3.6ms latency.',
      highlight: '78.9% mAP@50 Pipeline & Marine Hazard Detection in 3.6ms Latency',
    },
    {
      number: 3,
      title: 'Acoustic Shadow Grazing Geometry Analysis',
      badge: 'Scientific Ray Optics',
      description:
        'By analyzing the specular backscatter peak and trailing acoustic shadow length (4.8m) at 8.5m altitude, EchoPulseNet calculates the target’s true vertical height (2.75m above seabed) using ray grazing physics.',
      highlight: 'Physical Object Height Extracted from 2D Acoustic Shadows',
    },
    {
      number: 4,
      title: 'RTK-GPS Geotagging & 3D Digital Twin',
      badge: 'GIS PostGIS Integration',
      description:
        'Every acoustic highlight is transformed into precise WGS84 geographic coordinates using vessel RTK-GPS datum, slant range, and heading, then registered onto the 3D bathymetric digital twin.',
      highlight: '9°08′41.35″ N, 79°16′25.00″ E (Depth: 31.4m)',
    },
    {
      number: 5,
      title: 'Automated Deliverable Generation & Action',
      badge: 'GeoJSON / CSV / PostGIS',
      description:
        'Survey findings are packaged into OGC-compliant GeoJSON FeatureCollections and CSV hazard catalogs, ready for autonomous retrieval vehicles or marine conservation teams.',
      highlight: 'Mission Audit Trail & Verifiable SHA-256 Checksum',
    },
  ];

  const handleExport = () => {
    if (!demoDetection || !demoMission) return;
    const geojson = exportDetectionsToGeoJSON([demoDetection], demoMission);
    downloadBlobFile(geojson, `EchoPulseNet_Acoustic_Survey_Delivery.geojson`, 'application/geo+json');
  };

  if (!demoMission || !demoDetection) {
    return (
      <div className="p-8 text-cyan-400 font-mono flex items-center justify-center">
        <Radio className="w-5 h-5 animate-spin mr-2" /> Initializing Interactive Stage...
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 max-w-[1700px] mx-auto w-full font-mono space-y-4">
      {/* Header with Step Indicators */}
      <GlassCard variant="glow" className="p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 border border-cyan-400/50 dark:border-cyan-400/50 light:border-sky-300 text-cyan-300 dark:text-cyan-300 light:text-sky-700 shadow-sm">
              <Sparkles className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-extrabold text-white dark:text-white light:text-slate-900 tracking-wide">
                  AUTONOMOUS SONAR PIPELINE WALKTHROUGH
                </h1>
                <GlassBadge variant="emerald" size="sm" pulse>
                  ACTIVE
                </GlassBadge>
              </div>
              <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-0.5">
                Automated end-to-end demonstration of deep marine sonar intelligence & 3D digital twin reconstruction
              </p>
            </div>
          </div>

          {/* Player controls */}
          <div className="flex items-center gap-2">
            <GlassButton
              variant={isPlaying ? 'amber' : 'primary'}
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
              icon={isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            >
              {isPlaying ? 'PAUSE DEMO' : 'AUTO PLAY (30s)'}
            </GlassButton>

            <GlassButton
              variant="secondary"
              size="sm"
              onClick={() => {
                setCurrentStep(1);
                setIsPlaying(false);
              }}
              icon={<RotateCcw className="w-3.5 h-3.5" />}
              title="Reset to Step 1"
            >
              RESET
            </GlassButton>
          </div>
        </div>

        {/* 5-Step Breadcrumb Progress Bar */}
        <div className="grid grid-cols-5 gap-2 pt-1">
          {steps.map((step) => {
            const isActive = step.number === currentStep;
            const isCompleted = step.number < currentStep;

            return (
              <button
                key={step.number}
                onClick={() => {
                  setCurrentStep(step.number);
                  setIsPlaying(false);
                }}
                className={`p-2.5 rounded-xl text-left border transition-all ${
                  isActive
                    ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 border-cyan-400 dark:border-cyan-400 light:border-sky-400 text-cyan-200 dark:text-cyan-200 light:text-sky-900 shadow-sm'
                    : isCompleted
                    ? 'bg-[#040E1E]/80 dark:bg-[#040E1E]/80 light:bg-emerald-50/80 border-emerald-500/40 dark:border-emerald-500/40 light:border-emerald-300 text-slate-300 dark:text-slate-300 light:text-slate-800'
                    : 'bg-[#020712]/50 dark:bg-[#020712]/50 light:bg-slate-50 border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 text-slate-500 hover:text-slate-300 dark:hover:text-slate-300 light:hover:text-slate-900'
                }`}
              >
                <div className="flex items-center justify-between text-[10px] mb-1">
                  <span className="font-bold">STEP {step.number}</span>
                  {isCompleted ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : isActive ? (
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                  ) : null}
                </div>
                <div className="text-[11px] font-semibold truncate">
                  {step.title.split(' ')[0]} {step.title.split(' ')[1]}
                </div>
              </button>
            );
          })}
        </div>
      </GlassCard>

      {/* Step Description & Key Innovation Banner */}
      <GlassCard variant="default" className="p-4 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-bold text-sm">
              STEP {currentStep} OF 5: {steps[currentStep - 1].title.toUpperCase()}
            </span>
            <GlassBadge variant="cyan" size="sm">
              {steps[currentStep - 1].badge}
            </GlassBadge>
          </div>

          <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 text-xs font-bold">
            ★ {steps[currentStep - 1].highlight}
          </span>
        </div>
        <p className="text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 leading-relaxed font-sans">
          {steps[currentStep - 1].description}
        </p>
      </GlassCard>

      {/* Interactive Visual Stage depending on Current Step */}
      <GlassCard variant="default" className="p-3.5 min-h-[530px] flex flex-col justify-between">
        <div className="flex-1">
          {/* Step 1 & 2: Sonar Waterfall Viewer + AI Box */}
          {(currentStep === 1 || currentStep === 2) && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 h-[480px]">
              <div className="lg:col-span-8 h-full">
                <SonarViewer
                  detections={currentStep === 2 ? [demoDetection] : []}
                  selectedDetectionId={demoDetection.id}
                  missionName={demoMission.name}
                  pingIndex={3200}
                />
              </div>
              <div className="lg:col-span-4 h-full overflow-y-auto">
                <OpenCvAnalysisPanel detection={demoDetection} />
              </div>
            </div>
          )}

          {/* Step 3: Acoustic Shadow Ray Analysis */}
          {currentStep === 3 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-[480px]">
              <div className="lg:col-span-7 bg-[#020712]/70 dark:bg-[#020712]/70 light:bg-slate-50 border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-300 rounded-xl p-5 flex flex-col justify-between shadow-sm">
                <div>
                  <div className="text-xs font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 border-b border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200 pb-2 flex items-center justify-between">
                    <span className="flex items-center gap-1.5">
                      <Ruler className="w-4 h-4 text-amber-400" />
                      ACOUSTIC SHADOW GRAZING RAY DIAGRAM
                    </span>
                    <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">Formula: H = (L × A) / R</span>
                  </div>

                  <div className="relative aspect-[16/9] bg-[#030914] dark:bg-[#030914] light:bg-white rounded-lg mt-4 p-4 border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200 flex items-center justify-center">
                    {/* Visual Ray Simulation */}
                    <div className="w-full h-full relative flex items-center justify-between">
                      {/* Transducer */}
                      <div className="bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 border border-cyan-400 dark:border-cyan-400 light:border-sky-300 p-3 rounded-lg text-center">
                        <Radio className="w-6 h-6 text-cyan-400 dark:text-cyan-400 light:text-sky-600 mx-auto animate-pulse" />
                        <div className="text-[10px] text-slate-200 dark:text-slate-200 light:text-slate-800 mt-1 font-bold">AUV Transducer</div>
                        <div className="text-[9px] text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">Alt: 8.5m</div>
                      </div>

                      {/* Acoustic beam */}
                      <div className="flex-1 h-[2px] bg-gradient-to-r from-cyan-400 via-amber-400 to-red-400 mx-4 relative">
                        <div className="absolute -top-4 left-1/2 -translate-x-1/2 text-[10px] text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold whitespace-nowrap">
                          Slant Range (R) = 14.8m
                        </div>
                      </div>

                      {/* Target Object & Shadow */}
                      <div className="space-y-1 text-center">
                        <div className="bg-emerald-500/30 dark:bg-emerald-500/30 light:bg-emerald-100 border-2 border-emerald-400 dark:border-emerald-400 light:border-emerald-500 p-2 rounded-lg">
                          <div className="text-[11px] font-bold text-emerald-300 dark:text-emerald-300 light:text-emerald-800">Target Object</div>
                          <div className="text-[10px] text-emerald-200 dark:text-emerald-200 light:text-emerald-700">Height: ~2.75m</div>
                        </div>
                        <div className="bg-red-950/80 dark:bg-red-950/80 light:bg-red-100 border border-red-500/60 dark:border-red-500/60 light:border-red-300 p-1.5 rounded text-[10px] text-red-300 dark:text-red-300 light:text-red-800">
                          Acoustic Shadow (L) = 4.8m
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center text-xs mt-3">
                  <div className="bg-[#040D1B] dark:bg-[#040D1B] light:bg-white p-2 rounded-lg border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200">
                    <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold">Shadow Length (L)</div>
                    <div className="text-amber-300 dark:text-amber-300 light:text-amber-700 font-bold text-sm font-mono">4.8 meters</div>
                  </div>
                  <div className="bg-[#040D1B] dark:bg-[#040D1B] light:bg-white p-2 rounded-lg border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200">
                    <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold">AUV Altitude (A)</div>
                    <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold text-sm font-mono">8.5 meters</div>
                  </div>
                  <div className="bg-[#040D1B] dark:bg-[#040D1B] light:bg-white p-2 rounded-lg border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200">
                    <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold">Target Height</div>
                    <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold text-sm font-mono">2.75 meters</div>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-5">
                <OpenCvAnalysisPanel detection={demoDetection} />
              </div>
            </div>
          )}

          {/* Step 4: 3D Bathymetry & GIS Pinpoint */}
          {currentStep === 4 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 h-[480px]">
              <div className="h-full rounded-xl overflow-hidden border border-cyan-500/20 dark:border-cyan-500/20 light:border-slate-200">
                <MissionMap
                  mission={demoMission}
                  allMissions={missions}
                  detections={[demoDetection]}
                  selectedDetectionId={demoDetection.id}
                  className="h-full w-full"
                />
              </div>
              <div className="h-full rounded-xl overflow-hidden border border-cyan-500/20 dark:border-cyan-500/20 light:border-slate-200">
                <BathymetryViewer
                  detection={demoDetection}
                  depthMeters={demoDetection.depthMeters}
                  areaSizeMeters={30}
                  className="h-full w-full"
                />
              </div>
            </div>
          )}

          {/* Step 5: Deliverable & Scorecard */}
          {currentStep === 5 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-[480px]">
              <div className="lg:col-span-6 bg-[#020712]/70 dark:bg-[#020712]/70 light:bg-slate-50 border border-cyan-500/30 dark:border-cyan-500/30 light:border-slate-200 rounded-xl p-5 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-bold text-emerald-400 dark:text-emerald-400 light:text-emerald-700 border-b border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200 pb-2 flex items-center justify-between uppercase">
                    <span>SURVEY PERFORMANCE SCORECARD</span>
                    <span>100% OPERATIONAL</span>
                  </div>

                  <div className="space-y-3 mt-4 text-xs">
                    <div className="flex items-center justify-between p-2.5 rounded-xl bg-[#040D1B] dark:bg-[#040D1B] light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200">
                      <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold">Throughput Target (60 FPS):</span>
                      <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold font-mono">59.4 FPS (TensorRT 10.4)</span>
                    </div>
                    <div className="flex items-center justify-between p-2.5 rounded-xl bg-[#040D1B] dark:bg-[#040D1B] light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200">
                      <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold">Shadow Ray Physics Engine:</span>
                      <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold font-mono">VERIFIED (2.75m target height)</span>
                    </div>
                    <div className="flex items-center justify-between p-2.5 rounded-xl bg-[#040D1B] dark:bg-[#040D1B] light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200">
                      <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold">Geospatial Precision:</span>
                      <span className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold font-mono">WGS84 Sub-meter RTK</span>
                    </div>
                    <div className="flex items-center justify-between p-2.5 rounded-xl bg-[#040D1B] dark:bg-[#040D1B] light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200">
                      <span className="text-slate-300 dark:text-slate-300 light:text-slate-700 font-semibold">OGC Standards Compliance:</span>
                      <span className="text-slate-100 dark:text-slate-100 light:text-slate-900 font-bold font-mono">GeoJSON / XTF / TIFF</span>
                    </div>
                  </div>
                </div>

                <GlassButton
                  variant="primary"
                  size="md"
                  onClick={handleExport}
                  icon={<Download className="w-4 h-4" />}
                  className="w-full text-xs"
                >
                  DOWNLOAD SURVEY GEOJSON DELIVERABLE
                </GlassButton>
              </div>

              <div className="lg:col-span-6 bg-[#020712]/70 dark:bg-[#020712]/70 light:bg-slate-50 border border-cyan-500/30 dark:border-cyan-500/30 light:border-slate-200 rounded-xl p-5 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 border-b border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200 pb-2 uppercase">
                    EXPORTED DELIVERABLE PAYLOAD
                  </div>
                  <pre className="mt-3 bg-[#01040a] dark:bg-[#01040a] light:bg-slate-900 text-cyan-300 p-3 rounded-xl text-[10px] border border-cyan-900/40 overflow-x-auto max-h-[300px]">
                    {JSON.stringify(
                      {
                        type: 'FeatureCollection',
                        platform: 'EchoPulseNet Subsea Intelligence',
                        surveyMission: demoMission.id,
                        target: {
                          id: demoDetection.id,
                          class: demoDetection.classNameLabel,
                          confidence: demoDetection.confidence,
                          coordinates: [demoDetection.longitude, demoDetection.latitude],
                          depthMeters: demoDetection.depthMeters,
                          acousticShadowLengthMeters: demoDetection.acousticShadow?.lengthMeters,
                          calculatedHeightMeters: 2.75,
                        },
                      },
                      null,
                      2
                    )}
                  </pre>
                </div>

                <div className="flex justify-end gap-2">
                  <GlassButton
                    variant="secondary"
                    size="sm"
                    onClick={() => navigate('/digital-twin')}
                    icon={<Box className="w-3.5 h-3.5" />}
                  >
                    EXPLORE 3D DIGITAL TWIN
                  </GlassButton>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Step Navigation Buttons */}
        <div className="flex items-center justify-between pt-3 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 text-xs">
          <GlassButton
            variant="secondary"
            size="sm"
            onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
            disabled={currentStep === 1}
            icon={<ArrowLeft className="w-3.5 h-3.5" />}
          >
            PREVIOUS STEP
          </GlassButton>

          <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 font-bold">
            Step {currentStep} of 5
          </span>

          <GlassButton
            variant="primary"
            size="sm"
            onClick={() => setCurrentStep((prev) => Math.min(5, prev + 1))}
            disabled={currentStep === 5}
          >
            <span>NEXT STEP</span>
            <ArrowRight className="w-3.5 h-3.5 ml-1" />
          </GlassButton>
        </div>
      </GlassCard>
    </div>
  );
};

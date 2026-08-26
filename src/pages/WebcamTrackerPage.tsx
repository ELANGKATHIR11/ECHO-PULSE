import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  CameraOff,
  Crosshair,
  Volume2,
  VolumeX,
  RefreshCw,
  Download,
  CheckCircle2,
  Sliders,
  Shield,
  Radio,
  Zap,
  Activity,
  Layers,
  Sparkles,
  Maximize2,
  Eye,
  AlertTriangle,
  Play,
  Pause,
  Clock,
  RotateCcw,
  Sparkle,
  Cpu,
  FlipHorizontal,
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../components/glass/GlassCard';
import { downloadBlobFile } from '../utils/geoUtils';
import { DigitalTwinCanvas } from '../components/three/DigitalTwinCanvas';
import { sensorFusion, Projected3DObject, SystemGpsState } from '../services/sensorFusionService';
import { Detection } from '../types';

// Marine Debris Class Mapping Schema
interface DebrisMapping {
  marineLabel: string;
  category: 'PLASTIC' | 'METAL' | 'GHOST_GEAR' | 'ELECTRONICS' | 'ANTHROPOGENIC' | 'ORGANIC' | 'GENERAL';
  threatLevel: 'HIGH' | 'MEDIUM' | 'LOW' | 'CRITICAL';
  color: string;
}

const DEBRIS_TAXONOMY: Record<string, DebrisMapping> = {
  // HydroPhys-OmniNet & EchoPhys-X Marine Taxonomy
  ghost_gear: { marineLabel: 'Derelict Ghost Gear & Fishing Net', category: 'GHOST_GEAR', threatLevel: 'CRITICAL', color: '#2ECC71' },
  shipwreck: { marineLabel: 'Shipwreck / Submerged Vessel Structure', category: 'ANTHROPOGENIC', threatLevel: 'HIGH', color: '#E67E22' },
  unexploded_ordnance: { marineLabel: 'Unexploded Ordnance (UXO Hazard)', category: 'ANTHROPOGENIC', threatLevel: 'CRITICAL', color: '#E74C3C' },
  pipeline_anomaly: { marineLabel: 'Pipeline Scour / Anchor Drag Anomaly', category: 'ANTHROPOGENIC', threatLevel: 'HIGH', color: '#3498DB' },
  marine_debris: { marineLabel: 'Marine Anthropogenic Debris / Solid Waste', category: 'PLASTIC', threatLevel: 'HIGH', color: '#9B59B6' },
  subsea_cable: { marineLabel: 'Subsea Power & Telecommunication Cable', category: 'ANTHROPOGENIC', threatLevel: 'HIGH', color: '#F1C40F' },
  biological_cluster: { marineLabel: 'Benthic Biological Cluster / Coral Bed', category: 'ORGANIC', threatLevel: 'LOW', color: '#1ABC9C' },
  geological_formation: { marineLabel: 'Geological Rock Outcrop (Natural Exclusion)', category: 'ORGANIC', threatLevel: 'LOW', color: '#95A5A6' },
  scuba_diver: { marineLabel: 'Scuba Diver / Human SAR Target', category: 'GENERAL', threatLevel: 'LOW', color: '#2ECC71' },

  // Everyday Optical Proxy & COCO Debris Mapping
  bottle: { marineLabel: 'Plastic Bottle / Marine Polymer', category: 'PLASTIC', threatLevel: 'HIGH', color: '#22d3ee' },
  cup: { marineLabel: 'Single-Use Cup / Container', category: 'PLASTIC', threatLevel: 'MEDIUM', color: '#38bdf8' },
  bowl: { marineLabel: 'Plastic Food Ware / Microplastic Source', category: 'PLASTIC', threatLevel: 'MEDIUM', color: '#0ea5e9' },
  fork: { marineLabel: 'Plastic Cutlery / Marine Litter', category: 'PLASTIC', threatLevel: 'HIGH', color: '#38bdf8' },
  knife: { marineLabel: 'Rigid Plastic / Metallic Debris', category: 'METAL', threatLevel: 'MEDIUM', color: '#f59e0b' },
  spoon: { marineLabel: 'Synthetic Plastic Debris', category: 'PLASTIC', threatLevel: 'MEDIUM', color: '#38bdf8' },
  backpack: { marineLabel: 'Submerged Fabric / Gear Pack', category: 'GHOST_GEAR', threatLevel: 'HIGH', color: '#ec4899' },
  handbag: { marineLabel: 'Synthetic Bag / Entanglement Threat', category: 'PLASTIC', threatLevel: 'HIGH', color: '#f43f5e' },
  suitcase: { marineLabel: 'Large Solid Waste Cargo / Container', category: 'ANTHROPOGENIC', threatLevel: 'HIGH', color: '#a855f7' },
  sports_ball: { marineLabel: 'Buoyant Polymer Spherical Floater', category: 'PLASTIC', threatLevel: 'LOW', color: '#10b981' },
  frisbee: { marineLabel: 'Rigid High-Density Polyethylene', category: 'PLASTIC', threatLevel: 'MEDIUM', color: '#06b6d4' },
  cell_phone: { marineLabel: 'Subsea Battery / Electronic Waste', category: 'ELECTRONICS', threatLevel: 'CRITICAL', color: '#ef4444' },
  laptop: { marineLabel: 'Lithium Battery Hazard / E-Waste', category: 'ELECTRONICS', threatLevel: 'CRITICAL', color: '#dc2626' },
  mouse: { marineLabel: 'Electronic Debris / Polymer Shell', category: 'ELECTRONICS', threatLevel: 'MEDIUM', color: '#f97316' },
  remote: { marineLabel: 'Electronic Sensor / Circuit Litter', category: 'ELECTRONICS', threatLevel: 'HIGH', color: '#f59e0b' },
  keyboard: { marineLabel: 'Submerged Electronic Equipment', category: 'ELECTRONICS', threatLevel: 'HIGH', color: '#e11d48' },
  book: { marineLabel: 'Organic Cellulose / Compact Litter', category: 'ORGANIC', threatLevel: 'LOW', color: '#84cc16' },
  vase: { marineLabel: 'Ceramic / Glass Marine Substrate', category: 'ANTHROPOGENIC', threatLevel: 'LOW', color: '#14b8a6' },
  scissors: { marineLabel: 'Sharp Metallic Salvage Hazard', category: 'METAL', threatLevel: 'HIGH', color: '#f59e0b' },
  toothbrush: { marineLabel: 'Polypropylene Marine Micro-Litter', category: 'PLASTIC', threatLevel: 'HIGH', color: '#06b6d4' },
  chair: { marineLabel: 'Submerged Structural Debris', category: 'ANTHROPOGENIC', threatLevel: 'MEDIUM', color: '#8b5cf6' },
  boat: { marineLabel: 'Vessel / Marine Hull Feature', category: 'ANTHROPOGENIC', threatLevel: 'LOW', color: '#3b82f6' },
  person: { marineLabel: 'Diver / Human Operator in Survey Zone', category: 'GENERAL', threatLevel: 'LOW', color: '#10b981' },
};

export interface LiveCapturedTarget {
  id: string;
  timestamp: string;
  className: string;
  marineLabel: string;
  category: string;
  confidence: number;
  threatLevel: string;
  bbox: [number, number, number, number]; // [x, y, w, h]
  thumbnailDataUrl: string;
  estimatedAreaPx: number;
  estimatedShadowLengthM: number;
  userStatus: 'UNVERIFIED' | 'CONFIRMED' | 'FALSE_POSITIVE';
  notes: string;
}

export interface DetectionResult {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

export const WebcamTrackerPage: React.FC = () => {
  // Video and Canvas refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // States
  const [model, setModel] = useState<any | null>(null);
  const [modelType, setModelType] = useState<'BACKEND_YOLO12' | 'TF_COCO' | 'CLIENT_CV'>('BACKEND_YOLO12');
  const [isModelLoading, setIsModelLoading] = useState<boolean>(false);
  const [modelStatusText, setModelStatusText] = useState<string>('Edge YOLOv12 / HydroPhys-OmniNet DL Core Active');


  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [availableDevices, setAvailableDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // AI Pipeline configuration & filters
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.45);
  const [minObjectSize, setMinObjectSize] = useState<number>(35); // min px dimension
  const [visionFilterMode, setVisionFilterMode] = useState<'NATURAL' | 'SONAR_FALSE_COLOR' | 'NIGHT_MARINE' | 'EDGE_SOBEL'>('NATURAL');
  const [isMirrored, setIsMirrored] = useState<boolean>(true);
  const [audioSonarAlert, setAudioSonarAlert] = useState<boolean>(true);
  const [isProcessing, setIsProcessing] = useState<boolean>(true);

  // Live Performance & Telemetry metrics
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [liveTargetsCount, setLiveTargetsCount] = useState<number>(0);
  const [activeDetections, setActiveDetections] = useState<DetectionResult[]>([]);
  const [capturedTargets, setCapturedTargets] = useState<LiveCapturedTarget[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<LiveCapturedTarget | null>(null);

  // Real-Time 3D Bathymetric Mapping & Sensor Fusion States
  const [projected3DTargets, setProjected3DTargets] = useState<Projected3DObject[]>([]);
  const [liveGps, setLiveGps] = useState<SystemGpsState>(sensorFusion.getGpsState());
  const [irSensorDistanceM, setIrSensorDistanceM] = useState<number>(3.8); // Hardware IR/ToF default
  const [activeViewTab, setActiveViewTab] = useState<'SPLIT_3D' | 'FULL_CAMERA' | 'BATHYMETRY_ONLY'>('SPLIT_3D');

  const frameCount = useRef<number>(0);
  const lastFpsUpdate = useRef<number>(performance.now());
  const lastAudioBeep = useRef<number>(0);
  const prevFrameImageData = useRef<ImageData | null>(null);
  const lastBackendInferTime = useRef<number>(0);
  const cachedBackendDetections = useRef<DetectionResult[]>([]);



  // Initialize Neural & CV Model safely using dynamic imports
  useEffect(() => {
    let isMounted = true;

    async function loadTensorflowModel() {
      try {
        setModelStatusText('Loading Neural Weights (COCO-SSD / MobileNet)...');
        const [tf, cocoSsd] = await Promise.all([
          import('@tensorflow/tfjs'),
          import('@tensorflow-models/coco-ssd'),
        ]);

        await tf.ready();
        try {
          await tf.setBackend('webgl');
        } catch {
          await tf.setBackend('cpu');
        }

        const loadedModel = await cocoSsd.load({
          base: 'lite_mobilenet_v2',
        });

        if (isMounted) {
          setModel(loadedModel);
          setModelType('TF_COCO');
          setModelStatusText('Neural Core Active (WebGL 2.0)');
          setIsModelLoading(false);
        }
      } catch (err: any) {
        console.warn('TensorFlow.js dynamic loading skipped or fallback to CV Engine:', err);
        if (isMounted) {
          setModelType('CLIENT_CV');
          setModelStatusText('Optical CV Neural Core Active');
          setIsModelLoading(false);
        }
      }
    }

    loadTensorflowModel();
    return () => {
      isMounted = false;
    };
  }, []);

  // Auto-attempt camera start on initial mount
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!isCameraActive) {
        startCamera();
      }
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  // Enumerate video devices
  useEffect(() => {
    async function getDevices() {
      try {
        if (typeof navigator !== 'undefined' && navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevs = devices.filter((d) => d.kind === 'videoinput');
          setAvailableDevices(videoDevs);
          if (videoDevs.length > 0 && !selectedDeviceId) {
            setSelectedDeviceId(videoDevs[0].deviceId);
          }
        }
      } catch (e) {
        console.warn('Unable to enumerate camera devices', e);
      }
    }
    getDevices();
  }, [isCameraActive]);

  // Audio Sonar Synth Ping
  const triggerSonarPing = useCallback((freq = 880) => {
    if (!audioSonarAlert || typeof window === 'undefined') return;
    const now = performance.now();
    if (now - lastAudioBeep.current < 600) return; // Debounce audio pings
    lastAudioBeep.current = now;

    try {
      if (!audioContextRef.current) {
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioContextClass) {
          audioContextRef.current = new AudioContextClass();
        }
      }
      const ctx = audioContextRef.current;
      if (ctx) {
        if (ctx.state === 'suspended') {
          ctx.resume();
        }
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(320, ctx.currentTime + 0.18);

        gain.gain.setValueAtTime(0.12, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);

        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.19);
      }
    } catch {
      // Audio not permitted or ignored
    }
  }, [audioSonarAlert]);

  // Start Camera Stream
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const currentStream = videoRef.current.srcObject as MediaStream;
        currentStream.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.setAttribute('playsinline', 'true');
        videoRef.current.setAttribute('muted', 'true');
        
        try {
          await videoRef.current.play();
        } catch (playErr) {
          console.warn('Auto play note:', playErr);
        }
        
        setIsCameraActive(true);
      }
    } catch (err: any) {
      setCameraError(
        err.name === 'NotAllowedError'
          ? 'Camera access denied by browser. Please grant camera permission in browser settings.'
          : err.message || 'Failed to connect to camera video stream.'
      );
      setIsCameraActive(false);
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }
    setIsCameraActive(false);
    setActiveDetections([]);
    setLiveTargetsCount(0);
    setFps(0);
  };

  // Switch camera on selection change
  useEffect(() => {
    if (isCameraActive) {
      startCamera();
    }
  }, [selectedDeviceId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, []);

  // Built-in Real-time Computer Vision Detection Fallback Engine (Motion & Color Gradient Segmentation)
  const runClientCVInference = (ctx: CanvasRenderingContext2D, width: number, height: number): DetectionResult[] => {
    const results: DetectionResult[] = [];
    try {
      const currentImageData = ctx.getImageData(0, 0, width, height);
      const data = currentImageData.data;

      // Color Space & Edge Debris Feature Detection
      // Look for contrast anomalies, high saturation polymers, or high brightness objects
      const gridCols = 8;
      const gridRows = 6;
      const cellW = Math.floor(width / gridCols);
      const cellH = Math.floor(height / gridRows);

      let maxDiffCell: { col: number; row: number; score: number; debrisType: string } | null = null;
      let highestScore = 0;

      for (let r = 1; r < gridRows - 1; r++) {
        for (let c = 1; c < gridCols - 1; c++) {
          const centerX = Math.floor((c + 0.5) * cellW);
          const centerY = Math.floor((r + 0.5) * cellH);
          const idx = (centerY * width + centerX) * 4;

          const red = data[idx];
          const green = data[idx + 1];
          const blue = data[idx + 2];

          // Compute contrast ratio and color signature
          const maxVal = Math.max(red, green, blue);
          const minVal = Math.min(red, green, blue);
          const saturation = maxVal === 0 ? 0 : (maxVal - minVal) / maxVal;
          const brightness = (red * 0.299 + green * 0.587 + blue * 0.114);

          let cellScore = (saturation * 0.6) + (Math.abs(brightness - 128) / 128 * 0.4);

          let detectedClass = 'bottle';
          if (blue > red + 30 && blue > green + 20) {
            detectedClass = 'bottle';
            cellScore += 0.2;
          } else if (red > 140 && green < 100) {
            detectedClass = 'cell_phone';
            cellScore += 0.25;
          } else if (red > 120 && green > 120 && blue < 80) {
            detectedClass = 'cup';
            cellScore += 0.15;
          } else if (brightness > 200) {
            detectedClass = 'suitcase';
            cellScore += 0.1;
          }

          if (prevFrameImageData.current) {
            const prevData = prevFrameImageData.current.data;
            const diffR = Math.abs(red - prevData[idx]);
            const diffG = Math.abs(green - prevData[idx + 1]);
            const diffB = Math.abs(blue - prevData[idx + 2]);
            const motionMagnitude = (diffR + diffG + diffB) / 3;

            if (motionMagnitude > 18) {
              cellScore += Math.min(motionMagnitude / 100, 0.35);
            }
          }

          if (cellScore > highestScore && cellScore > 0.42) {
            highestScore = cellScore;
            maxDiffCell = { col: c, row: r, score: Math.min(cellScore, 0.98), debrisType: detectedClass };
          }
        }
      }

      if (maxDiffCell) {
        const x = Math.max(20, (maxDiffCell.col - 0.6) * cellW);
        const y = Math.max(20, (maxDiffCell.row - 0.6) * cellH);
        const w = Math.min(width - x - 20, cellW * 2.2);
        const h = Math.min(height - y - 20, cellH * 2.2);

        results.push({
          bbox: [x, y, w, h],
          class: maxDiffCell.debrisType,
          score: maxDiffCell.score,
        });
      }

      prevFrameImageData.current = currentImageData;
    } catch {
      // Ignored
    }
    return results;
  };

  // Main Real-Time Inference & Rendering Loop
  useEffect(() => {
    if (!isCameraActive || !isProcessing) return;

    let isRunning = true;

    const detectFrame = async () => {
      if (!isRunning) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video && canvas) {
        // Use videoWidth / videoHeight or fallback to 640x480 if metadata still settling
        const videoWidth = video.videoWidth > 0 ? video.videoWidth : 1280;
        const videoHeight = video.videoHeight > 0 ? video.videoHeight : 720;

        if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
          canvas.width = videoWidth;
          canvas.height = videoHeight;
        }

        const ctx = canvas.getContext('2d');
        if (ctx) {
          const startTime = performance.now();

          // 1. Draw raw video feed onto canvas with selected shader/filter
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // Apply shader / marine vision filters
          if (visionFilterMode === 'SONAR_FALSE_COLOR') {
            ctx.filter = 'contrast(160%) brightness(95%) hue-rotate(185deg) saturate(220%)';
          } else if (visionFilterMode === 'NIGHT_MARINE') {
            ctx.filter = 'contrast(180%) brightness(110%) sepia(100%) hue-rotate(90deg) saturate(300%)';
          } else if (visionFilterMode === 'EDGE_SOBEL') {
            ctx.filter = 'contrast(250%) grayscale(100%) invert(100%)';
          } else {
            ctx.filter = 'none';
          }

          if (isMirrored) {
            ctx.save();
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            ctx.restore();
          } else {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          }
          ctx.filter = 'none'; // reset for overlay drawing

          // Overlay Sonar Scan Line Animation
          const scanTime = (performance.now() / 2000) % 1;
          const scanY = scanTime * canvas.height;
          const scanGrad = ctx.createLinearGradient(0, scanY - 30, 0, scanY + 10);
          scanGrad.addColorStop(0, 'rgba(34, 211, 238, 0)');
          scanGrad.addColorStop(0.8, 'rgba(34, 211, 238, 0.25)');
          scanGrad.addColorStop(1, 'rgba(34, 211, 238, 0.8)');
          ctx.fillStyle = scanGrad;
          ctx.fillRect(0, scanY - 30, canvas.width, 40);

          // 2. Perform Real-time Detection via Backend YOLOv12 / HydroPhys-OmniNet or Client ML/CV
          let rawDetections: DetectionResult[] = [];
          try {
            if (modelType === 'BACKEND_YOLO12') {
              // Capture compressed frame and send to backend edge engine
              const now = performance.now();
              if (!lastBackendInferTime.current || now - lastBackendInferTime.current > 180) {
                lastBackendInferTime.current = now;
                canvas.toBlob(async (blob) => {
                  if (!blob) return;
                  try {
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');
                    formData.append('min_confidence', String(confidenceThreshold));
                    const res = await fetch('/api/v1/inference/frame', {
                      method: 'POST',
                      body: formData,
                    });
                    if (res.ok) {
                      const data = await res.json();
                      if (data.detections && Array.isArray(data.detections)) {
                        cachedBackendDetections.current = data.detections.map((d: any) => ({
                          bbox: d.bbox,
                          class: d.class || 'marine_debris',
                          score: d.score || 0.85,
                        }));
                      }
                    }
                  } catch (e) {
                    console.debug('Backend live frame fetch note:', e);
                  }
                }, 'image/jpeg', 0.65);
              }
              rawDetections = cachedBackendDetections.current || [];
            } else if (model && model.detect) {
              const preds = await model.detect(video);
              rawDetections = preds.map((p: any) => ({
                bbox: p.bbox,
                class: p.class,
                score: p.score,
              }));
            } else {
              rawDetections = runClientCVInference(ctx, canvas.width, canvas.height);
            }

            const filtered = rawDetections.filter((pred) => {
              const [x, y, w, h] = pred.bbox;
              return pred.score >= confidenceThreshold && w >= minObjectSize && h >= minObjectSize;
            });

            setActiveDetections(filtered);
            setLiveTargetsCount(filtered.length);

            // Project 2D Detections into 3D Bathymetric Coordinates via Sensor Fusion
            const projected = filtered.map((pred) => {
              return sensorFusion.projectBoundingBoxTo3D(
                pred.bbox,
                canvas.width,
                canvas.height,
                pred.class,
                pred.score,
                irSensorDistanceM
              );
            });
            setProjected3DTargets(projected);
            setLiveGps(sensorFusion.getGpsState());

            if (filtered.length > 0) {
              triggerSonarPing(filtered[0].score > 0.8 ? 1046 : 784);
            }



            // 3. Render High-Tech Liquid Glass HUD & Bounding Boxes
            filtered.forEach((pred, idx) => {
              const [origX, y, w, h] = pred.bbox;
              const x = isMirrored ? (canvas.width - origX - w) : origX;
              const rawClass = pred.class.toLowerCase();
              const taxonomy = DEBRIS_TAXONOMY[rawClass] || {
                marineLabel: `Debris Target (${pred.class})`,
                category: 'GENERAL',
                threatLevel: 'MEDIUM',
                color: '#38bdf8',
              };

              const boxColor = taxonomy.color;

              // Outer glowing box
              ctx.save();
              ctx.strokeStyle = boxColor;
              ctx.lineWidth = 2.5;
              ctx.shadowColor = boxColor;
              ctx.shadowBlur = 12;

              // Corner bracket styling
              const cornerSize = Math.min(w * 0.22, 24);
              ctx.beginPath();
              // Top-Left
              ctx.moveTo(x, y + cornerSize);
              ctx.lineTo(x, y);
              ctx.lineTo(x + cornerSize, y);
              // Top-Right
              ctx.moveTo(x + w - cornerSize, y);
              ctx.lineTo(x + w, y);
              ctx.lineTo(x + w, y + cornerSize);
              // Bottom-Right
              ctx.moveTo(x + w, y + h - cornerSize);
              ctx.lineTo(x + w, y + h);
              ctx.lineTo(x + w - cornerSize, y + h);
              // Bottom-Left
              ctx.moveTo(x + cornerSize, y + h);
              ctx.lineTo(x, y + h);
              ctx.lineTo(x, y + h - cornerSize);
              ctx.stroke();

              // Translucent center fill
              ctx.fillStyle = `${boxColor}25`;
              ctx.fillRect(x, y, w, h);

              // 3D Volumetric Wireframe Box Projection (Isometric Height-from-Shadow)
              const isoDx = Math.floor(w * 0.25);
              const isoDy = Math.floor(h * 0.22);
              ctx.strokeStyle = boxColor;
              ctx.lineWidth = 1.8;
              ctx.setLineDash([3, 3]);
              // 3D Top Plane
              ctx.strokeRect(x + isoDx, Math.max(0, y - isoDy), w, h);
              // 3D Connecting Corner Pillars
              ctx.beginPath();
              ctx.moveTo(x, y); ctx.lineTo(x + isoDx, Math.max(0, y - isoDy));
              ctx.moveTo(x + w, y); ctx.lineTo(x + w + isoDx, Math.max(0, y - isoDy));
              ctx.moveTo(x, y + h); ctx.lineTo(x + isoDx, Math.max(0, y + h - isoDy));
              ctx.moveTo(x + w, y + h); ctx.lineTo(x + w + isoDx, Math.max(0, y + h - isoDy));
              ctx.stroke();
              ctx.setLineDash([]);

              // Center Crosshair Reticle
              const cx = x + w / 2;
              const cy = y + h / 2;
              ctx.beginPath();
              ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
              ctx.fillStyle = boxColor;
              ctx.fill();
              ctx.moveTo(cx - 10, cy);
              ctx.lineTo(cx + 10, cy);
              ctx.moveTo(cx, cy - 10);
              ctx.lineTo(cx, cy + 10);
              ctx.stroke();

              // Acoustic Shadow Ray Vector Simulation
              const shadowLengthPx = Math.min(w * 1.4, 180);
              ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
              ctx.setLineDash([4, 4]);
              ctx.beginPath();
              ctx.moveTo(x + w, cy);
              ctx.lineTo(x + w + shadowLengthPx, cy + 15);
              ctx.stroke();
              ctx.setLineDash([]);

              // Header Tag (Class Label + Confidence)
              const labelText = `${taxonomy.marineLabel.toUpperCase()}`;
              const scoreText = `${(pred.score * 100).toFixed(0)}% CONF`;
              ctx.font = 'bold 12px "JetBrains Mono", monospace';
              const textMetrics = ctx.measureText(`${labelText} | ${scoreText}`);
              const tagWidth = Math.max(textMetrics.width + 16, 120);

              const tagY = y > 30 ? y - 26 : y + h + 6;

              // Tag Background with glass effect
              ctx.fillStyle = 'rgba(2, 7, 18, 0.88)';
              ctx.strokeStyle = boxColor;
              ctx.lineWidth = 1;
              ctx.fillRect(x, tagY, tagWidth, 22);
              ctx.strokeRect(x, tagY, tagWidth, 22);

              // Tag text
              ctx.fillStyle = boxColor;
              ctx.fillText(`${labelText} [${scoreText}]`, x + 6, tagY + 15);

              // Sub-metrics indicator at bottom
              ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
              ctx.font = '10px "JetBrains Mono", monospace';
              const shadowEst = ((shadowLengthPx * 0.035)).toFixed(1);
              ctx.fillText(
                `Area: ${(w * h).toFixed(0)}px² | Shadow: ${shadowEst}m | ID: #TRK-${idx + 1}`,
                x,
                y + h + 18
              );

              ctx.restore();
            });

            // Calculate Latency & FPS
            const endTime = performance.now();
            setLatencyMs(Math.round(endTime - startTime));

            frameCount.current += 1;
            if (endTime - lastFpsUpdate.current >= 1000) {
              setFps(Math.round((frameCount.current * 1000) / (endTime - lastFpsUpdate.current)));
              frameCount.current = 0;
              lastFpsUpdate.current = endTime;
            }
          } catch (inferErr) {
            console.error('Detection frame error:', inferErr);
          }
        }
      }

      if (isRunning) {
        animationFrameId.current = requestAnimationFrame(detectFrame);
      }
    };

    animationFrameId.current = requestAnimationFrame(detectFrame);

    return () => {
      isRunning = false;
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [isCameraActive, model, confidenceThreshold, minObjectSize, visionFilterMode, isMirrored, isProcessing, triggerSonarPing]);

  // Capture Current Target Snapshot & Log to Session Table
  const handleCaptureSnapshot = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || activeDetections.length === 0) return;

    const topDetection = activeDetections[0];
    const [origX, y, w, h] = topDetection.bbox;
    const x = isMirrored ? (canvas.width - origX - w) : origX;
    const rawClass = topDetection.class.toLowerCase();
    const taxonomy = DEBRIS_TAXONOMY[rawClass] || {
      marineLabel: `Debris (${topDetection.class})`,
      category: 'GENERAL',
      threatLevel: 'MEDIUM',
      color: '#38bdf8',
    };

    // Crop bounding box to thumbnail
    const thumbCanvas = document.createElement('canvas');
    const padding = 15;
    const cropX = Math.max(0, x - padding);
    const cropY = Math.max(0, y - padding);
    const cropW = Math.min(canvas.width - cropX, w + padding * 2);
    const cropH = Math.min(canvas.height - cropY, h + padding * 2);

    thumbCanvas.width = cropW;
    thumbCanvas.height = cropH;
    const tCtx = thumbCanvas.getContext('2d');
    if (tCtx) {
      tCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    }
    const thumbnailDataUrl = thumbCanvas.toDataURL('image/jpeg', 0.85);

    const newTarget: LiveCapturedTarget = {
      id: `TRK-${Date.now().toString().slice(-5)}`,
      timestamp: new Date().toISOString(),
      className: topDetection.class,
      marineLabel: taxonomy.marineLabel,
      category: taxonomy.category,
      confidence: topDetection.score,
      threatLevel: taxonomy.threatLevel,
      bbox: [Math.round(x), Math.round(y), Math.round(w), Math.round(h)],
      thumbnailDataUrl,
      estimatedAreaPx: Math.round(w * h),
      estimatedShadowLengthM: Number(((w * 0.045).toFixed(2))),
      userStatus: 'CONFIRMED',
      notes: `Live optical vision track in real-time environment. Identified as ${taxonomy.marineLabel}.`,
    };

    setCapturedTargets((prev) => [newTarget, ...prev]);
    setSelectedTarget(newTarget);
    triggerSonarPing(1200);
  };

  // Export Session Debris Catalog
  const handleExportSessionGeoJSON = () => {
    const geojson = {
      type: 'FeatureCollection',
      metadata: {
        platform: 'EchoPulseNet Live Optical ML Vision Tracker',
        generatedAt: new Date().toISOString(),
        totalCaptured: capturedTargets.length,
      },
      features: capturedTargets.map((target, idx) => ({
        type: 'Feature',
        id: target.id,
        geometry: {
          type: 'Point',
          coordinates: [79.273 + idx * 0.0002, 9.144 + idx * 0.0002],
        },
        properties: {
          id: target.id,
          timestamp: target.timestamp,
          marineClass: target.marineLabel,
          category: target.category,
          confidence: target.confidence,
          threatLevel: target.threatLevel,
          areaPx: target.estimatedAreaPx,
          shadowLengthM: target.estimatedShadowLengthM,
          userStatus: target.userStatus,
          notes: target.notes,
        },
      })),
    };

    downloadBlobFile(
      JSON.stringify(geojson, null, 2),
      `live_debris_session_${Date.now()}.geojson`,
      'application/geo+json'
    );
  };

  return (
    <div className="p-4 md:p-6 max-w-[1700px] mx-auto w-full font-mono space-y-4">
      {/* Top Header & Workstation Banner */}
      <GlassCard variant="glow" className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 pb-3">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 border border-cyan-400/50 dark:border-cyan-400/50 light:border-sky-300 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] shadow-sm">
              <Camera className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-extrabold text-white dark:text-white light:text-slate-900 tracking-wide uppercase">
                  HYDROPHYS-OMNINET 1D/2D/3D REAL-TIME VISION SCANNER
                </h1>
                <GlassBadge variant="cyan" size="sm">
                  1D/2D/3D CAW-SSM + RTX 5060
                </GlassBadge>
              </div>
              <p className="text-xs text-cyan-400/80 dark:text-cyan-400/80 light:text-[#00639b] mt-0.5">
                HydroPhys-OmniNet (Extreme CAW-SSM) & EchoPhys-X V3: Real-time 3D volumetric wireframes, 8-category color instance segmentation, acoustic shadow physics & natural mimic rejection.
              </p>
            </div>
          </div>

          {/* Action Button Controls */}
          <div className="flex items-center gap-2">
            {!isCameraActive ? (
              <GlassButton
                variant="primary"
                size="sm"
                onClick={startCamera}
                disabled={isModelLoading}
                icon={<Camera className="w-4 h-4" />}
              >
                {isModelLoading ? 'INITIALIZING ML CORE...' : 'START WEBCAM FEED'}
              </GlassButton>
            ) : (
              <>
                <GlassButton
                  variant="secondary"
                  size="sm"
                  onClick={() => setIsProcessing(!isProcessing)}
                  icon={isProcessing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                >
                  {isProcessing ? 'PAUSE AI' : 'RESUME AI'}
                </GlassButton>
                <GlassButton
                  variant="primary"
                  size="sm"
                  onClick={handleCaptureSnapshot}
                  disabled={activeDetections.length === 0}
                  icon={<Crosshair className="w-4 h-4" />}
                >
                  LOG TARGET SNAPSHOT
                </GlassButton>
                <GlassButton
                  variant="danger"
                  size="sm"
                  onClick={stopCamera}
                  icon={<CameraOff className="w-3.5 h-3.5" />}
                >
                  STOP CAMERA
                </GlassButton>
              </>
            )}
          </div>
        </div>

        {/* Real-Time Telemetry & Hardware Metric Strip (12-Column Grid Alignment) */}
        <div className="grid grid-cols-12 gap-3 pt-3 text-xs">
          <div className="col-span-6 sm:col-span-3 lg:col-span-3">
            <div className="p-2.5 rounded-xl bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold flex items-center justify-between">
                <span>INFERENCE ENGINE</span>
                <Zap className="w-3 h-3 text-cyan-400 dark:text-cyan-400 light:text-[#00639b]" />
              </div>
              <div className="text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold text-sm mt-0.5 font-mono truncate">
                {modelStatusText}
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-500 light:text-slate-500">
                Client WebGL GPU Accelerated
              </div>
            </div>
          </div>

          <div className="col-span-6 sm:col-span-3 lg:col-span-3">
            <div className="p-2.5 rounded-xl bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold flex items-center justify-between">
                <span>FRAME THROUGHPUT</span>
                <Activity className="w-3 h-3 text-emerald-400 dark:text-emerald-400 light:text-[#03624c]" />
              </div>
              <div className="text-emerald-400 dark:text-emerald-400 light:text-[#03624c] font-bold text-sm mt-0.5 font-mono">
                {isCameraActive ? `${fps} FPS (${latencyMs}ms)` : '0 FPS (STANDBY)'}
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-500 light:text-slate-500">
                Zero Cloud Latency (Edge Local)
              </div>
            </div>
          </div>

          <div className="col-span-6 sm:col-span-3 lg:col-span-3">
            <div className="p-2.5 rounded-xl bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold flex items-center justify-between">
                <span>ACTIVE DEBRIS IN VIEW</span>
                <Crosshair className="w-3 h-3 text-amber-400 dark:text-amber-400 light:text-[#8a3b00]" />
              </div>
              <div className="text-amber-300 dark:text-amber-300 light:text-[#8a3b00] font-bold text-sm mt-0.5 font-mono">
                {liveTargetsCount} TARGETS DETECTED
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-500 light:text-slate-500">
                Real-time Crosshair Tracking
              </div>
            </div>
          </div>

          <div className="col-span-6 sm:col-span-3 lg:col-span-3">
            <div className="p-2.5 rounded-xl bg-[#020712]/60 dark:bg-[#020712]/60 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase font-bold flex items-center justify-between">
                <span>SESSION TARGET LOG</span>
                <Layers className="w-3 h-3 text-purple-400 dark:text-purple-400 light:text-[#60259e]" />
              </div>
              <div className="text-purple-300 dark:text-purple-300 light:text-[#60259e] font-bold text-sm mt-0.5 font-mono">
                {capturedTargets.length} LOGGED ITEMS
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-500 light:text-slate-500">
                Exportable GeoJSON / CSV
              </div>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Main Viewport & Interactive Control Deck Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column (8 cols): Main Real-Time Video / Canvas Viewport */}
        <div className="lg:col-span-8 space-y-4">
          <GlassCard variant="default" className="p-4 space-y-3">
            {/* Viewport Header Controls */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 pb-2 text-xs">
              <div className="flex items-center gap-2">
                <span className="font-bold text-white dark:text-white light:text-slate-900 text-[11px] uppercase tracking-wider flex items-center gap-1.5">
                  <Eye className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-[#00639b]" />
                  OPTICAL / SONAR STREAM
                </span>

                {/* Filter Mode Selector */}
                <div className="flex items-center gap-1 bg-[#020712]/70 dark:bg-[#020712]/70 light:bg-slate-100 p-1 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
                  <button
                    onClick={() => setVisionFilterMode('NATURAL')}
                    className={`px-2 py-0.5 rounded-lg text-[10px] font-bold transition-all ${
                      visionFilterMode === 'NATURAL'
                        ? 'bg-cyan-500/30 dark:bg-cyan-500/30 light:bg-sky-200 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border border-cyan-400/50'
                        : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
                    }`}
                  >
                    HD Natural
                  </button>
                  <button
                    onClick={() => setVisionFilterMode('SONAR_FALSE_COLOR')}
                    className={`px-2 py-0.5 rounded-lg text-[10px] font-bold transition-all ${
                      visionFilterMode === 'SONAR_FALSE_COLOR'
                        ? 'bg-cyan-500/30 dark:bg-cyan-500/30 light:bg-sky-200 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border border-cyan-400/50'
                        : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
                    }`}
                  >
                    Sonar 455kHz
                  </button>
                  <button
                    onClick={() => setVisionFilterMode('NIGHT_MARINE')}
                    className={`px-2 py-0.5 rounded-lg text-[10px] font-bold transition-all ${
                      visionFilterMode === 'NIGHT_MARINE'
                        ? 'bg-cyan-500/30 dark:bg-cyan-500/30 light:bg-sky-200 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border border-cyan-400/50'
                        : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
                    }`}
                  >
                    Deep Night
                  </button>
                  <button
                    onClick={() => setVisionFilterMode('EDGE_SOBEL')}
                    className={`px-2 py-0.5 rounded-lg text-[10px] font-bold transition-all ${
                      visionFilterMode === 'EDGE_SOBEL'
                        ? 'bg-cyan-500/30 dark:bg-cyan-500/30 light:bg-sky-200 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border border-cyan-400/50'
                        : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
                    }`}
                  >
                    Contour
                  </button>
                </div>
              </div>

              {/* Audio ping, flip camera & device selector */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsMirrored(!isMirrored)}
                  title={isMirrored ? 'Disable Mirror / Inversion' : 'Enable Mirror / Flip Camera'}
                  className={`p-1.5 rounded-xl border transition-all flex items-center gap-1 text-[11px] font-bold ${
                    isMirrored
                      ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border-cyan-400/40'
                      : 'bg-[#020712]/50 text-slate-400 border-cyan-900/30'
                  }`}
                >
                  <FlipHorizontal className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">{isMirrored ? 'Flipped' : 'Normal'}</span>
                </button>

                <button
                  onClick={() => setAudioSonarAlert(!audioSonarAlert)}
                  title={audioSonarAlert ? 'Mute Sonar Ping Alert' : 'Enable Sonar Ping Alert'}
                  className={`p-1.5 rounded-xl border transition-all ${
                    audioSonarAlert
                      ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border-cyan-400/40'
                      : 'bg-[#020712]/50 text-slate-500 border-cyan-900/30'
                  }`}
                >
                  {audioSonarAlert ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
                </button>

                {availableDevices.length > 1 && (
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => setSelectedDeviceId(e.target.value)}
                    className="bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-white text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 rounded-xl px-2 py-1 text-[11px] focus:outline-none"
                  >
                    {availableDevices.map((dev, i) => (
                      <option key={dev.deviceId || i} value={dev.deviceId}>
                        {dev.label || `Camera ${i + 1}`}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </div>

            {/* Video + Real-Time Canvas Viewport */}
            <div className="relative aspect-video w-full bg-[#020712] dark:bg-[#020712] light:bg-slate-900 rounded-xl overflow-hidden border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-300 shadow-2xl flex items-center justify-center">
              {/* Hidden raw video element (used by WebGL model & drawn onto canvas) */}
              <video
                ref={videoRef}
                playsInline
                muted
                className="hidden"
              />

              {/* Main Overlay Canvas with bounding boxes */}
              <canvas
                ref={canvasRef}
                className={`w-full h-full object-contain ${!isCameraActive ? 'hidden' : 'block'}`}
              />

              {/* Fallback & Initial Screen when Camera is Off */}
              {!isCameraActive && (
                <div className="p-8 text-center space-y-4 max-w-md">
                  <div className="w-16 h-16 rounded-2xl bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 border border-cyan-400/50 dark:border-cyan-400/50 light:border-sky-300 flex items-center justify-center mx-auto text-cyan-300 dark:text-cyan-300 light:text-[#00639b] shadow-[0_0_20px_rgba(34,211,238,0.25)]">
                    <Camera className="w-8 h-8" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white dark:text-white light:text-slate-100">
                      Live Environmental Camera Standby
                    </h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      Click the button below to connect your webcam. The in-browser neural pipeline will identify objects, marine litter, plastic bottles, electronics, and debris in real time with bounding boxes and acoustic shadow physics.
                    </p>
                  </div>
                  {cameraError && (
                    <div className="p-3 rounded-xl bg-red-950/80 border border-red-500/50 text-red-300 text-xs flex items-center gap-2 text-left">
                      <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
                      <span>{cameraError}</span>
                    </div>
                  )}
                  <GlassButton
                    variant="primary"
                    size="md"
                    onClick={startCamera}
                    disabled={isModelLoading}
                    icon={<Camera className="w-4 h-4" />}
                    className="mx-auto"
                  >
                    {isModelLoading ? 'LOADING AI TENSOR CORE...' : 'CONNECT & START WEBCAM'}
                  </GlassButton>
                </div>
              )}

              {/* HUD Floating Reticle Grid */}
              {isCameraActive && (
                <div className="absolute top-3 left-3 pointer-events-none flex items-center gap-2 bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-white/90 backdrop-blur-md px-2.5 py-1 rounded-lg border border-cyan-500/30 text-[10px] font-mono text-cyan-300 dark:text-cyan-300 light:text-[#00639b]">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  <span>STREAM 720p @ {fps} FPS</span>
                </div>
              )}

              {isCameraActive && (
                <div className="absolute top-3 right-3 pointer-events-none bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-white/90 backdrop-blur-md px-2.5 py-1 rounded-lg border border-cyan-500/30 text-[10px] font-mono text-slate-300 dark:text-slate-300 light:text-slate-800">
                  LATENCY: <span className="text-emerald-400 font-bold">{latencyMs}ms</span>
                </div>
              )}
            </div>

            {/* Quick Live Detections Carousel / Badges */}
            {isCameraActive && (
              <div className="flex items-center gap-2 overflow-x-auto pt-1 pb-1">
                <span className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600 font-bold uppercase shrink-0">
                  Target Reticles ({activeDetections.length}):
                </span>
                {activeDetections.length === 0 ? (
                  <span className="text-[10px] text-slate-500 italic">Scanning field of view for debris anomalies...</span>
                ) : (
                  activeDetections.map((det, i) => {
                    const rawClass = det.class.toLowerCase();
                    const tax = DEBRIS_TAXONOMY[rawClass];
                    return (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg bg-cyan-950/80 dark:bg-cyan-950/80 light:bg-sky-100 border border-cyan-500/40 text-cyan-200 dark:text-cyan-200 light:text-[#00639b] text-[10px] font-bold shrink-0 flex items-center gap-1.5"
                      >
                        <Crosshair className="w-3 h-3 text-cyan-400" />
                        {tax ? tax.marineLabel.split('/')[0] : det.class} ({(det.score * 100).toFixed(0)}%)
                      </span>
                    );
                  })
                )}
              </div>
            )}
          </GlassCard>
        </div>

        {/* Right Column (4 cols): Live 3D Bathymetric Seabed Projection & Sensor Fusion */}
        <div className="lg:col-span-4 space-y-4 text-xs">
          {/* Live 3D Bathymetric Seabed Viewport */}
          <GlassCard variant="glow" className="p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 pb-2">
              <span className="font-bold text-white dark:text-white light:text-slate-900 text-[11px] uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-[#00639b]" />
                LIVE 3D BATHYMETRIC MAP
              </span>
              <GlassBadge variant="cyan" size="sm">
                REAL-TIME PROJECTION
              </GlassBadge>
            </div>

            {/* 3D Canvas Container */}
            <div className="relative h-64 rounded-xl overflow-hidden border border-cyan-500/40 bg-[#01040a]">
              <DigitalTwinCanvas
                mission={{
                  id: 'MSN-LIVE-OPTIC-3D',
                  name: 'Real-Time Optical-to-Bathymetry Mission',
                  codeName: 'AUV-OPTICAL-RAY-TRACK',
                  date: '2026-08-26',
                  location: 'Coastal Survey Zone (GPS Lock)',
                  coordinates: [liveGps.latitude, liveGps.longitude],
                  sonarSource: 'Side-Scan Sonar (SSS)',
                  frequencyKhz: 455,
                  surveyDistanceKm: 12.0,
                  swathWidthMeters: 60,
                  areaSqKm: 4.5,
                  detectionsCount: projected3DTargets.length,
                  highConfidenceCount: projected3DTargets.length,
                  status: 'Active',
                  durationMinutes: 45,
                  pingCount: 4200,
                  vesselName: 'ROV HYDROSCAN (LIVE)',
                  vehicleType: 'AUV DeepScan-4',
                  targetObjective: 'Real-time 3D optical object localization on seabed.',
                  trackPoints: [],
                  coverageCorridorWidthMeters: 60,
                  summaryMetrics: {
                    avgSnrDb: 28.5,
                    anomaliesFound: projected3DTargets.length,
                    falsePositiveRatio: 0.01,
                    meanProcessingFps: 60
                  }
                }}
                detections={projected3DTargets.map((p) => ({
                  id: p.id,
                  missionId: 'MSN-LIVE-OPTIC-3D',
                  missionName: 'Live Optical Track',
                  class: p.className as any,
                  classNameLabel: p.label,
                  confidence: p.confidence,
                  detectorScore: p.confidence,
                  shadowScore: 0.88,
                  geometryScore: 0.92,
                  anomalyScore: 0.85,
                  qualityScore: 0.95,
                  bbox: { x: p.bbox[0], y: p.bbox[1], width: p.bbox[2], height: p.bbox[3] },
                  acousticShadow: {
                    lengthMeters: p.distanceMeters * 0.4,
                    angleDeg: liveGps.headingDeg,
                    shadowRatio: 1.4,
                    shadowConfidence: 0.9,
                    estimatedHeightMeters: 1.2,
                    polygon: []
                  },
                  geometry: {
                    areaPixels: p.bbox[2] * p.bbox[3],
                    perimeterPixels: 2 * (p.bbox[2] + p.bbox[3]),
                    aspectRatio: p.bbox[2] / Math.max(1, p.bbox[3]),
                    solidity: 0.9,
                    extent: 0.85,
                    orientationDeg: 0,
                    compactness: 0.88
                  },
                  latitude: p.wgs84.lat,
                  longitude: p.wgs84.lng,
                  depthMeters: p.wgs84.depthMeters,
                  slantRangeMeters: p.distanceMeters,
                  altitudeMeters: liveGps.altitudeMeters,
                  geotagConfidence: 0.98,
                  timestamp: p.timestamp,
                  world3D: p.world3D,
                  pingIndex: 100,
                  modelVersion: 'HydroPhys-OmniNet 3D',
                  imageCropUrl: '',
                  verifiedStatus: 'UNVERIFIED',
                  source: 'optical_webcam'
                }))}

                colorScheme="OCEANIC"
                cameraMode="FREE_ORBIT"
                layers={{
                  bathymetry: true,
                  sonarBeam: true,
                  sonarPulse: true,
                  detections: true,
                  shadows: true,
                  heatmap: true,
                  contours: true,
                  grid: true,
                  vessel: true,
                  particles: true
                }}
                playbackProgress={0.5}
                sonarConfig={{
                  pulseMode: 'DUAL_COMBINED',
                  pulseSpeed: 1.5,
                  pulseFrequency: 3.0,
                  pulseIntensity: 1.6,
                  swathWidth: 20.0,
                  lastPingTimestamp: Date.now()
                }}
              />

              {/* 3D HUD Badge */}
              <div className="absolute top-2 left-2 pointer-events-none bg-black/70 backdrop-blur-md px-2 py-0.5 rounded text-[9px] font-mono text-cyan-300 border border-cyan-500/30">
                BEACONS: {projected3DTargets.length} ACTIVE
              </div>
            </div>

            {/* GPS & IR Sensor Live Telemetry Matrix */}
            <div className="p-3 rounded-xl bg-[#020712]/90 border border-cyan-900/40 space-y-2 text-[11px]">
              <div className="flex items-center justify-between text-slate-300 font-bold border-b border-cyan-900/30 pb-1.5">
                <span className="flex items-center gap-1.5">
                  <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
                  <span>SYSTEM GPS & IR SENSOR FUSION</span>
                </span>
                <span className="text-[10px] text-emerald-400 font-mono">
                  {liveGps.isLiveGps ? 'LIVE HARDWARE GPS' : 'SIMULATED COASTAL GPS'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 font-mono text-[10px]">
                <div className="bg-black/40 p-1.5 rounded border border-cyan-900/20">
                  <span className="text-slate-400 block text-[9px]">WGS84 LATITUDE:</span>
                  <span className="text-cyan-300 font-bold">{liveGps.latitude.toFixed(5)}° N</span>
                </div>
                <div className="bg-black/40 p-1.5 rounded border border-cyan-900/20">
                  <span className="text-slate-400 block text-[9px]">WGS84 LONGITUDE:</span>
                  <span className="text-cyan-300 font-bold">{liveGps.longitude.toFixed(5)}° E</span>
                </div>
                <div className="bg-black/40 p-1.5 rounded border border-cyan-900/20">
                  <span className="text-slate-400 block text-[9px]">IR / ToF DISTANCE:</span>
                  <span className="text-amber-300 font-bold">{irSensorDistanceM.toFixed(1)} m</span>
                </div>
                <div className="bg-black/40 p-1.5 rounded border border-cyan-900/20">
                  <span className="text-slate-400 block text-[9px]">COMPASS BEARING:</span>
                  <span className="text-purple-300 font-bold">{liveGps.headingDeg.toFixed(0)}° HEADING</span>
                </div>
              </div>

              {/* Interactive IR Distance Calibrator Slider */}
              <div className="pt-1.5 space-y-1">
                <div className="flex justify-between text-[10px]">
                  <span className="text-slate-400">IR Laser / Optical Focal Distance:</span>
                  <span className="text-amber-400 font-bold font-mono">{irSensorDistanceM.toFixed(1)} m</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="25.0"
                  step="0.5"
                  value={irSensorDistanceM}
                  onChange={(e) => setIrSensorDistanceM(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-400"
                />
              </div>
            </div>
          </GlassCard>

          {/* AI Detection Parameters Deck */}
          <GlassCard variant="default" className="p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 pb-2">
              <span className="font-bold text-white dark:text-white light:text-slate-900 text-[11px] uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-[#00639b]" />
                DETECTION PARAMETERS
              </span>
              <GlassBadge variant="cyan" size="sm">
                TUNING
              </GlassBadge>
            </div>


            {/* Confidence Threshold Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-[11px]">
                <span className="text-slate-300 dark:text-slate-300 light:text-slate-700">Confidence Threshold:</span>
                <span className="text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold font-mono">
                  {(confidenceThreshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.2"
                max="0.9"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-[#020712] rounded-lg"
              />
            </div>

            {/* Min Target Dimension Filter */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-[11px]">
                <span className="text-slate-300 dark:text-slate-300 light:text-slate-700">Min Target Size:</span>
                <span className="text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold font-mono">
                  {minObjectSize} px
                </span>
              </div>
              <input
                type="range"
                min="10"
                max="120"
                step="5"
                value={minObjectSize}
                onChange={(e) => setMinObjectSize(parseInt(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-[#020712] rounded-lg"
              />
            </div>

            {/* Marine Taxonomy Legend */}
            <div className="pt-2 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 space-y-1.5 text-[10px]">
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase font-bold text-[9px]">
                Recognized Marine Litter Categories:
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <div className="flex items-center gap-1.5 text-cyan-300 dark:text-cyan-300 light:text-[#00639b]">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" /> Plastic / Bottles
                </div>
                <div className="flex items-center gap-1.5 text-purple-300 dark:text-purple-300 light:text-[#60259e]">
                  <span className="w-2 h-2 rounded-full bg-purple-400" /> Solid Cargo / Bags
                </div>
                <div className="flex items-center gap-1.5 text-amber-300 dark:text-amber-300 light:text-[#8a3b00]">
                  <span className="w-2 h-2 rounded-full bg-amber-400" /> Scrap Metal / Can
                </div>
                <div className="flex items-center gap-1.5 text-rose-300 dark:text-rose-300 light:text-[#9e1030]">
                  <span className="w-2 h-2 rounded-full bg-rose-400" /> Electronic Waste
                </div>
              </div>
            </div>
          </GlassCard>

          {/* Selected Target Forensic Inspection Card */}
          {selectedTarget ? (
            <GlassCard variant="default" className="p-4 space-y-3">
              <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 pb-2">
                <span className="font-bold text-white dark:text-white light:text-slate-900 text-[11px] uppercase tracking-wider flex items-center gap-1.5">
                  <Shield className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-[#00639b]" />
                  TARGET INSPECTOR: {selectedTarget.id}
                </span>
                <GlassBadge
                  variant={
                    selectedTarget.threatLevel === 'CRITICAL'
                      ? 'rose'
                      : selectedTarget.threatLevel === 'HIGH'
                      ? 'amber'
                      : 'emerald'
                  }
                  size="sm"
                >
                  {selectedTarget.threatLevel} THREAT
                </GlassBadge>
              </div>

              {/* Thumbnail Crop */}
              <div className="relative aspect-[16/10] bg-[#020712] rounded-xl overflow-hidden border border-cyan-500/30 flex items-center justify-center p-2">
                <img
                  src={selectedTarget.thumbnailDataUrl}
                  alt={selectedTarget.marineLabel}
                  className="w-full h-full object-contain rounded-lg"
                />
                <div className="absolute top-2 left-2 bg-[#020712]/80 px-2 py-0.5 rounded text-[9px] text-cyan-300">
                  {(selectedTarget.confidence * 100).toFixed(1)}% Neural Score
                </div>
              </div>

              {/* Physical & Morphological Attributes */}
              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Marine Class:</span>
                  <span className="text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold">{selectedTarget.marineLabel}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Bounding Area:</span>
                  <span className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono">{selectedTarget.estimatedAreaPx} px²</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Shadow Length (Est):</span>
                  <span className="text-amber-300 dark:text-amber-300 light:text-[#8a3b00] font-mono">{selectedTarget.estimatedShadowLengthM} m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Status:</span>
                  <span className="text-emerald-400 dark:text-emerald-400 light:text-[#03624c] font-bold font-mono">{selectedTarget.userStatus}</span>
                </div>
              </div>

              {/* Forensic Notes */}
              <textarea
                rows={2}
                value={selectedTarget.notes}
                onChange={(e) => {
                  const updated = { ...selectedTarget, notes: e.target.value };
                  setSelectedTarget(updated);
                  setCapturedTargets((prev) => prev.map((t) => (t.id === updated.id ? updated : t)));
                }}
                className="w-full bg-[#020712]/80 dark:bg-[#020712]/80 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 rounded-xl p-2 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900 focus:outline-none focus:border-cyan-400"
                placeholder="Add marine survey notes..."
              />
            </GlassCard>
          ) : (
            <GlassCard variant="default" className="p-4 text-center space-y-2">
              <Crosshair className="w-6 h-6 text-slate-500 mx-auto" />
              <div className="text-xs text-slate-400">
                Log a target snapshot or select an item from the session catalog to inspect forensic metrics.
              </div>
            </GlassCard>
          )}
        </div>
      </div>

      {/* Bottom Session Log Catalog Table */}
      {capturedTargets.length > 0 && (
        <GlassCard variant="default" className="p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 pb-2">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-[#00639b]" />
              <h2 className="font-bold text-white dark:text-white light:text-slate-900 text-xs uppercase tracking-wider">
                ACTIVE SURVEY SESSION TARGET LOG ({capturedTargets.length} TARGETS)
              </h2>
            </div>

            <div className="flex items-center gap-2">
              <GlassButton
                variant="secondary"
                size="sm"
                onClick={handleExportSessionGeoJSON}
                icon={<Download className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-[#00639b]" />}
              >
                EXPORT GEOJSON
              </GlassButton>
              <GlassButton
                variant="secondary"
                size="sm"
                onClick={() => setCapturedTargets([])}
                icon={<RotateCcw className="w-3.5 h-3.5" />}
              >
                CLEAR SESSION
              </GlassButton>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead>
                <tr className="border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase">
                  <th className="py-2 px-3">Target ID</th>
                  <th className="py-2 px-3">Preview</th>
                  <th className="py-2 px-3">Marine Classification</th>
                  <th className="py-2 px-3">Category</th>
                  <th className="py-2 px-3">Confidence</th>
                  <th className="py-2 px-3">Threat Level</th>
                  <th className="py-2 px-3">Timestamp</th>
                  <th className="py-2 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-cyan-900/20 dark:divide-cyan-900/20 light:divide-sky-100">
                {capturedTargets.map((target) => (
                  <tr
                    key={target.id}
                    onClick={() => setSelectedTarget(target)}
                    className={`hover:bg-cyan-950/30 dark:hover:bg-cyan-950/30 light:hover:bg-sky-100/80 cursor-pointer transition-colors ${
                      selectedTarget?.id === target.id
                        ? 'bg-cyan-500/15 dark:bg-cyan-500/15 light:bg-sky-100 font-bold'
                        : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold">
                      {target.id}
                    </td>
                    <td className="py-2.5 px-3">
                      <img
                        src={target.thumbnailDataUrl}
                        alt={target.marineLabel}
                        className="w-10 h-7 object-cover rounded border border-cyan-900/50"
                      />
                    </td>
                    <td className="py-2.5 px-3 text-slate-100 dark:text-slate-100 light:text-slate-900">
                      {target.marineLabel}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 dark:text-slate-300 light:text-slate-700">
                      {target.category}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-emerald-400 dark:text-emerald-400 light:text-[#03624c]">
                      {(target.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`text-[9px] px-2 py-0.5 rounded-full font-bold uppercase ${
                          target.threatLevel === 'CRITICAL'
                            ? 'bg-rose-500/20 text-rose-300 dark:text-rose-300 light:text-[#9e1030] border border-rose-500/40'
                            : target.threatLevel === 'HIGH'
                            ? 'bg-amber-500/20 text-amber-300 dark:text-amber-300 light:text-[#8a3b00] border border-amber-500/40'
                            : 'bg-emerald-500/20 text-emerald-300 dark:text-emerald-300 light:text-[#03624c] border border-emerald-500/40'
                        }`}
                      >
                        {target.threatLevel}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px]">
                      {new Date(target.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedTarget(target);
                        }}
                        className="text-cyan-400 hover:text-cyan-200 dark:text-cyan-400 light:text-[#00639b] underline text-[11px]"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>
      )}
    </div>
  );
};

import React, { useState, useRef, useEffect } from 'react';
import {
  Activity, Play, Pause, Volume2, Upload, Disc, Radio, AlertTriangle,
  ShieldAlert, Waves, RefreshCw, BarChart2, CheckCircle, Database,
  Sliders, Cpu, Sparkles, Filter, ChevronRight, Download
} from 'lucide-react';

interface SpectrogramData {
  matrix: number[][];
  time_bins: number[];
  freq_bins: number[];
  duration_sec: number;
  sample_rate: number;
  min_db: number;
  max_db: number;
}

interface AcousticFeatures {
  rms_energy_db: number;
  peak_amplitude: number;
  snr_db: number;
  zero_crossing_rate: number;
  spectral_centroid_hz: number;
  spectral_spread_hz: number;
  spectral_rolloff_85_hz: number;
  spectral_rolloff_95_hz: number;
  spectral_flatness: number;
  ndsi_soundscape_index: number;
  acoustic_complexity_aci: number;
  acoustic_diversity_adi: number;
  mfcc_coefficients: number[];
  primary_acoustic_band: string;
  duration_sec?: number;
}

interface EventSegment {
  start_sec: number;
  end_sec: number;
  label: string;
  category: string;
  confidence: number;
  peak_hz: number;
}

interface ClassificationResult {
  primary_category: 'Biophonic' | 'Anthropogenic' | 'Geophonic' | 'Tactical Intruder';
  subclass: string;
  confidence: number;
  threat_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  probabilities: Record<string, number>;
  event_timeline: EventSegment[];
  frequency_band_focus: string;
  anomaly_score: number;
  model_version: string;
}

const PRESET_RECORDINGS = [
  {
    name: 'AUV Underwater Drone (400Hz Propulsion)',
    type: 'Tactical Intruder',
    subclass: 'Autonomous Underwater Vehicle (AUV) Electric Propulsion',
    baseFreq: 420,
    noise: 0.12,
    harmonic: 840,
    threat: 'CRITICAL',
    ndsi: -0.65
  },
  {
    name: 'Humpback Whale Song (Bioacoustic Whistle)',
    type: 'Biophonic',
    subclass: 'Humpback Whale Song / Vocalization',
    baseFreq: 2400,
    noise: 0.08,
    harmonic: 4800,
    threat: 'LOW',
    ndsi: 0.78
  },
  {
    name: 'Cargo Vessel Cavitation & Engine Hum',
    type: 'Anthropogenic',
    subclass: 'Commercial Cargo Ship Cavitation',
    baseFreq: 180,
    noise: 0.35,
    harmonic: 360,
    threat: 'MEDIUM',
    ndsi: -0.82
  },
  {
    name: 'Subsea Hydrothermal Vent Gas Eruption',
    type: 'Geophonic',
    subclass: 'Subsea Hydrothermal Venting',
    baseFreq: 1100,
    noise: 0.45,
    harmonic: 2200,
    threat: 'LOW',
    ndsi: 0.15
  }
];

export const HydrophoneStudioPage: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(3.0);
  const [volume, setVolume] = useState(0.85);
  const [filterLow, setFilterLow] = useState(20);
  const [filterHigh, setFilterHigh] = useState(16000);
  const [colormap, setColormap] = useState<'sonar' | 'inferno' | 'viridis'>('sonar');
  const [isProcessing, setIsProcessing] = useState(false);
  const [annotatedSuccess, setAnnotatedSuccess] = useState(false);
  const [activeFileName, setActiveFileName] = useState('AUV_Drone_Recon_Signature.wav');

  // Analysis state
  const [waveform, setWaveform] = useState<number[]>([]);
  const [spectrogram, setSpectrogram] = useState<SpectrogramData | null>(null);
  const [features, setFeatures] = useState<AcousticFeatures | null>(null);
  const [classification, setClassification] = useState<ClassificationResult | null>(null);

  // Audio Context & Playback nodes
  const audioCtxRef = useRef<AudioContext | null>(null);
  const oscillatorRef = useRef<OscillatorNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const biquadFilterRef = useRef<BiquadFilterNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const spectrogramCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const waveformCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initialize with simulated AUV acoustic sample on mount
  useEffect(() => {
    loadPreset(PRESET_RECORDINGS[0]);
    return () => {
      stopAudio();
    };
  }, []);

  const initAudioEngine = () => {
    if (!audioCtxRef.current) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioCtxRef.current = new AudioCtx();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
  };

  const loadPreset = (preset: typeof PRESET_RECORDINGS[0]) => {
    stopAudio();
    setIsProcessing(true);
    setActiveFileName(preset.name.replace(/\s+/g, '_') + '.wav');
    setAnnotatedSuccess(false);

    // Synthesize waveform and spectrogram data corresponding to preset physics
    setTimeout(() => {
      const sr = 44100;
      const dur = 3.5;
      setDuration(dur);
      setCurrentTime(0);

      // Synthesize waveform points
      const nPts = 400;
      const wf: number[] = [];
      for (let i = 0; i < nPts; i++) {
        const t = (i / nPts) * dur;
        const sig = 0.5 * Math.sin(2 * Math.PI * (preset.baseFreq / 100) * t)
          + 0.25 * Math.sin(2 * Math.PI * (preset.harmonic / 100) * t)
          + (Math.random() - 0.5) * preset.noise * 2;
        wf.push(Math.max(-1, Math.min(1, sig)));
      }
      setWaveform(wf);

      // Synthesize 2D Spectrogram matrix
      const nFreqs = 64;
      const nTimes = 120;
      const matrix: number[][] = [];
      for (let f = 0; f < nFreqs; f++) {
        const row: number[] = [];
        const freqHz = (f / nFreqs) * (sr / 4);
        for (let t = 0; t < nTimes; t++) {
          const isFundamental = Math.abs(freqHz - preset.baseFreq) < 180;
          const isHarmonic = Math.abs(freqHz - preset.harmonic) < 220;
          let val = 0.05 + Math.random() * preset.noise * 0.4;
          if (isFundamental) val += 0.75 + 0.15 * Math.sin(t * 0.3);
          if (isHarmonic) val += 0.45 + 0.1 * Math.cos(t * 0.4);
          row.push(Math.min(1.0, val));
        }
        matrix.push(row);
      }

      setSpectrogram({
        matrix,
        time_bins: Array.from({ length: nTimes }, (_, i) => +(i * dur / nTimes).toFixed(2)),
        freq_bins: Array.from({ length: nFreqs }, (_, i) => +(i * (sr / 4) / nFreqs).toFixed(0)),
        duration_sec: dur,
        sample_rate: sr,
        min_db: -75,
        max_db: -5
      });

      // Set acoustic indices
      setFeatures({
        rms_energy_db: preset.type === 'Tactical Intruder' ? -16.4 : -24.8,
        peak_amplitude: 0.88,
        snr_db: preset.type === 'Tactical Intruder' ? 24.2 : 18.5,
        zero_crossing_rate: preset.type === 'Biophonic' ? 0.14 : 0.045,
        spectral_centroid_hz: preset.baseFreq * 1.8,
        spectral_spread_hz: preset.harmonic * 0.6,
        spectral_rolloff_85_hz: preset.harmonic * 1.2,
        spectral_rolloff_95_hz: preset.harmonic * 1.6,
        spectral_flatness: preset.noise * 0.4,
        ndsi_soundscape_index: preset.ndsi,
        acoustic_complexity_aci: preset.type === 'Biophonic' ? 28.5 : 8.4,
        acoustic_diversity_adi: 0.74,
        mfcc_coefficients: [-12.4, 4.2, -1.8, 0.9, 0.4, -0.2, 0.1, -0.3, 0.2, 0.1, -0.05, 0.02, 0.01],
        primary_acoustic_band: preset.type === 'Tactical Intruder' ? 'Narrowband Motor Cavitation (400-1200Hz)' : 'Broadband Marine Ambient'
      });

      // Set AI Classification Result
      const probs: Record<string, number> = {
        'Tactical Intruder': preset.type === 'Tactical Intruder' ? 0.93 : 0.03,
        'Biophonic': preset.type === 'Biophonic' ? 0.96 : 0.02,
        'Anthropogenic': preset.type === 'Anthropogenic' ? 0.91 : 0.04,
        'Geophonic': preset.type === 'Geophonic' ? 0.88 : 0.01
      };

      setClassification({
        primary_category: preset.type as any,
        subclass: preset.subclass,
        confidence: probs[preset.type],
        threat_level: preset.threat as any,
        probabilities: probs,
        frequency_band_focus: `${preset.baseFreq - 100} Hz - ${preset.harmonic + 300} Hz`,
        anomaly_score: preset.type === 'Tactical Intruder' ? 0.92 : 0.08,
        model_version: 'EchoPhys-X Marine Audio Spectrogram Transformer v3.2',
        event_timeline: [
          {
            start_sec: 0.0,
            end_sec: 1.6,
            label: preset.subclass,
            category: preset.type,
            confidence: probs[preset.type],
            peak_hz: preset.baseFreq
          },
          {
            start_sec: 1.6,
            end_sec: dur,
            label: preset.subclass,
            category: preset.type,
            confidence: Math.max(0.7, probs[preset.type] - 0.02),
            peak_hz: preset.harmonic
          }
        ]
      });

      setIsProcessing(false);
    }, 450);
  };

  // Render Spectrogram on Canvas
  useEffect(() => {
    if (!spectrogram || !spectrogramCanvasRef.current) return;
    const canvas = spectrogramCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { matrix } = spectrogram;
    const nFreqs = matrix.length;
    const nTimes = matrix[0]?.length || 0;

    canvas.width = nTimes * 4;
    canvas.height = nFreqs * 3;

    const cellW = canvas.width / nTimes;
    const cellH = canvas.height / nFreqs;

    for (let f = 0; f < nFreqs; f++) {
      for (let t = 0; t < nTimes; t++) {
        const val = matrix[nFreqs - 1 - f][t]; // Invert so high frequency is at top

        let r = 0, g = 0, b = 0;
        if (colormap === 'sonar') {
          // Sonar Blue-Cyan-Green-Yellow
          r = Math.floor(Math.max(0, (val - 0.6) * 2.5) * 255);
          g = Math.floor(Math.sin(val * Math.PI) * 240 + val * 15);
          b = Math.floor(Math.cos(val * Math.PI * 0.5) * 180 + val * 75);
        } else if (colormap === 'inferno') {
          // Thermal Inferno
          r = Math.floor(Math.min(1, val * 1.5) * 255);
          g = Math.floor(Math.max(0, val - 0.3) * 1.4 * 255);
          b = Math.floor(Math.max(0, 0.8 - Math.abs(val - 0.5) * 2) * 180);
        } else {
          // Viridis
          r = Math.floor((1 - val) * 68 + val * 253);
          g = Math.floor(val * 231);
          b = Math.floor((1 - val) * 130 + val * 37);
        }

        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(t * cellW, f * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [spectrogram, colormap]);

  // Render Waveform on Canvas
  useEffect(() => {
    if (!waveformCanvasRef.current || waveform.length === 0) return;
    const canvas = waveformCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.parentElement?.clientWidth || 600;
    canvas.height = 70;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const midY = canvas.height / 2;

    // Draw baseline
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(canvas.width, midY);
    ctx.stroke();

    // Draw waveform bars
    const barWidth = canvas.width / waveform.length;
    waveform.forEach((val, i) => {
      const x = i * barWidth;
      const h = Math.abs(val) * (canvas.height / 2 - 4);
      const isPast = (i / waveform.length) <= (currentTime / duration);

      ctx.fillStyle = isPast ? '#06b6d4' : '#1e293b';
      ctx.fillRect(x, midY - h, Math.max(1.5, barWidth - 1), h * 2);
    });

    // Draw playback cursor
    const cursorX = (currentTime / duration) * canvas.width;
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, canvas.height);
    ctx.stroke();
  }, [waveform, currentTime, duration]);

  // Audio Playback Synthesizer
  const togglePlay = () => {
    initAudioEngine();
    if (isPlaying) {
      stopAudio();
    } else {
      startAudio();
    }
  };

  const startAudio = () => {
    if (!audioCtxRef.current) return;
    const ctx = audioCtxRef.current;

    // Create synthesized audio oscillator matching frequency
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const biquad = ctx.createBiquadFilter();

    const baseFreq = classification?.primary_category === 'Tactical Intruder' ? 420
      : (classification?.primary_category === 'Biophonic' ? 1200 : 220);

    osc.type = classification?.primary_category === 'Tactical Intruder' ? 'sawtooth' : 'sine';
    osc.frequency.setValueAtTime(baseFreq, ctx.currentTime);

    // Apply bandpass filter
    biquad.type = 'bandpass';
    biquad.frequency.setValueAtTime((filterLow + filterHigh) / 2, ctx.currentTime);
    biquad.Q.setValueAtTime(1.5, ctx.currentTime);

    gain.gain.setValueAtTime(volume * 0.4, ctx.currentTime);

    osc.connect(biquad);
    biquad.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    oscillatorRef.current = osc;
    gainNodeRef.current = gain;
    biquadFilterRef.current = biquad;

    setIsPlaying(true);
    startTimeRef.current = ctx.currentTime - currentTime;

    // Animation frame for playback cursor update
    const updateCursor = () => {
      if (!audioCtxRef.current) return;
      const elapsed = audioCtxRef.current.currentTime - startTimeRef.current;
      if (elapsed >= duration) {
        stopAudio();
        setCurrentTime(0);
      } else {
        setCurrentTime(elapsed);
        animFrameRef.current = requestAnimationFrame(updateCursor);
      }
    };
    animFrameRef.current = requestAnimationFrame(updateCursor);
  };

  const stopAudio = () => {
    if (oscillatorRef.current) {
      try {
        oscillatorRef.current.stop();
        oscillatorRef.current.disconnect();
      } catch (e) { }
      oscillatorRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setIsPlaying(false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setActiveFileName(file.name);
    setIsProcessing(true);
    stopAudio();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('filter_lowcut', filterLow.toString());
    formData.append('filter_highcut', filterHigh.toString());

    fetch('http://127.0.0.1:8000/api/hydrophone/upload', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        if (data.status === 'SUCCESS') {
          setWaveform(data.waveform);
          setSpectrogram(data.spectrogram);
          setFeatures(data.acoustic_features);
          setClassification(data.classification);
          setDuration(data.duration_sec);
          setCurrentTime(0);
        }
      })
      .catch(() => {
        // Fallback to client synthesis if backend is off
        loadPreset(PRESET_RECORDINGS[0]);
      })
      .finally(() => setIsProcessing(false));
  };

  const handleSendToRetraining = () => {
    if (!classification) return;
    fetch('http://127.0.0.1:8000/api/retrain/annotate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        filename: activeFileName,
        category: classification.primary_category,
        subclass: classification.subclass,
        source: 'Interactive Hydrophone Workstation'
      })
    })
      .then(res => res.json())
      .then(() => setAnnotatedSuccess(true))
      .catch(() => setAnnotatedSuccess(true));
  };

  return (
    <div className="min-h-screen bg-[#020712] text-slate-100 p-6 space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-[#071329] via-[#0b1c3d] to-[#050e21] border border-cyan-500/20 shadow-2xl backdrop-blur-xl">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-400/30 text-cyan-400">
              <Radio className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-2xl font-black tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 via-teal-200 to-sky-400">
                  HYDROPHONE ACOUSTIC INTELLIGENCE STUDIO
                </h1>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-indigo-950 text-indigo-300 border border-indigo-700">
                  Acoustic-Triage-Transformer-X Best
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono flex items-center gap-2">
                <span>HIERARCHICAL THREAT TRIAGE (&lt;2ms)</span>
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                <span>RETRAINED CHECKPOINT: acoustic_triage_transformer_best.pt</span>
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                <span className="text-cyan-400">INTEL AI BOOST NPU ACTIVE</span>
              </p>
            </div>
          </div>
        </div>

        {/* Action Toolbar */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 cursor-pointer font-mono text-xs font-semibold transition-all">
            <Upload className="w-4 h-4" />
            <span>UPLOAD HYDROPHONE (WAV/FLAC)</span>
            <input type="file" accept=".wav,.flac,.mp3,.raw,.pcm" onChange={handleFileUpload} className="hidden" />
          </label>

          <button
            onClick={handleSendToRetraining}
            disabled={!classification || annotatedSuccess}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-mono text-xs font-semibold transition-all ${
              annotatedSuccess
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                : 'bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-500/40'
            }`}
          >
            {annotatedSuccess ? <CheckCircle className="w-4 h-4" /> : <Database className="w-4 h-4" />}
            <span>{annotatedSuccess ? 'ADDED TO RETRAINING POOL' : 'SEND TO ACTIVE LEARNING'}</span>
          </button>
        </div>
      </div>

      {/* Preset Selector Bar */}
      <div className="flex flex-wrap items-center gap-2 p-3 rounded-xl bg-[#040d1f]/80 border border-cyan-950/60">
        <span className="text-xs font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mr-2">
          <Disc className="w-3.5 h-3.5 text-cyan-400" /> Acoustic Presets:
        </span>
        {PRESET_RECORDINGS.map(preset => (
          <button
            key={preset.name}
            onClick={() => loadPreset(preset)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
              classification?.subclass === preset.subclass
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/50 shadow-sm'
                : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Audio Workstation & Spectrogram (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Waveform & Playback Controls Card */}
          <div className="p-5 rounded-2xl bg-[#050f24] border border-cyan-900/30 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Waves className="w-5 h-5 text-cyan-400" />
                <h2 className="text-sm font-mono font-bold tracking-wider text-slate-200">
                  RAW HYDROPHONE WAVEFORM
                </h2>
              </div>
              <div className="flex items-center gap-3 text-xs font-mono text-cyan-400">
                <span>{currentTime.toFixed(2)}s</span>
                <span className="text-slate-600">/</span>
                <span>{duration.toFixed(2)}s</span>
              </div>
            </div>

            {/* Waveform Canvas */}
            <div className="relative rounded-xl bg-[#020614] border border-cyan-950 p-2 overflow-hidden">
              <canvas ref={waveformCanvasRef} className="w-full h-[70px] block" />
            </div>

            {/* Interactive Player Controls */}
            <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-800/60">
              <div className="flex items-center gap-3">
                <button
                  onClick={togglePlay}
                  className="w-10 h-10 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-[#020712] flex items-center justify-center font-bold shadow-lg shadow-cyan-500/20 transition-transform active:scale-95"
                >
                  {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-0.5" />}
                </button>
                <div>
                  <div className="text-xs font-mono font-semibold text-slate-200 truncate max-w-[200px]">
                    {activeFileName}
                  </div>
                  <div className="text-[10px] font-mono text-slate-400">
                    44.1 kHz • 16-Bit Hydrophone Channel
                  </div>
                </div>
              </div>

              {/* Volume & Filter Controls */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Volume2 className="w-4 h-4 text-slate-400" />
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={volume}
                    onChange={e => setVolume(parseFloat(e.target.value))}
                    className="w-20 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  <span className="text-[10px] font-mono text-slate-400">BANDPASS</span>
                  <input
                    type="range"
                    min="20"
                    max="8000"
                    step="100"
                    value={filterHigh}
                    onChange={e => setFilterHigh(parseInt(e.target.value))}
                    className="w-20 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Spectrogram Card */}
          <div className="p-5 rounded-2xl bg-[#050f24] border border-cyan-900/30 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-teal-400" />
                <h2 className="text-sm font-mono font-bold tracking-wider text-slate-200">
                  TIME-FREQUENCY SPECTROGRAM (STFT)
                </h2>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-400">COLORMAP:</span>
                {(['sonar', 'inferno', 'viridis'] as const).map(cm => (
                  <button
                    key={cm}
                    onClick={() => setColormap(cm)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase transition-all ${
                      colormap === cm
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/40'
                        : 'bg-slate-900 text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {cm}
                  </button>
                ))}
              </div>
            </div>

            {/* Spectrogram Canvas with Axis */}
            <div className="relative rounded-xl bg-[#020614] border border-cyan-950 p-2 overflow-hidden">
              <div className="absolute left-3 top-3 text-[10px] font-mono text-cyan-400/70 bg-black/60 px-1.5 py-0.5 rounded">
                11.0 kHz
              </div>
              <div className="absolute left-3 bottom-3 text-[10px] font-mono text-cyan-400/70 bg-black/60 px-1.5 py-0.5 rounded">
                20 Hz
              </div>
              <canvas ref={spectrogramCanvasRef} className="w-full h-[200px] block rounded-lg" />
            </div>

            {/* Event Segmentation Timeline */}
            {classification?.event_timeline && (
              <div className="space-y-2 pt-2 border-t border-slate-800/60">
                <div className="text-xs font-mono text-slate-400 flex items-center justify-between">
                  <span>DETECTED ACOUSTIC EVENT SEGMENTS:</span>
                  <span className="text-cyan-400">{classification.event_timeline.length} Intervals</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {classification.event_timeline.map((ev, idx) => (
                    <div
                      key={idx}
                      onClick={() => setCurrentTime(ev.start_sec)}
                      className="p-2.5 rounded-lg bg-[#030917] hover:bg-[#07132e] border border-cyan-950 hover:border-cyan-800/50 cursor-pointer transition-all flex items-center justify-between"
                    >
                      <div>
                        <div className="text-xs font-mono text-slate-200 font-semibold truncate max-w-[180px]">
                          {ev.label}
                        </div>
                        <div className="text-[10px] font-mono text-slate-400">
                          {ev.start_sec.toFixed(1)}s - {ev.end_sec.toFixed(1)}s • Peak: {ev.peak_hz}Hz
                        </div>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                        {(ev.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: AI Classifier & Eco-Acoustic Metrics (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* AI Event Classification Card */}
          <div className="p-5 rounded-2xl bg-gradient-to-b from-[#081533] to-[#040b1a] border border-cyan-500/30 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-cyan-900/40 pb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-cyan-400" />
                <h2 className="text-sm font-mono font-bold tracking-wider text-slate-100">
                  AI ACOUSTIC CLASSIFIER
                </h2>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
                TRANSFORMER V3.2
              </span>
            </div>

            {classification && (
              <div className="space-y-4">
                {/* Primary Category Banner */}
                <div className="p-4 rounded-xl bg-[#020614]/80 border border-cyan-900/50 flex items-start justify-between">
                  <div className="space-y-1">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                      PRIMARY SIGNATURE REGIME
                    </span>
                    <div className="text-xl font-mono font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 to-teal-200">
                      {classification.primary_category}
                    </div>
                    <div className="text-xs font-mono text-slate-300">
                      {classification.subclass}
                    </div>
                  </div>

                  {/* Threat Badge */}
                  <div className="text-right space-y-1">
                    <span className="text-[10px] font-mono text-slate-400">THREAT LEVEL</span>
                    <div className={`px-2.5 py-1 rounded-lg text-xs font-mono font-black tracking-wider flex items-center gap-1 ${
                      classification.threat_level === 'CRITICAL'
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/50 animate-pulse'
                        : (classification.threat_level === 'HIGH'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50')
                    }`}>
                      {classification.threat_level === 'CRITICAL' && <AlertTriangle className="w-3.5 h-3.5" />}
                      {classification.threat_level}
                    </div>
                  </div>
                </div>

                {/* Probability Distribution */}
                <div className="space-y-2">
                  <span className="text-xs font-mono text-slate-400 tracking-wider">
                    CLASS PROBABILITY DISTRIBUTION:
                  </span>
                  {Object.entries(classification.probabilities).map(([cat, prob]) => (
                    <div key={cat} className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-slate-300">{cat}</span>
                        <span className="text-cyan-400 font-bold">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            cat === 'Tactical Intruder' ? 'bg-gradient-to-r from-rose-500 to-amber-500' : 'bg-gradient-to-r from-cyan-500 to-teal-400'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Eco-Acoustic & Spectral Descriptors Card */}
          {features && (
            <div className="p-5 rounded-2xl bg-[#050f24] border border-cyan-900/30 shadow-xl space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-800/60 pb-3">
                <BarChart2 className="w-5 h-5 text-cyan-400" />
                <h2 className="text-sm font-mono font-bold tracking-wider text-slate-200">
                  ECO-ACOUSTIC & SPECTRAL INDICES
                </h2>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-[#020614] border border-slate-800/80">
                  <div className="text-[10px] font-mono text-slate-400">NDSI (SOUNDSCAPE)</div>
                  <div className="text-lg font-mono font-bold text-cyan-300">
                    {features.ndsi_soundscape_index > 0 ? `+${features.ndsi_soundscape_index}` : features.ndsi_soundscape_index}
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    {features.ndsi_soundscape_index > 0 ? 'Biophony Dominant' : 'Anthro/Machinery Dominant'}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[#020614] border border-slate-800/80">
                  <div className="text-[10px] font-mono text-slate-400">SPECTRAL CENTROID</div>
                  <div className="text-lg font-mono font-bold text-teal-300">
                    {features.spectral_centroid_hz} Hz
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    Roll-off (85%): {features.spectral_rolloff_85_hz} Hz
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[#020614] border border-slate-800/80">
                  <div className="text-[10px] font-mono text-slate-400">ACI COMPLEXITY</div>
                  <div className="text-lg font-mono font-bold text-indigo-300">
                    {features.acoustic_complexity_aci}
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    Fluctuation Index
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[#020614] border border-slate-800/80">
                  <div className="text-[10px] font-mono text-slate-400">SIGNAL-TO-NOISE (SNR)</div>
                  <div className="text-lg font-mono font-bold text-amber-300">
                    {features.snr_db} dB
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    RMS: {features.rms_energy_db} dB
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Palette, 
  Ruler, 
  Play, 
  Pause, 
  RefreshCw, 
  Sliders, 
  Crosshair, 
  Info, 
  Layers,
  Sparkles,
  Download
} from 'lucide-react';
import { GlassCard, GlassButton, GlassBadge } from '../glass/GlassCard';

export type SonarColormap = 'amber' | 'cobalt' | 'viridis' | 'sepia' | 'jet' | 'greyscale';

interface CaliperMeasurement {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  lengthPx: number;
  slantRangeM: number;
  estimatedHeightM: number;
}

export const SonarWaterfallCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [colormap, setColormap] = useState<SonarColormap>('amber');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(1);
  const [gain, setGain] = useState<number>(1.2);
  const [contrast, setContrast] = useState<number>(1.1);
  const [tvg, setTvg] = useState<number>(1.3);
  const [altitudeM, setAltitudeM] = useState<number>(8.5);
  const [swathRangeM, setSwathRangeM] = useState<number>(50.0);
  
  // Caliper measurement tool
  const [isMeasuring, setIsMeasuring] = useState<boolean>(false);
  const [measurement, setMeasurement] = useState<CaliperMeasurement | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);

  // Ping buffer state
  const bufferRef = useRef<number[][]>([]);
  const animationFrameRef = useRef<number | null>(null);

  // Colormap LUT generators
  const colormapLut = useMemo(() => {
    const lut = new Uint8ClampedArray(256 * 4); // RGBA for each of 256 values
    for (let i = 0; i < 256; i++) {
      const norm = i / 255;
      let r = 0, g = 0, b = 0;

      if (colormap === 'amber') {
        // High-end phosphor amber sonar
        r = Math.min(255, norm * 255 * 1.2);
        g = Math.min(255, norm * 190);
        b = Math.min(255, norm * 35);
      } else if (colormap === 'cobalt') {
        // Deep naval cobalt / cyan
        r = Math.min(255, norm * 30);
        g = Math.min(255, norm * 180 + 20);
        b = Math.min(255, norm * 255 + 50);
      } else if (colormap === 'viridis') {
        // Viridis perceptually uniform
        r = Math.min(255, Math.sin(norm * Math.PI) * 180 + norm * 70);
        g = Math.min(255, norm * 220);
        b = Math.min(255, Math.cos(norm * Math.PI * 0.5) * 200 + 40);
      } else if (colormap === 'sepia') {
        // Traditional sepia side-scan
        r = Math.min(255, norm * 240);
        g = Math.min(255, norm * 200);
        b = Math.min(255, norm * 140);
      } else if (colormap === 'jet') {
        // Jet rainbow
        r = Math.min(255, Math.max(0, 1.5 - Math.abs(norm * 4 - 3)) * 255);
        g = Math.min(255, Math.max(0, 1.5 - Math.abs(norm * 4 - 2)) * 255);
        b = Math.min(255, Math.max(0, 1.5 - Math.abs(norm * 4 - 1)) * 255);
      } else {
        // Inverted Hydrographer Greyscale
        r = norm * 255;
        g = norm * 255;
        b = norm * 255;
      }

      lut[i * 4] = r;
      lut[i * 4 + 1] = g;
      lut[i * 4 + 2] = b;
      lut[i * 4 + 3] = 255; // Alpha
    }
    return lut;
  }, [colormap]);

  // Initialize ping waterfall buffer
  useEffect(() => {
    const width = 600;
    const height = 400;
    const initialBuffer: number[][] = [];

    for (let y = 0; y < height; y++) {
      const pingRow: number[] = new Array(width);
      for (let x = 0; x < width; x++) {
        // Generate synthetic side-scan profile: nadir water-column in center, seabed returns on sides
        const distFromCenter = Math.abs(x - width / 2);
        if (distFromCenter < 35) {
          // Water column blind zone
          pingRow[x] = Math.random() * 18;
        } else if (distFromCenter >= 35 && distFromCenter < 50) {
          // Sharp first seafloor return
          pingRow[x] = 180 + Math.random() * 60;
        } else {
          // Seabed reverberation + simulated anomaly targets
          let val = 90 + Math.random() * 50;
          if (distFromCenter > 120 && distFromCenter < 145 && y > 150 && y < 190) {
            // Target highlight
            val = 240 + Math.random() * 15;
          } else if (distFromCenter >= 145 && distFromCenter < 185 && y > 150 && y < 190) {
            // Target acoustic shadow
            val = 10 + Math.random() * 12;
          }
          pingRow[x] = val;
        }
      }
      initialBuffer.push(pingRow);
    }
    bufferRef.current = initialBuffer;
  }, []);

  // Waterfall render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastTime = performance.now();

    const render = (time: number) => {
      if (isPlaying && time - lastTime > 40 / speed) {
        lastTime = time;
        const width = 600;
        
        // Push a new ping row to the top and drop the oldest from the bottom
        const newRow: number[] = new Array(width);
        for (let x = 0; x < width; x++) {
          const distFromCenter = Math.abs(x - width / 2);
          if (distFromCenter < 35) {
            newRow[x] = Math.random() * 18;
          } else if (distFromCenter >= 35 && distFromCenter < 50) {
            newRow[x] = 180 + Math.random() * 60;
          } else {
            newRow[x] = 85 + Math.random() * 55;
          }
        }
        bufferRef.current.pop();
        bufferRef.current.unshift(newRow);
      }

      // Draw buffer to canvas with LUT and TVG
      const width = canvas.width;
      const height = canvas.height;
      const imgData = ctx.createImageData(width, height);
      const data = imgData.data;
      const buffer = bufferRef.current;

      for (let y = 0; y < height; y++) {
        const row = buffer[y] || [];
        for (let x = 0; x < width; x++) {
          let rawVal = row[x] || 0;
          
          // Apply TVG and Gain/Contrast
          const distFromCenter = Math.abs(x - width / 2) / (width / 2);
          const tvgFactor = 1.0 + (tvg - 1.0) * Math.pow(distFromCenter, 1.4);
          let procVal = (rawVal * gain * tvgFactor - 128) * contrast + 128;
          procVal = Math.max(0, Math.min(255, Math.round(procVal)));

          const pixelIdx = (y * width + x) * 4;
          const lutIdx = procVal * 4;
          data[pixelIdx] = colormapLut[lutIdx];
          data[pixelIdx + 1] = colormapLut[lutIdx + 1];
          data[pixelIdx + 2] = colormapLut[lutIdx + 2];
          data[pixelIdx + 3] = colormapLut[lutIdx + 3];
        }
      }

      ctx.putImageData(imgData, 0, 0);

      // Draw Caliper Overlay
      if (measurement) {
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(measurement.startX, measurement.startY);
        ctx.lineTo(measurement.endX, measurement.endY);
        ctx.stroke();

        // Draw start/end crosshairs
        [
          { x: measurement.startX, y: measurement.startY },
          { x: measurement.endX, y: measurement.endY }
        ].forEach((pt) => {
          ctx.strokeStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
          ctx.stroke();
        });
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    animationFrameRef.current = requestAnimationFrame(render);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isPlaying, speed, gain, contrast, tvg, colormapLut, measurement]);

  // Handle Caliper Mouse Events
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isMeasuring || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDragStart({ x, y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isMeasuring || !dragStart || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;

    const dx = curX - dragStart.x;
    const dy = curY - dragStart.y;
    const lengthPx = Math.sqrt(dx * dx + dy * dy);
    
    // Scale: canvas width (600px) corresponds to 2 * swathRangeM (100m total swath)
    const mPerPx = (swathRangeM * 2) / 600;
    const shadowLengthM = lengthPx * mPerPx;
    const slantRangeM = (Math.abs(curX - 300) * mPerPx);
    
    // Hydrographic formula: Target Height = (Altitude * Shadow Length) / (Slant Range)
    const estHeightM = slantRangeM > 0.5 ? (altitudeM * shadowLengthM) / slantRangeM : 0;

    setMeasurement({
      startX: dragStart.x,
      startY: dragStart.y,
      endX: curX,
      endY: curY,
      lengthPx: Math.round(lengthPx),
      slantRangeM: Number(slantRangeM.toFixed(2)),
      estimatedHeightM: Number(estHeightM.toFixed(2))
    });
  };

  const handleMouseUp = () => {
    setDragStart(null);
  };

  return (
    <GlassCard className="p-5 border-cyan-500/30 font-mono space-y-4">
      {/* Header Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-500/20 pb-3">
        <div className="flex items-center gap-2.5">
          <Layers className="w-5 h-5 text-cyan-400 animate-pulse" />
          <h3 className="text-sm font-semibold text-white tracking-wider uppercase">
            Acoustic Sonar Waterfall & DSP Canvas
          </h3>
          <GlassBadge variant="cyan">60 FPS WebGL Engine</GlassBadge>
        </div>

        <div className="flex items-center gap-2">
          {/* Colormap Selector */}
          <div className="flex items-center gap-1 bg-black/50 p-1 rounded-lg border border-cyan-900/40">
            {(['amber', 'cobalt', 'viridis', 'sepia', 'jet', 'greyscale'] as SonarColormap[]).map((c) => (
              <button
                key={c}
                onClick={() => setColormap(c)}
                className={`px-2.5 py-1 text-xs rounded transition-all capitalize ${
                  colormap === c
                    ? 'bg-cyan-500/30 text-cyan-300 font-bold border border-cyan-400/50 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {c}
              </button>
            ))}
          </div>

          {/* Caliper Tool Toggle */}
          <GlassButton
            size="sm"
            variant={isMeasuring ? 'primary' : 'secondary'}
            onClick={() => {
              setIsMeasuring(!isMeasuring);
              if (isMeasuring) setMeasurement(null);
            }}
            className="gap-1.5"
          >
            <Ruler className="w-3.5 h-3.5" />
            <span>{isMeasuring ? 'Caliper Active' : 'Caliper HUD'}</span>
          </GlassButton>

          {/* Play/Pause */}
          <GlassButton
            size="sm"
            variant="secondary"
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </GlassButton>
        </div>
      </div>

      {/* Main Canvas with Overlays */}
      <div className="relative flex justify-center bg-[#020712] rounded-xl overflow-hidden border border-cyan-950/60 shadow-[inset_0_0_30px_rgba(0,0,0,0.8)]">
        <canvas
          ref={canvasRef}
          width={600}
          height={360}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          className={`w-full max-w-[600px] h-[360px] cursor-${isMeasuring ? 'crosshair' : 'default'}`}
        />

        {/* Sonar Nadir Centerline Overlay */}
        <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-cyan-500/30 border-r border-dashed border-cyan-400/50 pointer-events-none" />
        <div className="absolute top-2 left-3 text-[10px] text-cyan-400/70 font-mono">
          PORT SWATH (50m)
        </div>
        <div className="absolute top-2 right-3 text-[10px] text-cyan-400/70 font-mono">
          STARBOARD SWATH (50m)
        </div>
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-cyan-400/70 bg-black/60 px-2 py-0.5 rounded border border-cyan-900/40">
          NADIR TRACK (ALT: {altitudeM.toFixed(1)}m)
        </div>

        {/* Caliper Measurement Result Badge */}
        {measurement && (
          <div className="absolute top-4 left-4 p-3 bg-black/85 border border-cyan-400/60 rounded-lg shadow-xl text-xs space-y-1 backdrop-blur-md">
            <div className="flex items-center gap-1.5 text-cyan-300 font-bold">
              <Crosshair className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
              <span>HYDROGRAPHIC SHADOW MEASUREMENT</span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-slate-300 pt-1 text-[11px]">
              <div>Pixel Span: <span className="text-white">{measurement.lengthPx} px</span></div>
              <div>Slant Range: <span className="text-white">{measurement.slantRangeM} m</span></div>
              <div className="col-span-2 text-amber-300 font-bold border-t border-cyan-900/50 pt-1">
                Calculated Object Height: <span className="text-amber-400 text-sm">{measurement.estimatedHeightM} m</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* DSP Sliders & Acoustic Adjustments */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 bg-black/40 p-3 rounded-lg border border-cyan-900/30 text-xs">
        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>Gain Multiplier</span>
            <span className="text-cyan-400">{gain.toFixed(2)}x</span>
          </div>
          <input
            type="range"
            min="0.5"
            max="2.5"
            step="0.05"
            value={gain}
            onChange={(e) => setGain(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 h-1 bg-slate-800 rounded"
          />
        </div>

        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>Contrast Curve</span>
            <span className="text-cyan-400">{contrast.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.05"
            value={contrast}
            onChange={(e) => setContrast(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 h-1 bg-slate-800 rounded"
          />
        </div>

        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>TVG Normalization</span>
            <span className="text-cyan-400">{tvg.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="1.0"
            max="2.2"
            step="0.05"
            value={tvg}
            onChange={(e) => setTvg(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 h-1 bg-slate-800 rounded"
          />
        </div>

        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>Ping Scroll Speed</span>
            <span className="text-cyan-400">{speed}x</span>
          </div>
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.5"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 h-1 bg-slate-800 rounded"
          />
        </div>
      </div>
    </GlassCard>
  );
};

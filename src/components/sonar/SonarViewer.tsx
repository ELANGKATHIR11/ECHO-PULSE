import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  ColorPalette,
  Detection,
  SonarFrame,
  SonarViewerSettings,
} from '../../types';
import {
  generateSyntheticSonarCanvas,
  applyViewerFilters,
  computeImageHistogram,
} from '../../utils/sonarProcessor';
import {
  ZoomIn,
  ZoomOut,
  Maximize,
  Sliders,
  Eye,
  Layers,
  RotateCcw,
  Download,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Split,
} from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

interface SonarViewerProps {
  frame?: SonarFrame;
  detections?: Detection[];
  selectedDetectionId?: string | null;
  onSelectDetection?: (d: Detection) => void;
  onHistogramUpdate?: (hist: number[]) => void;
  missionName?: string;
  pingIndex?: number;
  onPingChange?: (newPing: number) => void;
}

export const SonarViewer: React.FC<SonarViewerProps> = ({
  frame,
  detections = [],
  selectedDetectionId,
  onSelectDetection,
  onHistogramUpdate,
  missionName = 'Gulf of Mannar Survey',
  pingIndex = 3200,
  onPingChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Viewer state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isPlaying, setIsPlaying] = useState(false);

  // Settings
  const [settings, setSettings] = useState<SonarViewerSettings>({
    brightness: 0,
    contrast: 0,
    gamma: 1.0,
    thresholdPreview: false,
    thresholdLevel: 128,
    invert: false,
    palette: 'copper',
    splitComparison: false,
    splitPosition: 50,
    showLayers: {
      raw: true,
      processed: true,
      detections: true,
      shadows: true,
      anomalies: true,
      confidence: true,
      track: true,
      grid: true,
    },
  });

  const [showSettingsDrawer, setShowSettingsDrawer] = useState(false);

  // Render Sonar Canvas
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 1024;
    const height = 580;
    canvas.width = width;
    canvas.height = height;

    // Generate Raw & Processed offscreen canvases
    const rawCanvas = generateSyntheticSonarCanvas(width, height, detections, settings.palette, false);
    const procCanvas = generateSyntheticSonarCanvas(width, height, detections, settings.palette, true);

    if (settings.splitComparison) {
      // Split mode
      const splitX = Math.floor((settings.splitPosition / 100) * width);

      // Draw Raw on left
      ctx.drawImage(rawCanvas, 0, 0, splitX, height, 0, 0, splitX, height);
      // Draw Processed on right
      ctx.drawImage(procCanvas, splitX, 0, width - splitX, height, splitX, 0, width - splitX, height);

      // Draw Split Divider Bar
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(splitX, 0);
      ctx.lineTo(splitX, height);
      ctx.stroke();

      // Split Labels
      ctx.fillStyle = 'rgba(7, 14, 23, 0.8)';
      ctx.fillRect(8, 8, 80, 22);
      ctx.fillRect(width - 110, 8, 102, 22);
      ctx.font = '11px "JetBrains Mono"';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('RAW SONAR', 14, 23);
      ctx.fillStyle = '#00f0ff';
      ctx.fillText('AI PROCESSED', width - 104, 23);
    } else {
      // Single view (processed or raw)
      const activeCanvas = settings.showLayers.processed ? procCanvas : rawCanvas;
      ctx.drawImage(activeCanvas, 0, 0);
    }

    // Apply Client Filter Adjustments (Brightness, Contrast, Gamma, Invert, Threshold)
    applyViewerFilters(ctx, width, height, settings);

    // Render Range Grid Lines
    if (settings.showLayers.grid) {
      ctx.strokeStyle = 'rgba(14, 165, 233, 0.18)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 6]);

      // Slant range arcs / vertical markers
      for (let x = 100; x < width; x += 100) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      // Ping time lines
      for (let y = 80; y < height; y += 80) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // Render Acoustic Shadows Polygon Overlays
    if (settings.showLayers.shadows) {
      detections.forEach((det) => {
        if (det.acousticShadow) {
          const bx = det.bbox.x * width;
          const by = det.bbox.y * height;
          const bw = det.bbox.width * width;
          const bh = det.bbox.height * height;
          const isRight = bx > width / 2;
          const shadowW = det.acousticShadow.lengthMeters * 12;

          ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
          ctx.lineWidth = 1.5;

          const sx = isRight ? bx + bw : bx - shadowW;
          ctx.beginPath();
          ctx.rect(sx, by, shadowW, bh);
          ctx.fill();
          ctx.stroke();

          // Shadow ray line
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
          ctx.setLineDash([2, 3]);
          ctx.beginPath();
          ctx.moveTo(isRight ? bx + bw : bx, by + bh / 2);
          ctx.lineTo(isRight ? bx + bw + shadowW : bx - shadowW, by + bh / 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    }

    // Render Detections Bounding Boxes & Confidence Tags
    if (settings.showLayers.detections) {
      detections.forEach((det) => {
        const bx = det.bbox.x * width;
        const by = det.bbox.y * height;
        const bw = det.bbox.width * width;
        const bh = det.bbox.height * height;
        const isSelected = det.id === selectedDetectionId;

        let boxColor = '#00f0ff';
        if (det.class === 'ghost_gear') boxColor = '#f59e0b';
        if (det.class === 'shipwreck') boxColor = '#ec4899';
        if (det.class === 'unexploded_ordnance') boxColor = '#ef4444';
        if (det.class === 'pipeline_anomaly') boxColor = '#8b5cf6';

        // Bounding Box
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = isSelected ? 3 : 1.5;
        ctx.strokeRect(bx, by, bw, bh);

        // Corner ticks
        const tick = 6;
        ctx.fillStyle = boxColor;
        ctx.fillRect(bx - 2, by - 2, tick, 2);
        ctx.fillRect(bx - 2, by - 2, 2, tick);
        ctx.fillRect(bx + bw - tick + 2, by - 2, tick, 2);
        ctx.fillRect(bx + bw, by - 2, 2, tick);
        ctx.fillRect(bx - 2, by + bh, tick, 2);
        ctx.fillRect(bx - 2, by + bh - tick + 2, 2, tick);
        ctx.fillRect(bx + bw - tick + 2, by + bh, tick, 2);
        ctx.fillRect(bx + bw, by + bh - tick + 2, 2, tick);

        // Confidence & Tag Label
        if (settings.showLayers.confidence) {
          const label = `${det.classNameLabel.split(' ')[0]} ${(det.confidence * 100).toFixed(0)}%`;
          ctx.font = 'bold 11px "JetBrains Mono"';
          const textWidth = ctx.measureText(label).width;

          ctx.fillStyle = 'rgba(7, 14, 23, 0.9)';
          ctx.fillRect(bx, Math.max(0, by - 20), textWidth + 12, 18);

          ctx.fillStyle = boxColor;
          ctx.fillText(label, bx + 6, Math.max(13, by - 6));
        }
      });
    }

    // Update real-time histogram to parent if requested
    if (onHistogramUpdate) {
      const hist = computeImageHistogram(ctx, width, height);
      onHistogramUpdate(hist);
    }
  }, [detections, selectedDetectionId, settings, onHistogramUpdate]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Ping playback animation
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      if (onPingChange) {
        onPingChange((pingIndex + 20) % 20000);
      }
    }, 200);
    return () => clearInterval(interval);
  }, [isPlaying, pingIndex, onPingChange]);

  // Mouse pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  const handleMouseUp = () => setIsDragging(false);

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const exportSnapshot = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `echopulse_sonar_ping_${pingIndex}.png`;
    a.click();
  };

  return (
    <div
      ref={containerRef}
      className="relative flex flex-col h-full bg-[#02060E] rounded-2xl border border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 overflow-hidden select-none shadow-[0_16px_40px_-10px_rgba(0,0,0,0.75),inset_0_1px_1.5px_0_rgba(255,255,255,0.25)] backdrop-blur-2xl"
    >
      {/* Top Workstation Header Bar */}
      <div className="h-12 bg-[#050B14]/90 dark:bg-[#050B14]/90 light:bg-slate-100/90 border-b border-cyan-900/35 dark:border-cyan-900/35 light:border-slate-200 px-3 flex items-center justify-between text-xs font-mono shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-cyan-950/40 dark:bg-cyan-950/40 light:bg-sky-100 border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-300 px-2.5 py-0.5 rounded-sm text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold text-[10px] uppercase tracking-wider">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 dark:bg-cyan-400 light:bg-sky-600 animate-ping inline-block" />
            <span>WATERFALL SSS</span>
          </div>
          <span className="text-slate-600 dark:text-slate-600 light:text-slate-300">|</span>
          <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px]">
            Ping: <strong className="text-white dark:text-white light:text-slate-900">#{pingIndex}</strong>
          </span>
          <span className="text-slate-600 dark:text-slate-600 light:text-slate-300 hidden sm:inline">|</span>
          <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] hidden sm:inline">
            455 kHz
          </span>
          <span className="text-slate-600 dark:text-slate-600 light:text-slate-300 hidden md:inline">|</span>
          <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[11px] hidden md:inline">
            Range: <strong className="text-cyan-400 dark:text-cyan-400 light:text-sky-700">50m</strong>
          </span>
        </div>

        {/* Viewport Action Controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setZoom((z) => Math.min(4, z + 0.25))}
            className="p-1 rounded bg-transparent hover:bg-cyan-950/40 dark:hover:bg-cyan-950/40 light:hover:bg-slate-200 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
            className="p-1 rounded bg-transparent hover:bg-cyan-950/40 dark:hover:bg-cyan-950/40 light:hover:bg-slate-200 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={resetView}
            className="p-1 rounded bg-transparent hover:bg-cyan-950/40 dark:hover:bg-cyan-950/40 light:hover:bg-slate-200 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <div className="w-[1px] h-4 bg-cyan-900/40 dark:bg-cyan-900/40 light:bg-slate-300 mx-1" />

          {/* Raw vs Processed Split Toggle */}
          <button
            onClick={() => setSettings({ ...settings, splitComparison: !settings.splitComparison })}
            className={`px-2.5 py-1 rounded-sm text-[10px] uppercase font-mono tracking-wider flex items-center gap-1.5 transition-all font-bold ${
              settings.splitComparison
                ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-400 dark:border-cyan-400 light:border-sky-300 shadow-[0_0_8px_rgba(34,211,238,0.2)]'
                : 'bg-cyan-950/20 dark:bg-cyan-950/20 light:bg-white text-slate-400 dark:text-slate-400 light:text-slate-600 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 hover:text-slate-200 dark:hover:text-slate-200 light:hover:text-slate-900'
            }`}
            title="Toggle Raw vs Processed Split Slider"
          >
            <Split className="w-3 h-3" />
            <span>Split View</span>
          </button>

          {/* Filter & Tuning Drawer Button */}
          <button
            onClick={() => setShowSettingsDrawer(!showSettingsDrawer)}
            className={`p-1.5 rounded-sm transition-all text-[10px] uppercase font-mono font-bold ${
              showSettingsDrawer
                ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-400 dark:border-cyan-400 light:border-sky-300'
                : 'bg-cyan-950/20 dark:bg-cyan-950/20 light:bg-white text-slate-400 dark:text-slate-400 light:text-slate-600 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800'
            }`}
            title="Sonar DSP & Image Enhancement"
          >
            <Sliders className="w-3.5 h-3.5" />
          </button>

          {/* Export Frame */}
          <button
            onClick={exportSnapshot}
            className="p-1.5 rounded-sm bg-cyan-950/20 dark:bg-cyan-950/20 light:bg-white text-slate-400 dark:text-slate-400 light:text-slate-600 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 transition-colors"
            title="Export Frame PNG"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Viewport Container */}
      <div
        className="relative flex-1 bg-black overflow-hidden flex items-center justify-center cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        {/* Sleek radial dot pattern background */}
        <div className="absolute inset-0 opacity-20 pointer-events-none bg-radial-dots" />

        {/* Live Sonar Feed Pill Overlay */}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-black/70 backdrop-blur px-3 py-1 rounded-full border border-cyan-500/30 pointer-events-none">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse shadow-[0_0_8px_#EF4444]" />
          <span className="text-[9px] font-mono tracking-widest uppercase text-cyan-300 font-bold">
            Live Sonar Feed
          </span>
        </div>

        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
          }}
          className="relative"
        >
          <canvas ref={canvasRef} className="rounded shadow-2xl max-w-none" />
        </div>

        {/* Split Position Slider Overlay */}
        {settings.splitComparison && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 bg-[#050B14]/90 dark:bg-[#050B14]/90 light:bg-white/95 backdrop-blur border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-300 px-4 py-2 rounded-full flex items-center gap-3 font-mono text-xs shadow-xl">
            <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase tracking-wider font-bold">
              Raw
            </span>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.splitPosition}
              onChange={(e) => setSettings({ ...settings, splitPosition: Number(e.target.value) })}
              className="w-44 accent-cyan-400 cursor-pointer"
            />
            <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 text-[10px] font-bold uppercase tracking-wider">
              Processed ({settings.splitPosition}%)
            </span>
          </div>
        )}

        {/* Floating Enhancement Drawer */}
        {showSettingsDrawer && (
          <div className="absolute top-3 right-3 z-30 w-72 bg-[#0A121E]/95 dark:bg-[#0A121E]/95 light:bg-white/95 backdrop-blur-md border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-200 rounded-lg p-3.5 font-mono text-xs space-y-3 shadow-2xl transition-colors">
            <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2">
              <span className="font-bold text-white dark:text-white light:text-slate-900 text-[10px] tracking-widest uppercase flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                DSP & FILTERS
              </span>
              <button
                onClick={() =>
                  setSettings({
                    ...settings,
                    brightness: 0,
                    contrast: 0,
                    gamma: 1.0,
                    thresholdPreview: false,
                    invert: false,
                  })
                }
                className="text-[9px] uppercase tracking-wider text-slate-400 dark:text-slate-400 light:text-slate-500 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-700 font-bold"
              >
                Reset
              </button>
            </div>

            {/* False Color Palettes */}
            <div>
              <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[10px] uppercase tracking-wider mb-1.5 font-semibold">
                Color Palette:
              </div>
              <div className="grid grid-cols-3 gap-1">
                {(
                  [
                    'copper',
                    'amber_sonar',
                    'oceanic_blue',
                    'thermal',
                    'emerald',
                    'grayscale',
                  ] as ColorPalette[]
                ).map((pal) => (
                  <button
                    key={pal}
                    onClick={() => setSettings({ ...settings, palette: pal })}
                    className={`px-2 py-1 rounded text-[9px] uppercase font-mono truncate transition-colors ${
                      settings.palette === pal
                        ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-400 dark:border-cyan-400 light:border-sky-300 font-bold shadow-[0_0_6px_rgba(34,211,238,0.2)]'
                        : 'bg-[#02060C] dark:bg-[#02060C] light:bg-slate-100 text-slate-400 dark:text-slate-400 light:text-slate-700 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 hover:text-slate-200 dark:hover:text-slate-200 light:hover:text-slate-900'
                    }`}
                  >
                    {pal.replace('_', ' ')}
                  </button>
                ))}
              </div>
            </div>

            {/* Brightness */}
            <div>
              <div className="flex justify-between text-[10px] uppercase text-slate-400 dark:text-slate-400 light:text-slate-600 mb-1 font-semibold">
                <span>Gain / Brightness</span>
                <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-700 font-bold">
                  {settings.brightness}
                </span>
              </div>
              <input
                type="range"
                min="-100"
                max="100"
                value={settings.brightness}
                onChange={(e) => setSettings({ ...settings, brightness: Number(e.target.value) })}
                className="w-full accent-cyan-400 h-1 bg-slate-800 dark:bg-slate-800 light:bg-slate-200 rounded appearance-none cursor-pointer"
              />
            </div>

            {/* Contrast */}
            <div>
              <div className="flex justify-between text-[10px] uppercase text-slate-400 dark:text-slate-400 light:text-slate-600 mb-1 font-semibold">
                <span>Contrast</span>
                <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-700 font-bold">
                  {settings.contrast}
                </span>
              </div>
              <input
                type="range"
                min="-100"
                max="100"
                value={settings.contrast}
                onChange={(e) => setSettings({ ...settings, contrast: Number(e.target.value) })}
                className="w-full accent-cyan-400 h-1 bg-slate-800 dark:bg-slate-800 light:bg-slate-200 rounded appearance-none cursor-pointer"
              />
            </div>

            {/* Gamma */}
            <div>
              <div className="flex justify-between text-[10px] uppercase text-slate-400 dark:text-slate-400 light:text-slate-600 mb-1 font-semibold">
                <span>Gamma</span>
                <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-700 font-bold">
                  {settings.gamma.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.2"
                max="3.0"
                step="0.05"
                value={settings.gamma}
                onChange={(e) => setSettings({ ...settings, gamma: Number(e.target.value) })}
                className="w-full accent-cyan-400 h-1 bg-slate-800 dark:bg-slate-800 light:bg-slate-200 rounded appearance-none cursor-pointer"
              />
            </div>

            {/* Otsu / Binary Threshold */}
            <div className="pt-1.5 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
              <label className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-300 dark:text-slate-300 light:text-slate-700 cursor-pointer font-semibold">
                <span>Otsu Threshold Mask</span>
                <input
                  type="checkbox"
                  checked={settings.thresholdPreview}
                  onChange={(e) => setSettings({ ...settings, thresholdPreview: e.target.checked })}
                  className="accent-cyan-400"
                />
              </label>
              {settings.thresholdPreview && (
                <div className="mt-1.5">
                  <input
                    type="range"
                    min="0"
                    max="255"
                    value={settings.thresholdLevel}
                    onChange={(e) => setSettings({ ...settings, thresholdLevel: Number(e.target.value) })}
                    className="w-full accent-cyan-400 h-1 bg-slate-800 dark:bg-slate-800 light:bg-slate-200 rounded appearance-none cursor-pointer"
                  />
                  <div className="text-right text-[9px] text-slate-400 dark:text-slate-400 light:text-slate-500 mt-1">
                    {settings.thresholdLevel} / 255
                  </div>
                </div>
              )}
            </div>

            {/* Invert */}
            <div>
              <label className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-300 dark:text-slate-300 light:text-slate-700 cursor-pointer font-semibold">
                <span>Invert Acoustic Values</span>
                <input
                  type="checkbox"
                  checked={settings.invert}
                  onChange={(e) => setSettings({ ...settings, invert: e.target.checked })}
                  className="accent-cyan-400"
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Ping Playback & Scrub Bar */}
      <div className="h-12 bg-slate-900/50 dark:bg-slate-900/50 light:bg-slate-100 border-t border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200 px-4 flex items-center justify-between gap-4 font-mono text-xs shrink-0 transition-colors">
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPingChange && onPingChange(Math.max(0, pingIndex - 500))}
            className="p-1.5 rounded hover:bg-cyan-950/40 dark:hover:bg-cyan-950/40 light:hover:bg-slate-200 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 transition-colors"
            title="Step Back 500 Pings"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-1.5 rounded-sm transition-all ${
              isPlaying
                ? 'bg-cyan-500 dark:bg-cyan-500 light:bg-sky-600 text-slate-950 light:text-white font-bold'
                : 'bg-cyan-500/10 dark:bg-cyan-500/10 light:bg-sky-100 border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-300 text-cyan-400 dark:text-cyan-400 light:text-sky-800 hover:bg-cyan-500 hover:text-black dark:hover:text-black light:hover:text-white'
            }`}
            title={isPlaying ? 'Pause Sonar Stream' : 'Play Live Sonar Waterfall'}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
          </button>
          <button
            onClick={() => onPingChange && onPingChange(pingIndex + 500)}
            className="p-1.5 rounded hover:bg-cyan-950/40 dark:hover:bg-cyan-950/40 light:hover:bg-slate-200 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 transition-colors"
            title="Step Forward 500 Pings"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Timeline Slider */}
        <div className="flex-1 flex items-center gap-3">
          <span className="text-[10px] text-slate-500 dark:text-slate-500 light:text-slate-500 uppercase tracking-wider shrink-0 font-bold">
            PING #{pingIndex}
          </span>
          <input
            type="range"
            min="0"
            max="18420"
            value={pingIndex}
            onChange={(e) => onPingChange && onPingChange(Number(e.target.value))}
            className="w-full accent-cyan-400 h-1 bg-slate-800 dark:bg-slate-800 light:bg-slate-200 rounded appearance-none cursor-pointer"
          />
          <span className="text-[10px] text-slate-500 dark:text-slate-500 light:text-slate-500 uppercase tracking-wider shrink-0 font-bold">
            #18420
          </span>
        </div>

        {/* Quick Layer Visibility Pills */}
        <div className="hidden lg:flex items-center gap-1.5">
          <button
            onClick={() =>
              setSettings({
                ...settings,
                showLayers: { ...settings.showLayers, detections: !settings.showLayers.detections },
              })
            }
            className={`px-2 py-0.5 rounded-sm text-[9px] uppercase tracking-wider font-mono transition-colors font-bold ${
              settings.showLayers.detections
                ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-500/40 dark:border-cyan-500/40 light:border-sky-300'
                : 'text-slate-500 hover:text-slate-300 dark:hover:text-slate-300 light:hover:text-slate-700'
            }`}
          >
            Boxes
          </button>
          <button
            onClick={() =>
              setSettings({
                ...settings,
                showLayers: { ...settings.showLayers, shadows: !settings.showLayers.shadows },
              })
            }
            className={`px-2 py-0.5 rounded-sm text-[9px] uppercase tracking-wider font-mono transition-colors font-bold ${
              settings.showLayers.shadows
                ? 'bg-red-950/60 dark:bg-red-950/60 light:bg-red-100 text-red-300 dark:text-red-300 light:text-red-800 border border-red-500/50 dark:border-red-500/50 light:border-red-300'
                : 'text-slate-500 hover:text-slate-300 dark:hover:text-slate-300 light:hover:text-slate-700'
            }`}
          >
            Shadows
          </button>
          <button
            onClick={() =>
              setSettings({
                ...settings,
                showLayers: { ...settings.showLayers, grid: !settings.showLayers.grid },
              })
            }
            className={`px-2 py-0.5 rounded-sm text-[9px] uppercase tracking-wider font-mono transition-colors font-bold ${
              settings.showLayers.grid
                ? 'bg-cyan-950 dark:bg-cyan-950 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-300'
                : 'text-slate-500 hover:text-slate-300 dark:hover:text-slate-300 light:hover:text-slate-700'
            }`}
          >
            Grid
          </button>
        </div>
      </div>
    </div>
  );
};

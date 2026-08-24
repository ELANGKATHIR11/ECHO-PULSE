import React from 'react';
import { Detection } from '../../types';
import { Activity, Shield } from 'lucide-react';

interface OpenCvAnalysisPanelProps {
  histogram?: number[];
  detection?: Detection | null;
  altitudeMeters?: number;
}

export const OpenCvAnalysisPanel: React.FC<OpenCvAnalysisPanelProps> = ({
  histogram = [],
  detection,
  altitudeMeters = 8.5,
}) => {
  // Compute acoustic target height formula: H = (L * Altitude) / Range
  const shadowLength = detection?.acousticShadow?.lengthMeters ?? 4.8;
  const slantRange = detection?.slantRangeMeters ?? 14.8;
  const calculatedTargetHeight = ((shadowLength * altitudeMeters) / Math.max(1, slantRange)).toFixed(2);

  return (
    <div className="bg-[#0A121E] dark:bg-[#0A121E] light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 rounded-lg p-3.5 font-mono text-xs flex flex-col gap-3 shadow-md transition-colors">
      <div className="flex items-center justify-between border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 pb-2">
        <div className="flex items-center gap-1.5 text-white dark:text-white light:text-slate-900 font-bold text-[10px] tracking-widest uppercase">
          <Activity className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
          <span>OPENCV ACOUSTIC TELEMETRY</span>
        </div>
        <span className="text-[9px] px-1.5 py-0.5 rounded font-mono uppercase bg-cyan-950/40 dark:bg-cyan-950/40 light:bg-sky-100 text-cyan-400 dark:text-cyan-400 light:text-sky-800 border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-300 font-bold">
          PYTORCH / C++
        </span>
      </div>

      {/* 256-Bin Intensity Histogram */}
      <div>
        <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-400 light:text-slate-600 mb-1.5 font-bold">
          <span>Backscatter Intensity Histogram</span>
          <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-bold">256 Bins</span>
        </div>
        <div className="h-16 bg-[#02060C] dark:bg-[#02060C] light:bg-slate-100 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded p-1 flex items-end gap-[1px] overflow-hidden">
          {histogram.length > 0
            ? histogram
                .filter((_, idx) => idx % 4 === 0)
                .map((val, idx) => (
                  <div
                    key={idx}
                    style={{ height: `${Math.max(4, Math.min(100, val))}%` }}
                    className={`flex-1 rounded-t-[1px] transition-all ${
                      idx > 48
                        ? 'bg-amber-400 dark:bg-amber-400 light:bg-amber-500' // Highlight peaks
                        : idx < 12
                        ? 'bg-slate-700 dark:bg-slate-700 light:bg-slate-300' // Shadow/water column
                        : 'bg-cyan-400 dark:bg-cyan-400 light:bg-sky-600' // Seabed backscatter
                    }`}
                  />
                ))
            : Array.from({ length: 64 }).map((_, idx) => (
                <div
                  key={idx}
                  style={{ height: `${Math.sin(idx * 0.1) * 40 + 20}%` }}
                  className="flex-1 bg-cyan-500/40 dark:bg-cyan-500/40 light:bg-sky-500/60 rounded-t-[1px]"
                />
              ))}
        </div>
        <div className="flex justify-between text-[8px] uppercase font-mono text-slate-500 dark:text-slate-500 light:text-slate-400 mt-1">
          <span>0 (Shadow)</span>
          <span>128 (Seabed Baseline)</span>
          <span>255 (Specular Peak)</span>
        </div>
      </div>

      {/* Acoustic Shadow Calculation Card */}
      <div className="bg-[#050B14] dark:bg-[#050B14] light:bg-slate-50 border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 rounded p-2.5 space-y-2">
        <div className="flex items-center justify-between text-[10px] uppercase tracking-wider">
          <span className="text-amber-400 dark:text-amber-400 light:text-amber-700 font-bold flex items-center gap-1">
            <Shield className="w-3.5 h-3.5" />
            Acoustic Shadow Geometry
          </span>
          <span className="text-[9px] text-slate-500 dark:text-slate-500 light:text-slate-400 font-semibold">Grazing Ray Analysis</span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="bg-[#02060C] dark:bg-[#02060C] light:bg-white p-1.5 rounded border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
            <div className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
              Shadow Length (L)
            </div>
            <div className="text-white dark:text-white light:text-slate-900 font-bold">{shadowLength}m</div>
          </div>
          <div className="bg-[#02060C] dark:bg-[#02060C] light:bg-white p-1.5 rounded border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
            <div className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
              Slant Range (R)
            </div>
            <div className="text-white dark:text-white light:text-slate-900 font-bold">{slantRange}m</div>
          </div>
          <div className="bg-[#02060C] dark:bg-[#02060C] light:bg-white p-1.5 rounded border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
            <div className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
              Tow Altitude (A)
            </div>
            <div className="text-white dark:text-white light:text-slate-900 font-bold">{altitudeMeters}m</div>
          </div>
          <div className="bg-[#02060C] dark:bg-[#02060C] light:bg-white p-1.5 rounded border border-cyan-500/40 dark:border-cyan-500/40 light:border-sky-300">
            <div className="text-cyan-400 dark:text-cyan-400 light:text-sky-700 text-[9px] uppercase tracking-wider font-bold">
              Target Height (H)
            </div>
            <div className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold">
              ~{calculatedTargetHeight}m above bed
            </div>
          </div>
        </div>
      </div>

      {/* Signal Quality & Morphological Metrics */}
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="bg-[#050B14] dark:bg-[#050B14] light:bg-slate-50 p-2 rounded border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
          <span className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
            SNR Ratio
          </span>
          <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold text-sm">24.2 dB</div>
        </div>
        <div className="bg-[#050B14] dark:bg-[#050B14] light:bg-slate-50 p-2 rounded border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
          <span className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
            Dynamic Range
          </span>
          <div className="text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-bold text-sm">48.5 dB</div>
        </div>
        <div className="bg-[#050B14] dark:bg-[#050B14] light:bg-slate-50 p-2 rounded border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
          <span className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
            Mean Backscatter
          </span>
          <div className="text-white dark:text-white light:text-slate-900 font-bold text-sm">124.6 / 255</div>
        </div>
        <div className="bg-[#050B14] dark:bg-[#050B14] light:bg-slate-50 p-2 rounded border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
          <span className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
            Sobel Gradient
          </span>
          <div className="text-purple-400 dark:text-purple-400 light:text-purple-700 font-bold text-sm">86.4 GxGy</div>
        </div>
      </div>

      {/* Target Selected Details */}
      {detection && (
        <div className="border-t border-cyan-900/20 dark:border-cyan-900/20 light:border-slate-200 pt-2 space-y-1 text-[11px]">
          <div className="text-slate-500 dark:text-slate-500 light:text-slate-500 text-[9px] uppercase tracking-wider">
            Contour Morphology:
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Area / Perimeter:</span>
            <span className="text-white dark:text-white light:text-slate-900 font-bold">
              {detection.geometry.areaPixels}px² / {detection.geometry.perimeterPixels}px
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Solidity / Extent:</span>
            <span className="text-white dark:text-white light:text-slate-900 font-bold">
              {detection.geometry.solidity.toFixed(2)} / {detection.geometry.extent.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">Aspect Ratio:</span>
            <span className="text-white dark:text-white light:text-slate-900 font-bold">
              {detection.geometry.aspectRatio.toFixed(2)} : 1
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Detection, DetectionClass } from '../../types';
import { formatDMS } from '../../utils/sonarProcessor';
import {
  Search,
  ArrowUpDown,
  ExternalLink,
  MapPin,
  FileSpreadsheet,
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton } from '../glass/GlassCard';

interface DetectionTableProps {
  detections: Detection[];
  onSelectDetection?: (d: Detection) => void;
  selectedId?: string | null;
  onExportCsv?: () => void;
}

export const DetectionTable: React.FC<DetectionTableProps> = ({
  detections,
  onSelectDetection,
  selectedId,
  onExportCsv,
}) => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedClass, setSelectedClass] = useState<string>('ALL');
  const [minConfidence, setMinConfidence] = useState<number>(0);
  const [geotagOnly, setGeotagOnly] = useState<boolean>(false);
  const [sortField, setSortField] = useState<'confidence' | 'timestamp' | 'anomalyScore' | 'class'>('confidence');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Filter & sort detections
  const filteredDetections = useMemo(() => {
    return detections
      .filter((d) => {
        if (searchTerm) {
          const matchSearch =
            d.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
            d.classNameLabel.toLowerCase().includes(searchTerm.toLowerCase()) ||
            d.missionName.toLowerCase().includes(searchTerm.toLowerCase()) ||
            (d.notes && d.notes.toLowerCase().includes(searchTerm.toLowerCase()));
          if (!matchSearch) return false;
        }

        if (selectedClass !== 'ALL' && d.class !== selectedClass) return false;
        if (d.confidence < minConfidence) return false;
        if (geotagOnly && (d.latitude === null || d.longitude === null)) return false;

        return true;
      })
      .sort((a, b) => {
        let valA: any = a[sortField];
        let valB: any = b[sortField];

        if (sortField === 'timestamp') {
          valA = new Date(a.timestamp).getTime();
          valB = new Date(b.timestamp).getTime();
        }

        if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
        if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
  }, [detections, searchTerm, selectedClass, minConfidence, geotagOnly, sortField, sortDirection]);

  const handleSort = (field: 'confidence' | 'timestamp' | 'anomalyScore' | 'class') => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const getClassBadge = (detClass: DetectionClass, label: string) => {
    let variant: 'cyan' | 'amber' | 'purple' | 'emerald' | 'slate' = 'cyan';
    if (detClass === 'ghost_gear') variant = 'amber';
    if (detClass === 'shipwreck') variant = 'purple';
    if (detClass === 'unexploded_ordnance') variant = 'amber';
    if (detClass === 'pipeline_anomaly') variant = 'purple';
    if (detClass === 'biological_cluster') variant = 'emerald';

    return (
      <GlassBadge variant={variant} size="sm" className="whitespace-nowrap truncate max-w-[170px]">
        {label}
      </GlassBadge>
    );
  };

  return (
    <GlassCard variant="default" className="flex flex-col font-sans text-xs overflow-hidden">
      {/* Search & Filter Bar */}
      <div className="p-3.5 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 flex flex-wrap items-center justify-between gap-3 bg-[#08121e]/40 dark:bg-[#08121e]/40 light:bg-slate-50/60">
        <div className="flex items-center gap-2 flex-1 min-w-[260px]">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search detection ID, class, mission, coordinates..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#050b14]/80 dark:bg-[#050b14]/80 light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-100 dark:text-slate-100 light:text-slate-900 placeholder-slate-500 focus:outline-none focus:border-cyan-400"
            />
          </div>

          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="bg-[#050b14]/80 dark:bg-[#050b14]/80 light:bg-white border border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-300 rounded-xl px-3 py-1.5 text-xs text-slate-200 dark:text-slate-200 light:text-slate-800 focus:outline-none focus:border-cyan-400"
          >
            <option value="ALL">All Classes</option>
            <option value="ghost_gear">Ghost Gear / Nets</option>
            <option value="shipwreck">Shipwrecks</option>
            <option value="unexploded_ordnance">UXO / Naval Mines</option>
            <option value="pipeline_anomaly">Pipeline Anomalies</option>
            <option value="subsea_cable">Subsea Cables</option>
            <option value="biological_cluster">Biological Reefs</option>
            <option value="marine_debris">Marine Debris</option>
          </select>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-slate-400 dark:text-slate-400 light:text-slate-600">
            <span className="font-semibold uppercase text-[10px]">Min Conf:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              className="w-20 accent-cyan-400 cursor-pointer h-1.5 bg-[#020712] dark:bg-[#020712] light:bg-slate-200 rounded-lg"
            />
            <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono font-bold">{(minConfidence * 100).toFixed(0)}%</span>
          </div>

          <label className="flex items-center gap-1.5 text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={geotagOnly}
              onChange={(e) => setGeotagOnly(e.target.checked)}
              className="accent-cyan-400 rounded"
            />
            <span>Geotagged Only</span>
          </label>

          {onExportCsv && (
            <GlassButton
              variant="secondary"
              size="sm"
              icon={<FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />}
              onClick={onExportCsv}
              title="Export filtered records to CSV"
            >
              Export CSV
            </GlassButton>
          )}
        </div>
      </div>

      {/* Virtualized Table Container */}
      <div className="overflow-x-auto max-h-[560px]">
        <table className="w-full text-left border-collapse">
          <thead className="bg-[#091522]/80 dark:bg-[#091522]/80 light:bg-slate-100/90 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 sticky top-0 z-10 select-none">
            <tr>
              <th className="py-3 px-3.5 font-bold uppercase text-[10px]">ID</th>
              <th
                onClick={() => handleSort('class')}
                className="py-3 px-3.5 cursor-pointer hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 font-bold uppercase text-[10px]"
              >
                <div className="flex items-center gap-1">
                  <span>Class</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th
                onClick={() => handleSort('confidence')}
                className="py-3 px-3.5 cursor-pointer hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 font-bold uppercase text-[10px]"
              >
                <div className="flex items-center gap-1">
                  <span>Confidence</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3 px-3.5 font-bold uppercase text-[10px]">Shadow (m)</th>
              <th
                onClick={() => handleSort('anomalyScore')}
                className="py-3 px-3.5 cursor-pointer hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 font-bold uppercase text-[10px]"
              >
                <div className="flex items-center gap-1">
                  <span>Anomaly</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3 px-3.5 font-bold uppercase text-[10px]">Depth / Range</th>
              <th className="py-3 px-3.5 font-bold uppercase text-[10px]">Coordinates (DMS)</th>
              <th
                onClick={() => handleSort('timestamp')}
                className="py-3 px-3.5 cursor-pointer hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 font-bold uppercase text-[10px]"
              >
                <div className="flex items-center gap-1">
                  <span>Timestamp</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3 px-3.5 text-right font-bold uppercase text-[10px]">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-cyan-900/20 dark:divide-cyan-900/20 light:divide-slate-200">
            {filteredDetections.length > 0 ? (
              filteredDetections.map((det, idx) => {
                const isSelected = det.id === selectedId;

                return (
                  <tr
                    key={`${det.id}-${idx}`}
                    onClick={() => onSelectDetection && onSelectDetection(det)}
                    className={`transition-colors cursor-pointer hover:bg-cyan-950/20 dark:hover:bg-cyan-950/20 light:hover:bg-sky-50 ${
                      isSelected ? 'bg-cyan-950/40 dark:bg-cyan-950/40 light:bg-sky-100/60 border-l-2 border-cyan-400' : ''
                    }`}
                  >
                    <td className="py-3 px-3.5 font-bold text-cyan-300 dark:text-cyan-300 light:text-sky-800 whitespace-nowrap font-mono">
                      {det.id}
                    </td>
                    <td className="py-3 px-3.5">
                      {getClassBadge(det.class, det.classNameLabel)}
                    </td>
                    <td className="py-3 px-3.5 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <div className="w-14 h-2 bg-[#142336] dark:bg-[#142336] light:bg-slate-200 rounded-full overflow-hidden">
                          <div
                            style={{ width: `${det.confidence * 100}%` }}
                            className={`h-full rounded-full ${
                              det.confidence > 0.9
                                ? 'bg-emerald-400'
                                : det.confidence > 0.75
                                ? 'bg-cyan-400'
                                : 'bg-amber-400'
                            }`}
                          />
                        </div>
                        <span className="font-bold text-white dark:text-white light:text-slate-900 font-mono">
                          {(det.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-3.5 whitespace-nowrap text-slate-300 dark:text-slate-300 light:text-slate-700 font-mono">
                      {det.acousticShadow ? (
                        <span className="text-amber-300 dark:text-amber-300 light:text-amber-700 font-bold">{det.acousticShadow.lengthMeters}m</span>
                      ) : (
                        <span className="text-slate-500">None</span>
                      )}
                    </td>
                    <td className="py-3 px-3.5 whitespace-nowrap">
                      <span className="text-purple-300 dark:text-purple-300 light:text-purple-700 font-bold font-mono">
                        {(det.anomalyScore * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="py-3 px-3.5 whitespace-nowrap text-slate-300 dark:text-slate-300 light:text-slate-700 font-mono">
                      <span>{det.depthMeters}m</span>
                      <span className="text-slate-500 mx-1">/</span>
                      <span className="text-slate-400 dark:text-slate-400 light:text-slate-600">{det.slantRangeMeters}m</span>
                    </td>
                    <td className="py-3 px-3.5 whitespace-nowrap text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 font-mono">
                      {det.latitude !== null && det.longitude !== null ? (
                        <div className="flex items-center gap-1 text-slate-300 dark:text-slate-300 light:text-slate-700">
                          <MapPin className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600 shrink-0" />
                          <span>
                            {formatDMS(det.latitude, true)}, {formatDMS(det.longitude, false)}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-500">Ungeotagged</span>
                      )}
                    </td>
                    <td className="py-3 px-3.5 whitespace-nowrap text-slate-400 dark:text-slate-400 light:text-slate-600 text-xs font-mono">
                      {det.timestamp.replace('T', ' ').substring(0, 19)}
                    </td>
                    <td className="py-3 px-3.5 text-right whitespace-nowrap">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/detections/${det.id}`);
                        }}
                        className="p-1.5 rounded-lg hover:bg-cyan-950 text-cyan-400 dark:text-cyan-400 light:text-sky-700 hover:text-cyan-300 border border-transparent hover:border-cyan-500/40 transition-colors"
                        title="Open Deep Investigation"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={9} className="py-8 text-center text-slate-500">
                  No sonar detections match the specified filter criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Table Footer Summary */}
      <div className="p-3 bg-[#08121e]/60 dark:bg-[#08121e]/60 light:bg-slate-50/80 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200 flex items-center justify-between text-xs text-slate-400 dark:text-slate-400 light:text-slate-600">
        <div>
          Showing <strong className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-mono">{filteredDetections.length}</strong> of{' '}
          <strong className="text-slate-200 dark:text-slate-200 light:text-slate-800 font-mono">{detections.length}</strong> acoustic targets
        </div>
        <div className="flex items-center gap-3">
          <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-semibold">
            {filteredDetections.filter((d) => d.confidence > 0.9).length} High-Confidence (&gt;90%)
          </span>
          <span className="text-slate-500">|</span>
          <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-semibold">
            {filteredDetections.filter((d) => d.latitude !== null).length} Geotagged
          </span>
        </div>
      </div>
    </GlassCard>
  );
};

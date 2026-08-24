import React, { useState, useEffect } from 'react';
import { Detection } from '../types';
import { detectionApi } from '../services/detectionApi';
import { DetectionTable } from '../components/detections/DetectionTable';
import { exportDetectionsToCSV, downloadBlobFile, exportDetectionsToGeoJSON } from '../utils/geoUtils';
import { Crosshair, Download, FileSpreadsheet } from 'lucide-react';
import { GlassButton } from '../components/glass/GlassCard';

export const DetectionsPage: React.FC = () => {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);

  useEffect(() => {
    detectionApi.getDetections().then((dets) => {
      setDetections(dets);
      if (dets.length > 0) setSelectedDetection(dets[0]);
    });
  }, []);

  const handleExportCSV = () => {
    const csv = exportDetectionsToCSV(detections);
    downloadBlobFile(csv, 'echopulsenet_acoustic_detections.csv', 'text/csv');
  };

  const handleExportGeoJSON = () => {
    const geojson = exportDetectionsToGeoJSON(detections);
    downloadBlobFile(geojson, 'echopulsenet_acoustic_detections.geojson', 'application/geo+json');
  };

  return (
    <div className="p-4 md:p-6 max-w-[1700px] mx-auto w-full font-sans space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200">
        <div>
          <h1 className="text-xl font-bold text-white dark:text-white light:text-slate-900 flex items-center gap-2.5">
            <Crosshair className="w-6 h-6 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
            ACOUSTIC TARGET DETECTIONS EXPLORER
          </h1>
          <p className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-600 mt-1">
            Real-time neural detections, acoustic shadow fusion geometry, and verified subsea anomalies
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <GlassButton
            variant="secondary"
            size="md"
            icon={<FileSpreadsheet className="w-4 h-4 text-emerald-400" />}
            onClick={handleExportCSV}
          >
            Export CSV
          </GlassButton>
          <GlassButton
            variant="primary"
            size="md"
            icon={<Download className="w-4 h-4" />}
            onClick={handleExportGeoJSON}
          >
            Export GeoJSON
          </GlassButton>
        </div>
      </div>

      {/* Main Virtualized Table */}
      <DetectionTable
        detections={detections}
        selectedId={selectedDetection?.id}
        onSelectDetection={(d) => setSelectedDetection(d)}
        onExportCsv={handleExportCSV}
      />
    </div>
  );
};

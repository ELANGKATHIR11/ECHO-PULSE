import { Detection, Mission } from '../types';

/**
 * Exports detections as standard GeoJSON FeatureCollection
 */
export function exportDetectionsToGeoJSON(detections: Detection[], mission?: Mission): string {
  const features = detections
    .filter((d) => d.latitude !== null && d.longitude !== null)
    .map((d) => ({
      type: 'Feature' as const,
      geometry: {
        type: 'Point' as const,
        coordinates: [d.longitude, d.latitude, -(d.depthMeters || 0)],
      },
      properties: {
        id: d.id,
        missionId: d.missionId,
        missionName: d.missionName,
        class: d.class,
        className: d.classNameLabel,
        confidence: d.confidence,
        detectorScore: d.detectorScore,
        shadowScore: d.shadowScore,
        anomalyScore: d.anomalyScore,
        qualityScore: d.qualityScore,
        depthMeters: d.depthMeters,
        slantRangeMeters: d.slantRangeMeters,
        shadowLengthMeters: d.acousticShadow?.lengthMeters ?? null,
        timestamp: d.timestamp,
        pingIndex: d.pingIndex,
        modelVersion: d.modelVersion,
        verifiedStatus: d.verifiedStatus,
        notes: d.notes || '',
      },
    }));

  const geoJson = {
    type: 'FeatureCollection',
    metadata: {
      generator: 'EchoPulseNet Marine Sonar Intelligence Platform',
      generatedAt: new Date().toISOString(),
      missionId: mission?.id || 'SURVEY-COLLECTION',
      featureCount: features.length,
    },
    features,
  };

  return JSON.stringify(geoJson, null, 2);
}

/**
 * Exports detections as CSV string
 */
export function exportDetectionsToCSV(detections: Detection[]): string {
  const headers = [
    'Detection ID',
    'Class',
    'Confidence',
    'Detector Score',
    'Shadow Score',
    'Anomaly Score',
    'Latitude',
    'Longitude',
    'Depth (m)',
    'Slant Range (m)',
    'Shadow Length (m)',
    'Ping Index',
    'Timestamp',
    'Mission ID',
    'Model Version',
    'Status',
  ];

  const rows = detections.map((d) => [
    `"${d.id}"`,
    `"${d.classNameLabel}"`,
    d.confidence.toFixed(4),
    d.detectorScore.toFixed(4),
    d.shadowScore.toFixed(4),
    d.anomalyScore.toFixed(4),
    d.latitude !== null ? d.latitude.toFixed(6) : 'N/A',
    d.longitude !== null ? d.longitude.toFixed(6) : 'N/A',
    d.depthMeters.toFixed(1),
    d.slantRangeMeters.toFixed(1),
    d.acousticShadow ? d.acousticShadow.lengthMeters.toFixed(2) : '0',
    d.pingIndex,
    `"${d.timestamp}"`,
    `"${d.missionId}"`,
    `"${d.modelVersion}"`,
    `"${d.verifiedStatus}"`,
  ]);

  return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
}

/**
 * Triggers browser file download
 */
export function downloadBlobFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

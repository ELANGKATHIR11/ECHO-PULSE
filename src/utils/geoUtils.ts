import { Detection, Mission } from '../types';

/**
 * Converts decimal degrees to DMS formatted string
 */
export function formatToDMS(dd: number | null | undefined, isLat: boolean): string {
  if (dd === null || dd === undefined || isNaN(dd)) return 'UNAVAILABLE';
  const direction = isLat ? (dd >= 0 ? 'N' : 'S') : dd >= 0 ? 'E' : 'W';
  const abs = Math.abs(dd);
  const degrees = Math.floor(abs);
  const minutes = Math.floor((abs - degrees) * 60);
  const seconds = (((abs - degrees) * 60 - minutes) * 60).toFixed(2);
  return `${degrees}°${minutes.toString().padStart(2, '0')}'${seconds.padStart(5, '0')}"${direction}`;
}

/**
 * Exports detections as an enhanced, complete JSON document with mission metadata & KPI summary
 */
export function exportDetectionsToJSON(detections: Detection[], mission?: Mission | null): string {
  const debrisCount = detections.filter((d) => d.isDebris ?? true).length;
  const naturalCount = detections.length - debrisCount;
  const avgConf =
    detections.length > 0
      ? (detections.reduce((acc, d) => acc + d.confidence, 0) / detections.length) * 100
      : 0;

  const categoryCounts: Record<string, number> = {};
  detections.forEach((d) => {
    const cat = d.guardrailCategory || 'UNKNOWN';
    categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
  });

  const payload = {
    metadata: {
      platform: 'EchoPulseNet Marine Sonar Intelligence Platform',
      problemStatement: 'SIH26057 — AI-Powered Automated Underwater Marine Debris and Anomaly Detection System Using Side-Scan Sonar Imagery',
      exportTimestampUtc: new Date().toISOString(),
      platformVersion: 'v2.6.0-PROD',
      crs: 'EPSG:4326 (WGS84 Ellipsoid)',
      classification: 'OFFICIAL HYDROGRAPHIC / RESEARCH ARCHIVE',
    },
    mission: {
      id: mission?.id || 'SURVEY-COLLECTION',
      name: mission?.name || 'Active Sonar Hydrographic Survey',
      sonarSource: mission?.sonarSource || 'Side-Scan Sonar (SSS)',
      frequencyKhz: mission?.frequencyKhz || 455,
      surveyDistanceKm: mission?.surveyDistanceKm || 18.4,
      swathWidthMeters: mission?.swathWidthMeters || 200.0,
      vesselName: mission?.vesselName || 'RV Sagar Nidhi (AUV DeepScan-4)',
    },
    kpiExecutiveSummary: {
      totalDetections: detections.length,
      anthropogenicDebrisCount: debrisCount,
      naturalSeabedFeaturesCount: naturalCount,
      averageFusedConfidencePct: Number(avgConf.toFixed(2)),
      taxonomyCategoryBreakdown: categoryCounts,
    },
    aiAndDspArchitecture: {
      primaryDetector: 'HydroPhys-OmniNet Extreme (CAW-SSM 1D/2D/3D Continuous Wavelet)',
      unifiedMambaScanner: 'EchoPhys-X v3 Unified (8-Channel Physics BiMamba)',
      physicsTensorChannels: '8-Channel (Intensity, Reverb, Highlight, Scatter, Range, Transmission Loss, Sound Speed c(T,S,D), Grazing Angle)',
      shadowInversion: 'Empirical Acoustic Shadow Geometric Inversion (H_target = (L_shadow * H_sensor) / (R_slant + L_shadow))',
    },
    detections: detections.map((d) => ({
      detectionId: d.id,
      missionId: d.missionId,
      timestampUtc: d.timestamp,
      pingIndex: d.pingIndex,
      classification: {
        classId: d.class,
        classLabel: d.classNameLabel,
        operationalCategory: d.guardrailCategory || 'UNKNOWN',
        isAnthropogenicDebris: d.isDebris ?? true,
        threatLevel: d.threatLevel || 'MODERATE',
      },
      confidenceMetrics: {
        fusedConfidencePct: Number((d.confidence * 100).toFixed(2)),
        detectorScore: Number(d.detectorScore.toFixed(4)),
        shadowPhysicsScore: Number(d.shadowScore.toFixed(4)),
        geometryScore: Number(d.geometryScore.toFixed(4)),
        anomalyScore: Number(d.anomalyScore.toFixed(4)),
        acousticQualityScore: Number(d.qualityScore.toFixed(4)),
        fusionModel: 'Multi-Factor Empirical Confidence Fusion (0.40D + 0.25S + 0.15G + 0.10A + 0.10Q)',
      },
      acousticShadowPhysics: {
        shadowDetected: !!(d.acousticShadow && d.acousticShadow.shadowConfidence > 0),
        shadowLengthMeters: d.acousticShadow ? Number(d.acousticShadow.lengthMeters.toFixed(2)) : 0.0,
        estimatedTargetHeightMeters: d.acousticShadow?.estimatedHeightMeters ?? null,
        shadowRatio: d.acousticShadow ? Number(d.acousticShadow.shadowRatio.toFixed(2)) : 0.0,
        shadowConfidence: d.acousticShadow ? Number(d.acousticShadow.shadowConfidence.toFixed(3)) : 0.0,
        shadowAngleDeg: d.acousticShadow?.angleDeg ?? 0.0,
        shadowPolygon: d.acousticShadow?.polygon ?? [],
      },
      geometryMorphology: {
        bboxPixels: d.bbox,
        areaPixels: d.geometry?.areaPixels ?? null,
        perimeterPixels: d.geometry?.perimeterPixels ?? null,
        aspectRatio: d.geometry?.aspectRatio ?? null,
        solidity: d.geometry?.solidity ?? null,
        extent: d.geometry?.extent ?? null,
        compactness: d.geometry?.compactness ?? null,
        orientationDeg: d.geometry?.orientationDeg ?? null,
      },
      hydrographicGeotag: {
        latitudeWgs84: d.latitude,
        longitudeWgs84: d.longitude,
        latitudeDms: formatToDMS(d.latitude, true),
        longitudeDms: formatToDMS(d.longitude, false),
        waterDepthMeters: d.depthMeters,
        slantRangeMeters: d.slantRangeMeters,
        sensorAltitudeMeters: d.altitudeMeters ?? null,
        geotagConfidencePct: Number((d.geotagConfidence * 100).toFixed(1)),
        positionSource: d.positionSource || (d.latitude !== null ? 'ESTIMATED_WGS84' : 'UNAVAILABLE'),
      },
      provenanceAndAudit: {
        modelVersion: d.modelVersion,
        inferenceSource: d.inferenceSource || 'NEURAL_VISION_CORE',
        dataSource: d.source || 'edge_pipeline',
        synthetic: d.synthetic ?? false,
        verificationStatus: d.verifiedStatus || 'UNVERIFIED',
        operatorNotes: d.notes || '',
        cropImageUrl: d.imageCropUrl,
      },
    })),
  };

  return JSON.stringify(payload, null, 2);
}

/**
 * Exports detections as standard GeoJSON FeatureCollection
 */
export function exportDetectionsToGeoJSON(detections: Detection[], mission?: Mission | null): string {
  const features = detections
    .filter((d) => d.latitude !== null && d.longitude !== null && !isNaN(d.latitude) && !isNaN(d.longitude))
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
        category: d.guardrailCategory || 'UNKNOWN',
        isDebris: d.isDebris ?? true,
        confidence: d.confidence,
        confidencePct: Number((d.confidence * 100).toFixed(1)),
        detectorScore: d.detectorScore,
        shadowScore: d.shadowScore,
        geometryScore: d.geometryScore,
        anomalyScore: d.anomalyScore,
        qualityScore: d.qualityScore,
        depthMeters: d.depthMeters,
        slantRangeMeters: d.slantRangeMeters,
        altitudeMeters: d.altitudeMeters ?? null,
        shadowLengthMeters: d.acousticShadow?.lengthMeters ?? 0,
        estimatedTargetHeightMeters: d.acousticShadow?.estimatedHeightMeters ?? null,
        shadowConfidence: d.acousticShadow?.shadowConfidence ?? 0,
        solidity: d.geometry?.solidity ?? null,
        aspectRatio: d.geometry?.aspectRatio ?? null,
        timestamp: d.timestamp,
        pingIndex: d.pingIndex,
        modelVersion: d.modelVersion,
        verificationStatus: d.verifiedStatus || 'UNVERIFIED',
        source: d.source || 'backend',
        synthetic: d.synthetic ?? false,
        notes: d.notes || '',
      },
    }));

  const geoJson = {
    type: 'FeatureCollection',
    metadata: {
      generator: 'EchoPulseNet Marine Sonar Intelligence Platform',
      problemStatement: 'SIH26057 Automated Underwater Marine Debris Detection',
      generatedAt: new Date().toISOString(),
      missionId: mission?.id || 'SURVEY-COLLECTION',
      featureCount: features.length,
      crs: 'EPSG:4326 (WGS84)',
    },
    features,
  };

  return JSON.stringify(geoJson, null, 2);
}

/**
 * Exports detections as an enhanced, complete CSV string
 */
export function exportDetectionsToCSV(detections: Detection[]): string {
  const headers = [
    'detection_id',
    'mission_id',
    'timestamp_utc',
    'ping_index',
    'class_id',
    'class_label',
    'category',
    'is_debris',
    'fused_confidence_pct',
    'detector_score',
    'shadow_physics_score',
    'geometry_score',
    'anomaly_score',
    'quality_score',
    'latitude_wgs84',
    'longitude_wgs84',
    'latitude_dms',
    'longitude_dms',
    'water_depth_m',
    'slant_range_m',
    'sensor_altitude_m',
    'shadow_detected',
    'shadow_length_m',
    'estimated_target_height_m',
    'shadow_ratio',
    'shadow_confidence',
    'area_pixels',
    'solidity',
    'aspect_ratio',
    'extent',
    'compactness',
    'geotag_confidence_pct',
    'position_source',
    'model_version',
    'verification_status',
    'notes',
  ];

  const rows = detections.map((d) => {
    const shadow = d.acousticShadow;
    const geom = d.geometry;

    return [
      `"${d.id}"`,
      `"${d.missionId}"`,
      `"${d.timestamp}"`,
      d.pingIndex,
      `"${d.class}"`,
      `"${d.classNameLabel}"`,
      `"${d.guardrailCategory || 'UNKNOWN'}"`,
      d.isDebris ? 1 : 0,
      (d.confidence * 100).toFixed(2),
      d.detectorScore.toFixed(4),
      d.shadowScore.toFixed(4),
      d.geometryScore.toFixed(4),
      d.anomalyScore.toFixed(4),
      d.qualityScore.toFixed(4),
      d.latitude !== null && !isNaN(d.latitude) ? d.latitude.toFixed(6) : 'UNAVAILABLE',
      d.longitude !== null && !isNaN(d.longitude) ? d.longitude.toFixed(6) : 'UNAVAILABLE',
      `"${formatToDMS(d.latitude, true)}"`,
      `"${formatToDMS(d.longitude, false)}"`,
      d.depthMeters.toFixed(1),
      d.slantRangeMeters.toFixed(1),
      d.altitudeMeters !== null && d.altitudeMeters !== undefined ? d.altitudeMeters.toFixed(1) : 'N/A',
      shadow && shadow.shadowConfidence > 0 ? 1 : 0,
      shadow ? shadow.lengthMeters.toFixed(2) : '0.00',
      shadow?.estimatedHeightMeters !== null && shadow?.estimatedHeightMeters !== undefined ? shadow.estimatedHeightMeters.toFixed(2) : 'N/A',
      shadow ? shadow.shadowRatio.toFixed(2) : '0.00',
      shadow ? shadow.shadowConfidence.toFixed(3) : '0.000',
      geom?.areaPixels ?? 'N/A',
      geom?.solidity !== null && geom?.solidity !== undefined ? geom.solidity.toFixed(3) : 'N/A',
      geom?.aspectRatio !== null && geom?.aspectRatio !== undefined ? geom.aspectRatio.toFixed(2) : 'N/A',
      geom?.extent !== null && geom?.extent !== undefined ? geom.extent.toFixed(3) : 'N/A',
      geom?.compactness !== null && geom?.compactness !== undefined ? geom.compactness.toFixed(3) : 'N/A',
      (d.geotagConfidence * 100).toFixed(1),
      `"${d.positionSource || (d.latitude !== null ? 'ESTIMATED_WGS84' : 'UNAVAILABLE')}"`,
      `"${d.modelVersion}"`,
      `"${d.verifiedStatus || 'UNVERIFIED'}"`,
      `"${(d.notes || '').replace(/"/g, '""')}"`,
    ];
  });

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

import { Mission, TrackPoint } from '../types';

export interface PingDensityNode {
  id: string;
  lat: number;
  lng: number;
  pingCount: number;
  pingsPerSqKm: number;
  densityLevel: 1 | 2 | 3 | 4; // 1: Low, 2: Moderate, 3: High, 4: Ultra/Multi-pass
  intensity: number; // 0.0 - 1.0
  color: string;
  fillOpacity: number;
  radiusMeters: number;
  contributingMissions: {
    id: string;
    name: string;
    codeName: string;
    date: string;
    sonarSource: string;
    frequencyKhz: number;
    pingsInCell: number;
  }[];
  overlapMultiplier: number;
  avgFrequencyKhz: number;
  dominantSonar: string;
  hasAnomalyCluster: boolean;
  anomalyCount: number;
  estimatedSnrDb: number;
}

export interface HistoricalHeatmapStats {
  totalAggregatedPings: number;
  totalMissionsIncluded: number;
  totalCoverageAreaSqKm: number;
  maxPingDensityPerSqKm: number;
  avgOverlapIndex: number;
  highDensityZonesCount: number;
  missionsList: { id: string; name: string; date: string; pingCount: number }[];
}

export type HeatmapMetricMode = 'PING_DENSITY' | 'SWATH_OVERLAP' | 'ANOMALY_RATE' | 'FREQUENCY_DIST';
export type HeatmapScope = 'ALL_HISTORICAL' | 'ACTIVE_ONLY';

/**
 * Returns color & opacity based on density level or normalized intensity
 */
export function getDensityColor(intensity: number, metric: HeatmapMetricMode = 'PING_DENSITY'): { color: string; level: 1 | 2 | 3 | 4 } {
  if (metric === 'SWATH_OVERLAP') {
    if (intensity > 0.75) return { color: '#f43f5e', level: 4 }; // >3x overlap (Bright Hot Rose)
    if (intensity > 0.45) return { color: '#f59e0b', level: 3 }; // 2x overlap (Amber)
    if (intensity > 0.2) return { color: '#10b981', level: 2 };  // 1.5x overlap (Emerald)
    return { color: '#06b6d4', level: 1 };                       // 1x pass (Cyan)
  }

  if (metric === 'ANOMALY_RATE') {
    if (intensity > 0.6) return { color: '#ef4444', level: 4 };
    if (intensity > 0.3) return { color: '#f97316', level: 3 };
    if (intensity > 0.1) return { color: '#eab308', level: 2 };
    return { color: '#22d3ee', level: 1 };
  }

  // Default PING_DENSITY gradient
  if (intensity > 0.75) return { color: '#f43f5e', level: 4 }; // Ultra high >40k pings/km²
  if (intensity > 0.5) return { color: '#f59e0b', level: 3 };  // High 25k-40k
  if (intensity > 0.25) return { color: '#10b981', level: 2 }; // Moderate 10k-25k
  return { color: '#06b6d4', level: 1 };                        // Base <10k
}

/**
 * Interpolates points along track segments with high density ping distribution
 */
function interpolateTrackSegments(
  points: TrackPoint[],
  totalPings: number,
  samplesPerSegment: number = 5
): { lat: number; lng: number; pingIndex: number; hasAnomaly: boolean; depth: number }[] {
  if (points.length === 0) return [];
  if (points.length === 1) {
    return [{ lat: points[0].latitude, lng: points[0].longitude, pingIndex: points[0].pingIndex, hasAnomaly: !!points[0].hasAnomaly, depth: points[0].depthMeters }];
  }

  const result: { lat: number; lng: number; pingIndex: number; hasAnomaly: boolean; depth: number }[] = [];

  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];

    for (let s = 0; s <= samplesPerSegment; s++) {
      const t = s / samplesPerSegment;
      const lat = p1.latitude + (p2.latitude - p1.latitude) * t;
      const lng = p1.longitude + (p2.longitude - p1.longitude) * t;
      const pingIndex = Math.round(p1.pingIndex + (p2.pingIndex - p1.pingIndex) * t);
      const hasAnomaly = (t < 0.3 && p1.hasAnomaly) || (t > 0.7 && p2.hasAnomaly);
      const depth = p1.depthMeters + (p2.depthMeters - p1.depthMeters) * t;

      result.push({ lat, lng, pingIndex, hasAnomaly: !!hasAnomaly, depth });
    }
  }

  return result;
}

/**
 * Calculates distance in meters between two lat/lng coordinates (Haversine)
 */
export function haversineDistanceMeters(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371e3; // Earth radius in meters
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δφ = ((lat2 - lat1) * Math.PI) / 180;
  const Δλ = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}

/**
 * Aggregates ping density across all historical missions into discrete spatial heatmap nodes
 */
export function generateAggregatedMissionHeatmap(
  missions: Mission[],
  activeMissionId?: string,
  scope: HeatmapScope = 'ALL_HISTORICAL',
  metric: HeatmapMetricMode = 'PING_DENSITY',
  kernelBandwidthMeters: number = 80
): { nodes: PingDensityNode[]; stats: HistoricalHeatmapStats } {
  const targetMissions = scope === 'ACTIVE_ONLY' && activeMissionId
    ? missions.filter((m) => m.id === activeMissionId)
    : missions;

  // Extract all track sample points from selected missions
  interface RawSample {
    missionId: string;
    mission: Mission;
    lat: number;
    lng: number;
    pingIndex: number;
    hasAnomaly: boolean;
    depth: number;
    swathRadiusMeters: number;
  }

  const rawSamples: RawSample[] = [];

  targetMissions.forEach((m) => {
    const swathRadius = (m.coverageCorridorWidthMeters || 200) * 0.5;
    const interpolated = interpolateTrackSegments(m.trackPoints, m.pingCount, 6);

    interpolated.forEach((pt) => {
      rawSamples.push({
        missionId: m.id,
        mission: m,
        lat: pt.lat,
        lng: pt.lng,
        pingIndex: pt.pingIndex,
        hasAnomaly: pt.hasAnomaly,
        depth: pt.depth,
        swathRadiusMeters: swathRadius,
      });

      // Add slight cross-track corridor offsets for side-scan swath footprint
      const offsetLat = 0.00035;
      const offsetLng = 0.00035;
      rawSamples.push({
        missionId: m.id,
        mission: m,
        lat: pt.lat + offsetLat,
        lng: pt.lng - offsetLng,
        pingIndex: pt.pingIndex + 20,
        hasAnomaly: false,
        depth: pt.depth,
        swathRadiusMeters: swathRadius,
      });
      rawSamples.push({
        missionId: m.id,
        mission: m,
        lat: pt.lat - offsetLat,
        lng: pt.lng + offsetLng,
        pingIndex: pt.pingIndex + 40,
        hasAnomaly: false,
        depth: pt.depth,
        swathRadiusMeters: swathRadius,
      });
    });
  });

  if (rawSamples.length === 0) {
    return {
      nodes: [],
      stats: {
        totalAggregatedPings: 0,
        totalMissionsIncluded: 0,
        totalCoverageAreaSqKm: 0,
        maxPingDensityPerSqKm: 0,
        avgOverlapIndex: 1.0,
        highDensityZonesCount: 0,
        missionsList: [],
      },
    };
  }

  // Spatial clustering grid (resolution roughly ~0.001 deg ≈ 110 meters)
  const cellSize = 0.0009;
  const gridMap = new Map<string, RawSample[]>();

  rawSamples.forEach((sample) => {
    const gridKey = `${Math.round(sample.lat / cellSize)}_${Math.round(sample.lng / cellSize)}`;
    if (!gridMap.has(gridKey)) {
      gridMap.set(gridKey, []);
    }
    gridMap.get(gridKey)!.push(sample);
  });

  // Convert grid clusters to PingDensityNodes
  const rawNodes: PingDensityNode[] = [];
  let maxPingCount = 1;
  let totalAggregatedPings = 0;
  let highDensityCount = 0;

  gridMap.forEach((samples, key) => {
    const avgLat = samples.reduce((acc, s) => acc + s.lat, 0) / samples.length;
    const avgLng = samples.reduce((acc, s) => acc + s.lng, 0) / samples.length;

    // Group by mission
    const missionGroups = new Map<string, { mission: Mission; count: number; anomalies: number }>();
    samples.forEach((s) => {
      if (!missionGroups.has(s.missionId)) {
        missionGroups.set(s.missionId, { mission: s.mission, count: 0, anomalies: 0 });
      }
      const g = missionGroups.get(s.missionId)!;
      g.count += 1;
      if (s.hasAnomaly) g.anomalies += 1;
    });

    const contributingMissions = Array.from(missionGroups.entries()).map(([mId, data]) => {
      // Scale sample count to approximate actual sonar pings emitted in this cell
      const pingsInCell = Math.round((data.count / samples.length) * (data.mission.pingCount / 18));
      return {
        id: mId,
        name: data.mission.name,
        codeName: data.mission.codeName,
        date: data.mission.date,
        sonarSource: data.mission.sonarSource,
        frequencyKhz: data.mission.frequencyKhz,
        pingsInCell: Math.max(850, pingsInCell),
      };
    });

    const cellPings = contributingMissions.reduce((acc, m) => acc + m.pingsInCell, 0);
    totalAggregatedPings += cellPings;
    if (cellPings > maxPingCount) maxPingCount = cellPings;

    const overlapMultiplier = contributingMissions.length >= 2 
      ? Number((1.4 + contributingMissions.length * 0.7).toFixed(1)) 
      : 1.0;

    const totalAnomalies = Array.from(missionGroups.values()).reduce((acc, g) => acc + g.anomalies, 0);
    const avgFreq = contributingMissions.reduce((acc, m) => acc + m.frequencyKhz, 0) / contributingMissions.length;
    const dominantSonar = contributingMissions[0]?.sonarSource || 'Side-Scan Sonar (SSS)';

    // Approximate pings per sq km (cell area roughly 0.015 sq km)
    const pingsPerSqKm = Math.round(cellPings / 0.018);

    rawNodes.push({
      id: `heat-cell-${key}`,
      lat: avgLat,
      lng: avgLng,
      pingCount: cellPings,
      pingsPerSqKm,
      densityLevel: 1, // will assign after normalization
      intensity: 0,
      color: '#06b6d4',
      fillOpacity: 0.35,
      radiusMeters: Math.max(50, kernelBandwidthMeters),
      contributingMissions,
      overlapMultiplier,
      avgFrequencyKhz: Math.round(avgFreq),
      dominantSonar,
      hasAnomalyCluster: totalAnomalies > 0,
      anomalyCount: totalAnomalies,
      estimatedSnrDb: Number((21.5 + (overlapMultiplier - 1) * 3.2).toFixed(1)),
    });
  });

  // Normalize intensities and assign styling
  const nodes: PingDensityNode[] = rawNodes.map((node) => {
    let metricValue = 0;

    if (metric === 'PING_DENSITY') {
      metricValue = node.pingCount / maxPingCount;
    } else if (metric === 'SWATH_OVERLAP') {
      metricValue = Math.min(1.0, (node.overlapMultiplier - 1.0) / 2.5);
    } else if (metric === 'ANOMALY_RATE') {
      metricValue = node.anomalyCount > 0 ? Math.min(1.0, node.anomalyCount / 3.0) : 0.05;
    } else {
      // FREQUENCY_DIST
      metricValue = Math.min(1.0, node.avgFrequencyKhz / 900);
    }

    const { color, level } = getDensityColor(metricValue, metric);
    if (level >= 3) highDensityCount++;

    return {
      ...node,
      intensity: metricValue,
      densityLevel: level,
      color,
      fillOpacity: 0.25 + metricValue * 0.45,
    };
  });

  const totalArea = targetMissions.reduce((acc, m) => acc + (m.areaSqKm || 3.5), 0);
  const avgOverlap = Number((nodes.reduce((acc, n) => acc + n.overlapMultiplier, 0) / Math.max(1, nodes.length)).toFixed(2));

  const stats: HistoricalHeatmapStats = {
    totalAggregatedPings: targetMissions.reduce((acc, m) => acc + m.pingCount, 0),
    totalMissionsIncluded: targetMissions.length,
    totalCoverageAreaSqKm: Number(totalArea.toFixed(1)),
    maxPingDensityPerSqKm: Math.round((maxPingCount / 0.018)),
    avgOverlapIndex: avgOverlap,
    highDensityZonesCount: highDensityCount,
    missionsList: targetMissions.map((m) => ({
      id: m.id,
      name: m.name,
      date: m.date,
      pingCount: m.pingCount,
    })),
  };

  return { nodes, stats };
}

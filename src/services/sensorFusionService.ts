/**
 * Sensor Fusion & 3D Spatial Triangulation Service
 * Calculates:
 * 1. Real-time IR / Optical Focal Range Distance (Meters)
 * 2. 3D World Coordinates (X, Y, Z) on Subsea Bathymetric Seabed Mesh
 * 3. Exact WGS84 Geodetic Coordinates (Lat, Lng, Altitude) from System GPS & Heading
 */

export interface SystemGpsState {
  latitude: number;
  longitude: number;
  altitudeMeters: number;
  headingDeg: number;
  accuracyMeters: number;
  isLiveGps: boolean;
}

export interface Projected3DObject {
  id: string;
  className: string;
  label: string;
  category: string;
  confidence: number;
  distanceMeters: number;
  irSensorRangeMeters: number;
  world3D: [number, number, number]; // [X, Y, Z] in Three.js coordinates
  wgs84: {
    lat: number;
    lng: number;
    depthMeters: number;
  };
  bbox: [number, number, number, number]; // [x, y, w, h]
  timestamp: string;
}

// Known physical reference heights (meters) for optical pinhole depth estimation
const OBJECT_PHYSICAL_HEIGHT_MAP: Record<string, number> = {
  // 1. Humans
  human: 1.8,
  person: 1.8,
  scuba_diver: 1.8,
  // 2. Electrical
  electrical: 0.4,
  subsea_cable: 0.4,
  power_cable: 0.4,
  cable: 0.4,
  // 3. Electronic
  electronic: 0.3,
  cell_phone: 0.15,
  laptop: 0.25,
  e_waste: 0.35,
  remote: 0.2,
  transponder: 0.6,
  // 4. Plastic
  plastic: 0.5,
  ghost_gear: 1.5,
  marine_debris: 0.6,
  bottle: 0.25,
  cup: 0.15,
  backpack: 0.55,
  handbag: 0.4,
  // 5. Metal Scraps
  metal_scrap: 1.2,
  metal: 1.0,
  shipwreck: 8.5,
  unexploded_ordnance: 0.9,
  pipeline_anomaly: 1.2,
  knife: 0.22,
  scissors: 0.20,
  default: 0.5,
};

class SensorFusionService {
  private gpsState: SystemGpsState = {
    latitude: 9.1524,
    longitude: 79.2819,
    altitudeMeters: 14.5,
    headingDeg: 84.0,
    accuracyMeters: 2.5,
    isLiveGps: false,
  };

  private watchId: number | null = null;
  private cameraFocalLengthPx: number = 650.0; // Standard calibrated 720p webcam focal length

  constructor() {
    this.initSystemGps();
  }

  /**
   * Initializes Web Geolocation API for real-time system GPS coordinates
   */
  public initSystemGps() {
    if (typeof window !== 'undefined' && 'geolocation' in navigator) {
      this.watchId = navigator.geolocation.watchPosition(
        (pos) => {
          this.gpsState = {
            latitude: pos.coords.latitude,
            longitude: pos.coords.longitude,
            altitudeMeters: pos.coords.altitude || 14.5,
            headingDeg: pos.coords.heading || this.gpsState.headingDeg,
            accuracyMeters: pos.coords.accuracy || 2.0,
            isLiveGps: true,
          };
        },
        (err) => {
          console.debug('[GPS] Geolocation fallback to simulated coastal WGS84 coordinates:', err.message);
        },
        { enableHighAccuracy: true, timeout: 5000, maximumAge: 1000 }
      );
    }
  }

  public getGpsState(): SystemGpsState {
    return this.gpsState;
  }

  public setManualHeading(heading: number) {
    this.gpsState.headingDeg = (heading + 360) % 360;
  }

  /**
   * Triangulates 3D World Position & WGS84 GPS from 2D Bounding Box + IR/Optical Range
   */
  public projectBoundingBoxTo3D(
    bbox: [number, number, number, number], // [x, y, w, h] in canvas pixels
    canvasWidth: number,
    canvasHeight: number,
    className: string,
    confidence: number,
    customIrRangeMeters?: number
  ): Projected3DObject {
    const [x, y, w, h] = bbox;
    const cx = x + w / 2;
    const cy = y + h / 2;

    // 1. Calculate Target Distance (IR Sensor or Optical Pinhole Trigonometry)
    const refHeightMeters = OBJECT_PHYSICAL_HEIGHT_MAP[className.toLowerCase()] || OBJECT_PHYSICAL_HEIGHT_MAP.default;
    let distanceMeters: number;

    if (customIrRangeMeters !== undefined && customIrRangeMeters > 0.1) {
      distanceMeters = customIrRangeMeters;
    } else {
      // D = (f * H_real) / h_pixels
      const pixelHeight = Math.max(12, h);
      distanceMeters = (this.cameraFocalLengthPx * refHeightMeters) / pixelHeight;
      // Clamp to reasonable subsea visibility range (0.5m to 45m)
      distanceMeters = Math.max(0.5, Math.min(45.0, distanceMeters));
    }

    // 2. Compute Angular Offsets from Optical Center
    const fX = this.cameraFocalLengthPx;
    const fY = this.cameraFocalLengthPx;
    const deltaXPixels = cx - canvasWidth / 2;
    const deltaYPixels = cy - canvasHeight / 2;

    const azimuthOffsetRad = Math.atan(deltaXPixels / fX);
    const pitchOffsetRad = Math.atan(deltaYPixels / fY);

    const totalHeadingRad = ((this.gpsState.headingDeg + (azimuthOffsetRad * 180) / Math.PI) * Math.PI) / 180.0;

    // 3. Three.js Bathymetric Mesh Coordinates
    // Scale distance for 3D digital twin viewport (1 unit = 2 meters)
    const normalizedDist = distanceMeters * 0.45;
    const worldX = Math.sin(totalHeadingRad) * normalizedDist;
    const worldZ = Math.cos(totalHeadingRad) * normalizedDist;
    // Seabed depth calculation (altitude minus target vertical drop)
    const worldY = -Math.max(0.2, (pitchOffsetRad * normalizedDist) + 0.5);

    // 4. Exact WGS84 Geodetic Coordinates
    const deltaNorthMeters = distanceMeters * Math.cos(totalHeadingRad);
    const deltaEastMeters = distanceMeters * Math.sin(totalHeadingRad);

    const latTarget = this.gpsState.latitude + deltaNorthMeters / 111320.0;
    const lngTarget =
      this.gpsState.longitude + deltaEastMeters / (111320.0 * Math.cos((this.gpsState.latitude * Math.PI) / 180.0));
    const targetDepthMeters = Math.round((this.gpsState.altitudeMeters + distanceMeters * 0.3) * 10) / 10;

    const norm = className.toLowerCase();
    let targetCat = 'PLASTIC';
    if (['human', 'person', 'scuba_diver'].includes(norm)) targetCat = 'HUMAN';
    else if (['electrical', 'subsea_cable', 'power_cable', 'cable'].includes(norm)) targetCat = 'ELECTRICAL';
    else if (['electronic', 'cell_phone', 'laptop', 'e_waste', 'remote', 'transponder', 'mouse', 'keyboard'].includes(norm)) targetCat = 'ELECTRONIC';
    else if (['metal_scrap', 'metal', 'shipwreck', 'unexploded_ordnance', 'pipeline_anomaly', 'knife', 'scissors'].includes(norm)) targetCat = 'METAL_SCRAP';
    else if (['biological_cluster', 'geological_formation', 'book', 'vase', 'chair', 'boat'].includes(norm)) targetCat = 'NOT_A_DEBRIS';

    return {
      id: `OPTIC-3D-${Math.random().toString(36).substring(2, 7).toUpperCase()}`,
      className,
      label: className.replace('_', ' ').toUpperCase(),
      category: targetCat,
      confidence: Math.round(confidence * 100) / 100,
      distanceMeters: Math.round(distanceMeters * 10) / 10,
      irSensorRangeMeters: Math.round((customIrRangeMeters || distanceMeters) * 10) / 10,
      world3D: [worldX, worldY, worldZ],
      wgs84: {
        lat: parseFloat(latTarget.toFixed(6)),
        lng: parseFloat(lngTarget.toFixed(6)),
        depthMeters: targetDepthMeters,
      },
      bbox,
      timestamp: new Date().toISOString(),
    };
  }
}

export const sensorFusion = new SensorFusionService();

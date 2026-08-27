import json
import csv
import io
import time
import math
from typing import List, Dict, Any, Optional
from ..schemas.contracts import DetectionSchema

def decdeg2dms(dd: Optional[float], is_lat: bool) -> str:
    if dd is None or math.isnan(dd):
        return "N/A"
    direction = ("N" if dd >= 0 else "S") if is_lat else ("E" if dd >= 0 else "W")
    dd = abs(dd)
    degrees = int(dd)
    minutes = int((dd - degrees) * 60)
    seconds = round(((dd - degrees) * 60 - minutes) * 60, 2)
    return f"{degrees}°{minutes:02d}'{seconds:05.2f}\"{direction}"


class ReportGenerator:
    """
    Enhanced Intelligence Report Generator for SIH26057:
    Exports hydrographic acoustic survey telemetry in structured JSON, GeoJSON, and CSV formats
    with complete physical metrics, directional shadow inversions, spatial uncertainties, and provenance.
    """

    @staticmethod
    def generate_json(detections: List[DetectionSchema], mission_meta: Dict[str, Any]) -> str:
        # Compute summary aggregations
        debris_count = len([d for d in detections if getattr(d, 'isDebris', True)])
        natural_count = len(detections) - debris_count
        avg_conf = (sum(d.confidence for d in detections) / max(1, len(detections))) * 100.0
        
        # Taxonomy breakdown
        cat_counts: Dict[str, int] = {}
        for d in detections:
            cat = getattr(d, 'guardrailCategory', 'UNKNOWN')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        formatted_detections = []
        for d in detections:
            shadow = d.acousticShadow
            geom = d.geometry
            
            det_record = {
                "detection_id": d.id,
                "mission_id": d.missionId,
                "timestamp_utc": d.timestamp,
                "ping_index": d.pingIndex,
                "classification": {
                    "class_id": d.class_name,
                    "class_label": d.classNameLabel,
                    "operational_category": getattr(d, 'guardrailCategory', 'UNKNOWN'),
                    "is_anthropogenic_debris": getattr(d, 'isDebris', True),
                    "guardrail_passed": getattr(d, 'guardrailPassed', True),
                    "guardrail_reason": getattr(d, 'guardrailReason', None)
                },
                "confidence_metrics": {
                    "fused_confidence_pct": round(d.confidence * 100.0, 2),
                    "raw_detector_score": round(d.detectorScore, 4),
                    "shadow_physics_score": round(d.shadowScore, 4),
                    "morphological_geometry_score": round(d.geometryScore, 4),
                    "seabed_anomaly_score": round(d.anomalyScore, 4),
                    "acoustic_quality_score": round(d.qualityScore, 4),
                    "fusion_model": "Empirically Tuned Multi-Factor Confidence Fusion (0.40D + 0.25S + 0.15G + 0.10A + 0.10Q)"
                },
                "acoustic_shadow_physics": {
                    "shadow_detected": shadow is not None and shadow.shadowConfidence > 0.0,
                    "shadow_length_m": round(shadow.lengthMeters, 2) if shadow else 0.0,
                    "shadow_ratio": round(shadow.shadowRatio, 2) if shadow else 0.0,
                    "shadow_angle_deg": round(shadow.angleDeg, 1) if shadow else 0.0,
                    "shadow_confidence": round(shadow.shadowConfidence, 3) if shadow else 0.0,
                    "estimated_target_height_m": shadow.estimatedHeightMeters if shadow else None,
                    "height_inversion_formula": "H_target = (L_shadow * H_sensor) / (R_slant + L_shadow)",
                    "polygon_points": [p.model_dump() if hasattr(p, 'model_dump') else dict(p) for p in shadow.polygon] if shadow else []
                },
                "geometry_and_bounding_box": {
                    "bbox_pixels": {
                        "x": d.bbox.x,
                        "y": d.bbox.y,
                        "width": d.bbox.width,
                        "height": d.bbox.height
                    },
                    "area_pixels": geom.areaPixels if geom else None,
                    "perimeter_pixels": geom.perimeterPixels if geom else None,
                    "aspect_ratio": geom.aspectRatio if geom else None,
                    "solidity": geom.solidity if geom else None,
                    "extent": geom.extent if geom else None,
                    "compactness": geom.compactness if geom else None,
                    "orientation_deg": geom.orientationDeg if geom else None
                },
                "geospatial_localization": {
                    "latitude_wgs84": d.latitude,
                    "longitude_wgs84": d.longitude,
                    "latitude_dms": decdeg2dms(d.latitude, is_lat=True),
                    "longitude_dms": decdeg2dms(d.longitude, is_lat=False),
                    "crs": "EPSG:4326 (WGS84)",
                    "water_depth_m": d.depthMeters,
                    "slant_range_m": d.slantRangeMeters,
                    "sensor_altitude_m": d.altitudeMeters,
                    "geotag_confidence_pct": round(d.geotagConfidence * 100.0, 1),
                    "position_uncertainty_m": getattr(d, 'positionUncertaintyMeters', None),
                    "position_source": getattr(d, 'positionSource', 'ESTIMATED_WGS84' if d.latitude else 'UNAVAILABLE')
                },
                "provenance_and_audit": {
                    "model_version": d.modelVersion,
                    "inference_source": getattr(d, 'inferenceSource', 'NEURAL_VISION_CORE'),
                    "data_source": getattr(d, 'source', 'backend'),
                    "synthetic": getattr(d, 'synthetic', False),
                    "review_status": getattr(d, 'verifiedStatus', 'UNVERIFIED'),
                    "operator_notes": getattr(d, 'notes', None),
                    "crop_image_url": d.imageCropUrl,
                    "raw_crop_url": getattr(d, 'rawCropUrl', None)
                }
            }
            formatted_detections.append(det_record)

        report_payload = {
            "metadata": {
                "report_title": "EchoPulseNet Hydrographic Sonar Debris & Anomaly Intelligence Report",
                "problem_statement": "SIH26057 — Automated Underwater Marine Debris and Anomaly Detection Using Side-Scan Sonar",
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "platform_version": "EchoPulseNet v2.6.0-PROD",
                "crs": "EPSG:4326 (WGS84)",
                "governance_classification": "OFFICIAL HYDROGRAPHIC / RESEARCH"
            },
            "survey_mission": {
                "id": mission_meta.get("id", "MSN-2026-0884"),
                "name": mission_meta.get("name", "Active Sonar Survey"),
                "code_name": mission_meta.get("codeName", "OPERATION NEPTUNE-SWEEP"),
                "vessel": mission_meta.get("vesselName", "RV Sagar Nidhi (AUV Unit-Alpha)"),
                "vehicle_type": mission_meta.get("vehicleType", "AUV DeepScan-4"),
                "location": mission_meta.get("location", "Gulf of Mannar"),
                "coordinates": mission_meta.get("coordinates", [9.1524, 79.2819]),
                "sonar_sensor": mission_meta.get("sonarSource", "Side-Scan Sonar (SSS)"),
                "frequency_khz": mission_meta.get("frequencyKhz", 455),
                "survey_distance_km": mission_meta.get("surveyDistanceKm", 18.4),
                "area_coverage_sq_km": mission_meta.get("areaSqKm", 3.68),
                "swath_width_m": mission_meta.get("swathWidthMeters", 200.0)
            },
            "kpi_executive_summary": {
                "total_detections_found": len(detections),
                "anthropogenic_debris_count": debris_count,
                "natural_seabed_features_count": natural_count,
                "average_fused_confidence_pct": round(avg_conf, 2),
                "category_breakdown": cat_counts
            },
            "ai_and_dsp_architecture": {
                "primary_detector": "HydroPhys-OmniNet Extreme (CAW-SSM Continuous Wavelet State-Space)",
                "unified_mamba_scanner": "EchoPhys-X v3 Unified (8-Channel Physics BiMamba)",
                "physics_acoustic_tensor": "8-Channel (Intensity, Reverb, Highlight, Scatter, Range, Transmission Loss, Mackenzie Sound Speed c(T,S,D), Grazing Angle)",
                "shadow_segmenter": "Lightweight Sonar U-Net",
                "anomaly_model": "Seabed Residual Autoencoder"
            },
            "detections": formatted_detections
        }

        return json.dumps(report_payload, indent=2)

    @staticmethod
    def generate_csv(detections: List[DetectionSchema]) -> str:
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        
        # Comprehensive CSV Column Headers
        writer.writerow([
            "detection_id",
            "mission_id",
            "timestamp_utc",
            "ping_index",
            "class_id",
            "class_label",
            "operational_category",
            "is_debris",
            "fused_confidence_pct",
            "detector_score",
            "shadow_physics_score",
            "geometry_score",
            "anomaly_score",
            "quality_score",
            "latitude_wgs84",
            "longitude_wgs84",
            "latitude_dms",
            "longitude_dms",
            "water_depth_m",
            "slant_range_m",
            "sensor_altitude_m",
            "shadow_detected",
            "shadow_length_m",
            "estimated_target_height_m",
            "shadow_ratio",
            "shadow_confidence",
            "area_pixels",
            "solidity",
            "aspect_ratio",
            "extent",
            "compactness",
            "geotag_confidence_pct",
            "position_source",
            "model_version",
            "review_status",
            "guardrail_reason",
            "notes"
        ])
        
        for d in detections:
            shadow = d.acousticShadow
            geom = d.geometry
            
            writer.writerow([
                d.id,
                d.missionId,
                d.timestamp,
                d.pingIndex,
                d.class_name,
                d.classNameLabel,
                getattr(d, 'guardrailCategory', 'UNKNOWN'),
                1 if getattr(d, 'isDebris', True) else 0,
                round(d.confidence * 100.0, 2),
                round(d.detectorScore, 4),
                round(d.shadowScore, 4),
                round(d.geometryScore, 4),
                round(d.anomalyScore, 4),
                round(d.qualityScore, 4),
                f"{d.latitude:.6f}" if d.latitude is not None else "UNAVAILABLE",
                f"{d.longitude:.6f}" if d.longitude is not None else "UNAVAILABLE",
                decdeg2dms(d.latitude, is_lat=True),
                decdeg2dms(d.longitude, is_lat=False),
                d.depthMeters,
                d.slantRangeMeters,
                d.altitudeMeters if d.altitudeMeters is not None else "N/A",
                1 if (shadow and shadow.shadowConfidence > 0.0) else 0,
                round(shadow.lengthMeters, 2) if shadow else 0.0,
                shadow.estimatedHeightMeters if (shadow and shadow.estimatedHeightMeters is not None) else "N/A",
                round(shadow.shadowRatio, 2) if shadow else 0.0,
                round(shadow.shadowConfidence, 3) if shadow else 0.0,
                geom.areaPixels if geom else "N/A",
                geom.solidity if geom else "N/A",
                geom.aspectRatio if geom else "N/A",
                geom.extent if geom else "N/A",
                geom.compactness if geom else "N/A",
                round(d.geotagConfidence * 100.0, 1),
                getattr(d, 'positionSource', 'ESTIMATED_WGS84' if d.latitude else 'UNAVAILABLE'),
                d.modelVersion,
                getattr(d, 'verifiedStatus', 'UNVERIFIED'),
                getattr(d, 'guardrailReason', ''),
                getattr(d, 'notes', '') or ''
            ])
            
        return output.getvalue()

    @staticmethod
    def generate_geojson(detections: List[DetectionSchema]) -> Dict[str, Any]:
        features = []
        for d in detections:
            if d.latitude is not None and d.longitude is not None:
                shadow = d.acousticShadow
                geom = d.geometry
                
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [d.longitude, d.latitude, -(d.depthMeters or 0.0)]
                    },
                    "properties": {
                        "id": d.id,
                        "missionId": d.missionId,
                        "missionName": d.missionName,
                        "class": d.class_name,
                        "classLabel": d.classNameLabel,
                        "category": getattr(d, 'guardrailCategory', 'UNKNOWN'),
                        "isDebris": getattr(d, 'isDebris', True),
                        "confidence": d.confidence,
                        "confidencePct": round(d.confidence * 100.0, 2),
                        "detectorScore": d.detectorScore,
                        "shadowScore": d.shadowScore,
                        "geometryScore": d.geometryScore,
                        "anomalyScore": d.anomalyScore,
                        "qualityScore": d.qualityScore,
                        "depthMeters": d.depthMeters,
                        "slantRangeMeters": d.slantRangeMeters,
                        "sensorAltitudeMeters": d.altitudeMeters,
                        "estimatedHeightMeters": shadow.estimatedHeightMeters if shadow else None,
                        "shadowLengthMeters": shadow.lengthMeters if shadow else None,
                        "shadowConfidence": shadow.shadowConfidence if shadow else None,
                        "solidity": geom.solidity if geom else None,
                        "aspectRatio": geom.aspectRatio if geom else None,
                        "geotagConfidence": d.geotagConfidence,
                        "timestamp": d.timestamp,
                        "pingIndex": d.pingIndex,
                        "modelVersion": d.modelVersion,
                        "verificationStatus": getattr(d, 'verifiedStatus', 'UNVERIFIED'),
                        "source": getattr(d, 'source', 'backend'),
                        "synthetic": getattr(d, 'synthetic', False)
                    }
                })
                
        return {
            "type": "FeatureCollection",
            "metadata": {
                "platform": "EchoPulseNet Marine Sonar Intelligence Platform",
                "problem": "SIH26057 Automated Underwater Marine Debris Detection",
                "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "featureCount": len(features)
            },
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

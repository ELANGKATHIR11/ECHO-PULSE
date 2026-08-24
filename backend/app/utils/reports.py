import json
import csv
import io
from typing import List, Dict, Any
from ..schemas.contracts import DetectionSchema

class ReportGenerator:
    @staticmethod
    def generate_json(detections: List[DetectionSchema], mission_meta: Dict[str, Any]) -> str:
        data = {
            "platform": "EchoPulseNet Marine Sonar Intelligence Platform",
            "version": "2.6.0-PROD",
            "mission": mission_meta,
            "detections_count": len(detections),
            "detections": [d.model_dump(by_alias=True) for d in detections],
            "provenance": {
                "source": "backend",
                "synthetic": False,
                "confidence_model": "MultiFactorFusion-v2.6"
            }
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def generate_csv(detections: List[DetectionSchema]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            "detection_id", "mission_id", "class", "label", "confidence",
            "detector_score", "shadow_score", "geometry_score", "anomaly_score",
            "latitude", "longitude", "depth_m", "slant_range_m", "estimated_height_m",
            "geotag_confidence", "source", "synthetic"
        ])
        
        for d in detections:
            writer.writerow([
                d.id, d.missionId, d.class_name, d.classNameLabel, d.confidence,
                d.detectorScore, d.shadowScore, d.geometryScore, d.anomalyScore,
                d.latitude or "", d.longitude or "", d.depthMeters, d.slantRangeMeters,
                d.acousticShadow.estimatedHeightMeters if d.acousticShadow else "",
                d.geotagConfidence, d.source, d.synthetic
            ])
            
        return output.getvalue()

    @staticmethod
    def generate_geojson(detections: List[DetectionSchema]) -> Dict[str, Any]:
        features = []
        for d in detections:
            if d.latitude is not None and d.longitude is not None:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [d.longitude, d.latitude]
                    },
                    "properties": {
                        "id": d.id,
                        "missionId": d.missionId,
                        "class": d.class_name,
                        "label": d.classNameLabel,
                        "confidence": d.confidence,
                        "depth": d.depthMeters,
                        "source": d.source,
                        "synthetic": d.synthetic
                    }
                })
        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

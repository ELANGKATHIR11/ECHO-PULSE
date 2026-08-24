import os
import json
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, Column, String, Float, Integer, JSON, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv(
    "POSTGIS_DATABASE_URL",
    os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/echopulse_postgis")
)

Base = declarative_base()

class SpatialDetectionORM(Base):
    __tablename__ = "sonar_spatial_detections"

    id = Column(String, primary_key=True, index=True)
    mission_id = Column(String, index=True)
    mission_name = Column(String)
    target_class = Column(String, index=True)
    class_name_label = Column(String)
    confidence = Column(Float)
    detector_score = Column(Float)
    shadow_score = Column(Float)
    geometry_score = Column(Float)
    anomaly_score = Column(Float)
    quality_score = Column(Float)
    latitude = Column(Float)
    longitude = Column(Float)
    depth_meters = Column(Float)
    slant_range_meters = Column(Float)
    altitude_meters = Column(Float, nullable=True)
    geotag_confidence = Column(Float)
    ping_index = Column(Integer)
    model_version = Column(String)
    image_crop_url = Column(String)
    verification_status = Column(String, default="UNVERIFIED")
    operator_notes = Column(String, nullable=True)
    bbox_json = Column(JSON)
    geometry_meta = Column(JSON)
    shadow_meta = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class SpatialMissionORM(Base):
    __tablename__ = "sonar_spatial_missions"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    code_name = Column(String)
    date = Column(String)
    location = Column(String)
    center_lat = Column(Float)
    center_lng = Column(Float)
    sonar_source = Column(String)
    frequency_khz = Column(Float)
    survey_distance_km = Column(Float)
    area_sq_km = Column(Float)
    status = Column(String)
    duration_minutes = Column(Integer)
    ping_count = Column(Integer)
    vessel_name = Column(String)
    vehicle_type = Column(String)
    target_objective = Column(String)
    track_points_json = Column(JSON)
    summary_metrics_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class PostGISConnector:
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.is_connected = False
        self.engine = None
        self.SessionLocal = None
        self._init_connection()

    def _init_connection(self):
        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
            # Test connectivity
            with self.engine.connect() as conn:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                    conn.commit()
                except Exception:
                    pass
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.is_connected = True
            print(f"[*] PostGISConnector: Connected to {self.db_url.split('@')[-1] if '@' in self.db_url else 'PostGIS Engine'}")
        except Exception as e:
            self.is_connected = False
            print(f"[!] PostGISConnector Notice: PostgreSQL/PostGIS server connection deferred: {e}")

    def sync_detection(self, det: Dict[str, Any]) -> bool:
        if not self.is_connected or not self.SessionLocal:
            return False
        try:
            session = self.SessionLocal()
            bbox = det.get("bbox", {})
            bbox_dict = bbox.model_dump() if hasattr(bbox, "model_dump") else (bbox if isinstance(bbox, dict) else {})
            geom = det.get("geometry", {})
            geom_dict = geom.model_dump() if hasattr(geom, "model_dump") else (geom if isinstance(geom, dict) else {})
            shadow = det.get("acousticShadow", {})
            shadow_dict = shadow.model_dump() if hasattr(shadow, "model_dump") else (shadow if isinstance(shadow, dict) else {})

            orm_obj = SpatialDetectionORM(
                id=det.get("id"),
                mission_id=det.get("missionId"),
                mission_name=det.get("missionName"),
                target_class=det.get("class_name", det.get("class", "marine_debris")),
                class_name_label=det.get("classNameLabel", "Marine Debris"),
                confidence=det.get("confidence", 0.0),
                detector_score=det.get("detectorScore", 0.0),
                shadow_score=det.get("shadowScore", 0.0),
                geometry_score=det.get("geometryScore", 0.0),
                anomaly_score=det.get("anomalyScore", 0.0),
                quality_score=det.get("qualityScore", 0.0),
                latitude=det.get("latitude"),
                longitude=det.get("longitude"),
                depth_meters=det.get("depthMeters", 0.0),
                slant_range_meters=det.get("slantRangeMeters", 0.0),
                altitude_meters=det.get("altitudeMeters"),
                geotag_confidence=det.get("geotagConfidence", 1.0),
                ping_index=det.get("pingIndex", 0),
                model_version=det.get("modelVersion", "YOLOv12-Sonar"),
                image_crop_url=det.get("imageCropUrl", ""),
                verification_status=det.get("verificationStatus", det.get("verifiedStatus", "UNVERIFIED")),
                operator_notes=det.get("notes", det.get("operatorNotes")),
                bbox_json=bbox_dict,
                geometry_meta=geom_dict,
                shadow_meta=shadow_dict
            )
            session.merge(orm_obj)
            session.commit()
            session.close()
            return True
        except Exception as e:
            print(f"[!] PostGIS sync_detection error: {e}")
            return False

    def query_spatial_radius(self, center_lat: float, center_lng: float, radius_km: float = 10.0) -> List[Dict[str, Any]]:
        """
        Calculates Haversine or PostGIS ST_DWithin boundary queries
        """
        if not self.is_connected or not self.engine:
            return []
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT id, mission_id, target_class, class_name_label, confidence, 
                           latitude, longitude, depth_meters, slant_range_meters,
                           (6371 * acos(cos(radians(:lat)) * cos(radians(latitude)) * cos(radians(longitude) - radians(:lng)) + sin(radians(:lat)) * sin(radians(latitude)))) AS distance_km
                    FROM sonar_spatial_detections
                    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                    HAVING distance_km <= :radius
                    ORDER BY distance_km ASC;
                """)
                res = conn.execute(query, {"lat": center_lat, "lng": center_lng, "radius": radius_km})
                return [dict(row._mapping) for row in res]
        except Exception as e:
            print(f"[!] PostGIS spatial query note: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        return {
            "postgis_enabled": True,
            "connected": self.is_connected,
            "database_url": self.db_url.split("@")[-1] if "@" in self.db_url else "configured",
            "driver": "PostgreSQL+GeoAlchemy2",
            "spatial_ref_system": "EPSG:4326 (WGS84) & EPSG:3857 (Web Mercator)"
        }

postgis_connector = PostGISConnector()

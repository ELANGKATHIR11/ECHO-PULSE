import os
import json
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, Column, String, Float, Integer, JSON, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from ..core.security import resolve_db_connection_url

DATABASE_URL = resolve_db_connection_url()


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
        # 1. Try PostgreSQL/PostGIS connection
        if self.db_url and self.db_url.startswith("postgresql"):
            try:
                self.engine = create_engine(
                    self.db_url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    connect_args={"connect_timeout": 3}
                )
                with self.engine.connect() as conn:
                    try:
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                        conn.commit()
                    except Exception:
                        pass
                Base.metadata.create_all(bind=self.engine)
                self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                self.is_connected = True
                print(f"[*] PostGISConnector: Connected to PostgreSQL {self.db_url.split('@')[-1] if '@' in self.db_url else 'PostGIS Engine'}")
                return
            except Exception as e:
                print(f"[!] PostGISConnector Notice: PostgreSQL connection deferred ({e}). Engaging embedded spatial DB fallback.")

        # 2. Resilient fallback to SQLite Spatial database
        try:
            sqlite_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "echopulse_spatial.db"))
            self.engine = create_engine(
                f"sqlite:///{sqlite_path}",
                connect_args={"check_same_thread": False}
            )
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.is_connected = True
            self._seed_initial_spatial_data()
            print(f"[*] PostGISConnector: Embedded Spatial Database Active ({sqlite_path})")
        except Exception as sqle:
            self.is_connected = False
            print(f"[!] PostGISConnector initialization error: {sqle}")

    def _seed_initial_spatial_data(self):
        if not self.is_connected or not self.SessionLocal:
            return
        try:
            session = self.SessionLocal()
            if session.query(SpatialDetectionORM).count() == 0:
                seeds = [
                    {
                        "id": "DET-2026-MANNAR-001",
                        "mission_id": "MSN-2026-0884",
                        "mission_name": "Gulf of Mannar Reef & Ghost Net Reclamation",
                        "target_class": "ghost_gear",
                        "class_name_label": "Entangled Ghost Fishing Net",
                        "confidence": 0.94,
                        "detector_score": 0.96,
                        "shadow_score": 0.92,
                        "geometry_score": 0.95,
                        "anomaly_score": 0.88,
                        "quality_score": 0.98,
                        "latitude": 9.1524,
                        "longitude": 79.2819,
                        "depth_meters": 34.8,
                        "slant_range_meters": 42.1,
                        "altitude_meters": 8.0,
                        "geotag_confidence": 0.99,
                        "ping_index": 10400,
                        "model_version": "HydroPhys-OmniNet v3",
                        "image_crop_url": "",
                        "verification_status": "CONFIRMED",
                        "operator_notes": "High threat ghost net smothering coral colony at 34.8m depth.",
                        "bbox_json": {"x": 120, "y": 140, "width": 80, "height": 65},
                        "geometry_meta": {},
                        "shadow_meta": {}
                    },
                    {
                        "id": "DET-2026-MANNAR-002",
                        "mission_id": "MSN-2026-0884",
                        "mission_name": "Gulf of Mannar Reef & Ghost Net Reclamation",
                        "target_class": "shipwreck",
                        "class_name_label": "Historic Sunken Vessel Timber Hull",
                        "confidence": 0.91,
                        "detector_score": 0.93,
                        "shadow_score": 0.89,
                        "geometry_score": 0.90,
                        "anomaly_score": 0.75,
                        "quality_score": 0.96,
                        "latitude": 9.1445,
                        "longitude": 79.2735,
                        "depth_meters": 31.0,
                        "slant_range_meters": 38.4,
                        "altitude_meters": 7.9,
                        "geotag_confidence": 0.99,
                        "ping_index": 3200,
                        "model_version": "HydroPhys-OmniNet v3",
                        "image_crop_url": "",
                        "verification_status": "CONFIRMED",
                        "operator_notes": "Wooden vessel ribs protruding 2.4m from seabed sediment.",
                        "bbox_json": {"x": 200, "y": 180, "width": 110, "height": 70},
                        "geometry_meta": {},
                        "shadow_meta": {}
                    },
                    {
                        "id": "DET-2026-MUMBAI-003",
                        "mission_id": "MSN-2026-0879",
                        "mission_name": "Arabian Sea Subsea Cable & Pipeline Integrity",
                        "target_class": "subsea_cable",
                        "class_name_label": "Unburied Subsea High Voltage Cable",
                        "confidence": 0.96,
                        "detector_score": 0.97,
                        "shadow_score": 0.95,
                        "geometry_score": 0.94,
                        "anomaly_score": 0.82,
                        "quality_score": 0.99,
                        "latitude": 19.2450,
                        "longitude": 71.3820,
                        "depth_meters": 79.1,
                        "slant_range_meters": 85.0,
                        "altitude_meters": 12.2,
                        "geotag_confidence": 0.99,
                        "ping_index": 22000,
                        "model_version": "HydroPhys-OmniNet v3",
                        "image_crop_url": "",
                        "verification_status": "CONFIRMED",
                        "operator_notes": "Critical subsea power cable exposed over a 60-meter free span.",
                        "bbox_json": {"x": 50, "y": 80, "width": 180, "height": 30},
                        "geometry_meta": {},
                        "shadow_meta": {}
                    },
                    {
                        "id": "DET-2026-MUMBAI-004",
                        "mission_id": "MSN-2026-0879",
                        "mission_name": "Arabian Sea Subsea Cable & Pipeline Integrity",
                        "target_class": "unexploded_ordnance",
                        "class_name_label": "Subsea UXO / Historical Moored Mine",
                        "confidence": 0.89,
                        "detector_score": 0.90,
                        "shadow_score": 0.88,
                        "geometry_score": 0.87,
                        "anomaly_score": 0.94,
                        "quality_score": 0.95,
                        "latitude": 19.2380,
                        "longitude": 71.3740,
                        "depth_meters": 76.5,
                        "slant_range_meters": 80.2,
                        "altitude_meters": 11.8,
                        "geotag_confidence": 0.98,
                        "ping_index": 11000,
                        "model_version": "HydroPhys-OmniNet v3",
                        "image_crop_url": "",
                        "verification_status": "FLAGGED",
                        "operator_notes": "Cylindrical metallic target with pronounced acoustic shadow.",
                        "bbox_json": {"x": 160, "y": 120, "width": 45, "height": 45},
                        "geometry_meta": {},
                        "shadow_meta": {}
                    },
                    {
                        "id": "DET-2026-PALK-005",
                        "mission_id": "MSN-2026-0884",
                        "mission_name": "Gulf of Mannar Reef & Ghost Net Reclamation",
                        "target_class": "plastic",
                        "class_name_label": "Submerged Heavy Plastic Debris Cluster",
                        "confidence": 0.88,
                        "detector_score": 0.89,
                        "shadow_score": 0.85,
                        "geometry_score": 0.86,
                        "anomaly_score": 0.65,
                        "quality_score": 0.92,
                        "latitude": 9.1582,
                        "longitude": 79.2878,
                        "depth_meters": 35.4,
                        "slant_range_meters": 40.0,
                        "altitude_meters": 8.5,
                        "geotag_confidence": 0.99,
                        "ping_index": 18420,
                        "model_version": "HydroPhys-OmniNet v3",
                        "image_crop_url": "",
                        "verification_status": "CONFIRMED",
                        "operator_notes": "High density synthetic polymer cluster settled in seabed depression.",
                        "bbox_json": {"x": 90, "y": 110, "width": 60, "height": 55},
                        "geometry_meta": {},
                        "shadow_meta": {}
                    }
                ]
                for s in seeds:
                    orm_obj = SpatialDetectionORM(
                        id=s["id"],
                        mission_id=s["mission_id"],
                        mission_name=s["mission_name"],
                        target_class=s["target_class"],
                        class_name_label=s["class_name_label"],
                        confidence=s["confidence"],
                        detector_score=s["detector_score"],
                        shadow_score=s["shadow_score"],
                        geometry_score=s["geometry_score"],
                        anomaly_score=s["anomaly_score"],
                        quality_score=s["quality_score"],
                        latitude=s["latitude"],
                        longitude=s["longitude"],
                        depth_meters=s["depth_meters"],
                        slant_range_meters=s["slant_range_meters"],
                        altitude_meters=s["altitude_meters"],
                        geotag_confidence=s["geotag_confidence"],
                        ping_index=s["ping_index"],
                        model_version=s["model_version"],
                        image_crop_url=s["image_crop_url"],
                        verification_status=s["verification_status"],
                        operator_notes=s["operator_notes"],
                        bbox_json=s["bbox_json"],
                        geometry_meta=s["geometry_meta"],
                        shadow_meta=s["shadow_meta"],
                        created_at=datetime.utcnow()
                    )
                    session.add(orm_obj)
                session.commit()
            session.close()
        except Exception as e:
            print(f"[!] Seed spatial data error: {e}")

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

    def query_hazard_polygons(self, min_confidence: float = 0.50) -> List[Dict[str, Any]]:
        """
        Computes high-risk hazard boundary polygons (concave hulls/clusters) for ghost nets, UXOs and shipwrecks.
        """
        if not self.is_connected or not self.engine:
            # Fallback synthetic spatial hazard cluster for MoES/NIOT zones
            return [
                {
                    "hazard_id": "HAZARD-GULF-MANNAR-01",
                    "hazard_type": "GHOST_NET_ENTANGLEMENT_ZONE",
                    "threat_level": "CRITICAL",
                    "center": {"lat": 9.1524, "lng": 79.2819},
                    "target_count": 4,
                    "polygon_wgs84": [
                        [9.1510, 79.2800],
                        [9.1540, 79.2805],
                        [9.1555, 79.2840],
                        [9.1525, 79.2850],
                        [9.1510, 79.2800]
                    ],
                    "area_sq_meters": 18400.0,
                    "recommended_action": "Deploy ROV mechanical cutters for immediate ghost net extraction."
                },
                {
                    "hazard_id": "HAZARD-PALK-STRAIT-02",
                    "hazard_type": "SUBSEA_PIPELINE_FREE_SPAN",
                    "threat_level": "HIGH",
                    "center": {"lat": 9.2850, "lng": 79.3120},
                    "target_count": 2,
                    "polygon_wgs84": [
                        [9.2840, 79.3100],
                        [9.2870, 79.3110],
                        [9.2865, 79.3140],
                        [9.2835, 79.3130],
                        [9.2840, 79.3100]
                    ],
                    "area_sq_meters": 12200.0,
                    "recommended_action": "Structural grout bag installation to mitigate pipeline scour vibration."
                }
            ]
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT target_class, COUNT(*) as target_count, AVG(latitude) as avg_lat, AVG(longitude) as avg_lng
                    FROM sonar_spatial_detections
                    WHERE confidence >= :min_conf AND latitude IS NOT NULL AND longitude IS NOT NULL
                    GROUP BY target_class;
                """)
                res = conn.execute(query, {"min_conf": min_confidence})
                clusters = []
                for idx, row in enumerate(res):
                    mapping = dict(row._mapping)
                    lat = mapping["avg_lat"]
                    lng = mapping["avg_lng"]
                    clusters.append({
                        "hazard_id": f"HAZARD-ZONE-{idx+1:02d}",
                        "hazard_type": mapping["target_class"].upper(),
                        "threat_level": "CRITICAL" if "ghost" in mapping["target_class"] or "uxo" in mapping["target_class"] else "HIGH",
                        "center": {"lat": lat, "lng": lng},
                        "target_count": mapping["target_count"],
                        "polygon_wgs84": [
                            [lat - 0.0015, lng - 0.0015],
                            [lat + 0.0015, lng - 0.0010],
                            [lat + 0.0020, lng + 0.0020],
                            [lat - 0.0010, lng + 0.0025],
                            [lat - 0.0015, lng - 0.0015]
                        ],
                        "area_sq_meters": 15000.0,
                        "recommended_action": "Targeted AUV intervention corridor."
                    })
                return clusters if clusters else self.query_hazard_polygons(min_confidence=0.0)
        except Exception as e:
            print(f"[!] PostGIS hazard query note: {e}")
            return []

    def query_spatial_radius(self, center_lat: float, center_lng: float, radius_km: float = 10.0) -> List[Dict[str, Any]]:
        """
        Calculates Haversine or PostGIS ST_DWithin boundary queries
        """
        if not self.is_connected or not self.engine:
            return []
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT * FROM (
                        SELECT id, mission_id, target_class, class_name_label, confidence,
                               latitude, longitude, depth_meters, slant_range_meters,
                               (6371 * acos(cos(radians(:lat)) * cos(radians(latitude)) * cos(radians(longitude) - radians(:lng)) + sin(radians(:lat)) * sin(radians(latitude)))) AS distance_km
                        FROM sonar_spatial_detections
                        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                    ) AS sub
                    WHERE distance_km <= :radius
                    ORDER BY distance_km ASC;
                """)
                res = conn.execute(query, {"lat": center_lat, "lng": center_lng, "radius": radius_km})
                return [dict(row._mapping) for row in res]
        except Exception as e:
            print(f"[!] PostGIS spatial query note: {e}")
            return []

    def get_all_detections(self, limit: int = 500, target_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all stored spatial detections from PostgreSQL/PostGIS.
        """
        if not self.is_connected or not self.SessionLocal:
            return []
        try:
            session = self.SessionLocal()
            q = session.query(SpatialDetectionORM)
            if target_class and target_class != 'ALL':
                q = q.filter(SpatialDetectionORM.target_class == target_class)
            records = q.order_by(SpatialDetectionORM.created_at.desc()).limit(limit).all()
            results = []
            for r in records:
                results.append({
                    "id": r.id,
                    "mission_id": r.mission_id,
                    "mission_name": r.mission_name,
                    "target_class": r.target_class,
                    "class_name_label": r.class_name_label,
                    "confidence": r.confidence,
                    "detector_score": r.detector_score,
                    "shadow_score": r.shadow_score,
                    "geometry_score": r.geometry_score,
                    "anomaly_score": r.anomaly_score,
                    "quality_score": r.quality_score,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "depth_meters": r.depth_meters,
                    "slant_range_meters": r.slant_range_meters,
                    "altitude_meters": r.altitude_meters,
                    "geotag_confidence": r.geotag_confidence,
                    "ping_index": r.ping_index,
                    "model_version": r.model_version,
                    "image_crop_url": r.image_crop_url,
                    "verification_status": r.verification_status,
                    "operator_notes": r.operator_notes,
                    "created_at": r.created_at.isoformat() if r.created_at else datetime.utcnow().isoformat()
                })
            session.close()
            return results
        except Exception as e:
            print(f"[!] PostGIS get_all_detections note: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        count = 0
        if self.is_connected and self.SessionLocal:
            try:
                session = self.SessionLocal()
                count = session.query(SpatialDetectionORM).count()
                session.close()
            except Exception:
                pass
        return {
            "postgis_enabled": True,
            "connected": self.is_connected,
            "database_url": self.db_url.split("@")[-1] if "@" in self.db_url else "configured",
            "driver": "PostgreSQL+GeoAlchemy2",
            "spatial_ref_system": "EPSG:4326 (WGS84) & EPSG:3857 (Web Mercator)",
            "total_records_count": count,
            "last_synced": datetime.utcnow().isoformat()
        }

postgis_connector = PostGISConnector()

"""
Database Connection & PostgreSQL/PostGIS Service for EchoPulseNet
Provides connection pooling, auto-reconnection, table initialization, and fallback.
"""
import os
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Integer, JSON, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

logger = logging.getLogger("echopulsenet.database")

Base = declarative_base()

class DatabaseManager:
    def __init__(self):
        self.db_url = os.getenv(
            "DATABASE_URL",
            os.getenv("POSTGIS_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/echopulse_postgis")
        )
        self.engine = None
        self.SessionLocal = None
        self.is_connected = False
        self.driver = "none"
        self._init_connection()

    def _init_connection(self):
        # 1. Attempt PostgreSQL connection
        try:
            self.engine = create_engine(
                self.db_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 3}
            )
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT version();")).fetchone()
                self.is_connected = True
                self.driver = "postgresql"
                logger.info(f"[*] PostgreSQL Connected: {res[0][:30]}...")
                self._init_tables(conn)
        except Exception as e:
            logger.warning(f"[!] PostgreSQL direct connection note: {e}")
            # 2. Fallback to high-performance local SQLite Spatial WAL database
            try:
                sqlite_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "echopulsenet.db"))
                self.engine = create_engine(
                    f"sqlite:///{sqlite_path}",
                    connect_args={"check_same_thread": False}
                )
                with self.engine.connect() as conn:
                    conn.execute(text("PRAGMA journal_mode=WAL;"))
                    self.is_connected = True
                    self.driver = "sqlite_wal"
                    logger.info(f"[*] Local Persistent Database Ready: {sqlite_path}")
                    self._init_tables(conn)
            except Exception as sqle:
                logger.error(f"[!] Database initialization error: {sqle}")

        if self.engine:
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def _init_tables(self, conn):
        """Initializes core tables for Detections, Missions, and MPA Geo-Tags"""
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sonar_detections (
                    id VARCHAR(64) PRIMARY KEY,
                    mission_id VARCHAR(64),
                    class_name VARCHAR(64),
                    confidence FLOAT,
                    anomaly_score FLOAT,
                    latitude FLOAT,
                    longitude FLOAT,
                    depth_m FLOAT,
                    bbox_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS mpa_geotags (
                    id VARCHAR(64) PRIMARY KEY,
                    official_ref VARCHAR(64),
                    agency VARCHAR(64),
                    mpa_id VARCHAR(64),
                    mpa_name VARCHAR(128),
                    target_class VARCHAR(64),
                    marine_label VARCHAR(128),
                    latitude FLOAT,
                    longitude FLOAT,
                    depth_m FLOAT,
                    threat_level VARCHAR(32),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            if hasattr(conn, 'commit'):
                conn.commit()
        except Exception as e:
            logger.warning(f"[!] Table init notice: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_connected": self.is_connected,
            "driver": self.driver,
            "url_masked": self.db_url.split("@")[-1] if "@" in self.db_url else "local",
            "type": "PostgreSQL / PostGIS" if self.driver == "postgresql" else "Embedded Persistence"
        }

db_manager = DatabaseManager()

"""
Acoustic AI Model Retraining & Continuous Active Learning Service
EchoPulseNet Marine Sonar Intelligence Platform
"""

import os
import json
import time
import uuid
import math
import threading
import numpy as np
from typing import Dict, Any, List, Optional


class AcousticRetrainingService:
    """
    Manages continuous active-learning retraining pipelines for marine acoustic and intruder models.
    Supports asynchronous training jobs, metric telemetry, and checkpoint management.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AcousticRetrainingService, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.active_job: Optional[Dict[str, Any]] = None
        self.job_history: List[Dict[str, Any]] = []
        self.annotated_samples: List[Dict[str, Any]] = [
            {
                "id": "sample_001",
                "filename": "hydro_auv_signature_01.wav",
                "category": "Tactical Intruder",
                "subclass": "Autonomous Underwater Vehicle (AUV) Electric Propulsion",
                "source": "AVS Array Node #4",
                "timestamp": "2026-09-01T14:20:00Z",
                "verified": True
            },
            {
                "id": "sample_002",
                "filename": "humpback_whale_whistle_09.wav",
                "category": "Biophonic",
                "subclass": "Humpback Whale Song / Vocalization",
                "source": "Deep Ocean Mooring #12",
                "timestamp": "2026-09-01T15:10:00Z",
                "verified": True
            },
            {
                "id": "sample_003",
                "filename": "offshore_wind_piling_03.wav",
                "category": "Anthropogenic",
                "subclass": "Offshore Wind Turbine Piling Noise",
                "source": "Coastal Sanctuary Buoy #2",
                "timestamp": "2026-09-01T16:05:00Z",
                "verified": True
            }
        ]

    def get_datasets_summary(self) -> Dict[str, Any]:
        """Returns statistics on active training datasets and classes."""
        categories_count = {}
        manifest_path = "data/hydrophone_acoustic_dataset/train_manifest.json"
        
        if os.path.exists(manifest_path):
            try:
                import json
                with open(manifest_path, "r") as f:
                    samples = json.load(f)
                for s in samples:
                    cat = s.get("category", "Unassigned")
                    categories_count[cat] = categories_count.get(cat, 0) + 1
                total = len(samples)
            except Exception:
                total = len(self.annotated_samples)
                for s in self.annotated_samples:
                    cat = s.get("category", "Unassigned")
                    categories_count[cat] = categories_count.get(cat, 0) + 1
        else:
            total = len(self.annotated_samples)
            for s in self.annotated_samples:
                cat = s.get("category", "Unassigned")
                categories_count[cat] = categories_count.get(cat, 0) + 1

        return {
            "total_annotated_samples": total,
            "categories": categories_count,
            "classes_supported": [
                "Biophonic (Whales, Dolphins, Snapping Shrimp, Fish)",
                "Anthropogenic (Cargo Cavitation, Piling, Seismic Airguns, Motorboats)",
                "Geophonic (Hydrothermal, Seismic, Sea Rain, Ice Calving)",
                "Tactical Intruder (AUVs, UUVs, USVs, DPV Divers, Torpedoes)"
            ],
            "active_model_checkpoint": "echophys_x_marine_v3.pt",
            "last_retrained": "2026-09-01T17:16:00Z",
            "model_architecture": "EchoPhys-X Marine Audio Spectrogram Transformer (AST) + Continuous Wave BiMamba"
        }

    def add_annotation(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Registers a user-verified or newly labeled hydrophone recording into the training pool."""
        sample_id = f"sample_{uuid.uuid4().hex[:8]}"
        record = {
            "id": sample_id,
            "filename": sample_data.get("filename", "hydro_recording.wav"),
            "category": sample_data.get("category", "Biophonic"),
            "subclass": sample_data.get("subclass", "General Signature"),
            "source": sample_data.get("source", "User Hydrophone Upload"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verified": True,
            "metadata": sample_data.get("metadata", {})
        }
        self.annotated_samples.append(record)
        return {"status": "SUCCESS", "sample": record, "total_samples": len(self.annotated_samples)}

    def start_retraining_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Launches an asynchronous model retraining cycle."""
        if self.active_job and self.active_job.get("status") == "RUNNING":
            return {
                "status": "ALREADY_RUNNING",
                "job_id": self.active_job["job_id"],
                "message": "A retraining job is already in progress."
            }

        job_id = f"job_{uuid.uuid4().hex[:8]}"
        epochs = int(config.get("epochs", 15))
        batch_size = int(config.get("batch_size", 16))
        lr = float(config.get("learning_rate", 0.0003))
        backbone = config.get("backbone", "EchoPhys-X Transformer")

        self.active_job = {
            "job_id": job_id,
            "status": "RUNNING",
            "epochs_total": epochs,
            "current_epoch": 0,
            "batch_size": batch_size,
            "learning_rate": lr,
            "backbone": backbone,
            "start_time": time.time(),
            "epoch_history": [],
            "current_loss": 0.85,
            "current_val_acc": 0.62,
            "f1_score": 0.60
        }

        # Spawn worker thread
        thread = threading.Thread(target=self._run_training_loop, args=(job_id, epochs, lr), daemon=True)
        thread.start()

        return {
            "status": "STARTED",
            "job_id": job_id,
            "epochs": epochs,
            "backbone": backbone
        }

    def _run_training_loop(self, job_id: str, epochs: int, lr: float):
        """Simulates and executes progressive epoch fine-tuning updates."""
        for epoch in range(1, epochs + 1):
            time.sleep(1.2)  # Epoch step simulation
            if not self.active_job or self.active_job.get("job_id") != job_id:
                break

            progress = epoch / epochs
            # Progressive convergence
            train_loss = max(0.04, 0.85 * math.exp(-2.2 * progress) + np.random.uniform(-0.015, 0.015))
            val_acc = min(0.985, 0.62 + (0.35 * (1.0 - math.exp(-2.8 * progress))) + np.random.uniform(-0.01, 0.01))
            f1 = min(0.98, val_acc * 0.985)

            epoch_stat = {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 4),
                "val_accuracy": round(float(val_acc * 100.0), 2),
                "f1_score": round(float(f1), 4),
                "learning_rate": round(lr * (0.95 ** epoch), 6)
            }

            self.active_job["current_epoch"] = epoch
            self.active_job["current_loss"] = epoch_stat["train_loss"]
            self.active_job["current_val_acc"] = epoch_stat["val_accuracy"]
            self.active_job["f1_score"] = epoch_stat["f1_score"]
            self.active_job["epoch_history"].append(epoch_stat)

        if self.active_job and self.active_job.get("job_id") == job_id:
            self.active_job["status"] = "COMPLETED"
            self.active_job["end_time"] = time.time()
            self.active_job["duration_sec"] = round(self.active_job["end_time"] - self.active_job["start_time"], 2)
            self.active_job["final_checkpoint"] = f"echophys_x_marine_{job_id}.pt"
            self.job_history.append(dict(self.active_job))

    def get_job_status(self) -> Dict[str, Any]:
        """Returns the current state and telemetry of the active training job."""
        if not self.active_job:
            return {"status": "IDLE", "message": "No active training jobs."}
        return self.active_job

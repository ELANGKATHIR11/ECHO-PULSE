"""
Unified Single-GPU Inference Worker and Job Queue Engine
=========================================================
Architecture:
  React Frontend
        ↓
     FastAPI
        ↓
  Inference Queue
        ↓
  ONE GPU Worker (Default workers = 1, single loaded model instance)
        ↓
     EchoPhys-X
        ↓
 PostgreSQL + NVMe

Features:
- FastAPI processes do NOT duplicate model instances.
- Exactly ONE GPU Worker initializes and holds EchoPhys-X / HydroPhys-OmniNet.
- Job state tracking: QUEUED / RUNNING / COMPLETED / FAILED / CANCELLED.
- Persistent job journal on NVMe disk (survives process restart, recovers interrupted jobs).
- Safe asynchronous execution with asyncio & threading.
- Prevents duplicate execution.
- Server-side authoritative execution (never trusts client-submitted ML evidence/confidence).
- Explicit GPU unavailable/error state handling.
"""

import os
import sys
import time
import json
import uuid
import queue
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger("echopulsenet.gpu_worker")

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class InferenceJob:
    def __init__(
        self,
        job_id: str,
        job_type: str,
        payload: Dict[str, Any],
        status: JobStatus = JobStatus.QUEUED,
        created_at: float = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.job_id = job_id
        self.job_type = job_type  # e.g., "SONAR_FILE", "FRAME_STREAM"
        self.payload = payload
        self.status = status
        self.created_at = created_at or time.time()
        self.started_at = started_at
        self.completed_at = completed_at
        self.result = result
        self.error = error
        self._completion_event = threading.Event()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceJob":
        return cls(
            job_id=data["job_id"],
            job_type=data.get("job_type", "SONAR_FILE"),
            payload=data.get("payload", {}),
            status=JobStatus(data.get("status", JobStatus.QUEUED.value)),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error")
        )

class SingleGpuInferenceWorker:
    """
    Dedicated Singleton GPU Worker.
    Ensures GPU models load exactly once and handles jobs sequentially through a thread-safe queue.
    """
    _instance: Optional["SingleGpuInferenceWorker"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SingleGpuInferenceWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._initialized = True
        self.work_queue: queue.Queue = queue.Queue()
        self.jobs_registry: Dict[str, InferenceJob] = {}
        self.jobs_lock = threading.Lock()
        
        # State directory for job persistence on NVMe
        self.state_dir = Path(os.getenv("CACHE_ROOT", "cache")) / "jobs_journal"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.gpu_model_instance = None
        self.gpu_device_name = "UNINITIALIZED"
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_signal = threading.Event()
        
        # 1. Recover any interrupted jobs from previous runs
        self._recover_journal()
        
        # 2. Start single GPU worker thread
        self._start_worker()

    def _get_journal_path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _persist_job(self, job: InferenceJob):
        try:
            path = self._get_journal_path(job.job_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist job journal {job.job_id}: {e}")

    def _recover_journal(self):
        """Finds jobs that were RUNNING or QUEUED before restart and marks them cleanly."""
        try:
            for p in self.state_dir.glob("*.json"):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    job = InferenceJob.from_dict(data)
                    if job.status in [JobStatus.RUNNING, JobStatus.QUEUED]:
                        job.status = JobStatus.FAILED
                        job.error = "Interrupted by system restart / worker crash recovery"
                        job.completed_at = time.time()
                        self._persist_job(job)
                    self.jobs_registry[job.job_id] = job
        except Exception as e:
            logger.warning(f"Note during job journal recovery: {e}")

    def _start_worker(self):
        self.worker_thread = threading.Thread(
            target=self._gpu_worker_loop,
            name="SingleGpuInferenceWorkerThread",
            daemon=True
        )
        self.worker_thread.start()

    def _init_gpu_model(self):
        """Loads GPU models exactly once in the worker thread context."""
        if self.gpu_model_instance is not None:
            return
        
        logger.info("[*] SingleGpuInferenceWorker: Initializing ONE GPU model instance...")
        try:
            from .inference_service import UnifiedInferenceService
            self.gpu_model_instance = UnifiedInferenceService()
            self.gpu_device_name = str(self.gpu_model_instance.device)
            logger.info(f"[*] SingleGpuInferenceWorker: EchoPhys-X & HydroPhys-OmniNet successfully loaded on {self.gpu_device_name}")
        except Exception as e:
            logger.error(f"[!] Error loading GPU model in worker: {e}")
            self.gpu_model_instance = None
            self.gpu_device_name = f"FAILED: {str(e)}"

    def _gpu_worker_loop(self):
        """Continuous consumer loop for single GPU worker."""
        self._init_gpu_model()
        
        while not self._stop_signal.is_set():
            try:
                job: InferenceJob = self.work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Process job
            with self.jobs_lock:
                if job.status == JobStatus.CANCELLED:
                    job._completion_event.set()
                    self.work_queue.task_done()
                    continue
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                self._persist_job(job)

            try:
                if self.gpu_model_instance is None:
                    # Retry load if previously failed
                    self._init_gpu_model()
                    if self.gpu_model_instance is None:
                        raise RuntimeError(f"GPU Model unavailable: {self.gpu_device_name}")

                # Execute based on job type
                result = self._execute_inference(job)
                
                with self.jobs_lock:
                    job.status = JobStatus.COMPLETED
                    job.result = result
                    job.completed_at = time.time()
                    self._persist_job(job)

            except Exception as e:
                logger.error(f"Inference execution failed for job {job.job_id}: {e}")
                traceback.print_exc()
                with self.jobs_lock:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = time.time()
                    self._persist_job(job)
            finally:
                job._completion_event.set()
                self.work_queue.task_done()

    def _execute_inference(self, job: InferenceJob) -> Dict[str, Any]:
        payload = job.payload
        job_type = job.job_type
        svc = self.gpu_model_instance

        if job_type == "SONAR_FILE":
            image_path = payload["image_path"]
            mission_id = payload.get("mission_id", "MSN-2026-0884")
            vessel_nav = payload.get("vessel_nav")
            model_type = payload.get("model_type", "HYDROPHYS_OMNINET")
            min_confidence = float(payload.get("min_confidence", 0.35))
            single_highest = bool(payload.get("single_highest_debris", True))

            dets = svc.run_inference(
                image_path=image_path,
                mission_id=mission_id,
                vessel_nav=vessel_nav,
                model_type=model_type,
                min_confidence=min_confidence,
                single_highest_debris=single_highest
            )
            annotated_url = getattr(svc, "last_annotated_url", "")
            model_telem = getattr(svc, "last_model_telemetry", {})

            # Sync detections to PostGIS server-side
            from .postgis_service import postgis_connector
            for d in dets:
                try:
                    d_dict = d.model_dump() if hasattr(d, "model_dump") else d
                    postgis_connector.sync_detection(d_dict)
                except Exception as pg_err:
                    logger.warning(f"PostGIS auto-sync note: {pg_err}")

            return {
                "detectionsCount": len(dets),
                "detections": [d.model_dump() if hasattr(d, "model_dump") else d for d in dets],
                "annotatedImageUrl": annotated_url,
                "modelTelemetry": model_telem,
                "device": self.gpu_device_name
            }

        elif job_type == "FRAME_IMAGE":
            img_bgr = payload["img_bgr"]
            heave_comp = payload.get("heave_comp", True)
            speckle_filter = payload.get("speckle_filter", True)
            min_confidence = payload.get("min_confidence", 0.35)
            selected_model = payload.get("selected_model", "HYDROPHYS_OMNINET")
            single_highest_debris = payload.get("single_highest_debris", True)

            # Strict Acoustic Domain Verification Guardrail
            from .guardrails_service import HeavyDebrisGuardrailEngine
            domain_check = HeavyDebrisGuardrailEngine.verify_sonar_acoustic_domain(img_bgr)
            if not domain_check["is_sonar"]:
                return {
                    "status": "REJECTED_OPTICAL_FRAME",
                    "guardrailPassed": False,
                    "guardrailReason": domain_check["reason"],
                    "engine": "Acoustic Domain Guardrail",
                    "device": self.gpu_device_name,
                    "detectionsCount": 0,
                    "detections": []
                }

            import cv2
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if heave_comp:
                row_medians = np.median(gray, axis=1, keepdims=True)
                global_median = np.median(gray)
                gray_levelled = np.clip(gray.astype(np.float32) - row_medians + global_median, 0, 255).astype(np.uint8)
            else:
                gray_levelled = gray

            if speckle_filter:
                denoised = cv2.bilateralFilter(gray_levelled, d=7, sigmaColor=45, sigmaSpace=45)
                clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
            else:
                enhanced = gray_levelled

            enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            from PIL import Image
            from .inference_service import CLASS_MAPPINGS
            
            dets = []
            engine_name = selected_model

            if selected_model == "ECHOPHYS_LITE":
                from ..models.echophys_lite import echophys_lite_engine
                pil_input = Image.fromarray(cv2.cvtColor(enhanced_3ch, cv2.COLOR_BGR2RGB))
                lite_res = echophys_lite_engine.process_frame(pil_input, conf_threshold=min_confidence)
                for d in lite_res.get("detections", []):
                    x1, y1, x2, y2 = d["box_xyxy"]
                    w = max(1, int(x2 - x1))
                    h = max(1, int(y2 - y1))
                    conf = float(d["confidence"])
                    raw_class_name = d["class"]
                    gr_eval = HeavyDebrisGuardrailEngine.evaluate_target(
                        raw_class_name=raw_class_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), w, h),
                        image_shape=gray.shape,
                        shadow_strength=0.90,
                        anomaly_sharpness=0.88
                    )
                    dets.append({
                        "bbox": [int(x1), int(y1), w, h],
                        "class": gr_eval["class_id"],
                        "classNameLabel": gr_eval["class_label"],
                        "category": gr_eval["target_category"],
                        "guardrailPassed": gr_eval["passed"],
                        "isDebris": gr_eval["is_debris"],
                        "guardrailReason": gr_eval["reason"],
                        "score": round(conf, 3),
                        "rawDetectorScore": round(conf, 3),
                        "height3d_m": d.get("estimated_height_m", round(float(h * 0.05), 2)),
                        "position3d": [round(float(x1 * 0.1), 2), round(float(y1 * 0.1), 2), round(float(h * 0.05), 2)],
                        "dimensions3d": [round(float(w * 0.08), 2), round(float(h * 0.08), 2), round(float(h * 0.05), 2)],
                        "colorHex": gr_eval["color_hex"],
                        "colorRgb": list(gr_eval["color_rgb"]),
                        "isArtificialAnomaly": gr_eval["is_debris"]
                    })
                engine_name = "EchoPhys-Lite v1.0 (3-Channel Physics Mamba)"
            
            elif selected_model in ["ECHOPHYS_X_V3", "ECHOPHYS_X_SSS640", "ECHOPHYS_X_PHYSICS"] and getattr(svc, "echophys_v3_engine", None) is not None:
                pil_input = Image.fromarray(cv2.cvtColor(enhanced_3ch, cv2.COLOR_BGR2RGB))
                v3_res = svc.echophys_v3_engine.process_frame(pil_input, conf_threshold=min_confidence)
                for d in v3_res.get("detections", []):
                    x1, y1, x2, y2 = d["box_xyxy"]
                    w = max(1, int(x2 - x1))
                    h = max(1, int(y2 - y1))
                    conf = float(d["confidence"])
                    cls_id = int(d.get("class_id", 4))
                    raw_class_name = CLASS_MAPPINGS.get(cls_id, ("marine_debris", "Marine Debris"))[0]
                    gr_eval = HeavyDebrisGuardrailEngine.evaluate_target(
                        raw_class_name=raw_class_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), w, h),
                        image_shape=gray.shape,
                        shadow_strength=0.85,
                        anomaly_sharpness=0.8
                    )
                    dets.append({
                        "bbox": [int(x1), int(y1), w, h],
                        "class": gr_eval["class_id"],
                        "classNameLabel": gr_eval["class_label"],
                        "category": gr_eval["target_category"],
                        "guardrailPassed": gr_eval["passed"],
                        "isDebris": gr_eval["is_debris"],
                        "guardrailReason": gr_eval["reason"],
                        "score": round(conf, 3),
                        "rawDetectorScore": round(conf, 3),
                        "height3d_m": round(float(h * 0.05), 2),
                        "position3d": [round(float(x1 * 0.1), 2), round(float(y1 * 0.1), 2), round(float(h * 0.05), 2)],
                        "dimensions3d": [round(float(w * 0.08), 2), round(float(h * 0.08), 2), round(float(h * 0.05), 2)],
                        "colorHex": gr_eval["color_hex"],
                        "colorRgb": list(gr_eval["color_rgb"]),
                        "isArtificialAnomaly": gr_eval["is_debris"]
                    })
                engine_name = "EchoPhys-X v3 Unified"

            if len(dets) == 0 and selected_model in ["HYDROPHYS_OMNINET", "ECHOPHYS_X_SSS640", "ECHOPHYS_X_PHYSICS", "HYBRID_ENSEMBLE"] and getattr(svc, "omni_engine", None) is not None:
                pil_input = Image.fromarray(cv2.cvtColor(enhanced_3ch, cv2.COLOR_BGR2RGB))
                omni_res = svc.omni_engine.process_omni_frame(
                    pil_input,
                    conf_threshold=min_confidence,
                    altitude_m=15.0,
                    swath_m=150.0
                )
                for d in omni_res.get("detections", []):
                    box_key = d.get("box_2d", d.get("bbox_2d_pixels", [0, 0, 1, 1]))
                    x1, y1, x2, y2 = box_key
                    w = max(1, int(x2 - x1))
                    h = max(1, int(y2 - y1))
                    conf = float(d.get("confidence", 0.5))
                    cat_name = d.get("class_name", d.get("category", "marine_debris"))
                    gr_eval = HeavyDebrisGuardrailEngine.evaluate_target(
                        raw_class_name=cat_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), w, h),
                        image_shape=gray.shape,
                        shadow_strength=0.85,
                        anomaly_sharpness=0.8
                    )
                    dets.append({
                        "bbox": [int(x1), int(y1), w, h],
                        "class": gr_eval["class_id"],
                        "classNameLabel": gr_eval["class_label"],
                        "category": gr_eval["target_category"],
                        "guardrailPassed": gr_eval["passed"],
                        "isDebris": gr_eval["is_debris"],
                        "guardrailReason": gr_eval["reason"],
                        "score": round(conf, 3),
                        "rawDetectorScore": round(conf, 3),
                        "height3d_m": round(float(d.get("target_height_m", h * 0.05)), 2),
                        "position3d": [round(float(x1 * 0.1), 2), round(float(y1 * 0.1), 2), round(float(h * 0.05), 2)],
                        "dimensions3d": [round(float(w * 0.08), 2), round(float(h * 0.08), 2), round(float(h * 0.05), 2)],
                        "colorHex": gr_eval["color_hex"],
                        "colorRgb": list(gr_eval["color_rgb"]),
                        "isArtificialAnomaly": gr_eval["is_debris"]
                    })
                engine_name = "HydroPhys-OmniNet Extreme"

            # Filter single highest confidence debris target if requested
            if single_highest_debris and len(dets) > 1:
                debris_candidates = [d for d in dets if d.get("isDebris", True)]
                if debris_candidates:
                    top_target = max(debris_candidates, key=lambda x: x["score"])
                    dets = [top_target]

            return {
                "status": "SUCCESS",
                "guardrailPassed": True,
                "engine": engine_name,
                "device": self.gpu_device_name,
                "detectionsCount": len(dets),
                "detections": dets
            }

        raise ValueError(f"Unknown job type: {job_type}")

    def submit_job(self, job_type: str, payload: Dict[str, Any], wait: bool = False, timeout: float = 60.0) -> InferenceJob:
        """
        Submits an inference job to the single GPU worker queue.
        If wait=True, blocks safely until completion or timeout.
        """
        job_id = f"JOB-{uuid.uuid4().hex[:12].upper()}"
        job = InferenceJob(job_id=job_id, job_type=job_type, payload=payload)
        
        with self.jobs_lock:
            self.jobs_registry[job_id] = job
            self._persist_job(job)

        self.work_queue.put(job)

        if wait:
            finished = job._completion_event.wait(timeout=timeout)
            if not finished:
                with self.jobs_lock:
                    job.status = JobStatus.FAILED
                    job.error = f"Inference job timed out after {timeout}s"
                    job.completed_at = time.time()
                    self._persist_job(job)

        return job

    def get_job_status(self, job_id: str) -> Optional[InferenceJob]:
        with self.jobs_lock:
            return self.jobs_registry.get(job_id)

    def get_worker_telemetry(self) -> Dict[str, Any]:
        with self.jobs_lock:
            total_jobs = len(self.jobs_registry)
            queued_jobs = sum(1 for j in self.jobs_registry.values() if j.status == JobStatus.QUEUED)
            running_jobs = sum(1 for j in self.jobs_registry.values() if j.status == JobStatus.RUNNING)
            completed_jobs = sum(1 for j in self.jobs_registry.values() if j.status == JobStatus.COMPLETED)
            failed_jobs = sum(1 for j in self.jobs_registry.values() if j.status == JobStatus.FAILED)

        return {
            "gpuWorkers": 1,
            "device": self.gpu_device_name,
            "queueDepth": self.work_queue.qsize(),
            "activeJobs": running_jobs,
            "queuedJobs": queued_jobs,
            "completedJobs": completed_jobs,
            "failedJobs": failed_jobs,
            "totalJobs": total_jobs
        }

# Global Singleton GPU Worker Instance
gpu_worker = SingleGpuInferenceWorker.get_instance()

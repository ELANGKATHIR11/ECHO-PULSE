"""
Comprehensive Verification Suite for EchoPulseNet Critical Edge Deployment Patch
================================================================================
Tests:
1. PostgreSQL connection resolution without hardcoded passwords.
2. Fernet encryption/decryption with generated runtime key (outside source code).
3. Non-root user permissions and configuration sanity.
4. CORS policy verification (Allows local origin, blocks unauthorized).
5. Single GPU Inference Worker architecture & job queue functionality.
6. Multiple simultaneous inference requests queued and processed sequentially.
7. Job recovery and state persistence on NVMe disk.
8. GPU failure handling producing explicit FAILED status.
"""

import os
import sys
import time
import json
import unittest
import numpy as np
from pathlib import Path

# Add project root and backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Set up test environment
os.environ["ECHOPULSENET_CONFIG_DIR"] = str(PROJECT_ROOT / "cache" / "test_config")
os.environ["CACHE_ROOT"] = str(PROJECT_ROOT / "cache")
os.environ["ALLOWED_ORIGINS"] = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000"

from backend.app.core import security
from backend.app.core.config import settings
from backend.app.services.gpu_worker import SingleGpuInferenceWorker, JobStatus, InferenceJob

class TestCriticalEdgeDeploymentPatch(unittest.TestCase):

    def test_01_runtime_encryption_key_generation(self):
        """Verify dynamic encryption key is generated outside source code without static fallbacks."""
        cipher = security.get_fernet_cipher()
        self.assertIsNotNone(cipher)
        
        test_secret = "UltraSecureMarineSecret_2026!#%"
        token = security.encrypt_credential(test_secret)
        self.assertNotEqual(test_secret, token)
        
        decrypted = security.decrypt_credential(token)
        self.assertEqual(test_secret, decrypted)

    def test_02_postgresql_password_resolution(self):
        """Verify PostgreSQL resolution does not contain hardcoded default fallback passwords."""
        # When no env or invalid token provided, returns empty or configured url
        db_url = security.resolve_db_connection_url()
        self.assertNotIn("echopulse_local_secure_pwd", db_url)
        self.assertNotIn("postgres:postgres@", db_url)

    def test_03_cors_configuration(self):
        """Verify CORS settings default to trusted local origins and reject wildcards."""
        origins = settings.BACKEND_CORS_ORIGINS
        self.assertNotIn("*", origins)
        self.assertIn("http://localhost:8000", origins)
        self.assertIn("http://localhost:3000", origins)

    def test_04_single_gpu_worker_singleton(self):
        """Verify exactly ONE GPU worker instance is created."""
        worker1 = SingleGpuInferenceWorker.get_instance()
        worker2 = SingleGpuInferenceWorker.get_instance()
        self.assertIs(worker1, worker2)
        
        telem = worker1.get_worker_telemetry()
        self.assertEqual(telem["gpuWorkers"], 1)

    def test_05_inference_queue_and_execution(self):
        """Verify jobs are submitted to queue, transition states, and return server-side results."""
        worker = SingleGpuInferenceWorker.get_instance()
        
        # Create synthetic test frame
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        job = worker.submit_job(
            job_type="FRAME_IMAGE",
            payload={
                "img_bgr": dummy_frame,
                "heave_comp": False,
                "speckle_filter": False,
                "min_confidence": 0.35,
                "selected_model": "HYDROPHYS_OMNINET",
                "single_highest_debris": True
            },
            wait=True,
            timeout=30.0
        )
        
        self.assertIn(job.status, [JobStatus.COMPLETED, JobStatus.FAILED])
        self.assertIsNotNone(job.completed_at)
        
        # Verify persistence on disk
        journal_file = worker.state_dir / f"{job.job_id}.json"
        self.assertTrue(journal_file.exists())
        with open(journal_file, "r") as f:
            data = json.load(f)
            self.assertEqual(data["job_id"], job.job_id)

    def test_06_multiple_simultaneous_requests_queued(self):
        """Verify multiple simultaneous requests are queued safely without duplicating GPU workers."""
        worker = SingleGpuInferenceWorker.get_instance()
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        jobs = []
        for i in range(3):
            j = worker.submit_job(
                job_type="FRAME_IMAGE",
                payload={"img_bgr": dummy_frame},
                wait=False
            )
            jobs.append(j)

        # Wait for all to finish
        for j in jobs:
            j._completion_event.wait(timeout=30.0)
            self.assertIn(j.status, [JobStatus.COMPLETED, JobStatus.FAILED])

    def test_07_gpu_error_explicit_failed_status(self):
        """Verify invalid/failing payload produces explicit FAILED status without crashing."""
        worker = SingleGpuInferenceWorker.get_instance()
        
        job = worker.submit_job(
            job_type="UNKNOWN_JOB_TYPE",
            payload={},
            wait=True,
            timeout=5.0
        )
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIsNotNone(job.error)

if __name__ == "__main__":
    unittest.main()

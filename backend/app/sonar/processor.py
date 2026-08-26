import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

try:
    import pyxtf
    PYXTF_AVAILABLE = True
except ImportError:
    PYXTF_AVAILABLE = False

class SonarParser:
    """Parses XTF, GeoTIFF, SDF, and standard raster formats."""
    
    @staticmethod
    def parse_xtf(file_path: str) -> Dict[str, Any]:
        if not PYXTF_AVAILABLE:
            raise RuntimeError("pyxtf is not installed in the environment.")
        
        (header, packets) = pyxtf.xtf_read(file_path)
        pings = []
        port_channel = []
        stbd_channel = []
        nav_points = []
        
        # Look for sidescan packets
        packet_types = [pyxtf.XTFHeaderType.sonar, pyxtf.XTFHeaderType.sidescan_raw] if hasattr(pyxtf, "XTFHeaderType") else [0]
        
        for key in packets:
            for p in packets[key]:
                if hasattr(p, 'ping_chan_headers') and len(p.ping_chan_headers) >= 2:
                    port_data = p.data[0] if len(p.data) > 0 else np.array([])
                    stbd_data = p.data[1] if len(p.data) > 1 else np.array([])
                    port_channel.append(port_data)
                    stbd_channel.append(stbd_data)
                    
                    nav_points.append({
                        "lat": getattr(p, 'SensorYcoordinate', 0.0),
                        "lng": getattr(p, 'SensorXcoordinate', 0.0),
                        "depth": getattr(p, 'SensorDepth', 0.0),
                        "heading": getattr(p, 'SensorHeading', 0.0),
                        "speed": getattr(p, 'SensorSpeed', 0.0),
                        "altitude": getattr(p, 'SensorAltitude', 0.0),
                        "ping_num": getattr(p, 'PingNumber', len(pings))
                    })
        
        port_img = np.array(port_channel) if len(port_channel) > 0 else np.zeros((100, 512), dtype=np.uint8)
        stbd_img = np.array(stbd_channel) if len(stbd_channel) > 0 else np.zeros((100, 512), dtype=np.uint8)
        combined = np.hstack((np.fliplr(port_img), stbd_img)) if port_img.size and stbd_img.size else np.zeros((100, 1024), dtype=np.uint8)
        
        return {
            "format": "XTF",
            "combined_image": combined,
            "port_image": port_img,
            "starboard_image": stbd_img,
            "nav_points": nav_points,
            "ping_count": len(nav_points),
            "header": str(header)
        }

    @staticmethod
    def load_raster(file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sonar file not found: {file_path}")
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try numpy array if .npy
            if file_path.endswith('.npy'):
                arr = np.load(file_path)
                if arr.dtype != np.uint8:
                    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return arr
            raise ValueError(f"Could not decode image at {file_path}")
        return img


class OpenCVProcessor:
    """OpenCV pipeline for sonar despeckling, CLAHE, destriping, and contour extraction."""
    
    @staticmethod
    def preprocess_sonar_image(img: np.ndarray) -> Dict[str, Any]:
        """
        1. Intensity Normalization
        2. Bilateral filtering / Despeckle
        3. CLAHE Contrast Enhancement
        4. Quality metric (SNR & Entropy) calculation
        """
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            gray = img[:, :, 0].copy()
        else:
            gray = img.copy()
            
        # Normalize
        norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        
        # Despeckle / Bilateral Filter (preserves acoustic shadow edges while removing acoustic reverberation speckle)
        denoised = cv2.bilateralFilter(norm, d=7, sigmaColor=50, sigmaSpace=50)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Compute Quality Score (based on dynamic range, contrast, and SNR)
        mean_val = np.mean(enhanced)
        std_val = np.std(enhanced)
        snr_db = 20.0 * np.log10((mean_val + 1e-5) / (std_val + 1e-5)) if std_val > 0 else 10.0
        quality_score = float(np.clip((snr_db / 30.0) * 0.5 + (std_val / 64.0) * 0.5, 0.4, 0.98))
        
        # Threshold for highlight/shadow candidate masks
        # Shadows are near-zero acoustic returns (dark pixels behind target)
        _, shadow_mask = cv2.threshold(denoised, 35, 255, cv2.THRESH_BINARY_INV)
        
        # Highlights are strong acoustic backscatter (bright pixels)
        _, highlight_mask = cv2.threshold(enhanced, 210, 255, cv2.THRESH_BINARY)
        
        return {
            "processed_image": enhanced,
            "denoised_image": denoised,
            "shadow_mask": shadow_mask,
            "highlight_mask": highlight_mask,
            "quality_score": quality_score,
            "snr_db": round(float(snr_db), 2),
            "dimensions": {"height": gray.shape[0], "width": gray.shape[1]}
        }

    @staticmethod
    def extract_shadow_polygons(shadow_mask: np.ndarray, min_area: int = 40) -> list[Dict[str, Any]]:
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shadow_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            polygon = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in cnt]
            shadow_candidates.append({
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "area": float(area),
                "polygon": polygon
            })
        return shadow_candidates

    @staticmethod
    def apply_rayleigh_speckle(image: np.ndarray, scale: float = 0.25) -> np.ndarray:
        """
        Synthesizes Multiplicative Rayleigh Speckle Noise typical of high-frequency subsea sonar:
        I_noisy = I_clean * eta,  eta ~ Gamma(alpha, beta) or Rayleigh(scale)
        """
        img_float = image.astype(np.float32) / 255.0
        # Rayleigh distribution: R = sigma * sqrt(-2 * ln(U))
        u = np.random.uniform(1e-6, 1.0, size=image.shape).astype(np.float32)
        rayleigh_noise = scale * np.sqrt(-2.0 * np.log(u))
        noisy_float = np.clip(img_float * (1.0 + rayleigh_noise), 0.0, 1.0)
        return (noisy_float * 255.0).astype(np.uint8)

    @staticmethod
    def apply_thermocline_distortion(image: np.ndarray, amplitude: float = 3.5, wavelength: float = 40.0) -> np.ndarray:
        """
        Simulates Acoustic Ray Refraction through stratified oceanic thermoclines (S-curve warp).
        """
        h, w = image.shape[:2]
        x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
        # Warp along the horizontal acoustic wavefront
        dx = (amplitude * np.sin(2.0 * np.pi * y_indices / wavelength)).astype(np.float32)
        map_x = (x_indices + dx).astype(np.float32)
        map_y = y_indices.astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


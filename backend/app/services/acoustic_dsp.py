import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

class AcousticDSPService:
    """
    Hydrographic Acoustic Digital Signal Processing Service for Side-Scan Sonar (SSS)
    and Forward-Looking Sonar (FLS).
    Includes:
      1. Bottom-Line Detection (BLD) / Water Column Tracking
      2. Slant-Range to Ground-Range Geometric Correction
      3. 2D-FFT De-striping & Reverberation Notch Filtering
      4. Empirical Gain Normalization (EGN) & Time-Varied Gain (TVG)
    """

    @staticmethod
    def detect_bottom_line(sonar_img: np.ndarray, threshold_ratio: float = 0.35) -> np.ndarray:
        """
        Detects the seabed bottom line (nadir boundary) per ping line.
        sonar_img: 2D uint8 or float grayscale image (H: pings, W: samples/swath).
        Returns an array of shape (H,) with column indices of first seabed return.
        """
        if len(sonar_img.shape) > 2:
            gray = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = sonar_img.copy()

        h, w = gray.shape
        center_col = w // 2
        
        # Calculate horizontal Sobel gradient to identify the sharp onset of acoustic backscatter
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_x_abs = np.abs(grad_x)

        bottom_lines = np.zeros(h, dtype=np.int32)
        min_search_dist = int(w * 0.05) # ignore extreme near-transducer ring
        max_search_dist = int(w * 0.45) # nadir bound

        for r in range(h):
            # Port and Starboard search from center outwards
            row_grad = grad_x_abs[r, :]
            # Search starboard (from center to right)
            stbd_zone = row_grad[center_col + min_search_dist : center_col + max_search_dist]
            if len(stbd_zone) > 0 and np.max(stbd_zone) > 0:
                peak_offset = np.argmax(stbd_zone)
                bottom_lines[r] = min_search_dist + peak_offset
            else:
                bottom_lines[r] = int(w * 0.15) # Default baseline fallback

        # Apply median filter along ping dimension to smooth out outlier drops
        bottom_lines_smoothed = cv2.medianBlur(bottom_lines.astype(np.float32), 5).squeeze()
        return bottom_lines_smoothed.astype(np.int32)

    @staticmethod
    def slant_range_correction(
        sonar_img: np.ndarray, 
        altitude_px: Optional[np.ndarray] = None,
        default_alt_ratio: float = 0.15
    ) -> np.ndarray:
        """
        Applies geometric Slant-Range Correction (SRC) projecting slant ranges to ground ranges:
        Y_ground = sqrt(R_slant^2 - Altitude^2)
        """
        if len(sonar_img.shape) > 2:
            gray = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = sonar_img.copy()

        h, w = gray.shape
        center = w // 2
        half_w = center

        if altitude_px is None:
            altitude_px = np.full(h, int(half_w * default_alt_ratio), dtype=np.float32)

        corrected = np.zeros_like(gray)

        # Process Port and Starboard channels
        for r in range(h):
            alt = max(1.0, float(altitude_px[r]))
            if alt >= half_w:
                alt = half_w * 0.5

            # Maximum ground range
            max_ground_range = np.sqrt(max(1.0, float(half_w**2 - alt**2)))
            
            # Non-linear sampling grid for ground range
            ground_grid = np.linspace(0, max_ground_range, half_w)
            slant_grid = np.sqrt(ground_grid**2 + alt**2)
            
            # Map Starboard
            stbd_orig = gray[r, center:]
            if len(stbd_orig) == half_w:
                mapped_stbd = np.interp(slant_grid, np.arange(half_w), stbd_orig, left=0, right=0)
                corrected[r, center:] = mapped_stbd.astype(gray.dtype)

            # Map Port (inverted horizontally)
            port_orig = gray[r, :center][::-1]
            if len(port_orig) == half_w:
                mapped_port = np.interp(slant_grid, np.arange(half_w), port_orig, left=0, right=0)
                corrected[r, :center] = mapped_port[::-1].astype(gray.dtype)

        return corrected

    @staticmethod
    def fft_destripe_filter(sonar_img: np.ndarray, notch_width: int = 4, notch_cutoff: float = 0.05) -> np.ndarray:
        """
        2D-FFT Frequency Domain Filter to remove horizontal/vertical acoustic stripe noise
        caused by boat propeller wash, acoustic multipath, and electrical transducer ripple.
        """
        if len(sonar_img.shape) > 2:
            gray = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = sonar_img.astype(np.float32)

        h, w = gray.shape
        # Compute 2D Discrete Fourier Transform
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        # Create frequency mask
        crow, ccol = h // 2, w // 2
        mask = np.ones((h, w), dtype=np.float32)

        # Filter out horizontal striping frequency axis (near vertical centerline in FFT)
        v_cutoff = int(h * notch_cutoff)
        mask[crow - v_cutoff : crow + v_cutoff, ccol - notch_width : ccol + notch_width] = 0.1
        # Preserve DC component
        mask[crow - 2 : crow + 2, ccol - 2 : ccol + 2] = 1.0

        # Apply mask and compute inverse FFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize back to 0-255 uint8
        norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

    @staticmethod
    def apply_tvg_gain(sonar_img: np.ndarray, gain_db_per_sample: float = 0.08) -> np.ndarray:
        """
        Time-Varied Gain (TVG) compensation to restore high-frequency acoustic attenuation
        and geometric spreading loss at the outer margins of the sonar swath.
        """
        if len(sonar_img.shape) > 2:
            gray = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = sonar_img.astype(np.float32)

        h, w = gray.shape
        center = w // 2
        
        # Distance from nadir track
        distances = np.abs(np.arange(w) - center).astype(np.float32)
        
        # Exponential acoustic absorption curve: G(r) = 1.0 + alpha * (r / center)^1.4
        gain_curve = 1.0 + (gain_db_per_sample * (distances / max(1, center)) ** 1.35)
        gain_matrix = np.tile(gain_curve, (h, 1))

        amplified = gray * gain_matrix
        amplified = np.clip(amplified, 0, 255)
        return amplified.astype(np.uint8)

    @classmethod
    def process_full_hydrographic_pipeline(
        cls, 
        raw_img: np.ndarray, 
        apply_bld: bool = True,
        apply_src: bool = True,
        apply_destripe: bool = True,
        apply_tvg: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the complete hydrographic acoustic DSP pipeline.
        """
        curr = raw_img.copy()
        if len(curr.shape) > 2:
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        metrics = {
            "originalMeanIntensity": float(np.mean(curr)),
            "originalStdDev": float(np.std(curr)),
            "waterColumnTracked": False,
            "slantRangeCorrected": False,
            "destripeFiltered": False,
            "tvgApplied": False,
            "snrImprovementDb": 0.0
        }

        # 1. 2D-FFT De-striping
        if apply_destripe:
            curr = cls.fft_destripe_filter(curr)
            metrics["destripeFiltered"] = True

        # 2. Bottom Line Detection & Slant Range Correction
        if apply_bld:
            bottom_lines = cls.detect_bottom_line(curr)
            metrics["waterColumnTracked"] = True
            if apply_src:
                curr = cls.slant_range_correction(curr, altitude_px=bottom_lines)
                metrics["slantRangeCorrected"] = True

        # 3. TVG Gain Normalization
        if apply_tvg:
            curr = cls.apply_tvg_gain(curr)
            metrics["tvgApplied"] = True

        # Calculate SNR gain
        final_std = float(np.std(curr))
        metrics["enhancedMeanIntensity"] = float(np.mean(curr))
        metrics["enhancedStdDev"] = final_std
        metrics["snrImprovementDb"] = round(float(20.0 * np.log10(max(1.0, final_std) / max(1.0, metrics["originalStdDev"]))), 2)

        return {
            "enhanced_image": curr,
            "metrics": metrics
        }

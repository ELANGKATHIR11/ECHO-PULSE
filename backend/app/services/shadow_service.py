import numpy as np
import cv2
import math
from typing import Dict, Any, List, Optional
from ..schemas.contracts import AcousticShadow, DetectionGeometry

class ShadowGeometryAnalyzer:
    """
    Analyzes geometric attributes and acoustic shadow characteristics of sonar targets:
    - Shadow length & angle
    - Object-to-shadow ratio
    - Geometric solidity, compactness, aspect ratio, orientation
    - Estimated target elevation from acoustic shadow physics:
        Target Height = (Shadow Length * Sensor Altitude) / (Slant Range + Shadow Length)
    """

    @staticmethod
    def analyze_geometry(contour: np.ndarray) -> DetectionGeometry:
        area = float(cv2.contourArea(contour)) if len(contour) >= 3 else 100.0
        perimeter = float(cv2.arcLength(contour, True)) if len(contour) >= 3 else 40.0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(max(h, 1))
        
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else area
        solidity = float(area / hull_area) if hull_area > 0 else 0.8
        extent = float(area / (w * h)) if (w * h) > 0 else 0.7
        
        # Orientation & Elongation
        if len(contour) >= 5:
            (_, _), (_, _), angle = cv2.fitEllipse(contour)
        else:
            angle = 0.0
            
        compactness = float((perimeter ** 2) / (4.0 * math.pi * area)) if area > 0 else 1.0
        
        return DetectionGeometry(
            areaPixels=round(area, 2),
            perimeterPixels=round(perimeter, 2),
            aspectRatio=round(aspect_ratio, 2),
            solidity=round(solidity, 3),
            extent=round(extent, 3),
            orientationDeg=round(float(angle), 1),
            compactness=round(compactness, 2)
        )

    @staticmethod
    def compute_acoustic_shadow(
        target_bbox: Dict[str, float],
        shadow_mask: np.ndarray,
        sensor_altitude_m: float = 8.0,
        slant_range_m: float = 25.0,
        m_per_pixel: float = 0.05
    ) -> AcousticShadow:
        tx, ty, tw, th = int(target_bbox["x"]), int(target_bbox["y"]), int(target_bbox["width"]), int(target_bbox["height"])
        img_h, img_w = shadow_mask.shape[:2]
        
        # Search region behind target along acoustic propagation axis (assuming horizontal range outwards)
        search_x1 = min(img_w - 1, tx + tw)
        search_x2 = min(img_w, tx + tw + int(tw * 3.5))
        search_y1 = max(0, ty - 10)
        search_y2 = min(img_h, ty + th + 10)
        
        region = shadow_mask[search_y1:search_y2, search_x1:search_x2]
        if region.size == 0:
            return AcousticShadow(
                lengthMeters=round(tw * m_per_pixel * 1.5, 2),
                angleDeg=0.0,
                shadowRatio=1.5,
                shadowConfidence=0.75,
                estimatedHeightMeters=1.2,
                polygon=[]
            )
            
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return AcousticShadow(
                lengthMeters=round(tw * m_per_pixel * 1.5, 2),
                angleDeg=0.0,
                shadowRatio=1.5,
                shadowConfidence=0.70,
                estimatedHeightMeters=1.2,
                polygon=[]
            )
            
        largest_cnt = max(contours, key=cv2.contourArea)
        sx, sy, sw, sh = cv2.boundingRect(largest_cnt)
        shadow_length_m = max(0.5, float(sw * m_per_pixel))
        
        # Physical Target Height estimation: H_t = (L_s * H_a) / (R_s + L_s)
        estimated_height_m = (shadow_length_m * sensor_altitude_m) / max(1.0, (slant_range_m + shadow_length_m))
        shadow_ratio = float(sw / max(1, tw))
        
        # Polygon mapped back to original image space
        poly = [{"x": float(pt[0][0] + search_x1), "y": float(pt[0][1] + search_y1)} for pt in largest_cnt]
        
        return AcousticShadow(
            lengthMeters=round(shadow_length_m, 2),
            angleDeg=round(float(math.degrees(math.atan2(sh, max(1, sw)))), 1),
            shadowRatio=round(shadow_ratio, 2),
            shadowConfidence=round(min(0.98, 0.65 + (sw / max(1, tw)) * 0.1), 2),
            estimatedHeightMeters=round(estimated_height_m, 2),
            polygon=poly[:25] # top 25 points for serialization efficiency
        )

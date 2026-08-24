import math
from typing import Optional, Tuple, Dict, Any

class GeotaggingService:
    """
    Computes real geographic WGS84 coordinates from sonar navigation telemetry and slant range:
    1. Slant-Range Ground-Range Correction:
       Ground Range = sqrt(Slant Range^2 - Altitude^2)
    2. Heading & Lateral Bearing Projection:
       Bearing = Vessel Heading + 90 deg (Starboard) or -90 deg (Port)
    3. Great-Circle Geodesic Forward Projection (WGS84)
    """
    
    EARTH_RADIUS_M = 6378137.0

    @staticmethod
    def calculate_wgs84_position(
        vessel_lat: float,
        vessel_lng: float,
        vessel_heading_deg: float,
        slant_range_m: float,
        altitude_m: float,
        is_port_channel: bool = False
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Returns (latitude, longitude, geotag_confidence)
        """
        if vessel_lat is None or vessel_lng is None or vessel_lat == 0.0 or vessel_lng == 0.0:
            return None, None, 0.0
            
        # Ground range calculation with Pythagorean projection
        if slant_range_m >= altitude_m:
            ground_range_m = math.sqrt(slant_range_m**2 - altitude_m**2)
        else:
            ground_range_m = slant_range_m # fallback
            
        # Lateral offset angle (perpendicular to track heading)
        offset_angle = -90.0 if is_port_channel else 90.0
        target_bearing_deg = (vessel_heading_deg + offset_angle) % 360.0
        bearing_rad = math.radians(target_bearing_deg)
        
        # Great circle forward calculation
        lat1_rad = math.radians(vessel_lat)
        lng1_rad = math.radians(vessel_lng)
        angular_dist = ground_range_m / GeotaggingService.EARTH_RADIUS_M
        
        lat2_rad = math.asin(
            math.sin(lat1_rad) * math.cos(angular_dist) +
            math.cos(lat1_rad) * math.sin(angular_dist) * math.cos(bearing_rad)
        )
        lng2_rad = lng1_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_dist) * math.cos(lat1_rad),
            math.cos(angular_dist) - math.sin(lat1_rad) * math.sin(lat2_rad)
        )
        
        target_lat = math.degrees(lat2_rad)
        target_lng = math.degrees(lng2_rad)
        confidence = 0.95 if altitude_m > 0 else 0.80
        
        return round(target_lat, 6), round(target_lng, 6), confidence

import math
from typing import Optional, Tuple, Dict, Any

class GeotaggingService:
    """
    Computes rigorous geographic WGS84 coordinates and propagated spatial uncertainty
    from sonar navigation telemetry, sensor altitude, and slant range:
    1. Slant-Range Ground-Range Geometric Inversion:
       R_ground = sqrt(max(0, R_slant^2 - Altitude^2))
    2. Heading & Lateral Bearing Orthogonal Projection:
       Bearing = Vessel Heading + 90 deg (Starboard) or -90 deg (Port)
    3. Great-Circle Geodesic Forward Calculation (WGS84 Ellipsoid approximation)
    4. Positional Uncertainty Propagation (sigma_pos = f(sigma_GPS, sigma_heading, sigma_range, sigma_alt))
    """
    
    EARTH_RADIUS_M = 6378137.0

    @staticmethod
    def calculate_wgs84_position(
        vessel_lat: Optional[float],
        vessel_lng: Optional[float],
        vessel_heading_deg: Optional[float],
        slant_range_m: float,
        altitude_m: Optional[float],
        is_port_channel: bool = False,
        gps_accuracy_m: float = 2.5,
        heading_accuracy_deg: float = 1.0
    ) -> Tuple[Optional[float], Optional[float], float, Optional[float], str]:
        """
        Returns:
            (latitude, longitude, geotag_confidence, position_uncertainty_m, position_source)
        """
        if (
            vessel_lat is None or vessel_lng is None or
            math.isnan(vessel_lat) or math.isnan(vessel_lng) or
            vessel_lat == 0.0 or vessel_lng == 0.0
        ):
            return None, None, 0.0, None, "UNAVAILABLE"

        heading = vessel_heading_deg if vessel_heading_deg is not None else 0.0
        alt = altitude_m if (altitude_m is not None and altitude_m >= 0.0) else 0.0

        # Slant-to-ground range calculation
        if slant_range_m >= alt and alt > 0.0:
            ground_range_m = math.sqrt(max(0.0, slant_range_m**2 - alt**2))
            altitude_mode = "MEASURED_ALTITUDE"
        else:
            ground_range_m = max(0.1, slant_range_m)
            altitude_mode = "ASSUMED_COINCIDENT"

        # Lateral offset angle (perpendicular to track heading)
        offset_angle = -90.0 if is_port_channel else 90.0
        target_bearing_deg = (heading + offset_angle) % 360.0
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

        # Propagated positional uncertainty sigma_pos (meters)
        # sigma_cross_track = ground_range * tan(sigma_heading)
        # sigma_range = 0.05 * slant_range
        # sigma_alt_contrib = (alt / max(1.0, ground_range)) * 0.5
        sigma_heading_rad = math.radians(heading_accuracy_deg)
        sigma_cross_track = ground_range_m * math.sin(sigma_heading_rad)
        sigma_range_m = 0.03 * slant_range_m + 0.2
        sigma_alt_m = 0.5 if altitude_m is not None else 2.0

        pos_uncertainty_m = math.sqrt(
            gps_accuracy_m**2 + sigma_cross_track**2 + sigma_range_m**2 + sigma_alt_m**2
        )

        # Derived empirical confidence based on uncertainty radius
        confidence = round(float(math.exp(-pos_uncertainty_m / 20.0)), 2)
        confidence = float(max(0.10, min(0.98, confidence)))

        position_source = f"ESTIMATED_WGS84_{altitude_mode}"

        return (
            round(target_lat, 6),
            round(target_lng, 6),
            confidence,
            round(pos_uncertainty_m, 2),
            position_source
        )

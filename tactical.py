import numpy as np
import math

class ThreatZone:
    """
    Represents a circular area of denial.
    """
    def __init__(self, lat, lon, radius_km, name="Hostile Zone"):
        self.lat = float(lat)
        self.lon = float(lon)
        self.radius_km = float(radius_km)
        self.name = name

    def to_dict(self):
        return {
            'lat': self.lat,
            'lon': self.lon,
            'radius_km': self.radius_km,
            'name': self.name
        }

class TacticalRouter:
    """
    Handles routing logic in the presence of threats.
    """

    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def latlon_to_cartesian(lat, lon):
        """Convert lat/lon to approximate local cartesian (km) relative to a reference."""
        # Note: For global scale, this is inaccurate, but for local routing (100-500km),
        # treating lat/lon as roughly cartesian with scaling factors is acceptable for a prototype.
        # Better: Work in 3D cartesian, or just use Haversine for distance and bearing.
        # For intersection checks, we need a coordinate system.
        # Let's use a local projection around the start point of a segment?
        # No, simpler: Convert everything to radians, work on sphere? Too hard for circle intersection.
        # Let's use Equirectangular approximation: x = lon * cos(avg_lat), y = lat.
        # Scale by Earth Radius.

        # We'll use the 'start' of a route segment as the origin for local calculations if needed,
        # but to keep it consistent, we'll project everything relative to (0,0) which is bad near poles.
        # Let's standard projection:
        # x = R * lon_rad * cos(lat_rad)
        # y = R * lat_rad

        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        x = TacticalRouter.EARTH_RADIUS_KM * lon_rad * math.cos(lat_rad)
        y = TacticalRouter.EARTH_RADIUS_KM * lat_rad
        return np.array([x, y])

    @staticmethod
    def cartesian_to_latlon(point):
        """Reverse approximation."""
        x, y = point
        lat_rad = y / TacticalRouter.EARTH_RADIUS_KM

        # Avoid division by zero at poles
        cos_lat = math.cos(lat_rad)
        if abs(cos_lat) < 1e-6:
            lon_rad = 0
        else:
            lon_rad = x / (TacticalRouter.EARTH_RADIUS_KM * cos_lat)

        return math.degrees(lat_rad), math.degrees(lon_rad)

    @staticmethod
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def intersect_segment_circle(start, end, center, radius):
        """
        Check if segment start-end intersects circle (center, radius).
        Returns bool.
        """
        d = end - start
        f = start - center

        # Quadratic equation for t: |Start + t*d - Center|^2 = R^2
        # (f + t*d).(f + t*d) = R^2
        # f.f + 2t(f.d) + t^2(d.d) - R^2 = 0

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return False # No intersection

        # Check if intersection points are within segment t=[0,1]
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True

        # Also need to check if circle is completely inside (not possible if radius is threat)
        # or if segment is inside... (assumed threat > 0 radius)
        return False

    @staticmethod
    def get_evasion_point(start, end, center, radius):
        """
        Calculate a waypoint to bypass the circle.
        Strategy: Move perpendicular to the direct path, far enough to clear the circle.
        """
        # Vector from start to end
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            return start

        unit_dir = direction / length

        # Perpendicular vector (rotate 90 degrees in 2D)
        # (x, y) -> (-y, x)
        perp = np.array([-unit_dir[1], unit_dir[0]])

        # Project center onto line to find closest point on infinite line
        v_center = center - start
        dist_along_line = np.dot(v_center, unit_dir)
        closest_point_on_line = start + unit_dir * dist_along_line

        # Distance from center to line
        dist_to_line = np.linalg.norm(closest_point_on_line - center)

        # If we are intersecting, dist_to_line < radius.
        # We want to be at radius + buffer distance from center.
        buffer_dist = radius * 1.2 # 20% buffer

        # We need to move 'out' from the center along the vector (closest_point -> center) reversed?
        # No, simpler: We want a point 'sideways' from the obstacle.

        # Which side? The side that `center` is NOT on?
        # Vector Line->Center is (Center - ClosestPoint).
        vec_to_center = center - closest_point_on_line

        # If center is ON the line, pick arbitrary perp.
        if np.linalg.norm(vec_to_center) < 1e-6:
            offset_dir = perp
        else:
            # We want to go AWAY from center.
            # So direction is - (Center - ClosestPoint)
            offset_dir = -vec_to_center / np.linalg.norm(vec_to_center)

        # The waypoint should be at dist = buffer from center in that direction
        evasion_point = center + offset_dir * buffer_dist

        return evasion_point

    @staticmethod
    def compute_safe_route(start_coord, end_coord, threats, recursion_depth=0):
        """
        Recursive function to find a path avoiding threats.
        Args:
            start_coord: (lat, lon)
            end_coord: (lat, lon)
            threats: list of ThreatZone
        Returns:
            list of points [(lat, lon), ...] including start and end.
            float: total distance km
        """
        # Convert to local cartesian for geometry
        # Use average lat for projection of this segment to minimize distortion
        avg_lat = (start_coord[0] + end_coord[0]) / 2.0

        # Helper to project specifically for this calculation context?
        # No, let's stick to the static method projection for simplicity across recursion.
        p_start = TacticalRouter.latlon_to_cartesian(*start_coord)
        p_end = TacticalRouter.latlon_to_cartesian(*end_coord)

        # Find first threat that intersects
        # Sort threats by distance to start? Or just check all.
        closest_collision_t = float('inf')
        hit_threat = None

        # We just need ANY collision to start subdividing, but picking the "closest" intersection along the path is better.
        # Or checking which threat center is closest to the segment?

        for threat in threats:
            # Project threat center
            p_center = TacticalRouter.latlon_to_cartesian(threat.lat, threat.lon)

            if TacticalRouter.intersect_segment_circle(p_start, p_end, p_center, threat.radius_km):
                # We hit this threat.
                # Prioritize handling? Let's just handle the first one we find for now.
                hit_threat = threat
                break

        if not hit_threat or recursion_depth > 3:
            # No threats or too deep, return straight line
            dist = np.linalg.norm(p_end - p_start) # Cartesian approx dist
            # Better to recalculate true haversine for final output?
            # The 'distance' return here is used for cost calculation.
            # Let's use haversine for the final scalar.
            import planning
            true_dist = planning.RoutePlanner.haversine_distance(start_coord, end_coord)
            return [start_coord, end_coord], true_dist

        # Evasion logic
        p_threat_center = TacticalRouter.latlon_to_cartesian(hit_threat.lat, hit_threat.lon)
        p_evade = TacticalRouter.get_evasion_point(p_start, p_end, p_threat_center, hit_threat.radius_km)

        evade_coord = TacticalRouter.cartesian_to_latlon(p_evade)

        # Recurse: Start -> Evade
        path1, dist1 = TacticalRouter.compute_safe_route(start_coord, evade_coord, threats, recursion_depth + 1)
        # Recurse: Evade -> End
        path2, dist2 = TacticalRouter.compute_safe_route(evade_coord, end_coord, threats, recursion_depth + 1)

        # Combine
        # path1 ends with evade_coord, path2 starts with evade_coord. Don't duplicate.
        full_path = path1[:-1] + path2
        total_dist = dist1 + dist2

        return full_path, total_dist

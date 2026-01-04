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

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return False # No intersection

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True

        return False

    @staticmethod
    def get_evasion_point(start, end, center, radius, stealth_factor=0.0):
        """
        Calculate a waypoint to bypass the circle.
        Strategy: Move perpendicular to the direct path, far enough to clear the circle.

        Args:
            start, end, center (np.array): Cartesian points
            radius (float): Radius of threat
            stealth_factor (float): 0.0 (visible) to 1.0 (invisible).
                                    Higher stealth reduces the required buffer distance.
        """
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            return start

        unit_dir = direction / length
        perp = np.array([-unit_dir[1], unit_dir[0]])

        v_center = center - start
        dist_along_line = np.dot(v_center, unit_dir)
        closest_point_on_line = start + unit_dir * dist_along_line

        vec_to_center = center - closest_point_on_line

        if np.linalg.norm(vec_to_center) < 1e-6:
            offset_dir = perp
        else:
            offset_dir = -vec_to_center / np.linalg.norm(vec_to_center)

        # Base buffer is 20% extra.
        # Stealth factor reduces this buffer.
        # If stealth = 1.0, buffer = 0% extra (skims the edge).
        # If stealth = 0.0, buffer = 50% extra (wider berth for safety).

        base_buffer = 1.5 - (0.5 * stealth_factor) # 1.5x radius to 1.0x radius

        # Ensure we are at least outside the radius
        evasion_dist = radius * base_buffer

        evasion_point = center + offset_dir * evasion_dist

        return evasion_point

    @staticmethod
    def compute_safe_route(start_coord, end_coord, threats, asset=None, recursion_depth=0):
        """
        Recursive function to find a path avoiding threats.
        Args:
            start_coord: (lat, lon)
            end_coord: (lat, lon)
            threats: list of ThreatZone
            asset: Asset object (optional)
        Returns:
            list of points [(lat, lon), ...] including start and end.
            float: total distance km
        """
        p_start = TacticalRouter.latlon_to_cartesian(*start_coord)
        p_end = TacticalRouter.latlon_to_cartesian(*end_coord)

        hit_threat = None

        # Check all threats to find intersection
        for threat in threats:
            p_center = TacticalRouter.latlon_to_cartesian(threat.lat, threat.lon)
            if TacticalRouter.intersect_segment_circle(p_start, p_end, p_center, threat.radius_km):
                hit_threat = threat
                break

        if not hit_threat or recursion_depth > 3:
            import planning
            true_dist = planning.RoutePlanner.haversine_distance(start_coord, end_coord)
            return [start_coord, end_coord], true_dist

        # Evasion logic
        p_threat_center = TacticalRouter.latlon_to_cartesian(hit_threat.lat, hit_threat.lon)

        stealth = asset.stealth_factor if asset else 0.0

        p_evade = TacticalRouter.get_evasion_point(p_start, p_end, p_threat_center, hit_threat.radius_km, stealth_factor=stealth)

        evade_coord = TacticalRouter.cartesian_to_latlon(p_evade)

        path1, dist1 = TacticalRouter.compute_safe_route(start_coord, evade_coord, threats, asset, recursion_depth + 1)
        path2, dist2 = TacticalRouter.compute_safe_route(evade_coord, end_coord, threats, asset, recursion_depth + 1)

        full_path = path1[:-1] + path2
        total_dist = dist1 + dist2

        return full_path, total_dist

    @staticmethod
    def analyze_route_risk(route, threats):
        """
        Calculates a 'Risk Score' for a given route based on proximity to threats.
        Score is dimensionless but proportional to time spent near threats.
        """
        risk_score = 0.0
        if not threats or not route:
            return 0.0

        # Sampling points along the route
        # For simplicity, we just check the waypoints themselves.
        # For better resolution, we should interpolate.

        for point in route:
            p_point = TacticalRouter.latlon_to_cartesian(*point)

            for threat in threats:
                p_center = TacticalRouter.latlon_to_cartesian(threat.lat, threat.lon)
                dist_km = np.linalg.norm(p_point - p_center)

                # Risk formula: (Radius / Distance)^2
                # If inside radius, risk is high (but we should have avoided it).
                # If distance == radius, risk = 1.0.
                # If distance is large, risk -> 0.

                # Avoid division by zero
                if dist_km < 0.1: dist_km = 0.1

                ratio = threat.radius_km / dist_km
                if ratio > 0.2: # Only count if reasonably close (within 5x radius)
                    risk_score += ratio ** 2

        return round(risk_score, 2)

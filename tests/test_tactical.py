import unittest
import numpy as np
from tactical import TacticalRouter, ThreatZone
import planning

class TestTacticalRouting(unittest.TestCase):

    def test_intersection(self):
        # Line from (0,0) to (10,0)
        start = np.array([0, 0])
        end = np.array([10, 0])

        # Circle at (5,0) radius 2. Should intersect.
        center = np.array([5, 0])
        radius = 2.0

        self.assertTrue(TacticalRouter.intersect_segment_circle(start, end, center, radius))

        # Circle at (5, 5) radius 1. Should not intersect.
        center_miss = np.array([5, 5])
        self.assertFalse(TacticalRouter.intersect_segment_circle(start, end, center_miss, 1.0))

    def test_evasion_point(self):
        start = np.array([0, 0])
        end = np.array([10, 0])
        center = np.array([5, 0])
        radius = 1.0

        evade = TacticalRouter.get_evasion_point(start, end, center, radius)

        # Should be at x=5, y = +/- (1 * 1.2)
        self.assertAlmostEqual(evade[0], 5.0, delta=0.1)
        self.assertTrue(abs(evade[1]) >= 1.2)

    def test_safe_route_integration(self):
        # Paris coords
        start = (48.8566, 2.3522)
        end = (48.8606, 2.3376)

        # Threat right in the middle
        # Distance is approx 1.15km.
        # Midpoint
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2

        # Radius 0.2km (200m). Should block path but start/end should be outside (dist ~0.57km from mid).
        threat = ThreatZone(mid_lat, mid_lon, radius_km=0.2)

        path, dist = TacticalRouter.compute_safe_route(start, end, [threat])

        # If threat intersects, we should have evaded.
        # Path should have > 2 points (Start, Evasion, End)
        self.assertTrue(len(path) > 2, f"Path length is {len(path)}. Expected > 2. Path: {path}")

        # Direct distance
        direct_dist = planning.RoutePlanner.haversine_distance(start, end)
        # Evasion path should be longer
        self.assertGreater(dist, direct_dist)

if __name__ == '__main__':
    unittest.main()

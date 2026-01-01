import unittest
from planning import RoutePlanner

class TestRoutePlanner(unittest.TestCase):

    def test_haversine_distance(self):
        # Rome
        rome = (41.9028, 12.4964)
        # Milan
        milan = (45.4642, 9.1900)

        # Expected distance approx 477 km
        dist = RoutePlanner.haversine_distance(rome, milan)
        self.assertTrue(470 < dist < 490, f"Distance {dist} not in expected range")

    def test_optimize_route_basic(self):
        # A simple triangle: A -> B (close) -> C (far)
        site_a = {'Site Name': 'A', 'Latitude': 0, 'Longitude': 0}
        site_b = {'Site Name': 'B', 'Latitude': 0, 'Longitude': 1} # Approx 111km away
        site_c = {'Site Name': 'C', 'Latitude': 10, 'Longitude': 0} # Approx 1110km away

        # Start at A
        others = [site_c, site_b]

        # Nearest neighbor should pick B then C
        result = RoutePlanner.optimize_route(site_a, others)
        route_names = [s['Site Name'] for s in result['route']]

        self.assertEqual(route_names, ['A', 'B', 'C'])

    def test_optimize_route_empty(self):
        site_a = {'Site Name': 'A', 'Latitude': 0, 'Longitude': 0}
        result = RoutePlanner.optimize_route(site_a, [])
        self.assertEqual(len(result['route']), 1)
        self.assertEqual(result['total_distance_km'], 0)

if __name__ == '__main__':
    unittest.main()

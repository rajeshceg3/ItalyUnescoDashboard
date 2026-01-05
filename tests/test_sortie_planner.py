import unittest
import numpy as np
from planning import RoutePlanner, NumpyKMeans, StrategicSortiePlanner
from assets import Asset

class TestSortiePlanner(unittest.TestCase):
    def setUp(self):
        # Create dummy sites
        self.base = {'Site Name': 'Base', 'Latitude': 0, 'Longitude': 0}
        self.sites = [
            {'Site Name': 'A', 'Latitude': 1, 'Longitude': 1}, # Near
            {'Site Name': 'B', 'Latitude': 1.1, 'Longitude': 1.1}, # Near A
            {'Site Name': 'C', 'Latitude': 10, 'Longitude': 10}, # Far
            {'Site Name': 'D', 'Latitude': 10.1, 'Longitude': 10.1}, # Near C
        ]
        # Create a test asset
        self.asset = Asset("Test Asset", speed_kmh=100, max_range_km=5000, stealth_factor=0.5)

    def test_haversine(self):
        # Dist between 0,0 and 0,1 degree (approx 111km)
        dist = RoutePlanner.haversine_distance((0,0), (0,1))
        self.assertTrue(100 < dist < 120)

    def test_kmeans(self):
        data = np.array([
            [1, 1],
            [1.1, 1.1],
            [10, 10],
            [10.1, 10.1]
        ])
        kmeans = NumpyKMeans(k=2)
        labels = kmeans.fit(data)

        # Check that 0 and 1 are grouped, and 2 and 3 are grouped
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_single_sortie_plan(self):
        planner = StrategicSortiePlanner(self.base, self.sites)
        # Updated signature: pass asset instead of avg_speed_kmh
        sorties = planner.plan_sorties(num_sorties=1, asset=self.asset)

        self.assertEqual(len(sorties), 1)
        self.assertEqual(sorties[0]['site_count'], 4) # Should visit all 4 unique sites

        # Route should have Base -> ... -> Base.
        self.assertEqual(sorties[0]['route'][0]['Site Name'], 'Base')
        self.assertEqual(sorties[0]['route'][-1]['Site Name'], 'Base')

    def test_multi_sortie_plan(self):
        planner = StrategicSortiePlanner(self.base, self.sites)
        sorties = planner.plan_sorties(num_sorties=2, asset=self.asset)

        self.assertEqual(len(sorties), 2)
        # One sortie should have 2 sites, the other 2 sites
        counts = sorted([s['site_count'] for s in sorties])
        self.assertEqual(counts, [2, 2])

    def test_fleet_plan(self):
        planner = StrategicSortiePlanner(self.base, self.sites)
        fleet = [self.asset, self.asset]
        sorties = planner.plan_fleet_mission(fleet=fleet)

        self.assertEqual(len(sorties), 2)
        counts = sorted([s['site_count'] for s in sorties])
        self.assertEqual(counts, [2, 2])

if __name__ == '__main__':
    unittest.main()

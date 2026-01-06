import unittest
import numpy as np
from planning import StrategicSortiePlanner
from wargame import MonteCarloEngine, SimulationResult
from assets import ASSET_REGISTRY, Asset
from tactical import ThreatZone

class TestWargame(unittest.TestCase):
    def setUp(self):
        # Mock Data
        self.base_site = {'Site Name': 'Base', 'Latitude': 0, 'Longitude': 0}
        self.target_sites = [
            {'Site Name': 'T1', 'Latitude': 0.1, 'Longitude': 0.1}, # Close
            {'Site Name': 'T2', 'Latitude': 1.0, 'Longitude': 1.0}  # Far
        ]
        self.fleet = [
            Asset("Fast", 100, 1000, 0.5, 10.0),
            Asset("Slow", 50, 1000, 0.5, 10.0)
        ]
        self.threats = [ThreatZone(0.5, 0.5, 10.0, "TestThreat")] # Threat in middle

    def test_monte_carlo_engine_init(self):
        engine = MonteCarloEngine(num_runs=50)
        self.assertEqual(engine.num_runs, 50)

    def test_simulation_run_structure(self):
        # Create a simple plan first
        planner = StrategicSortiePlanner(self.base_site, self.target_sites)
        sorties = planner.plan_fleet_mission(self.fleet, threats=[])

        engine = MonteCarloEngine(num_runs=10)
        stats = engine.simulate_plan(sorties, threats=[])

        self.assertIsInstance(stats.success_rate, float)
        self.assertIsInstance(stats.avg_time, float)
        self.assertIsInstance(stats.risk_variance, float)
        self.assertTrue(0 <= stats.success_rate <= 100)

    def test_threat_interception(self):
        # Force a path through a threat
        # Route from (0,0) to (1,1) goes through (0.5, 0.5)
        planner = StrategicSortiePlanner(self.base_site, [{'Site Name': 'T2', 'Latitude': 1.0, 'Longitude': 1.0}])
        # We pass NO threats to planner so it makes a straight line
        sorties = planner.plan_fleet_mission([self.fleet[0]], threats=[])

        # Now we simulate WITH threats
        # The threat at 0.5,0.5 with radius 10km (~0.1 deg) should intercept
        # 1 deg lat is approx 111km. 0.1 deg is 11km.
        # Radius 10km is about 0.09 degrees.
        # Line from 0,0 to 1,1 passes exactly through 0.5,0.5.

        threat_radius_km = 20 # Make it big enough to definitely hit
        # 20km is roughly 0.18 degrees.

        threat = ThreatZone(0.5, 0.5, threat_radius_km)

        engine = MonteCarloEngine(num_runs=5)
        stats = engine.simulate_plan(sorties, [threat])

        # Should fail most of the time
        # success_rate should be low or 0
        self.assertLess(stats.success_rate, 100.0)

    def test_coa_generation(self):
        planner = StrategicSortiePlanner(self.base_site, self.target_sites)
        coas = planner.generate_coas(self.fleet)

        self.assertIn('SPEED', coas)
        self.assertIn('STEALTH', coas)
        self.assertIn('EFFICIENCY', coas)

        # Check that 'SPEED' plan exists and is a list
        self.assertIsInstance(coas['SPEED'], list)

if __name__ == '__main__':
    unittest.main()

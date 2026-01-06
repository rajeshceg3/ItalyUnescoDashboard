import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SimulationResult:
    run_id: int
    success: bool
    total_time: float
    total_fuel: float
    failure_reason: str = "None"
    events: List[str] = field(default_factory=list)

@dataclass
class AggregatedStats:
    success_rate: float
    avg_time: float
    avg_fuel: float
    risk_variance: float
    p90_time: float # 90th percentile time (worst case)

class MonteCarloEngine:
    """
    Simulates mission plans under probabilistic conditions to determine robustness.
    """

    def __init__(self, num_runs=100):
        self.num_runs = num_runs

    def simulate_plan(self, sorties: List[Dict[str, Any]], threats: List[Any]) -> AggregatedStats:
        """
        Runs Monte Carlo simulations on the provided set of sorties.

        Args:
            sorties: List of sortie dictionaries (from planning.py).
            threats: List of ThreatZone objects.

        Returns:
            AggregatedStats object.
        """
        results = []

        for i in range(self.num_runs):
            res = self._run_single_simulation(i, sorties, threats)
            results.append(res)

        return self._aggregate_results(results)

    def _run_single_simulation(self, run_id, sorties, threats) -> SimulationResult:
        """
        Executes one simulation run with randomized parameters.
        """
        # Noise parameters
        speed_factor = np.random.normal(1.0, 0.05) # +/- 5% speed variation
        fuel_burn_factor = np.random.normal(1.0, 0.10) # +/- 10% fuel efficiency variation
        threat_radius_factor = np.random.normal(1.0, 0.05) # +/- 5% threat expansion

        simulation_success = True
        total_time = 0.0
        total_fuel = 0.0
        events = []
        failure_reason = "None"

        # Check each sortie in the plan
        for sortie in sorties:
            if sortie['status'] == "RED":
                simulation_success = False
                failure_reason = "Plan Initial Invalid"
                break

            # Recalculate metrics with noise
            # Base duration
            base_duration = sortie['est_duration_hrs']
            real_duration = base_duration / speed_factor

            # Base fuel
            base_fuel = sortie['fuel_used']
            real_fuel = base_fuel * fuel_burn_factor

            # Update totals
            total_time += real_duration
            total_fuel += real_fuel

            # Check Fuel Exhaustion (simplified max range check)
            # Retrieve max range from asset (need to infer or pass it.
            # The sortie dict doesn't carry the asset object, but usually has status check.
            # We'll assume if real_fuel exceeds a buffer, it might fail,
            # but we don't have asset max fuel here directly without looking up registry.
            # Let's use a simpler heuristic: if efficiency drops too much and dist is high.

            # Critical: Threat Intersections
            # We re-check the path against "expanded" threats
            path = sortie.get('detailed_path', [])
            if self._check_threat_collision(path, threats, threat_radius_factor):
                simulation_success = False
                failure_reason = f"Sortie #{sortie['id']} Intercepted"
                events.append(f"Sortie #{sortie['id']} intercepted by hostile element.")
                break

        return SimulationResult(
            run_id=run_id,
            success=simulation_success,
            total_time=total_time,
            total_fuel=total_fuel,
            failure_reason=failure_reason,
            events=events
        )

    def _check_threat_collision(self, path, threats, radius_multiplier):
        """
        Checks if the path intersects with any threat given the noise multiplier.
        """
        if not threats or not path:
            return False

        # Simplified check: Just check waypoints against expanded radii
        # A more expensive check would do segment-circle intersection again

        from tactical import TacticalRouter

        for i in range(len(path) - 1):
            p1 = TacticalRouter.latlon_to_cartesian(*path[i])
            p2 = TacticalRouter.latlon_to_cartesian(*path[i+1])

            for threat in threats:
                center = TacticalRouter.latlon_to_cartesian(threat.lat, threat.lon)
                eff_radius = threat.radius_km * radius_multiplier

                if TacticalRouter.intersect_segment_circle(p1, p2, center, eff_radius):
                    return True

        return False

    def _aggregate_results(self, results: List[SimulationResult]) -> AggregatedStats:
        successes = [r for r in results if r.success]

        success_rate = len(successes) / len(results) if results else 0.0

        if successes:
            avg_time = np.mean([r.total_time for r in successes])
            avg_fuel = np.mean([r.total_fuel for r in successes])
            times = [r.total_time for r in successes]
            p90_time = np.percentile(times, 90)
        else:
            avg_time = 0.0
            avg_fuel = 0.0
            p90_time = 0.0

        # Risk variance (std dev of success vs fail - essentially just success rate metric,
        # but let's use fuel variance as a proxy for operational instability in successful missions)
        risk_variance = np.var([r.total_fuel for r in successes]) if successes else 0.0

        return AggregatedStats(
            success_rate=round(success_rate * 100, 1),
            avg_time=round(avg_time, 2),
            avg_fuel=round(avg_fuel, 1),
            risk_variance=round(risk_variance, 2),
            p90_time=round(p90_time, 2)
        )

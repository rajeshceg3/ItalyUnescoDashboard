import math
import numpy as np
try:
    from tactical import TacticalRouter
except ImportError:
    # If tactical.py is not available/run independently
    TacticalRouter = None

from assets import Asset
from scipy.optimize import linear_sum_assignment

class RoutePlanner:
    """
    Handles the calculation of optimal routes between multiple geographic locations.
    """

    @staticmethod
    def haversine_distance(coord1, coord2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).

        Args:
            coord1 (tuple): (latitude, longitude)
            coord2 (tuple): (latitude, longitude)

        Returns:
            float: Distance in kilometers.
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371 # Radius of earth in kilometers
        return c * r

    @staticmethod
    def optimize_route(start_site, other_sites, return_to_start=False, threats=None, asset=None):
        """
        Optimizes the route to visit all sites using a Nearest Neighbor algorithm.

        Args:
            start_site (dict): The starting site dictionary.
            other_sites (list): List of other site dictionaries to visit.
            return_to_start (bool): If True, adds the start site as the final destination.
            threats (list): Optional list of ThreatZone objects.
            asset (Asset): Optional asset object for stealth routing.

        Returns:
            dict: {
                'route': list of site dictionaries in order,
                'total_distance_km': float,
                'detailed_path': list of (lat, lon) tuples (including waypoints),
                'risk_score': float
            }
        """
        if not other_sites:
            return {'route': [start_site], 'total_distance_km': 0, 'detailed_path': [], 'risk_score': 0.0}

        # Initialize
        current_site = start_site
        route = [start_site]
        unvisited = other_sites[:]
        total_distance = 0.0

        # We also want to track the *detailed* path (with evasion waypoints)
        detailed_path = [(float(start_site['Latitude']), float(start_site['Longitude']))]

        while unvisited:
            nearest_site = None
            min_dist = float('inf')
            best_segment_path = None

            current_coords = (float(current_site['Latitude']), float(current_site['Longitude']))

            for site in unvisited:
                site_coords = (float(site['Latitude']), float(site['Longitude']))

                # If threats exist, use tactical routing
                if threats and TacticalRouter:
                    path_seg, dist = TacticalRouter.compute_safe_route(current_coords, site_coords, threats, asset=asset)
                else:
                    dist = RoutePlanner.haversine_distance(current_coords, site_coords)
                    path_seg = [current_coords, site_coords]

                if dist < min_dist:
                    min_dist = dist
                    nearest_site = site
                    best_segment_path = path_seg

            # Move to nearest site
            if nearest_site:
                route.append(nearest_site)
                total_distance += min_dist

                # Append path segments (excluding start which is already in detailed_path)
                if best_segment_path:
                    detailed_path.extend(best_segment_path[1:])

                current_site = nearest_site
                unvisited.remove(nearest_site)

        if return_to_start:
            # Add distance back to start
            last_coords = (float(current_site['Latitude']), float(current_site['Longitude']))
            start_coords = (float(start_site['Latitude']), float(start_site['Longitude']))

            if threats and TacticalRouter:
                path_seg, dist = TacticalRouter.compute_safe_route(last_coords, start_coords, threats, asset=asset)
            else:
                dist = RoutePlanner.haversine_distance(last_coords, start_coords)
                path_seg = [last_coords, start_coords]

            total_distance += dist
            if path_seg:
                detailed_path.extend(path_seg[1:])

            route.append(start_site)

        # Calculate Risk Score
        risk_score = 0.0
        if threats and TacticalRouter:
            risk_score = TacticalRouter.analyze_route_risk(detailed_path, threats)

        return {
            'route': route,
            'total_distance_km': round(total_distance, 2),
            'detailed_path': detailed_path,
            'risk_score': risk_score
        }

class NumpyKMeans:
    """
    Simple K-Means implementation using NumPy to avoid heavy sklearn dependency.
    """
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """
        Fit K-Means to the data X (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        # Randomly initialize centroids
        # Use a fixed seed for reproducibility in missions
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Assign clusters
            # distances: (n_samples, k)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])

            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        return labels

class StrategicSortiePlanner:
    """
    Advanced Mission Planner for generating multi-sortie logistics plans.
    """
    def __init__(self, base_site, target_sites):
        self.base_site = base_site
        self.target_sites = target_sites # List of dicts

    def plan_sorties(self, num_sorties=1, asset=None, threats=None):
        """
        Generates a mission plan with 'num_sorties' sorties.
        Uses K-Means to cluster targets, but assumes identical assets.
        Kept for backward compatibility.
        """
        # Create a fleet of identical assets
        if asset is None:
             # Default fallback
             asset = Asset("Generic", 60, 500, 0.0, 10.0)

        fleet = [asset] * int(num_sorties)
        return self.plan_fleet_mission(fleet, threats)

    def generate_coas(self, fleet, threats=None):
        """
        Generates three Courses of Action (COAs):
        1. SPEED: Maximizes speed (standard behavior).
        2. STEALTH: Prioritizes avoidance, even at high distance cost.
        3. EFFICIENCY: Prioritizes fuel economy (shortest distance).
        """
        coas = {}

        # 1. SPEED (Standard)
        coas['SPEED'] = self.plan_fleet_mission(fleet, threats, strategy='speed')

        # 2. STEALTH (High caution)
        # We simulate this by effectively increasing threat radius perception in the planner or logic
        # But since optimize_route takes 'threats', we rely on the asset's stealth factor primarily.
        # To make a distinct COA, we can modify the cost matrix to penalize assigning low-stealth assets to dangerous clusters.
        coas['STEALTH'] = self.plan_fleet_mission(fleet, threats, strategy='stealth')

        # 3. EFFICIENCY
        coas['EFFICIENCY'] = self.plan_fleet_mission(fleet, threats, strategy='efficiency')

        return coas

    def plan_fleet_mission(self, fleet, threats=None, strategy='speed'):
        """
        Generates a mission plan for a specific heterogeneous fleet.

        Args:
            fleet (list of Asset): The list of available assets.
            threats (list): Optional list of ThreatZone objects.
            strategy (str): 'speed', 'stealth', or 'efficiency'.

        Returns:
            list of dicts (sorties).
        """
        if not self.target_sites:
            return []

        num_assets = len(fleet)
        num_targets = len(self.target_sites)

        # If we have more assets than targets, we only use the best N assets.
        # But for simplicity, we cluster into min(num_assets, num_targets)
        k = min(num_assets, num_targets)

        # 1. Cluster Targets
        coords = np.array([[float(s['Latitude']), float(s['Longitude'])] for s in self.target_sites])
        kmeans = NumpyKMeans(k=k)
        labels = kmeans.fit(coords)

        clusters = []
        for i in range(k):
             cluster_indices = np.where(labels == i)[0]
             cluster_sites = [self.target_sites[idx] for idx in cluster_indices]

             # Calculate centroid (naive mean of lat/lon)
             centroid = np.mean([[float(s['Latitude']), float(s['Longitude'])] for s in cluster_sites], axis=0)
             clusters.append({
                 'sites': cluster_sites,
                 'centroid': centroid
             })

        # 2. Assign Clusters to Assets (Linear Assignment Problem)
        # We need a Cost Matrix where C[i, j] is cost of Asset i handling Cluster j.
        # Cost = Time taken. If impossible (Range exceeded), Cost = Infinite.

        cost_matrix = np.zeros((num_assets, k))

        # Pre-calculate distances from Base to Centroids
        base_coords = (float(self.base_site['Latitude']), float(self.base_site['Longitude']))

        for i, asset in enumerate(fleet):
            for j, cluster in enumerate(clusters):
                # Approximate distance: Base -> Centroid -> Base + intra-cluster travel
                # Distance to centroid
                centroid_dist = RoutePlanner.haversine_distance(base_coords, tuple(cluster['centroid']))

                # Estimate total route distance ~ 2 * centroid_dist + intra-cluster
                # Intra-cluster is roughly (N-1) * avg_dist_between_sites.
                # Let's simplify: 2.5 * centroid_dist usually covers the tour if dense.
                # A better heuristic: 2 * centroid_dist + sum of distances from centroid to points.

                intra_dist_sum = sum([RoutePlanner.haversine_distance(tuple(cluster['centroid']), (float(s['Latitude']), float(s['Longitude']))) for s in cluster['sites']])

                est_total_dist = (2 * centroid_dist) + intra_dist_sum

                # --- STRATEGY ADJUSTMENTS ---
                # Default cost is Time
                base_cost = est_total_dist / asset.speed_kmh

                if strategy == 'efficiency':
                    # Cost is Fuel
                    base_cost = est_total_dist / asset.fuel_efficiency

                elif strategy == 'stealth':
                    # Cost is Risk (Distance / Stealth)
                    # Low stealth factor (0.2) -> High Cost.
                    # High stealth factor (0.8) -> Low Cost.
                    stealth_mod = max(0.1, asset.stealth_factor)
                    base_cost = (est_total_dist / asset.speed_kmh) * (1.0 / stealth_mod)

                # Check Range
                if est_total_dist > asset.max_range_km:
                    cost_matrix[i, j] = 1e9 # Impossible
                else:
                    cost_matrix[i, j] = base_cost

        # Solve Assignment
        # If num_assets > k, some rows will be unassigned.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        final_sorties = []

        # Process assignments
        for r, c in zip(row_ind, col_ind):
            asset = fleet[r]
            cluster = clusters[c]

            # If cost was infinite, this assignment is invalid (mission failure for this cluster)
            # But linear_sum_assignment forces an assignment.
            # We handle the "Red" status in the output.

            route_result = RoutePlanner.optimize_route(
                self.base_site,
                cluster['sites'],
                return_to_start=True,
                threats=threats,
                asset=asset
            )

            dist = route_result['total_distance_km']
            transit_time = dist / asset.speed_kmh
            mission_time = transit_time + (len(cluster['sites']) * 2.0) # 2 hrs on site

            fuel_used = dist / asset.fuel_efficiency

            status = "GREEN"
            status_msg = "Mission Feasible"
            if dist > asset.max_range_km:
                status = "RED"
                status_msg = f"CRITICAL: Exceeds Max Range ({asset.max_range_km} km)"
            elif cost_matrix[r, c] >= 1e9:
                 status = "RED"
                 status_msg = "CRITICAL: No suitable asset available"

            final_sorties.append({
                'id': r + 1, # Asset ID
                'asset_name': asset.name,
                'route': route_result['route'],
                'distance': dist,
                'est_duration_hrs': round(mission_time, 2),
                'site_count': len(cluster['sites']),
                'detailed_path': route_result.get('detailed_path', []),
                'risk_score': route_result.get('risk_score', 0.0),
                'fuel_used': round(fuel_used, 1),
                'status': status,
                'status_msg': status_msg
            })

        return final_sorties

class MissionSimulator:
    """
    Simulates the mission execution over time.
    """
    def __init__(self, sorties):
        self.sorties = sorties # The output from plan_fleet_mission
        # Pre-process trajectories
        self.trajectories = {} # map sortie_id -> list of (t, lat, lon)

        for sortie in sorties:
            path = sortie['detailed_path']
            if not path:
                continue

            total_dist = sortie['distance']
            total_time = sortie['est_duration_hrs']

            # Simple assumption: constant speed over the whole path
            # t goes from 0 to 1 (normalized progress)

            traj = []
            num_points = len(path)
            if num_points < 2:
                 traj = [(0.0, path[0][0], path[0][1]), (1.0, path[0][0], path[0][1])]
            else:
                # We distribute time proportional to index (approximation)
                # Better: distribute by segment length
                # Calculate cumulative distance
                cum_dist = [0.0]
                for i in range(1, num_points):
                    d = RoutePlanner.haversine_distance(path[i-1], path[i])
                    cum_dist.append(cum_dist[-1] + d)

                total_d = cum_dist[-1]
                if total_d == 0:
                     traj = [(0.0, path[0][0], path[0][1]), (1.0, path[0][0], path[0][1])]
                else:
                    for i in range(num_points):
                        t_norm = cum_dist[i] / total_d
                        traj.append((t_norm, path[i][0], path[i][1]))

            self.trajectories[sortie['id']] = traj

    def get_state_at_progress(self, progress_pct):
        """
        Returns the positions of all assets at a given progress percentage (0-100).
        """
        t_req = progress_pct / 100.0
        states = []

        for s_id, traj in self.trajectories.items():
            # Find segment [i, i+1] where traj[i].t <= t_req <= traj[i+1].t
            # Binary search or linear scan

            found = False
            for i in range(len(traj) - 1):
                t1, lat1, lon1 = traj[i]
                t2, lat2, lon2 = traj[i+1]

                if t1 <= t_req <= t2:
                    # Interpolate
                    span = t2 - t1
                    if span == 0:
                        lat, lon = lat1, lon1
                    else:
                        ratio = (t_req - t1) / span
                        lat = lat1 + (lat2 - lat1) * ratio
                        lon = lon1 + (lon2 - lon1) * ratio

                    states.append({
                        'id': s_id,
                        'lat': lat,
                        'lon': lon
                    })
                    found = True
                    break

            if not found:
                # Must be at end
                states.append({
                    'id': s_id,
                    'lat': traj[-1][1],
                    'lon': traj[-1][2]
                })

        return states

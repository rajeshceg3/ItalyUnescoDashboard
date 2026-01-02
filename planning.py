import math
import numpy as np

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
    def optimize_route(start_site, other_sites, return_to_start=False):
        """
        Optimizes the route to visit all sites using a Nearest Neighbor algorithm.

        Args:
            start_site (dict): The starting site dictionary.
            other_sites (list): List of other site dictionaries to visit.
            return_to_start (bool): If True, adds the start site as the final destination.

        Returns:
            dict: {
                'route': list of site dictionaries in order,
                'total_distance_km': float
            }
        """
        if not other_sites:
            return {'route': [start_site], 'total_distance_km': 0}

        # Initialize
        current_site = start_site
        route = [start_site]
        unvisited = other_sites[:]
        total_distance = 0.0

        while unvisited:
            nearest_site = None
            min_dist = float('inf')

            current_coords = (float(current_site['Latitude']), float(current_site['Longitude']))

            for site in unvisited:
                site_coords = (float(site['Latitude']), float(site['Longitude']))
                dist = RoutePlanner.haversine_distance(current_coords, site_coords)

                if dist < min_dist:
                    min_dist = dist
                    nearest_site = site

            # Move to nearest site
            if nearest_site:
                route.append(nearest_site)
                total_distance += min_dist
                current_site = nearest_site
                unvisited.remove(nearest_site)

        if return_to_start:
            # Add distance back to start
            last_coords = (float(current_site['Latitude']), float(current_site['Longitude']))
            start_coords = (float(start_site['Latitude']), float(start_site['Longitude']))
            dist = RoutePlanner.haversine_distance(last_coords, start_coords)
            total_distance += dist
            route.append(start_site)

        return {
            'route': route,
            'total_distance_km': round(total_distance, 2)
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

    def plan_sorties(self, num_sorties=1, avg_speed_kmh=60):
        """
        Generates a mission plan with 'num_sorties' sorties.

        Args:
            num_sorties (int): Number of sorties (clusters) to split the targets into.
            avg_speed_kmh (float): Average operational speed in km/h.

        Returns:
            list of dicts, where each dict is a sortie plan:
            {
                'id': int,
                'route': list of sites,
                'distance': float,
                'est_duration_hrs': float
            }
        """
        if not self.target_sites:
            return []

        # If only 1 sortie, just run standard optimization (Round Trip)
        if num_sorties == 1:
            raw_route = RoutePlanner.optimize_route(self.base_site, self.target_sites, return_to_start=True)
            dist = raw_route['total_distance_km']
            # Estimate time: Driving + 2 hours per site
            # Note: This is a rough estimation.
            transit_time = dist / avg_speed_kmh
            # Excluding start/end base visits from "site time" if it appears twice?
            # Route has N+2 nodes (Start, S1...Sn, Start). Unique sites visited = N.
            # Let's assume 2 hours per target site.
            unique_targets = len(self.target_sites)
            mission_time = transit_time + (unique_targets * 2.0)

            return [{
                'id': 1,
                'route': raw_route['route'],
                'distance': dist,
                'est_duration_hrs': round(mission_time, 2),
                'site_count': unique_targets
            }]

        # Prepare data for clustering
        # We use Latitude/Longitude.
        # Note: Euclidean distance on Lat/Lon is not perfect but sufficient for clustering at this scale.
        coords = np.array([[float(s['Latitude']), float(s['Longitude'])] for s in self.target_sites])

        # Handle case where k > n_samples
        k = min(num_sorties, len(self.target_sites))

        kmeans = NumpyKMeans(k=k)
        labels = kmeans.fit(coords)

        sorties = []
        for i in range(k):
            # Get sites belonging to this cluster
            cluster_indices = np.where(labels == i)[0]
            cluster_sites = [self.target_sites[idx] for idx in cluster_indices]

            # Optimize route for this cluster (Round Trip from Base)
            route_result = RoutePlanner.optimize_route(self.base_site, cluster_sites, return_to_start=True)

            dist = route_result['total_distance_km']
            transit_time = dist / avg_speed_kmh
            mission_time = transit_time + (len(cluster_sites) * 2.0)

            sorties.append({
                'id': i + 1,
                'route': route_result['route'],
                'distance': dist,
                'est_duration_hrs': round(mission_time, 2),
                'site_count': len(cluster_sites)
            })

        return sorties

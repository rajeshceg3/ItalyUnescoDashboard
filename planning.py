import math

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
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    @staticmethod
    def optimize_route(start_site, other_sites):
        """
        Optimizes the route to visit all sites using a Nearest Neighbor algorithm.

        Args:
            start_site (dict): The starting site dictionary (must have 'Latitude', 'Longitude').
            other_sites (list): List of other site dictionaries to visit.

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

        return {
            'route': route,
            'total_distance_km': round(total_distance, 2)
        }

from models.models import BranchAndATM
from math import radians, sin, cos, sqrt, atan2

class LocationService:
    @staticmethod
    def find_nearby_locations(latitude, longitude, radius_km=5):
        def calculate_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c

        locations = BranchAndATM.query.all()
        nearby = []
        
        for loc in locations:
            distance = calculate_distance(
                float(latitude), float(longitude),
                float(loc.Latitude), float(loc.Longitude)
            )
            if distance <= radius_km:
                nearby.append({
                    'name': loc.LocationName,
                    'address': loc.Address,
                    'type': loc.Type,
                    'distance': round(distance, 2)
                })
        
        return sorted(nearby, key=lambda x: x['distance'])
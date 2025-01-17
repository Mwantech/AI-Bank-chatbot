# routes/location_routes.py
from flask import Blueprint, jsonify, request
from services.location_service import LocationService

location_bp = Blueprint('location', __name__)

@location_bp.route('/nearby', methods=['GET'])
def find_nearby():
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))
    radius = float(request.args.get('radius', 5))
    
    locations = LocationService.find_nearby_locations(latitude, longitude, radius)
    return jsonify({'locations': locations})


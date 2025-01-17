from flask import Blueprint, jsonify, request
from services.customer_support_service import CustomerSupportService

support_bp = Blueprint('support', __name__)

@support_bp.route('/request', methods=['POST'])
def create_request():
    data = request.get_json()
    success, message = CustomerSupportService.create_support_request(
        data['user_id'],
        data['request_type'],
        data['details']
    )
    return jsonify({'success': success, 'message': message})

@support_bp.route('/requests/<int:user_id>', methods=['GET'])
def get_requests(user_id):
    requests = CustomerSupportService.get_support_requests(user_id)
    return jsonify({'requests': requests})
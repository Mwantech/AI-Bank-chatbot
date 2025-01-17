
from flask import Blueprint, jsonify, request
from services.investment_service import InvestmentService

investment_bp = Blueprint('investment', __name__)

@investment_bp.route('/create', methods=['POST'])
def create_investment():
    data = request.get_json()
    success, message = InvestmentService.create_investment(
        data['user_id'],
        data['investment_type'],
        data['amount']
    )
    return jsonify({'success': success, 'message': message})

@investment_bp.route('/<int:user_id>', methods=['GET'])
def get_investments(user_id):
    investments = InvestmentService.get_investments(user_id)
    return jsonify({'investments': investments})
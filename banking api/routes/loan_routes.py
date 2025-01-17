from flask import Blueprint, jsonify, request
from services.loan_service import LoanService

loan_bp = Blueprint('loan', __name__)

@loan_bp.route('/apply', methods=['POST'])
def apply_loan():
    data = request.get_json()
    success, message = LoanService.apply_for_loan(
        data['user_id'],
        data['loan_type'],
        data['amount']
    )
    return jsonify({'success': success, 'message': message})

@loan_bp.route('/status/<int:user_id>', methods=['GET'])
def loan_status(user_id):
    loans = LoanService.get_loan_status(user_id)
    return jsonify({'loans': loans})

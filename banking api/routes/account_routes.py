from flask import Blueprint, jsonify, request
from services.account_service import AccountService

account_bp = Blueprint('account', __name__)

@account_bp.route('/balance/<int:user_id>', methods=['GET'])
def get_balance(user_id):
    balances = AccountService.get_balance(user_id)
    return jsonify({'accounts': balances})

@account_bp.route('/transactions/<int:account_id>', methods=['GET'])
def get_transactions(account_id):
    transactions = AccountService.get_transactions(account_id)
    return jsonify({'transactions': transactions})

@account_bp.route('/transfer', methods=['POST'])
def transfer():
    data = request.get_json()
    success, message = AccountService.transfer_funds(
        data['from_account_id'],
        data['to_account_number'],
        data['amount']
    )
    return jsonify({'success': success, 'message': message})
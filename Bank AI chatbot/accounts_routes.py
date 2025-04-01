from flask import Blueprint, request, jsonify
from models import db, User, Account, Transaction
from auth import token_required
from datetime import datetime

account_bp = Blueprint('account', __name__, url_prefix='/api')

@account_bp.route('/accounts', methods=['GET'])
@token_required
def get_accounts(current_user):
    try:
        accounts = Account.query.filter_by(UserID=current_user.UserID).all()
        
        accounts_data = [{
            'AccountID': account.AccountID,
            'AccountType': account.AccountType,
            'AccountNumber': account.AccountNumber,
            'Balance': str(account.Balance),
            'Currency': account.Currency,
            'Status': account.Status,
            'CreatedAt': account.CreatedAt.isoformat()
        } for account in accounts]
        
        return jsonify({
            'accounts': accounts_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@account_bp.route('/accounts/<int:account_id>', methods=['GET'])
@token_required
def get_account(current_user, account_id):
    try:
        account = Account.query.filter_by(AccountID=account_id, UserID=current_user.UserID).first()
        
        if not account:
            return jsonify({'error': 'Account not found or access denied'}), 404
        
        account_data = {
            'AccountID': account.AccountID,
            'AccountType': account.AccountType,
            'AccountNumber': account.AccountNumber,
            'Balance': str(account.Balance),
            'Currency': account.Currency,
            'Status': account.Status,
            'CreatedAt': account.CreatedAt.isoformat()
        }
        
        return jsonify({
            'account': account_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@account_bp.route('/accounts/<int:account_id>/transactions', methods=['GET'])
@token_required
def get_transactions(current_user, account_id):
    try:
        # Verify the account belongs to the current user
        account = Account.query.filter_by(AccountID=account_id, UserID=current_user.UserID).first()
        
        if not account:
            return jsonify({'error': 'Account not found or access denied'}), 404
        
        transactions = Transaction.query.filter_by(AccountID=account_id).order_by(Transaction.TransactionDate.desc()).all()
        
        transactions_data = [{
            'TransactionID': transaction.TransactionID,
            'Type': transaction.Type,
            'Amount': str(transaction.Amount),
            'Description': transaction.Description,
            'Status': transaction.Status,
            'TransactionDate': transaction.TransactionDate.isoformat()
        } for transaction in transactions]
        
        return jsonify({
            'transactions': transactions_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@account_bp.route('/accounts/<int:account_id>/transactions', methods=['POST'])
@token_required
def create_transaction(current_user, account_id):
    try:
        # Verify the account belongs to the current user
        account = Account.query.filter_by(AccountID=account_id, UserID=current_user.UserID).first()
        
        if not account:
            return jsonify({'error': 'Account not found or access denied'}), 404
        
        data = request.get_json()
        
        # Validate transaction data
        if not all(k in data for k in ['type', 'amount']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        transaction_type = data['type'].capitalize()
        amount = float(data['amount'])
        
        if amount <= 0:
            return jsonify({'error': 'Amount must be greater than zero'}), 400
        
        # Type conversion for database - ensure Decimal type
        from decimal import Decimal
        decimal_amount = Decimal(str(amount))
        
        # Update account balance
        if transaction_type.lower() == 'deposit':
            account.Balance = account.Balance + decimal_amount
        elif transaction_type.lower() == 'withdrawal':
            if decimal_amount > account.Balance:
                return jsonify({'error': 'Insufficient funds'}), 400
            account.Balance = account.Balance - decimal_amount
        else:
            return jsonify({'error': 'Invalid transaction type'}), 400
        
        # Create new transaction
        new_transaction = Transaction(
            AccountID=account_id,
            Type=transaction_type,
            Amount=decimal_amount,
            Description=data.get('description', ''),
            Status='Completed',
            TransactionDate=datetime.utcnow()
        )
        
        db.session.add(new_transaction)
        db.session.commit()
        
        # Return the new transaction and updated account balance
        transaction_data = {
            'TransactionID': new_transaction.TransactionID,
            'Type': new_transaction.Type,
            'Amount': str(new_transaction.Amount),
            'Description': new_transaction.Description,
            'Status': new_transaction.Status,
            'TransactionDate': new_transaction.TransactionDate.isoformat()
        }
        
        return jsonify({
            'message': f'{transaction_type} processed successfully',
            'transaction': transaction_data,
            'newBalance': str(account.Balance)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        # Log the full error for debugging
        import traceback
        print(f"Transaction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
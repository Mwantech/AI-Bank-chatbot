from models.models import db, Account, Transaction
from decimal import Decimal

class AccountService:
    @staticmethod
    def get_balance(user_id):
        accounts = Account.query.filter_by(UserID=user_id).all()
        return [{
            'account_number': acc.AccountNumber,
            'account_type': acc.AccountType,
            'balance': float(acc.Balance)
        } for acc in accounts]

    @staticmethod
    def get_transactions(account_id):
        transactions = Transaction.query.filter_by(AccountID=account_id).all()
        return [{
            'type': t.TransactionType,
            'amount': float(t.Amount),
            'date': t.TransactionDate,
            'description': t.Description
        } for t in transactions]

    @staticmethod
    def transfer_funds(from_account_id, to_account_number, amount):
        try:
            from_account = Account.query.get(from_account_id)
            to_account = Account.query.filter_by(AccountNumber=to_account_number).first()
            
            if not from_account or not to_account:
                return False, "Invalid account"
            
            if from_account.Balance < Decimal(str(amount)):
                return False, "Insufficient funds"
            
            # Create transactions
            from_transaction = Transaction(
                AccountID=from_account.AccountID,
                TransactionType='Transfer',
                Amount=-amount,
                Description=f"Transfer to {to_account_number}"
            )
            
            to_transaction = Transaction(
                AccountID=to_account.AccountID,
                TransactionType='Transfer',
                Amount=amount,
                Description=f"Transfer from {from_account.AccountNumber}"
            )
            
            # Update balances
            from_account.Balance -= Decimal(str(amount))
            to_account.Balance += Decimal(str(amount))
            
            db.session.add(from_transaction)
            db.session.add(to_transaction)
            db.session.commit()
            
            return True, "Transfer successful"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
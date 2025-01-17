from models.models import db, Investment
from datetime import datetime, timedelta

class InvestmentService:
    @staticmethod
    def create_investment(user_id, investment_type, amount):
        try:
            investment = Investment(
                UserID=user_id,
                InvestmentType=investment_type,
                InvestmentAmount=amount,
                InterestRate=6.5,  # Example fixed rate
                StartDate=datetime.now().date(),
                MaturityDate=(datetime.now() + timedelta(days=365)).date()
            )
            db.session.add(investment)
            db.session.commit()
            return True, "Investment created successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)

    @staticmethod
    def get_investments(user_id):
        investments = Investment.query.filter_by(UserID=user_id).all()
        return [{
            'type': inv.InvestmentType,
            'amount': float(inv.InvestmentAmount),
            'interest_rate': float(inv.InterestRate),
            'start_date': inv.StartDate,
            'maturity_date': inv.MaturityDate,
            'status': inv.Status
        } for inv in investments]
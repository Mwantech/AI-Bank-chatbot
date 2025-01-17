from models.models import db, Loan

class LoanService:
    @staticmethod
    def apply_for_loan(user_id, loan_type, amount):
        try:
            loan = Loan(
                UserID=user_id,
                LoanType=loan_type,
                LoanAmount=amount,
                InterestRate=8.5  # Example fixed rate
            )
            db.session.add(loan)
            db.session.commit()
            return True, "Loan application submitted successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)

    @staticmethod
    def get_loan_status(user_id):
        loans = Loan.query.filter_by(UserID=user_id).all()
        return [{
            'loan_type': loan.LoanType,
            'amount': float(loan.LoanAmount),
            'status': loan.LoanStatus,
            'interest_rate': float(loan.InterestRate)
        } for loan in loans]
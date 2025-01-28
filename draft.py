import traceback
from models.models import (
    User, Account, Transaction, 
    Loan, CustomerSupport, Investment, db
)
from banking_chatbot import BankingChatbot
from sqlalchemy.exc import SQLAlchemyError
import logging

class BankingNLPService:
    def __init__(self):
        """Initialize NLP service with trained chatbot"""
        self.chatbot = BankingChatbot()
        self.logger = logging.getLogger(__name__)

    def process_user_request(self, user_id, user_input):
        """
        Process user input and map to banking operations
        
        Args:
            user_id (int): Authenticated user's ID
            user_input (str): User's natural language input
        
        Returns:
            dict: Contextual response with operation results
        """
        try:
            # Get intent and context from chatbot
            response = self.chatbot.get_response(str(user_id), user_input)
            intent = response['intent']
            entities = response.get('entities', {})

            # Mapping of intents to database operations
            intent_handlers = {
                'balance_inquiry': self._handle_balance_inquiry,
                'transaction_inquiry': self._handle_transaction_inquiry,
                'loan_inquiry': self._handle_loan_inquiry,
                'investment_inquiry': self._handle_investment_inquiry,
                'customer_support': self._handle_customer_support
            }

            # Execute appropriate handler if intent matches
            handler = intent_handlers.get(intent)
            if handler:
                operation_result = handler(user_id, entities)
                response['operation_result'] = operation_result

            return response

        except Exception as e:
            self.logger.error(f"NLP Processing Error: {str(e)}")
            traceback.print_exc()
            return {
                'response': "I'm sorry, I couldn't complete your request.",
                'intent': 'error',
                'entities': {},
                'operation_result': None
            }

    def _handle_balance_inquiry(self, user_id, entities):
        """Retrieve account balance"""
        try:
            account_type = entities.get('account_type', 'Savings').capitalize()
            account = Account.query.filter_by(
                UserID=user_id, 
                AccountType=account_type
            ).first()

            return {
                'account_type': account.AccountType,
                'balance': float(account.Balance),
                'account_number': account.AccountNumber
            } if account else {'error': 'Account not found'}

        except SQLAlchemyError as e:
            self.logger.error(f"Balance Inquiry Error: {str(e)}")
            return {'error': 'Unable to retrieve balance'}

    def _handle_transaction_inquiry(self, user_id, entities):
        """Retrieve recent transactions"""
        try:
            account = Account.query.filter_by(UserID=user_id).first()
            
            if account:
                transactions = Transaction.query.filter_by(
                    AccountID=account.AccountID
                ).order_by(Transaction.TransactionDate.desc()).limit(5).all()

                return {
                    'transactions': [
                        {
                            'type': t.TransactionType,
                            'amount': float(t.Amount),
                            'date': t.TransactionDate.isoformat(),
                            'description': t.Description
                        } for t in transactions
                    ]
                }
            return {'error': 'No transactions found'}

        except SQLAlchemyError as e:
            self.logger.error(f"Transaction Inquiry Error: {str(e)}")
            return {'error': 'Unable to retrieve transactions'}

    def _handle_loan_inquiry(self, user_id, entities):
        """Retrieve loan details"""
        try:
            loan_type = entities.get('loan_type', '').capitalize()
            
            loan = (Loan.query.filter_by(UserID=user_id, LoanType=loan_type).first() 
                    if loan_type 
                    else Loan.query.filter_by(UserID=user_id).first())

            return {
                'loan_type': loan.LoanType,
                'amount': float(loan.LoanAmount),
                'interest_rate': float(loan.InterestRate),
                'status': loan.LoanStatus
            } if loan else {'error': 'No loans found'}

        except SQLAlchemyError as e:
            self.logger.error(f"Loan Inquiry Error: {str(e)}")
            return {'error': 'Unable to retrieve loan details'}

    def _handle_investment_inquiry(self, user_id, entities):
        """Retrieve investment details"""
        try:
            investment_type = entities.get('investment_type', '').capitalize()
            
            investment = (Investment.query.filter_by(UserID=user_id, InvestmentType=investment_type).first() 
                          if investment_type 
                          else Investment.query.filter_by(UserID=user_id).first())

            return {
                'investment_type': investment.InvestmentType,
                'amount': float(investment.InvestmentAmount),
                'interest_rate': float(investment.InterestRate),
                'start_date': investment.StartDate.isoformat(),
                'maturity_date': investment.MaturityDate.isoformat(),
                'status': investment.Status
            } if investment else {'error': 'No investments found'}

        except SQLAlchemyError as e:
            self.logger.error(f"Investment Inquiry Error: {str(e)}")
            return {'error': 'Unable to retrieve investment details'}

    def _handle_customer_support(self, user_id, entities):
        """Create customer support request"""
        try:
            request_type = entities.get('support_type', 'FAQ')
            
            support_request = CustomerSupport(
                UserID=user_id,
                RequestType=request_type,
                RequestDetails=entities.get('request_details', 'General inquiry')
            )
            
            db.session.add(support_request)
            db.session.commit()

            return {
                'support_id': support_request.SupportID,
                'request_type': support_request.RequestType,
                'status': support_request.Status
            }

        except SQLAlchemyError as e:
            db.session.rollback()
            self.logger.error(f"Customer Support Error: {str(e)}")
            return {'error': 'Unable to process support request'}

        
from flask import Blueprint, request, jsonify
from functools import wraps
import jwt
import os
from models.models import User
from services.nlp_service import BankingNLPService

chatbot_bp = Blueprint('chatbot', __name__)

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').split(" ")[-1]
        
        if not token:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            request.user_id = payload.get('user_id')
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

@chatbot_bp.route('/process', methods=['POST'])
@token_required
def process_nlp_request():
    user_id = str(request.user_id)
    user_input = request.json.get('input')
    
    if not user_input:
        return jsonify({'error': 'Missing input'}), 400
    
    nlp_service = BankingNLPService()
    result = nlp_service.process_user_request(user_id, user_input)
    
    return jsonify(result)


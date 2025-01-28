
from flask import Blueprint, request, jsonify
from functools import wraps
import jwt
import os
from datetime import datetime
from banking_chatbot.models.models import User, Account, Transaction, CustomerSupport, Loan
from banking_chatbot.services.Nlp_service import BankingNLPService
from config import db

chatbot_bp = Blueprint('chatbot', __name__)

# Create a single instance of NLPService to maintain conversation state
nlp_service = BankingNLPService()

# JWT setup remains the same...
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY must be set in environment variables")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        try:
            token = auth_header.split(" ")[-1]
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(payload['user_id'])
            
            if not current_user:
                return jsonify({'error': 'User not found'}), 404
            
            if current_user.AccountStatus != 'Active':
                return jsonify({'error': 'Account is inactive'}), 403
                
            kwargs['current_user'] = current_user
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
    return decorated

@chatbot_bp.route('/process', methods=['POST'])
@token_required
def process_message(current_user):
    """Process a banking query and return the response"""
    data = request.json
    user_input = data.get('input')
    
    if not user_input:
        return jsonify({'error': 'Missing input message'}), 400
    
    try:
        # Get user's account information for context
        accounts = Account.query.filter_by(UserID=current_user.UserID).all()
        account_info = [{
            'account_number': acc.AccountNumber,
            'account_type': acc.AccountType,
            'balance': float(acc.Balance)
        } for acc in accounts]

        # Get conversation state from request and ensure it has required fields
        conversation_state = data.get('context', {}).get('conversation_state', {})
        if not conversation_state:
            conversation_state = {
                'current_intent': None,
                'awaiting_response': False,
                'last_question': None,
                'loan_details': None
            }
        
        # Ensure all required fields exist in the state
        required_fields = ['current_intent', 'awaiting_response', 'last_question', 'loan_details']
        for field in required_fields:
            if field not in conversation_state:
                conversation_state[field] = None
        
        # Update NLP service state
        nlp_service.conversation_state[current_user.UserID] = conversation_state

        # Process with NLP service
        response = nlp_service.process_user_request(
            current_user.UserID,
            user_input,
            context={
                'user_name': f"{current_user.FirstName} {current_user.LastName}",
                'accounts': account_info,
                'previous_context': data.get('context', {})
            }
        )
        
        # Get updated conversation state
        updated_state = nlp_service.conversation_state.get(current_user.UserID, {})
        
        return jsonify({
            'response': response['response'],
            'intent': response.get('intent'),
            'context': {
                'user_id': current_user.UserID,
                'conversation_state': updated_state,
                'accounts': account_info,
                'suggested_actions': response.get('suggested_actions', [])
            }
        })
        
    except Exception as e:
        # Add better error logging
        import traceback
        current_app.logger.error(f"Error processing request: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
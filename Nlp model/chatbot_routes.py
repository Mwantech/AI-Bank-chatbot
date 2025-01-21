from flask import Blueprint, jsonify, request
from services.nlp_service import NLPService
import os

chatbot_bp = Blueprint('chatbot', __name__)
nlp_service = NLPService(model_dir='models')  # Use the trained model instead of OpenAI

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_id = data.get('user_id', 'default')
        response = nlp_service.get_response(data['message'], user_id)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'intent': 'error',
            'response': "I apologize, but I'm experiencing technical difficulties. Please try again later."
        }), 500
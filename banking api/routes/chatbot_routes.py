from flask import Blueprint, jsonify, request
from services.nlp_service import NLPService
import os

chatbot_bp = Blueprint('chatbot', __name__)
nlp_service = NLPService(api_key=os.environ.get("OPENAI_API_KEY"))

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
        return jsonify({'error': str(e)}), 500
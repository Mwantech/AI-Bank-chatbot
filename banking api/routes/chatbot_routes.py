
from flask import Blueprint, jsonify, request
from services.nlp_service import NLPService

chatbot_bp = Blueprint('chatbot', __name__)
nlp_service = NLPService('your-openai-api-key')

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['message']
    
    # Get response from cache or generate new one
    response = nlp_service.get_cached_response(user_input)
    
    return jsonify(response)
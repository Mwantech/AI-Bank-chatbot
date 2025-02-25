from flask import Blueprint, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from models import db, ChatSession, ChatMessage
from auth import token_required, verify_token
from datetime import datetime
import logging
import jwt
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint with unique name
chat_bp = Blueprint('chat_routes', __name__, url_prefix='/api/chat')

# Initialize None socketio instance that will be set in init_app
socketio = None

# Initialize chatbot
from chatbot import BankingChatbot
chatbot = BankingChatbot()

# Import InquiryHandler
from inquiry_handler import InquiryHandler
inquiry_handler = InquiryHandler(chatbot)

# Active sessions store
active_sessions = {}

def socket_auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'token' in request.args:
            token = request.args['token']
            
        if not token:
            return emit('error', {'message': 'Token is missing'})
            
        try:
            current_user = verify_token(token)
            return f(current_user, *args, **kwargs)
        except jwt.ExpiredSignatureError:
            emit('error', {'message': 'Token has expired'})
        except jwt.InvalidTokenError:
            emit('error', {'message': 'Invalid token'})
    return decorated

@chat_bp.route('/start', methods=['POST'])
@token_required
def start_chat_session(current_user):
    try:
        session = ChatSession(UserID=current_user.UserID)
        db.session.add(session)
        db.session.commit()
        
        # Initialize session state
        active_sessions[str(session.SessionID)] = {
            'user_id': str(current_user.UserID),
            'current_intent': None,
            'collected_entities': {},
            'missing_entities': []
        }
        
        return jsonify({
            'message': 'Chat session started',
            'session_id': session.SessionID
        }), 201
    except Exception as e:
        logger.error(f"Error starting chat session: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to start chat session'}), 500

def register_socket_handlers(socketio_instance):
    """Register all socket event handlers"""
    @socketio_instance.on('connect')
    @socket_auth_required
    def handle_connect(current_user):
        user_room = f"user_{current_user.UserID}"
        join_room(user_room)
        emit('connected', {'message': 'Connected to chat server'})

    @socketio_instance.on('disconnect')
    def handle_disconnect():
        logger.info("Client disconnected")

    @socketio_instance.on('message')
    @socket_auth_required
    def handle_message(current_user, data):
        try:
            session_id = data.get('session_id')
            message = data.get('message')
            
            if not session_id or not message:
                emit('error', {'message': 'Session ID and message are required'})
                return
            
            # Validate session
            session = ChatSession.query.filter_by(
                SessionID=session_id,
                UserID=current_user.UserID,
                Status='Active'
            ).first()
            
            if not session:
                emit('error', {'message': 'Invalid or inactive session'})
                return
            
            # Get session state or initialize if not exists
            session_state = active_sessions.get(str(session_id))
            if not session_state:
                session_state = {
                    'user_id': str(current_user.UserID),
                    'current_intent': None,
                    'collected_entities': {},
                    'missing_entities': []
                }
                active_sessions[str(session_id)] = session_state
            
            # Get response from chatbot
            try:
                response_data = chatbot.get_response(
                    message,
                    str(current_user.UserID)
                )
                
                # Ensure required keys exist in response_data
                if not isinstance(response_data, dict):
                    response_data = {
                        'response': 'Error: Received invalid response format',
                        'intent': None,
                        'entities': {},
                        'missing_entities': []
                    }
                
                # Add default values for missing keys
                if 'response' not in response_data:
                    response_data['response'] = "I couldn't process that request properly."
                    logger.warning(f"Missing 'response' key in chatbot response: {response_data}")
                
                if 'intent' not in response_data:
                    response_data['intent'] = None
                
                if 'entities' not in response_data:
                    response_data['entities'] = {}
                
                if 'missing_entities' not in response_data:
                    response_data['missing_entities'] = []
                
                # Check if intent has changed - if so, reset entities
                if response_data.get('intent') and response_data.get('intent') != session_state['current_intent']:
                    # Clear previous state when intent changes
                    session_state['current_intent'] = response_data.get('intent')
                    session_state['collected_entities'] = {}
                    session_state['missing_entities'] = []
                else:
                    # Update current intent if not already set
                    if not session_state['current_intent'] and response_data.get('intent'):
                        session_state['current_intent'] = response_data.get('intent')
                
                # Update collected entities
                if response_data['entities']:
                    session_state['collected_entities'].update(response_data['entities'])
                
                # Handle missing entities case
                if response_data['missing_entities']:
                    session_state['missing_entities'] = response_data['missing_entities']
                    
                    chat_message = ChatMessage(
                        SessionID=session_id,
                        Message=message,
                        Response=response_data['response'],
                        Intent=response_data['intent'] if response_data['intent'] else 'unknown'
                    )
                    db.session.add(chat_message)
                    db.session.commit()
                    
                    emit('response', {
                        'message_id': chat_message.MessageID,
                        'response': response_data['response'],
                        'intent': response_data['intent'],
                        'missing_entities': response_data['missing_entities'],
                        'conversation_active': True
                    })
                    return
                
                # Handle completion of entity collection
                if session_state['missing_entities'] and not response_data['missing_entities']:
                    session_state['missing_entities'] = []
                
                # Store message
                chat_message = ChatMessage(
                    SessionID=session_id,
                    Message=message,
                    Response=response_data['response'],
                    Intent=response_data['intent'] if response_data['intent'] else 'unknown'
                )
                db.session.add(chat_message)
                db.session.commit()
                
                emit('response', {
                    'message_id': chat_message.MessageID,
                    'response': response_data['response'],
                    'intent': response_data['intent'],
                    'entities': response_data['entities'],
                    'conversation_active': bool(session_state['missing_entities'])
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                
                # Create a fallback response
                fallback_response = "I'm sorry, I encountered an error processing your message."
                
                # Store the error message
                chat_message = ChatMessage(
                    SessionID=session_id,
                    Message=message,
                    Response=fallback_response,
                    Intent='error'
                )
                db.session.add(chat_message)
                db.session.commit()
                
                emit('response', {
                    'message_id': chat_message.MessageID,
                    'response': fallback_response,
                    'intent': 'error',
                    'entities': {},
                    'conversation_active': False,
                    'error': str(e)
                })
                
        except Exception as e:
            logger.error(f"Error in handle_message: {str(e)}")
            emit('error', {'message': f'Failed to process message: {str(e)}'})

    @socketio_instance.on('end_session')
    @socket_auth_required
    def handle_end_session(current_user, data):
        try:
            session_id = data.get('session_id')
            
            if not session_id:
                emit('error', {'message': 'Session ID is required'})
                return
            
            session = ChatSession.query.filter_by(
                SessionID=session_id,
                UserID=current_user.UserID,
                Status='Active'
            ).first()
            
            if not session:
                emit('error', {'message': 'Invalid or inactive session'})
                return
            
            # Check if there's an ongoing operation
            session_state = active_sessions.get(str(session_id))
            if session_state and session_state.get('missing_entities'):
                emit('warning', {
                    'message': 'There is an ongoing operation. Do you want to cancel it?',
                    'pending_entities': session_state['missing_entities']
                })
                return
            
            # Close session
            session.Status = 'Completed'
            session.EndTime = datetime.utcnow()
            db.session.commit()
            
            # Clean up session state
            if str(session_id) in active_sessions:
                del active_sessions[str(session_id)]
            
            # Clean up any pending inquiries
            if hasattr(inquiry_handler, 'pending_inquiries') and int(current_user.UserID) in inquiry_handler.pending_inquiries:
                del inquiry_handler.pending_inquiries[int(current_user.UserID)]
            
            chatbot.clear_user_data(str(current_user.UserID))
            
            emit('session_ended', {'message': 'Chat session ended successfully'})
            
        except Exception as e:
            logger.error(f"Error ending chat session: {str(e)}")
            emit('error', {'message': f'Failed to end chat session: {str(e)}'})

    @socketio_instance.on('force_end_session')
    @socket_auth_required
    def handle_force_end_session(current_user, data):
        try:
            session_id = data.get('session_id')
            
            if not session_id:
                emit('error', {'message': 'Session ID is required'})
                return
            
            session = ChatSession.query.filter_by(
                SessionID=session_id,
                UserID=current_user.UserID,
                Status='Active'
            ).first()
            
            if not session:
                emit('error', {'message': 'Invalid or inactive session'})
                return
            
            # Force close session
            session.Status = 'Completed'
            session.EndTime = datetime.utcnow()
            db.session.commit()
            
            # Clean up session state
            if str(session_id) in active_sessions:
                del active_sessions[str(session_id)]
            
            # Clean up any pending inquiries
            if hasattr(inquiry_handler, 'pending_inquiries') and int(current_user.UserID) in inquiry_handler.pending_inquiries:
                del inquiry_handler.pending_inquiries[int(current_user.UserID)]
            
            chatbot.clear_user_data(str(current_user.UserID))
            
            emit('session_ended', {'message': 'Chat session forcefully ended'})
            
        except Exception as e:
            logger.error(f"Error force ending chat session: {str(e)}")
            emit('error', {'message': f'Failed to force end chat session: {str(e)}'})

def init_app(app, socketio_instance):
    """
    Initialize the chat functionality with the Flask app
    
    Args:
        app: Flask application instance
        socketio_instance: Existing SocketIO instance
    """
    try:
        global socketio
        socketio = socketio_instance
        
        # Register socket event handlers
        register_socket_handlers(socketio)
        
        # Register blueprint
        if not any(bp.name == 'chat_routes' for bp in app.blueprints.values()):
            app.register_blueprint(chat_bp)
        
        logger.info("Chat functionality initialized successfully")
        return socketio
        
    except Exception as e:
        logger.error(f"Failed to initialize chat functionality: {str(e)}")
        raise

@chat_bp.route('/history', methods=['GET'])
@token_required
def get_chat_history(current_user):
    try:
        session_id = request.args.get('session_id')
        
        query = ChatMessage.query.join(ChatSession).filter(
            ChatSession.UserID == current_user.UserID
        )
        
        if session_id:
            query = query.filter(ChatMessage.SessionID == session_id)
        
        messages = query.order_by(ChatMessage.Timestamp.desc()).limit(50).all()
        
        return jsonify({
            'messages': [{
                'message_id': msg.MessageID,
                'session_id': msg.SessionID,
                'message': msg.Message,
                'response': msg.Response,
                'intent': msg.Intent,
                'timestamp': msg.Timestamp.isoformat(),
                'additional_data': msg.AdditionalData if hasattr(msg, 'AdditionalData') else None
            } for msg in messages]
        })
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({'error': f'Failed to retrieve chat history: {str(e)}'}), 500
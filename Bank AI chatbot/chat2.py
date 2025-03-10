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

    # In your chat.py file, update the handle_message function

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
                    'missing_entities': [],
                    'verification_step': 0,
                    'verification_category': None
                }
                active_sessions[str(session_id)] = session_state
            
            # Log the current session state for debugging
            logger.info(f"Session state before processing: {session_state}")
            
            # Check if we're in the middle of collecting entities
            if session_state['missing_entities']:
                logger.info(f"In entity collection mode, missing: {session_state['missing_entities']}")
                
                # Parse the user's response as potential entity values
                entity_values = [val.strip() for val in message.split(',')]
                
                # If we have the same number of values as missing entities, match them up
                if len(entity_values) == len(session_state['missing_entities']):
                    entities = {}
                    for i, entity_name in enumerate(session_state['missing_entities']):
                        entities[entity_name] = entity_values[i]
                    
                    # Update collected entities
                    session_state['collected_entities'].update(entities)
                    
                    # Use the inquiry handler with the current intent and collected entities
                    intent = session_state['current_intent']
                    all_entities = session_state['collected_entities']
                    
                    # Clear missing entities since we've collected them
                    session_state['missing_entities'] = []
                    
                    if intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                        logger.info(f"Processing {intent} with entities: {all_entities}")
                        
                        try:
                            # Use InquiryHandler to process the request
                            inquiry_response = inquiry_handler.handle_inquiry(
                                int(current_user.UserID),
                                message,
                                intent,
                                all_entities
                            )
                            
                            if inquiry_response:
                                # Prevent switching to goodbye intent during active entity collection
                                if intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location'] and not inquiry_response.get('process_complete', False):
                                    # Force continuation of the current intent flow
                                    response_text = inquiry_response.get('display_text', inquiry_response.get('message', 'Request processed'))
                                    missing_fields = inquiry_response.get('missing_fields', [])
                                    
                                    # Update session state with any new missing fields
                                    if missing_fields:
                                        session_state['missing_entities'] = missing_fields
                                    
                                    # Store message with the intent preserved as the original intent
                                    chat_message = ChatMessage(
                                        SessionID=session_id,
                                        Message=message,
                                        Response=response_text,
                                        Intent=intent  # Preserve the original intent
                                    )
                                    db.session.add(chat_message)
                                    db.session.commit()
                                    
                                    emit('response', {
                                        'message_id': chat_message.MessageID,
                                        'response': response_text,
                                        'intent': intent,  # Keep the original intent
                                        'entities': all_entities,
                                        'missing_entities': missing_fields if missing_fields else [],
                                        'conversation_active': True  # Force conversation to stay active
                                    })
                                    logger.info(f"Session state after entity processing: {session_state}")
                                    return
                                
                                response_text = inquiry_response.get('display_text', inquiry_response.get('message', 'Request processed'))
                                missing_fields = inquiry_response.get('missing_fields', [])
                                
                                # Check if this is a verification question
                                if inquiry_response.get('is_verification_question'):
                                    # Update verification state
                                    session_state['verification_step'] = inquiry_response.get('verification_step', 0)
                                    session_state['verification_category'] = inquiry_response.get('verification_category')
                                    
                                    # Store and emit response
                                    chat_message = ChatMessage(
                                        SessionID=session_id,
                                        Message=message,
                                        Response=response_text,
                                        Intent=intent
                                    )
                                    db.session.add(chat_message)
                                    db.session.commit()
                                    
                                    emit('response', {
                                        'message_id': chat_message.MessageID,
                                        'response': response_text,
                                        'intent': intent,
                                        'entities': all_entities,
                                        'is_verification': True,
                                        'conversation_active': True
                                    })
                                    logger.info(f"Session state after verification setup: {session_state}")
                                    return
                                # If still incomplete, update missing entities
                                elif missing_fields:
                                    session_state['missing_entities'] = missing_fields
                                    response_data = {
                                        'response': response_text,
                                        'intent': intent,
                                        'entities': all_entities,
                                        'missing_entities': missing_fields
                                    }
                                else:
                                    # Request completed successfully
                                    response_data = {
                                        'response': response_text,
                                        'intent': intent,
                                        'entities': all_entities
                                    }
                                    
                                # Store message with the actual response from inquiry handler
                                chat_message = ChatMessage(
                                    SessionID=session_id,
                                    Message=message,
                                    Response=response_data['response'],
                                    Intent=intent
                                )
                                db.session.add(chat_message)
                                db.session.commit()
                                
                                emit('response', {
                                    'message_id': chat_message.MessageID,
                                    'response': response_data['response'],
                                    'intent': intent,
                                    'entities': all_entities,
                                    'missing_entities': missing_fields if missing_fields else [],
                                    'conversation_active': bool(missing_fields) or intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']
                                })
                                logger.info(f"Session state after processing inquiry: {session_state}")
                                return
                        except Exception as e:
                            logger.error(f"Error processing inquiry: {str(e)}")
                            # Add stack trace for better debugging
                            import traceback
                            logger.error(traceback.format_exc())
                            emit('error', {'message': f'Failed to process inquiry: {str(e)}'})
                            return
                else:
                    # If we couldn't match entities, maybe it's freeform input or a single value
                    if len(session_state['missing_entities']) == 1:
                        # If only one entity is missing, treat the whole message as that entity
                        entity_name = session_state['missing_entities'][0]
                        session_state['collected_entities'][entity_name] = message
                        
                        # Process with the single entity
                        intent = session_state['current_intent']
                        all_entities = session_state['collected_entities']
                        session_state['missing_entities'] = []
                        
                        # Process the inquiry with the updated entities
                        if intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                            try:
                                inquiry_response = inquiry_handler.handle_inquiry(
                                    int(current_user.UserID),
                                    message,
                                    intent,
                                    all_entities
                                )
                                
                                if inquiry_response:
                                    # Prevent switching to goodbye intent during active entity collection
                                    if intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location'] and not inquiry_response.get('process_complete', False):
                                        # Force continuation of the current intent flow
                                        response_text = inquiry_response.get('display_text', inquiry_response.get('message', 'Request processed'))
                                        missing_fields = inquiry_response.get('missing_fields', [])
                                        
                                        # Update session state with any new missing fields
                                        if missing_fields:
                                            session_state['missing_entities'] = missing_fields
                                        
                                        # Store message with the intent preserved as the original intent
                                        chat_message = ChatMessage(
                                            SessionID=session_id,
                                            Message=message,
                                            Response=response_text,
                                            Intent=intent  # Preserve the original intent
                                        )
                                        db.session.add(chat_message)
                                        db.session.commit()
                                        
                                        emit('response', {
                                            'message_id': chat_message.MessageID,
                                            'response': response_text,
                                            'intent': intent,  # Keep the original intent
                                            'entities': all_entities,
                                            'missing_entities': missing_fields if missing_fields else [],
                                            'conversation_active': True  # Force conversation to stay active
                                        })
                                        logger.info(f"Session state after single entity processing: {session_state}")
                                        return
                                    
                                    response_text = inquiry_response.get('display_text', inquiry_response.get('message', 'Request processed'))
                                    missing_fields = inquiry_response.get('missing_fields', [])
                                    
                                    # Check if this is a verification question
                                    if inquiry_response.get('is_verification_question'):
                                        # Update verification state
                                        session_state['verification_step'] = inquiry_response.get('verification_step', 0)
                                        session_state['verification_category'] = inquiry_response.get('verification_category')
                                        
                                        # Store and emit response
                                        chat_message = ChatMessage(
                                            SessionID=session_id,
                                            Message=message,
                                            Response=response_text,
                                            Intent=intent
                                        )
                                        db.session.add(chat_message)
                                        db.session.commit()
                                        
                                        emit('response', {
                                            'message_id': chat_message.MessageID,
                                            'response': response_text,
                                            'intent': intent,
                                            'entities': all_entities,
                                            'is_verification': True,
                                            'conversation_active': True
                                        })
                                        logger.info(f"Session state after verification setup: {session_state}")
                                        return
                                    # Update with any still missing fields
                                    elif missing_fields:
                                        session_state['missing_entities'] = missing_fields
                                        response_data = {
                                            'response': response_text,
                                            'intent': intent,
                                            'entities': all_entities,
                                            'missing_entities': missing_fields
                                        }
                                    else:
                                        # Request completed successfully
                                        response_data = {
                                            'response': response_text,
                                            'intent': intent,
                                            'entities': all_entities
                                        }
                                        
                                    # Store message
                                    chat_message = ChatMessage(
                                        SessionID=session_id,
                                        Message=message,
                                        Response=response_data['response'],
                                        Intent=intent
                                    )
                                    db.session.add(chat_message)
                                    db.session.commit()
                                    
                                    emit('response', {
                                        'message_id': chat_message.MessageID,
                                        'response': response_data['response'],
                                        'intent': intent,
                                        'entities': all_entities,
                                        'missing_entities': missing_fields if missing_fields else [],
                                        'conversation_active': bool(missing_fields) or intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']
                                    })
                                    logger.info(f"Session state after processing single entity inquiry: {session_state}")
                                    return
                            except Exception as e:
                                logger.error(f"Error processing single entity inquiry: {str(e)}")
                                import traceback
                                logger.error(traceback.format_exc())
                                emit('error', {'message': f'Failed to process inquiry: {str(e)}'})
                                return
            
                            # In the verification handling section, modify this part:
                if session_state.get('verification_step', 0) > 0:
                    try:
                        # Process verification response
                        intent = session_state['current_intent']
                        all_entities = session_state['collected_entities']
                        
                        # Add the verification answer to entities
                        if session_state.get('verification_category') and session_state.get('verification_step'):
                            category = session_state['verification_category']
                            step = session_state['verification_step'] - 1  # Convert to 0-indexed
                            
                            # Get the field name to store this answer
                            field_name = inquiry_handler.verification_questions[category][step]['field']
                            all_entities[field_name] = message
                            
                        # Update collected entities in session state
                        session_state['collected_entities'] = all_entities
                        
                        # Process with inquiry handler
                        inquiry_response = inquiry_handler.handle_verification(
                            int(current_user.UserID),
                            message,
                            intent,
                            all_entities,
                            session_state['verification_category'],
                            session_state['verification_step']
                        )
                        
                        if inquiry_response:
                            # Extract the display_text if it exists
                            response_text = inquiry_response.get('display_text', inquiry_response.get('message', 'Request processed'))
                            # Prevent switching to goodbye intent during verification
                            if intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location'] and not inquiry_response.get('verification_complete', False):
                                # Force continuation of verification
                                # Check if we have another verification question or need to continue verification
                                if inquiry_response.get('is_verification_question') or inquiry_response.get('continue_verification', False):
                                    # Update verification state - only update if provided in response
                                    if 'verification_step' in inquiry_response:
                                        session_state['verification_step'] = inquiry_response.get('verification_step', 0)
                                    if 'verification_category' in inquiry_response:
                                        session_state['verification_category'] = inquiry_response.get('verification_category')
                                    
                                    # Store and emit response
                                    chat_message = ChatMessage(
                                        SessionID=session_id,
                                        Message=message,
                                        Response=inquiry_response['message'],
                                        Intent=intent
                                    )
                                    db.session.add(chat_message)
                                    db.session.commit()
                                    
                                    emit('response', {
                                        'message_id': chat_message.MessageID,
                                        'response': inquiry_response['message'],
                                        'intent': intent,
                                        'entities': all_entities,
                                        'is_verification': True,
                                        'conversation_active': True
                                    })
                                    logger.info(f"Session state after verification processing: {session_state}")
                                    return
                                
                                # Continue verification by default
                                chat_message = ChatMessage(
                                    SessionID=session_id,
                                    Message=message,
                                    Response=inquiry_response['message'],
                                    Intent=intent
                                )
                                db.session.add(chat_message)
                                db.session.commit()
                                
                                emit('response', {
                                    'message_id': chat_message.MessageID,
                                    'response': inquiry_response['message'],
                                    'intent': intent,
                                    'entities': all_entities,
                                    'is_verification': True,
                                    'conversation_active': True
                                })
                                logger.info(f"Session state after continued verification: {session_state}")
                                return
                            
                            # Check if we have another verification question or need to continue verification
                            if inquiry_response.get('is_verification_question') or inquiry_response.get('continue_verification', False):
                                # Update verification state - only update if provided in response
                                if 'verification_step' in inquiry_response:
                                    session_state['verification_step'] = inquiry_response.get('verification_step', 0)
                                if 'verification_category' in inquiry_response:
                                    session_state['verification_category'] = inquiry_response.get('verification_category')
                                
                                # Store and emit response
                                chat_message = ChatMessage(
                                    SessionID=session_id,
                                    Message=message,
                                    Response=inquiry_response['message'],
                                    Intent=intent
                                )
                                db.session.add(chat_message)
                                db.session.commit()
                                
                                emit('response', {
                                    'message_id': chat_message.MessageID,
                                    'response': inquiry_response['message'],
                                    'intent': intent,
                                    'entities': all_entities,
                                    'is_verification': True,
                                    'conversation_active': True
                                })
                                logger.info(f"Session state after verification question: {session_state}")
                                return
                            # Only clear verification state if we've explicitly completed ALL verification questions
                            elif inquiry_response.get('verification_complete', False):
                                # Verification completed, clear verification state
                                session_state['verification_step'] = 0
                                session_state['verification_category'] = None
                                
                                # Store final response
                                chat_message = ChatMessage(
                                    SessionID=session_id,
                                    Message=message,
                                    Response=inquiry_response['message'],
                                    Intent=intent
                                )
                                db.session.add(chat_message)
                                db.session.commit()
                                
                                emit('response', {
                                    'message_id': chat_message.MessageID,
                                    'response': inquiry_response['message'],
                                    'intent': intent,
                                    'entities': all_entities,
                                    'conversation_active': False
                                })
                                logger.info(f"Session state after verification completion: {session_state}")
                                return
                            else:
                                # Continue verification by default unless explicitly told to stop
                                # This prevents premature termination of the verification sequence
                                chat_message = ChatMessage(
                                    SessionID=session_id,
                                    Message=message,
                                    Response=inquiry_response['message'],
                                    Intent=intent
                                )
                                db.session.add(chat_message)
                                db.session.commit()
                                
                                emit('response', {
                                    'message_id': chat_message.MessageID,
                                    'response': inquiry_response['message'],
                                    'intent': intent,
                                    'entities': all_entities,
                                    'is_verification': True,
                                    'conversation_active': True
                                })
                                logger.info(f"Session state after default continuation: {session_state}")
                                return
                    except Exception as e:
                        logger.error(f"Error processing verification: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        emit('error', {'message': f'Failed to process verification: {str(e)}'})
                        return
            
            # Check if we're awaiting confirmation for intent switching
            if session_state.get('awaiting_confirmation'):
                if session_state['confirmation_type'] == 'intent_switch':
                    if message.lower() in ['yes', 'y', 'sure', 'ok', 'okay']:
                        # User confirmed switch, update intent and entities
                        new_intent = session_state.pop('potential_new_intent')
                        new_entities = session_state.pop('potential_entities', {})
                        
                        session_state['current_intent'] = new_intent
                        session_state['collected_entities'] = new_entities
                        session_state['missing_entities'] = []
                        session_state.pop('awaiting_confirmation')
                        session_state.pop('confirmation_type')
                        
                        # Check for required entities for the new intent
                        if new_intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                            try:
                                required_entities = inquiry_handler._get_required_entities(new_intent)
                                collected_entities = set(session_state['collected_entities'].keys())
                                missing_entities = [e for e in required_entities if e not in collected_entities]
                                
                                if missing_entities:
                                    logger.info(f"Missing entities for {new_intent}: {missing_entities}")
                                    session_state['missing_entities'] = missing_entities
                                    
                                    # Generate response asking for missing entities
                                    response_text = f"Please provide the following information: {', '.join(missing_entities)}"
                                    
                                    chat_message = ChatMessage(
                                        SessionID=session_id,
                                        Message=message,
                                        Response=response_text,
                                        Intent=new_intent
                                    )
                                    db.session.add(chat_message)
                                    db.session.commit()
                                    
                                    emit('response', {
                                        'message_id': chat_message.MessageID,
                                        'response': response_text,
                                        'intent': new_intent,
                                        'entities': session_state['collected_entities'],
                                        'missing_entities': missing_entities,
                                        'conversation_active': True
                                    })
                                    logger.info(f"Session state after intent switch - missing entities: {session_state}")
                                    return
                            except Exception as e:
                                logger.error(f"Error getting required entities: {str(e)}")
                        
                        # If all entities are collected, proceed with normal flow
                        response_text = f"Switched to processing your {new_intent} request."
                        
                        chat_message = ChatMessage(
                            SessionID=session_id,
                            Message=message,
                            Response=response_text,
                            Intent=new_intent
                        )
                        db.session.add(chat_message)
                        db.session.commit()
                        
                        emit('response', {
                            'message_id': chat_message.MessageID,
                            'response': response_text,
                            'intent': new_intent,
                            'entities': session_state['collected_entities'],
                            'conversation_active': True
                        })
                        logger.info(f"Session state after intent switch completion: {session_state}")
                        return
                    else:
                        # User declined to switch, continue with original intent
                        original_intent = session_state['current_intent']
                        session_state.pop('potential_new_intent', None)
                        session_state.pop('potential_entities', None) 
                        session_state.pop('awaiting_confirmation')
                        session_state.pop('confirmation_type')
                        
                        response_text = f"Continuing with your {original_intent} request."
                        
                        if session_state['missing_entities']:
                            response_text += f" Please provide: {', '.join(session_state['missing_entities'])}"
                        
                        chat_message = ChatMessage(
                            SessionID=session_id,
                            Message=message,
                            Response=response_text,
                            Intent=original_intent
                        )
                        db.session.add(chat_message)
                        db.session.commit()
                        
                        emit('response', {
                            'message_id': chat_message.MessageID,
                            'response': response_text,
                            'intent': original_intent,
                            'entities': session_state['collected_entities'],
                            'missing_entities': session_state['missing_entities'],
                            'conversation_active': True
                        })
                        logger.info(f"Session state after declining intent switch: {session_state}")
                        return
            
            # Normal processing for new intents or when entity matching failed
            try:
                response_data = chatbot.get_response(
                    message,
                    str(current_user.UserID)
                )
                
                logger.info(f"Chatbot response data: {response_data}")
                
                # Make sure response contains required fields
                if 'response' not in response_data:
                    response_data['response'] = "I couldn't process that properly. Can you try again?"
                
                if 'intent' not in response_data:
                    response_data['intent'] = session_state.get('current_intent', 'unknown')
                
                # Check if we're getting a 'goodbye' intent during an important process
                if response_data['intent'] == 'goodbye' and session_state.get('current_intent') in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                    # Check if we're in the middle of an important process
                    if session_state.get('verification_step', 0) > 0 or session_state.get('missing_entities') or session_state.get('collected_entities'):
                        # Override the goodbye intent to continue the process
                        response_data['intent'] = session_state['current_intent']
                        response_data['response'] = "Let's continue with your " + session_state['current_intent'].replace('_', ' ') + " process. " 
                        
                        # If we have missing entities, ask for them
                        if session_state.get('missing_entities'):
                            response_data['response'] += f"Please provide the following information: {', '.join(session_state['missing_entities'])}"
                        # Otherwise, provide a generic continuation message
                        else:
                            response_data['response'] += "How can I assist you further with this request?"
                
                # Handle intent switching - if there's a current intent and missing entities,
                # confirm with user if they want to switch
                if (session_state['current_intent'] and 
                    session_state['current_intent'] not in ['unknown', 'fallback', 'goodbye'] and
                    response_data['intent'] != session_state['current_intent'] and
                    response_data['intent'] not in ['unknown', 'fallback', 'goodbye']):
                    
                    # If we were in the middle of collecting entities, ask for confirmation
                    if session_state.get('collected_entities'):
                        # Store the potential new intent in session state
                        session_state['potential_new_intent'] = response_data['intent']
                        session_state['potential_entities'] = response_data.get('entities', {})
                        
                        confirmation_msg = (
                            f"You were in the middle of a {session_state['current_intent']} request. "
                            f"Would you like to switch to handling your {response_data['intent']} request instead? "
                            f"(Yes/No)"
                        )
                        
                        # Store confirmation message
                        chat_message = ChatMessage(
                            SessionID=session_id,
                            Message=message,
                            Response=confirmation_msg,
                            Intent='confirmation'
                        )
                        db.session.add(chat_message)
                        db.session.commit()
                        
                        # Set a confirmation state flag
                        session_state['awaiting_confirmation'] = True
                        session_state['confirmation_type'] = 'intent_switch'
                        
                        emit('response', {
                            'message_id': chat_message.MessageID,
                            'response': confirmation_msg,
                            'intent': 'confirmation',
                            'original_intent': session_state['current_intent'],
                            'new_intent': response_data['intent'],
                            'conversation_active': True
                        })
                        logger.info(f"Session state after intent switch confirmation request: {session_state}")
                        return
                    # Check if this is a new intent
                new_intent = response_data.get('intent')
                if new_intent and (not session_state['current_intent'] or new_intent != session_state['current_intent']) and new_intent not in ['unknown', 'fallback', 'goodbye']:
                    logger.info(f"New intent detected: {new_intent}")
                    session_state['current_intent'] = new_intent
                    session_state['collected_entities'] = response_data.get('entities', {})
                    
                    # Check if we need to collect more entities for this intent
                    if new_intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                        try:
                            required_entities = inquiry_handler._get_required_entities(new_intent)
                            collected_entities = set(session_state['collected_entities'].keys())
                            missing_entities = [e for e in required_entities if e not in collected_entities]
                            
                            if missing_entities:
                                logger.info(f"Missing entities for {new_intent}: {missing_entities}")
                                session_state['missing_entities'] = missing_entities
                                response_data['missing_entities'] = missing_entities
                                response_data['response'] = f"Please provide the following information: {', '.join(missing_entities)}"
                        except Exception as e:
                            logger.error(f"Error getting required entities: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                            response_data['response'] = "I couldn't process that request. Please try again."
                
                # Store message
                chat_message = ChatMessage(
                    SessionID=session_id,
                    Message=message,
                    Response=response_data['response'],
                    Intent=response_data['intent']
                )
                db.session.add(chat_message)
                db.session.commit()
                
                emit('response', {
                    'message_id': chat_message.MessageID,
                    'response': response_data['response'],
                    'intent': response_data['intent'],
                    'entities': response_data.get('entities', {}),
                    'missing_entities': response_data.get('missing_entities', []),
                    'conversation_active': bool(session_state.get('missing_entities', []))
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                emit('error', {'message': f'Failed to process message: {str(e)}'})
                
        except Exception as e:
            logger.error(f"Error in handle_message: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            
            # Clean up any pending inquiries - safely
            try:
                if hasattr(inquiry_handler, 'pending_inquiries') and int(current_user.UserID) in inquiry_handler.pending_inquiries:
                    del inquiry_handler.pending_inquiries[int(current_user.UserID)]
            except Exception as e:
                logger.warning(f"Error clearing pending inquiries: {str(e)}")
            
            # Safely clear user data
            try:
                chatbot.clear_user_data(str(current_user.UserID))
            except Exception as e:
                logger.warning(f"Error clearing chatbot user data: {str(e)}")
            
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
            
            # Safely clean up any pending inquiries
            try:
                if hasattr(inquiry_handler, 'pending_inquiries') and int(current_user.UserID) in inquiry_handler.pending_inquiries:
                    del inquiry_handler.pending_inquiries[int(current_user.UserID)]
            except Exception as e:
                logger.warning(f"Error clearing pending inquiries in force end: {str(e)}")
            
            # Safely clear chatbot data
            try:
                chatbot.clear_user_data(str(current_user.UserID))
            except Exception as e:
                logger.warning(f"Error clearing chatbot user data in force end: {str(e)}")
            
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
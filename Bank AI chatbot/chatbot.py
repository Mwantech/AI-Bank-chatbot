from typing import Dict, Any, Optional
import random
from datetime import datetime
from models import ChatSession, ChatMessage, db
from inquiry_handler import InquiryHandler
import pickle
import spacy
import re

class BankingChatbot:
    def __init__(self, model_dir='models'):
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load all saved components
        with open(f'{model_dir}/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        with open(f'{model_dir}/classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
            
        with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        with open(f'{model_dir}/responses.pkl', 'rb') as f:
            self.responses = pickle.load(f)
            
        with open(f'{model_dir}/entities.pkl', 'rb') as f:
            self.entities = pickle.load(f)
            
        with open(f'{model_dir}/required_entities.pkl', 'rb') as f:
            self.required_entities = pickle.load(f)
        
        # Initialize inquiry handler
        self.inquiry_handler = InquiryHandler(self)
        
        # Enhanced mapping for specific inquiries
        self.entity_mappings = {
            'account_activation': {
                'account_type': ['checking', 'savings', 'credit', 'investment', 'business'],
                'account_number': r'\b\d{10}\b'
            },
            'account_deactivation': {
                'account_type': ['checking', 'savings', 'credit', 'investment', 'business'],
                'reason_code': ['suspicious', 'fraud', 'security', 'lost', 'stolen'],
                'account_number': r'\b\d{10}\b'
            },
            'loan_status': {
                'application_id': r'APP-\d{5}',
                'loan_type': ['personal', 'home', 'auto', 'business']
            },
            'atm_location': {
                'location': r'\b(?:in|at)\s+([A-Za-z\s]+)(?:\s|$)'
            }
        }
        
        # Define intent descriptions
        self.intent_descriptions = {
            'account_activation': 'Account Activation Request',
            'account_deactivation': 'Account Deactivation Request',
            'loan_status': 'Loan Application Status',
            'atm_location': 'ATM Location Finder',
            'general_query': 'General Banking Information',
            'goodbye': 'End Conversation'
        }
        
        # Active sessions storage
        self.active_sessions: Dict[int, Dict[str, Any]] = {}

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])

    def classify_and_extract(self, text: str) -> Dict[str, Any]:
        """
        Classify intent and extract entities from text
        """
        # Preprocess input text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the text
        X = self.vectorizer.transform([processed_text])
        
        # Predict intent
        intent_idx = self.classifier.predict(X)[0]
        intent = self.label_encoder.inverse_transform([intent_idx])[0]
        
        # Extract entities based on intent
        entities = self.extract_entities(text, intent)
        
        return {
            'intent': intent,
            'entities': entities
        }

    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """
        Extract entities based on intent-specific patterns
        """
        entities = {}
        
        if intent in self.entity_mappings:
            mapping = self.entity_mappings[intent]
            
            for entity_type, patterns in mapping.items():
                if isinstance(patterns, list):
                    # Handle word lists (e.g., account types)
                    for pattern in patterns:
                        if pattern in text.lower():
                            entities[entity_type] = pattern
                            break
                else:
                    # Handle regex patterns
                    if entity_type == 'location':
                        # Special handling for location extraction
                        match = re.search(patterns, text)
                        if match:
                            entities[entity_type] = match.group(1)
                    else:
                        # General regex matching
                        match = re.search(patterns, text)
                        if match:
                            entities[entity_type] = match.group()
        
        # Add general NER entities
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'MONEY', 'GPE']:
                entities[ent.label_.lower()] = ent.text
        
        return entities

    def get_response(self, text: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get response for user message with inquiry handling
        """
        try:
            # Get intent and entities
            analysis = self.classify_and_extract(text)
            intent = analysis['intent']
            entities = analysis['entities']
            
            # Handle specific inquiries through inquiry handler
            if intent in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                if user_id:
                    inquiry_response = self.inquiry_handler.handle_inquiry(
                        user_id, 
                        text,
                        intent=intent,
                        entities=entities
                    )
                    
                    if inquiry_response['status'] != 'error':
                        return {
                            'intent': intent,
                            'intent_description': self.intent_descriptions.get(intent),
                            'entities': entities,
                            'response': inquiry_response['message'],
                            'status': inquiry_response['status'],
                            'data': inquiry_response.get('data', {})
                        }
            
            # Handle other intents with standard responses
            response_template = random.choice(self.responses[intent])
            
            try:
                response = response_template.format(**entities)
            except KeyError:
                response = response_template
            
            return {
                'intent': intent,
                'intent_description': self.intent_descriptions.get(intent),
                'entities': entities,
                'response': response,
                'status': 'success'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error generating response: {str(e)}'
            }

    def start_session(self, user_id: int) -> Dict[str, Any]:
        """
        Start a new chat session for a user
        """
        try:
            # Create new session in database
            session = ChatSession(
                UserID=user_id,
                StartTime=datetime.utcnow(),
                Status='Active'
            )
            db.session.add(session)
            db.session.commit()
            
            # Store session info in memory
            self.active_sessions[user_id] = {
                'session_id': session.SessionID,
                'context': {},
                'last_intent': None,
                'last_entities': None
            }
            
            return {
                'status': 'success',
                'session_id': session.SessionID,
                'message': 'Welcome to banking support. How can I help you today?'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'status': 'error',
                'message': 'Error starting chat session'
            }

    def process_message(self, user_id: int, message: str) -> Dict[str, Any]:
        """
        Process incoming message with context management
        """
        try:
            # Check for active session
            if user_id not in self.active_sessions:
                return {'status': 'error', 'message': 'No active session found'}
            
            session_info = self.active_sessions[user_id]
            
            # Get response based on intent and context
            response = self.get_response(message, user_id)
            
            # Save message and response to database
            chat_message = ChatMessage(
                SessionID=session_info['session_id'],
                Message=message,
                Response=response.get('response', ''),
                Intent=response.get('intent', ''),
                Timestamp=datetime.utcnow()
            )
            db.session.add(chat_message)
            db.session.commit()
            
            # Update session context
            session_info['last_intent'] = response.get('intent')
            session_info['last_entities'] = response.get('entities', {})
            
            return response
            
        except Exception as e:
            db.session.rollback()
            return {
                'status': 'error',
                'message': f'Error processing message: {str(e)}'
            }

    def end_session(self, user_id: int) -> Dict[str, Any]:
        """
        End a user's chat session
        """
        try:
            if user_id in self.active_sessions:
                session_id = self.active_sessions[user_id]['session_id']
                
                # Update session in database
                session = db.session.query(ChatSession).get(session_id)
                if session:
                    session.EndTime = datetime.utcnow()
                    session.Status = 'Completed'
                    db.session.commit()
                
                # Clear session from memory
                del self.active_sessions[user_id]
                
                return {
                    'status': 'success',
                    'message': 'Thank you for using our banking support. Have a great day!'
                }
            
            return {
                'status': 'error',
                'message': 'No active session found'
            }
                
        except Exception as e:
            db.session.rollback()
            return {
                'status': 'error',
                'message': 'Error ending chat session'
            }
from typing import Dict, Any, Optional
import random
from datetime import datetime
from models import ChatSession, ChatMessage, db
from inquiry_handler import InquiryHandler
import pickle
import spacy
import re
import os
import torch
import numpy as np
from pathlib import Path

class BankingChatbot:
    def __init__(self, model_dir='neural_model'):
        self.nlp = spacy.load('en_core_web_sm')
        
        # Auto-detect model type from directory content
        self.model_type = self.detect_model_type(model_dir)
        
        # Load the appropriate model based on detected type
        if self.model_type == 'neural':
            self._load_neural_model(model_dir)
        else:
            self._load_rule_based_model(model_dir)
        
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

    def _load_rule_based_model(self, model_dir):
        """Load rule-based sklearn model components"""
        try:
            # Check if this directory actually contains rule-based model files
            if not os.path.exists(f'{model_dir}/vectorizer.pkl'):
                raise FileNotFoundError(f"Rule-based model files not found in {model_dir}")
                
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
                
            print(f"Successfully loaded rule-based model from {model_dir}")
        except Exception as e:
            raise Exception(f"Error loading rule-based model: {str(e)}")

    def _load_neural_model(self, model_dir):
        """Load neural PyTorch model components"""
        try:
            # Check if this directory actually contains neural model files
            if not os.path.exists(f'{model_dir}/model.pt'):
                raise FileNotFoundError(f"Neural model files not found in {model_dir}")
                
            # Device for PyTorch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load vocabulary and other data
            with open(f'{model_dir}/word_to_ix.pkl', 'rb') as f:
                self.word_to_ix = pickle.load(f)
            
            with open(f'{model_dir}/tag_to_ix.pkl', 'rb') as f:
                self.tag_to_ix = pickle.load(f)
            
            with open(f'{model_dir}/responses.pkl', 'rb') as f:
                self.responses = pickle.load(f)
            
            with open(f'{model_dir}/entities.pkl', 'rb') as f:
                self.entities = pickle.load(f)
            
            with open(f'{model_dir}/required_entities.pkl', 'rb') as f:
                self.required_entities = pickle.load(f)
            
            # Load configuration
            with open(f'{model_dir}/config.pkl', 'rb') as f:
                config = pickle.load(f)
                self.embedding_dim = config['embedding_dim']
                self.hidden_size = config['hidden_size']
                self.max_length = config.get('max_length', 20)
            
            # Define and load model (same as in NeuralBankingChatbot class)
            self.model = self._create_intent_classifier()
            self.model.load_state_dict(torch.load(f'{model_dir}/model.pt', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Successfully loaded neural model from {model_dir}")
            
        except Exception as e:
            raise Exception(f"Error loading neural model: {str(e)}")

    def _create_intent_classifier(self):
        """Create the neural intent classifier model with the same architecture as the training script"""
        class IntentClassifier(torch.nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
                super(IntentClassifier, self).__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                self.fc = torch.nn.Linear(hidden_dim, tagset_size)
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, (hidden, cell) = self.lstm(embedded)
                out = self.fc(hidden[-1])
                return out
        
        # Create the model with the same architecture
        vocab_size = len(self.word_to_ix)
        tagset_size = len(self.tag_to_ix)
        
        return IntentClassifier(vocab_size, self.embedding_dim, self.hidden_size, tagset_size)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])

    def prepare_sequence(self, sentence):
        """Prepare sequence for neural model (same as in NeuralBankingChatbot)"""
        # Simple tokenization
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        tokens = sentence.split()
        
        # Convert to indices
        idxs = [self.word_to_ix.get(w, self.word_to_ix.get("<UNK>", 1)) for w in tokens]
        
        # Pad or truncate sequence to fixed length
        if len(idxs) < self.max_length:
            idxs = idxs + [self.word_to_ix.get("<PAD>", 0)] * (self.max_length - len(idxs))
        else:
            idxs = idxs[:self.max_length]
            
        return torch.tensor(idxs, dtype=torch.long)

    def classify_and_extract(self, text: str) -> Dict[str, Any]:
        """
        Classify intent and extract entities from text
        """
        if self.model_type == 'neural':
            # Neural model classification
            with torch.no_grad():
                sequence = self.prepare_sequence(text).unsqueeze(0).to(self.device)
                outputs = self.model(sequence)
                _, predicted = torch.max(outputs, 1)
                intent = list(self.tag_to_ix.keys())[predicted.item()]
        else:
            # Rule-based classification
            processed_text = self.preprocess_text(text)
            X = self.vectorizer.transform([processed_text])
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
        
        # Custom entity extraction from NeuralBankingChatbot
        doc = self.nlp(text)
        
        # Use spaCy's NER to identify entities
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                entities["amount"] = ent.text
            elif ent.label_ == "DATE":
                entities["date"] = ent.text
            elif ent.label_ == "PERSON":
                entities["person"] = ent.text
            elif ent.label_ == "ORG":
                entities["organization"] = ent.text
            elif ent.label_ == "GPE":
                entities["location"] = ent.text
        
        # Entity mappings for specific intents
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
                    if entity_type == 'location' and 'location' not in entities:
                        # Special handling for location extraction
                        match = re.search(patterns, text)
                        if match:
                            entities[entity_type] = match.group(1)
                    else:
                        # General regex matching
                        match = re.search(patterns, text)
                        if match:
                            entities[entity_type] = match.group()
        
        # Custom entity extraction based on patterns from training data
        for entity_type, patterns in self.entities.items():
            if entity_type not in entities and isinstance(patterns, list):
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            entities[entity_type] = match.group(0)
        
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
            response_templates = self.responses.get(intent, ["I'm not sure how to respond to that."])
            response_template = random.choice(response_templates)
            
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
                'message': f'Error starting chat session: {str(e)}'
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
                'message': f'Error ending chat session: {str(e)}'
            }

    @staticmethod
    def detect_model_type(model_dir):
        """
        Detect which type of model exists in the given directory
        """
        # Check if the directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist")
            
        # First, try to find paths based on model type
        neural_path = os.path.join(model_dir, 'model.pt')
        rule_based_path = os.path.join(model_dir, 'vectorizer.pkl')
            
        # Check for neural model files
        if os.path.exists(neural_path):
            print(f"Neural model detected in {model_dir}")
            return 'neural'
        # Check for rule-based model files
        elif os.path.exists(rule_based_path):
            print(f"Rule-based model detected in {model_dir}")
            return 'rule_based'
        else:
            # Fallback: Check for subdirectories
            neural_dir = os.path.join(model_dir, 'neural_model')
            rule_based_dir = os.path.join(model_dir, 'rule_based_model')
            
            if os.path.exists(neural_dir) and os.path.exists(os.path.join(neural_dir, 'model.pt')):
                print(f"Neural model detected in {neural_dir}")
                return 'neural'
            elif os.path.exists(rule_based_dir) and os.path.exists(os.path.join(rule_based_dir, 'vectorizer.pkl')):
                print(f"Rule-based model detected in {rule_based_dir}")
                return 'rule_based'
            else:
                # Last attempt: look for any model directories
                for root, dirs, files in os.walk(model_dir):
                    if 'model.pt' in files:
                        print(f"Neural model detected in {root}")
                        return 'neural'
                    if 'vectorizer.pkl' in files:
                        print(f"Rule-based model detected in {root}")
                        return 'rule_based'
                        
            raise ValueError(f"No valid model found in {model_dir} or its subdirectories")
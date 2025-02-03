import pickle
import random
import spacy
from pathlib import Path
import re
from datetime import datetime

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
        
        # Enhanced mapping for entity extraction
        self.entity_mappings = {
            'account_activation': {
                'account_type': ['checking', 'savings', 'credit', 'investment', 'business'],
                'card_number': r'\d{6,}'
            },
            'account_deactivation': {
                'account_type': ['checking', 'savings', 'credit', 'investment', 'business'],
                'reason_code': ['suspicious', 'fraud', 'security', 'lost', 'stolen']
            },
            'loan_status': {
                'application_id': r'\d+',
                'loan_type': ['personal', 'home', 'auto', 'business']
            }
        }
        
        self.context = {}
        self.user_info_collection = {}
        
        # Define more explicit intent descriptions
        self.intent_descriptions = {
            'account_balance': 'Checking Account Balance',
            'account_activation': 'Account Activation',
            'account_deactivation': 'Account Deactivation',
            'atm_location': 'ATM Location Finder',
            'loan_status': 'Loan Status Inquiry',
            'goodbye': 'Ending Conversation',
            'general_query': 'General Banking Information'
        }
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])
    
    def extract_entities(self, text, intent=None):
        """Enhanced entity extraction with intent-specific logic"""
        extracted_entities = {}
        
        # If intent is specified, use intent-specific mappings
        if intent and intent in self.entity_mappings:
            mappings = self.entity_mappings[intent]
            
            # Check account types
            if 'account_type' in mappings:
                for account_type in mappings['account_type']:
                    if account_type in text.lower():
                        extracted_entities['account_type'] = account_type
            
            # Check card numbers or application IDs
            if 'card_number' in mappings or 'application_id' in mappings:
                number_pattern = mappings.get('card_number', mappings.get('application_id'))
                numbers = re.findall(number_pattern, text)
                if numbers:
                    key = 'card_number' if 'card_number' in mappings else 'application_id'
                    extracted_entities[key] = numbers[0]
            
            # Check reason codes
            if 'reason_code' in mappings:
                for reason in mappings['reason_code']:
                    if reason in text.lower():
                        extracted_entities['reason_code'] = reason
            
            # Check loan types
            if 'loan_type' in mappings:
                for loan_type in mappings['loan_type']:
                    if loan_type in text.lower():
                        extracted_entities['loan_type'] = loan_type
        
        # Fallback to general extraction
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                extracted_entities['date'] = ent.text
            elif ent.label_ == 'MONEY':
                extracted_entities['amount'] = ent.text
            elif ent.label_ == 'GPE':
                extracted_entities['location'] = ent.text
        
        return extracted_entities
    
    def collect_user_info(self, user_id, intent, entities):
        """Handle the collection of required user information"""
        if user_id not in self.user_info_collection:
            self.user_info_collection[user_id] = {
                'intent': intent,
                'collected_entities': entities,
                'missing_entities': self.check_required_entities(intent, entities)
            }
        else:
            # Update with any new entities
            self.user_info_collection[user_id]['collected_entities'].update(entities)
            self.user_info_collection[user_id]['missing_entities'] = self.check_required_entities(
                intent,
                self.user_info_collection[user_id]['collected_entities']
            )
        
        return self.user_info_collection[user_id]
    
    def check_required_entities(self, intent, entities):
        if intent in self.required_entities:
            missing_entities = []
            for required_entity in self.required_entities[intent]:
                if required_entity not in entities:
                    missing_entities.append(required_entity)
            return missing_entities
        return []
    
    def get_response(self, text, user_id=None):
        # Preprocess input text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the text
        X = self.vectorizer.transform([processed_text])
        
        # Predict intent
        intent_idx = self.classifier.predict(X)[0]
        intent = self.label_encoder.inverse_transform([intent_idx])[0]
        
        # Check if there's an ongoing intent from previous interaction
        if user_id in self.user_info_collection:
            ongoing_intent = self.user_info_collection[user_id]['intent']
            ongoing_missing = self.user_info_collection[user_id]['missing_entities']
            
            # If there's an ongoing intent, use that for entity extraction
            if ongoing_missing:
                entities = self.extract_entities(text, ongoing_intent)
                
                # Update user info collection with new entities
                user_info = self.collect_user_info(user_id, ongoing_intent, entities)
                
                # Check if all required entities are now collected
                if not user_info['missing_entities']:
                    # Prepare response for the original intent
                    response_template = random.choice(self.responses[ongoing_intent])
                    
                    try:
                        response = response_template.format(**user_info['collected_entities'])
                    except KeyError:
                        response = response_template
                    
                    # Clear collected info
                    del self.user_info_collection[user_id]
                    
                    return {
                        'intent': ongoing_intent,
                        'intent_description': self.intent_descriptions.get(ongoing_intent, 'Banking Operation'),
                        'response': response,
                        'entities': user_info['collected_entities']
                    }
                else:
                    # Still missing some entities
                    return {
                        'intent': ongoing_intent,
                        'intent_description': self.intent_descriptions.get(ongoing_intent, 'Banking Operation'),
                        'response': f"To proceed with {self.intent_descriptions.get(ongoing_intent, 'this operation')}, I need the following information: {', '.join(user_info['missing_entities'])}",
                        'missing_entities': user_info['missing_entities']
                    }
        
        # Normal entity extraction for new intent
        entities = self.extract_entities(text, intent)
        
        # Intents that require specific information collection
        info_collection_intents = [
            'loan_status', 'account_activation', 
            'account_deactivation', 'atm_location'
        ]
        
        # For restricted operations that require user info collection
        if intent in info_collection_intents:
            user_info = self.collect_user_info(user_id, intent, entities)
            
            if user_info['missing_entities']:
                return {
                    'intent': intent,
                    'intent_description': self.intent_descriptions.get(intent, 'Banking Operation'),
                    'response': f"To proceed with {self.intent_descriptions.get(intent, 'this operation')}, I need the following information: {', '.join(user_info['missing_entities'])}",
                    'missing_entities': user_info['missing_entities']
                }
        
        # Select and format response
        response_template = random.choice(self.responses[intent])
        
        try:
            response = response_template.format(**entities)
        except KeyError:
            response = response_template
        
        # Clear collected info after successful response
        if user_id in self.user_info_collection:
            del self.user_info_collection[user_id]
        
        return {
            'intent': intent,
            'intent_description': self.intent_descriptions.get(intent, 'Banking Operation'),
            'response': response,
            'entities': entities
        }
    
    def clear_user_data(self, user_id):
        """Clear all user-related data"""
        if user_id in self.user_info_collection:
            del self.user_info_collection[user_id]
        if user_id in self.context:
            del self.context[user_id]
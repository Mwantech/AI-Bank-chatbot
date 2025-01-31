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
        
        self.context = {}
        self.user_info_collection = {}
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])
    
    def extract_entities(self, text):
        doc = self.nlp(text)
        extracted_entities = {}
        
        # Extract numbers (for amounts, account numbers)
        numbers = re.findall(r'\d+', text)
        if numbers:
            extracted_entities['number'] = numbers[0]
            if len(numbers[0]) >= 4:
                extracted_entities['card_last_4'] = numbers[0][-4:]
        
        # Extract account types
        account_types = ['checking', 'savings', 'credit', 'investment', 'retirement', 'business']
        for word in doc:
            if word.text.lower() in account_types:
                extracted_entities['account_type'] = word.text.lower()
        
        # Extract dates
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
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # For restricted operations that require user info collection
        if intent in ['loan_status', 'account_activation', 'account_deactivation', 'atm_location']:
            user_info = self.collect_user_info(user_id, intent, entities)
            
            if user_info['missing_entities']:
                return {
                    'intent': intent,
                    'response': f"I need more information. Please provide: {', '.join(user_info['missing_entities'])}",
                    'missing_entities': user_info['missing_entities']
                }
            
            # All required info collected, use it for response
            entities = user_info['collected_entities']
        else:
            # For non-restricted operations, provide guidance
            return {
                'intent': intent,
                'response': random.choice(self.responses[intent]),
                'entities': entities
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
            'response': response,
            'entities': entities
        }
    
    def clear_user_data(self, user_id):
        """Clear all user-related data"""
        if user_id in self.user_info_collection:
            del self.user_info_collection[user_id]
        if user_id in self.context:
            del self.context[user_id]
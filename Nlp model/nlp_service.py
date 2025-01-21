import json
from pathlib import Path
import os
from datetime import datetime, timedelta
import pickle
import re
from typing import Dict, Any
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging

class NLPService:
    def __init__(self, model_dir: str = 'models'):
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__)))
        self.cache_file = self.base_path / 'data' / 'response_cache.pkl'
        self.cache_duration = timedelta(hours=24)
        self.conversation_context = {}  # Store context for each user
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en_core_web_sm')

        # Create data directory if it doesn't exist
        data_dir = self.base_path / 'data'
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        # Load model and components
        self.model_dir = self.base_path / model_dir
        self.load_model()
        self.load_intents()
        self.load_cache()

    def load_model(self) -> None:
        """Load the trained model components"""
        try:
            with open(self.model_dir / 'vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(self.model_dir / 'patterns.pkl', 'rb') as f:
                patterns_data = pickle.load(f)
                self.patterns = patterns_data['patterns']
                self.pattern_classes = patterns_data['pattern_classes']
            self.X = self.vectorizer.transform(self.patterns)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def load_cache(self):
        """Load cached responses from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    current_time = datetime.now()
                    self.cache = {
                        k: v for k, v in cache_data.items()
                        if current_time - v['timestamp'] < self.cache_duration
                    }
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self):
        """Save responses to cache file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    def load_intents(self):
        """Load intent patterns from JSON file"""
        intent_file = self.base_path / 'data' / 'intent_patterns.json'
        
        if not intent_file.exists():
            default_intents = {
                "intents": {
                    "greeting": {
                        "patterns": ["hello", "hi", "hey", "how are you"],
                        "responses": ["Hello! I'm your banking assistant. How can I help you today?"]
                    },
                    "transfer_money": {
                        "patterns": ["transfer", "send money", "pay someone"],
                        "responses": ["I can help you transfer money. Who would you like to send money to?"]
                    },
                    "loan_inquiry": {
                        "patterns": ["loan", "borrow", "credit"],
                        "responses": ["I can help you with a loan application. What amount are you looking to borrow?"]
                    },
                    "location_search": {
                        "patterns": ["atm", "branch", "location", "find"],
                        "responses": ["I'll locate the closest branch/ATM for you. Where are you currently?"]
                    },
                    "balance_inquiry": {
                        "patterns": ["balance", "how much", "check account"],
                        "responses": ["I'll help you check your balance. Please log in to your account first."]
                    },
                    "general": {
                        "patterns": [],
                        "responses": ["I'll help you with that. Could you provide more details?"]
                    }
                }
            }
            
            intent_file.parent.mkdir(parents=True, exist_ok=True)
            with open(intent_file, 'w') as f:
                json.dump(default_intents, f, indent=4)
            
            self.intent_data = default_intents
        else:
            with open(intent_file, 'r') as f:
                self.intent_data = json.load(f)

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text"""
        try:
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {str(e)}")
            return text

    def get_response(self, user_input: str, user_id: str = 'default') -> Dict[str, Any]:
        """Generate response based on user input"""
        # Initialize or get conversation context
        if user_id not in self.conversation_context:
            self.conversation_context[user_id] = {
                'current_intent': None,
                'conversation_history': [],
                'current_state': None
            }

        context = self.conversation_context[user_id]
        context['conversation_history'].append({"role": "user", "content": user_input})

        try:
            # Check cache first
            cache_key = f"{user_id}:{user_input}"
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < self.cache_duration:
                    return cache_entry['response']

            # Process input and get intent
            processed_text = self.preprocess_text(user_input)
            input_vector = self.vectorizer.transform([processed_text])
            similarities = cosine_similarity(input_vector, self.X)
            most_similar = np.argmax(similarities)
            
            intent = self.pattern_classes[most_similar]
            confidence = float(similarities[0][most_similar])

            # Use context for better understanding
            if confidence < 0.2 and context['current_intent']:
                intent = context['current_intent']
                confidence = 0.3

            # Extract entities and generate response
            entities = self.extract_entities(user_input)
            response = self.format_response(intent, entities, context)

            # Update cache and context
            self.cache[cache_key] = {'response': response, 'timestamp': datetime.now()}
            self.save_cache()
            
            if confidence >= 0.2:
                context['current_intent'] = intent
            
            context['conversation_history'].append({
                "role": "assistant",
                "content": response['response']
            })

            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "intent": "error",
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later."
            }

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract relevant entities from text"""
        try:
            doc = self.nlp(text)
            entities = {
                'amount': None,
                'recipient': None,
                'account_type': None,
                'date': None
            }
            
            # Extract amounts
            amount_pattern = re.compile(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?')
            amounts = amount_pattern.findall(text)
            if amounts:
                entities['amount'] = amounts[0]
            
            # Extract other entities using spaCy
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['recipient'] = ent.text
                elif ent.label_ == 'DATE':
                    entities['date'] = ent.text
                
            return entities
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return {}

    def format_response(self, intent: str, entities: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Format the final response based on intent and entities"""
        try:
            response = self.get_random_response(intent)
            
            # Add context-specific information
            if intent == 'loan_inquiry' and entities.get('amount'):
                amount = entities['amount'].replace('$', '')
                response = f"I see you're interested in a loan of ${amount}. Let me check your eligibility. Would you like to proceed with the loan application?"
            elif intent == 'transfer_money':
                if entities.get('recipient') and entities.get('amount'):
                    response = f"I'll help you transfer ${entities['amount']} to {entities['recipient']}. Please confirm this transaction."
                elif not entities.get('recipient'):
                    response = "Who would you like to send money to?"
                elif not entities.get('amount'):
                    response = f"How much would you like to send to {entities['recipient']}?"

            # Add time-based greeting
            hour = datetime.now().hour
            greeting = ""
            if 5 <= hour < 12:
                greeting = "Good morning! "
            elif 12 <= hour < 17:
                greeting = "Good afternoon! "
            elif 17 <= hour < 22:
                greeting = "Good evening! "

            return {
                'intent': intent,
                'response': f"{greeting}{response}",
                'entities': entities
            }

        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return {
                'intent': 'error',
                'response': "I apologize, but I'm having trouble understanding. Could you please rephrase that?"
            }

    def get_random_response(self, intent: str) -> str:
        """Get a random response for the given intent"""
        import random
        responses = self.intent_data['intents'].get(intent, {}).get('responses', [])
        if not responses:
            responses = ["I understand you need help with that. Let me assist you."]
        return random.choice(responses)
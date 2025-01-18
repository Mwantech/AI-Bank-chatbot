import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
import spacy
import datetime
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

class BankingChatbot:
    def __init__(self, intents_file: str = 'intents.json', model_dir: str = 'models'):
        """
        Initialize the Banking Chatbot with necessary NLP components and intents
        
        Args:
            intents_file (str): Path to the intents JSON file
            model_dir (str): Directory to save/load model files
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup paths
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__)))
        self.model_dir = self.base_path / model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load spaCy model
            self.nlp = spacy.load('en_core_web_sm')
            
            # Load or create model
            if self.model_exists():
                self.load_model()
            else:
                # Load intents and prepare training data
                self.load_intents(intents_file)
                self.patterns: List[str] = []
                self.pattern_classes: List[str] = []
                self.prepare_training_data()
                
                # Initialize and fit vectorizer
                self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
                self.X = self.vectorizer.fit_transform(self.patterns)
                
                # Save the model
                self.save_model()
            
            # Initialize conversation management
            self.conversation_context: Dict[str, Dict] = {}
            self.user_profiles: Dict[str, Dict] = {}
            
            self.logger.info("Banking Chatbot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing chatbot: {str(e)}")
            raise

    def model_exists(self) -> bool:
        """Check if saved model files exist"""
        vectorizer_path = self.model_dir / 'vectorizer.pkl'
        patterns_path = self.model_dir / 'patterns.pkl'
        intents_path = self.model_dir / 'intents.pkl'
        return all(path.exists() for path in [vectorizer_path, patterns_path, intents_path])

    def save_model(self) -> None:
        """Save the trained model components"""
        try:
            with open(self.model_dir / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(self.model_dir / 'patterns.pkl', 'wb') as f:
                pickle.dump({'patterns': self.patterns, 'pattern_classes': self.pattern_classes}, f)
            with open(self.model_dir / 'intents.pkl', 'wb') as f:
                pickle.dump(self.intents, f)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self) -> None:
        """Load the trained model components"""
        try:
            with open(self.model_dir / 'vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(self.model_dir / 'patterns.pkl', 'rb') as f:
                patterns_data = pickle.load(f)
                self.patterns = patterns_data['patterns']
                self.pattern_classes = patterns_data['pattern_classes']
            with open(self.model_dir / 'intents.pkl', 'rb') as f:
                self.intents = pickle.load(f)
            self.X = self.vectorizer.transform(self.patterns)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def load_intents(self, intents_file: str) -> None:
        """Load intents from JSON file"""
        try:
            file_path = self.base_path / intents_file
            with open(file_path, 'r', encoding='utf-8') as file:
                self.intents = json.load(file)
            self.logger.info(f"Intents loaded successfully from {intents_file}")
        except Exception as e:
            self.logger.error(f"Error loading intents file: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> List[str]:
        """Clean and preprocess input text"""
        try:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words]
            return tokens
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {str(e)}")
            return []

    def prepare_training_data(self) -> None:
        """Prepare training data from intents"""
        try:
            for intent in self.intents['intents']:
                for pattern in self.intents['intents'][intent]['patterns']:
                    self.patterns.append(pattern)
                    self.pattern_classes.append(intent)
            self.logger.info("Training data prepared successfully")
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def get_response(self, user_id: str, text: str) -> Dict[str, Any]:
        """Generate appropriate response based on user input"""
        try:
            # Initialize or get user context
            if user_id not in self.conversation_context:
                self.conversation_context[user_id] = {
                    'last_intent': None,
                    'turns': 0,
                    'current_state': None,
                    'conversation_history': []
                }
            
            context = self.conversation_context[user_id]
            context['turns'] += 1
            
            # Add user input to history
            context['conversation_history'].append({
                'role': 'user',
                'content': text,
                'timestamp': datetime.datetime.now()
            })
            
            # Process input and get intent
            processed_text = ' '.join(self.preprocess_text(text))
            input_vector = self.vectorizer.transform([processed_text])
            similarities = cosine_similarity(input_vector, self.X)
            most_similar = np.argmax(similarities)
            
            intent = self.pattern_classes[most_similar]
            confidence = float(similarities[0][most_similar])
            
            # Use conversation history for better context
            if confidence < 0.2 and context['last_intent']:
                # If low confidence but we have context, continue with previous intent
                intent = context['last_intent']
                confidence = 0.3  # Set a baseline confidence
            
            # Extract entities and generate response
            entities = self.extract_banking_entities(text)
            response = self.get_contextual_response(intent, entities, context)
            
            # Update context
            context['last_intent'] = intent
            context['conversation_history'].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.datetime.now()
            })
            
            return {
                'response': response,
                'intent': intent,
                'entities': entities,
                'confidence': confidence,
                'requires_clarification': confidence < 0.2 and not context['last_intent']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                'intent': 'error',
                'confidence': 0.0,
                'entities': {},
                'requires_clarification': True
            }

    def get_contextual_response(self, intent: str, entities: Dict[str, Any], context: Dict) -> str:
        """Generate response based on intent, entities, and conversation context"""
        try:
            # Get base response
            base_response = random.choice(self.intents['intents'][intent]['responses'])
            
            # Handle specific intents with context
            if intent == 'loan_inquiry':
                if entities['amount']:
                    amount = float(re.sub(r'[^\d.]', '', entities['amount']))
                    if amount > 0:
                        return f"I see you're interested in a loan of ${amount:,.2f}. Let me check your eligibility. Would you like to proceed with the loan application?"
                
            elif intent == 'balance_inquiry':
                if not context.get('authenticated'):
                    return "To check your balance, you'll need to log in first. Would you like to proceed with authentication?"
            
            # Add time-based greeting and conversational elements
            greeting = self.get_time_based_greeting()
            filler = random.choice([
                "Let me help you with that. ",
                "I understand what you need. ",
                "I'll assist you with that right away. "
            ])
            
            return f"{greeting}{filler}{base_response}"
            
        except Exception as e:
            self.logger.error(f"Error generating contextual response: {str(e)}")
            return "I apologize, but I'm having trouble understanding. Could you please rephrase that?"

    def get_time_based_greeting(self) -> str:
        """Get appropriate greeting based on time of day"""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning! "
        elif 12 <= hour < 17:
            return "Good afternoon! "
        elif 17 <= hour < 22:
            return "Good evening! "
        return "Hello! "

    def extract_banking_entities(self, text: str) -> Dict[str, Optional[str]]:
        """Extract banking-specific entities from text"""
        try:
            doc = self.nlp(text)
            entities = {
                'amount': None,
                'recipient': None,
                'account_type': None,
                'date': None,
                'transaction_type': None
            }
            
            # Extract amounts
            amount_pattern = re.compile(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?')
            amounts = amount_pattern.findall(text)
            if amounts:
                entities['amount'] = amounts[0]
            
            # Extract other entities
            text_tokens = text.lower().split()
            
            # Account types
            account_types = ['savings', 'checking', 'current', 'credit']
            for acc_type in account_types:
                if acc_type in text_tokens:
                    entities['account_type'] = acc_type
            
            # Transaction types
            transaction_types = ['deposit', 'withdrawal', 'transfer', 'payment', 'loan']
            for trans_type in transaction_types:
                if trans_type in text_tokens:
                    entities['transaction_type'] = trans_type
            
            # Named entities
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['recipient'] = ent.text
                elif ent.label_ == 'DATE':
                    entities['date'] = ent.text
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return {
                'amount': None,
                'recipient': None,
                'account_type': None,
                'date': None,
                'transaction_type': None
            }

def main():
    """Main function to demonstrate chatbot usage"""
    try:
        print("Initializing Banking Chatbot...")
        chatbot = BankingChatbot()
        
        print("\nBanking Chatbot is ready! Type 'quit' to exit.")
        print("=" * 50)
        
        user_id = "demo_user"  # You can generate unique IDs for different users
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'quit':
                print("Thank you for using Banking Chatbot. Goodbye!")
                break
            
            response = chatbot.get_response(user_id, user_input)
            print(f"Bot: {response['response']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
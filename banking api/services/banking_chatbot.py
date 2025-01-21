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
        
        # Setup paths - modified to handle the correct directory structure
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__))).parent
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
                # Modified to look for intents.json in the data directory
                self.load_intents(self.base_path / 'data' / intents_file)
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

    def load_intents(self, intents_file: Path) -> None:
        """Load intents from JSON file"""
        try:
            if not intents_file.exists():
                raise FileNotFoundError(f"Intents file not found at {intents_file}")
                
            with open(intents_file, 'r', encoding='utf-8') as file:
                self.intents = json.load(file)
            self.logger.info(f"Intents loaded successfully from {intents_file}")
        except Exception as e:
            self.logger.error(f"Error loading intents file: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> List[str]:
        """
        Clean and preprocess input text
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            List[str]: List of preprocessed tokens
        """
        try:
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            
            # Tokenize and remove stopwords
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

    def extract_banking_entities(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract banking-specific entities from text
        
        Args:
            text (str): Input text to extract entities from
            
        Returns:
            Dict[str, Optional[str]]: Dictionary of extracted entities
        """
        try:
            doc = self.nlp(text)
            entities = {
                'amount': None,
                'recipient': None,
                'account_type': None,
                'date': None,
                'transaction_type': None
            }
            
            # Extract amounts (looking for currency patterns)
            amount_pattern = re.compile(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?')
            amounts = amount_pattern.findall(text)
            if amounts:
                entities['amount'] = amounts[0]
            
            # Extract account types
            account_types = ['savings', 'checking', 'current', 'credit']
            text_tokens = text.lower().split()
            for acc_type in account_types:
                if acc_type in text_tokens:
                    entities['account_type'] = acc_type
            
            # Extract transaction types
            transaction_types = ['deposit', 'withdrawal', 'transfer', 'payment']
            for trans_type in transaction_types:
                if trans_type in text_tokens:
                    entities['transaction_type'] = trans_type
            
            # Extract recipient names using spaCy
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

    def get_humanized_response(self, intent: str, entities: Dict[str, Any], text: str) -> str:
        """
        Generate a human-like response based on intent and entities
        
        Args:
            intent (str): Detected intent
            entities (Dict[str, Any]): Extracted entities
            text (str): Original input text
            
        Returns:
            str: Humanized response
        """
        try:
            base_response = random.choice(self.intents['intents'][intent]['responses'])
            
            # Add time-based greeting
            hour = datetime.datetime.now().hour
            greeting = ""
            if 5 <= hour < 12:
                greeting = "Good morning! "
            elif 12 <= hour < 17:
                greeting = "Good afternoon! "
            elif 17 <= hour < 22:
                greeting = "Good evening! "
            
            # Personalize response based on entities
            response = base_response
            if entities['amount']:
                response = response.replace("the amount", f"${entities['amount']}")
            if entities['recipient']:
                response = response.replace("the recipient", entities['recipient'])
            
            # Add conversational fillers
            fillers = [
                "Let me see... ",
                "Just a moment... ",
                "Alright, ",
                "Sure thing! ",
                "I understand. ",
                "Of course! "
            ]
            
            # Add empathy phrases for certain intents
            if intent in ['customer_support', 'account_security']:
                empathy_phrases = [
                    "I understand your concern. ",
                    "I'll help you sort this out. ",
                    "Don't worry, I'm here to help. "
                ]
                response = f"{random.choice(empathy_phrases)}{response}"
            
            return f"{greeting}{random.choice(fillers)}{response}"
        except Exception as e:
            self.logger.error(f"Error generating humanized response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response. How else can I assist you?"

    def get_response(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        Generate appropriate response based on user input
        
        Args:
            user_id (str): Unique identifier for the user
            text (str): User input text
            
        Returns:
            Dict[str, Any]: Response containing intent, entities, and generated text
        """
        try:
            # Preprocess input
            processed_text = ' '.join(self.preprocess_text(text))
            
            # Vectorize input and find most similar pattern
            input_vector = self.vectorizer.transform([processed_text])
            similarities = cosine_similarity(input_vector, self.X)
            most_similar = np.argmax(similarities)
            
            # Get intent and confidence score
            intent = self.pattern_classes[most_similar]
            confidence = float(similarities[0][most_similar])
            
            # Update user context
            if user_id not in self.conversation_context:
                self.conversation_context[user_id] = {'last_intent': None, 'turns': 0}
            self.conversation_context[user_id]['last_intent'] = intent
            self.conversation_context[user_id]['turns'] += 1
            
            # Only proceed if confidence is high enough
            if confidence < 0.2:
                return {
                    'response': "I'm not quite sure what you're asking. Could you please rephrase that?",
                    'intent': 'unknown',
                    'confidence': confidence,
                    'entities': {},
                    'requires_clarification': True
                }
            
            # Extract entities
            entities = self.extract_banking_entities(text)
            
            # Generate response
            response = self.get_humanized_response(intent, entities, text)
            
            return {
                'response': response,
                'intent': intent,
                'entities': entities,
                'confidence': confidence,
                'requires_clarification': False
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

    def model_exists(self) -> bool:
        """Check if trained model files exist"""
        try:
            vectorizer_path = self.model_dir / 'vectorizer.pkl'
            patterns_path = self.model_dir / 'patterns.pkl'
            intents_path = self.model_dir / 'intents.pkl'
            
            return all([
                vectorizer_path.exists(),
                patterns_path.exists(),
                intents_path.exists()
            ])
        except Exception as e:
            self.logger.error(f"Error checking model existence: {str(e)}")
            return False

    def save_model(self) -> None:
        """Save the trained model components"""
        try:
            # Save vectorizer
            with open(self.model_dir / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f, protocol=4)
            
            # Save patterns data
            patterns_data = {
                'patterns': self.patterns,
                'pattern_classes': self.pattern_classes
            }
            with open(self.model_dir / 'patterns.pkl', 'wb') as f:
                pickle.dump(patterns_data, f, protocol=4)
            
            # Save intents
            with open(self.model_dir / 'intents.pkl', 'wb') as f:
                pickle.dump(self.intents, f, protocol=4)
                
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self) -> None:
        """Load the trained model components"""
        try:
            # Load vectorizer
            with open(self.model_dir / 'vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load patterns data
            with open(self.model_dir / 'patterns.pkl', 'rb') as f:
                patterns_data = pickle.load(f)
                self.patterns = patterns_data['patterns']
                self.pattern_classes = patterns_data['pattern_classes']
            
            # Load intents
            with open(self.model_dir / 'intents.pkl', 'rb') as f:
                self.intents = pickle.load(f)
            
            # Get the transformed patterns matrix
            self.X = self.vectorizer.transform(self.patterns)
                
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    """Main function to demonstrate chatbot usage"""
    try:
        print("Initializing Banking Chatbot...")
        chatbot = BankingChatbot()
        
        print("\nBanking Chatbot is ready! Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'quit':
                print("Thank you for using Banking Chatbot. Goodbye!")
                break
            
            response = chatbot.get_response("demo_user", user_input)
            print(f"Bot: {response['response']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
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
<<<<<<< HEAD
from typing import Dict, List, Any, Optional, Tuple
=======
from typing import Dict, List, Any, Optional
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0

class BankingChatbot:
    def __init__(self, intents_file: str = 'intents.json', model_dir: str = 'models'):
        """
<<<<<<< HEAD
        Initialize the Banking Chatbot with necessary components
=======
        Initialize the Banking Chatbot with necessary NLP components and intents
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
        
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
        
<<<<<<< HEAD
        # Setup paths
=======
        # Setup paths - modified to handle the correct directory structure
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__))).parent
        self.model_dir = self.base_path / model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
<<<<<<< HEAD
            # Initialize NLP components
=======
            # Initialize NLTK components
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load spaCy model
<<<<<<< HEAD
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.warning("Downloading spacy model...")
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
            
            # Initialize conversation management
            self.conversation_flows = {}
            self.user_contexts = {}
            self.session_data = {}
            
            # Define entity patterns
            self.entity_patterns = {
                'amount': r'\$?\d+(?:,\d{3})*(?:\.\d{2})?',
                'duration': r'\d+\s*(month|year|months|years)',
                'account_type': r'(savings|checking|current|credit)',
                'loan_type': r'(personal|home|auto|business|student)\s*loan'
            }
=======
            self.nlp = spacy.load('en_core_web_sm')
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
            
            # Load or create model
            if self.model_exists():
                self.load_model()
            else:
<<<<<<< HEAD
                self.load_intents(self.base_path / 'data' / intents_file)
                self.prepare_training_data()
                self.train_model()
                self.save_model()
            
=======
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
            
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
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
<<<<<<< HEAD
                
            self.logger.info(f"Intents loaded successfully from {intents_file}")
            
=======
            self.logger.info(f"Intents loaded successfully from {intents_file}")
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
        except Exception as e:
            self.logger.error(f"Error loading intents file: {str(e)}")
            raise

<<<<<<< HEAD
    def prepare_training_data(self) -> None:
        """Prepare patterns and their respective classes for training"""
        try:
            self.patterns = []
            self.pattern_classes = []
            
            for intent, data in self.intents['intents'].items():
                for pattern in data['patterns']:
                    self.patterns.append(pattern)
                    self.pattern_classes.append(intent)
                    
            self.logger.info("Training data prepared successfully")
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def train_model(self) -> None:
        """Train the TF-IDF vectorizer"""
        try:
            self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
            self.X = self.vectorizer.fit_transform(self.patterns)
            self.logger.info("Model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

=======
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
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
<<<<<<< HEAD
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
=======
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
            
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words]
            
            return tokens
<<<<<<< HEAD
            
=======
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {str(e)}")
            return []

<<<<<<< HEAD
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract relevant entities from text using patterns and spaCy
=======
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
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
        
        Args:
            text (str): Input text to extract entities from
            
        Returns:
<<<<<<< HEAD
            Dict[str, Any]: Dictionary of extracted entities
        """
        try:
            entities = {
                'amount': None,
                'duration': None,
                'account_type': None,
                'loan_type': None,
                'location': None,
                'date': None
            }
            
            # Use spaCy for named entity recognition
            doc = self.nlp(text)
            
            # Extract entities using spaCy
            for ent in doc.ents:
                if ent.label_ == 'GPE':
                    entities['location'] = ent.text
                elif ent.label_ == 'DATE':
                    entities['date'] = ent.text
            
            # Extract entities using patterns
            for entity_type, pattern in self.entity_patterns.items():
                matches = re.findall(pattern, text.lower())
                if matches:
                    entities[entity_type] = matches[0]
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return {}

    def get_intent(self, text: str) -> Tuple[str, float]:
        """
        Determine intent from input text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[str, float]: Intent and confidence score
        """
        try:
            # Vectorize input text
            input_vector = self.vectorizer.transform([text])
            
            # Calculate similarities
            similarities = cosine_similarity(input_vector, self.X)
            most_similar = np.argmax(similarities)
            
            return (
                self.pattern_classes[most_similar],
                float(similarities[0][most_similar])
            )
            
        except Exception as e:
            self.logger.error(f"Error getting intent: {str(e)}")
            return ('error', 0.0)

    def get_response(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        Generate appropriate response based on user input and context
=======
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
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
        
        Args:
            user_id (str): Unique identifier for the user
            text (str): User input text
            
        Returns:
            Dict[str, Any]: Response containing intent, entities, and generated text
        """
        try:
<<<<<<< HEAD
            # Get or initialize user context
            context = self.get_user_context(user_id)
            
            # Process input and get intent
            processed_text = ' '.join(self.preprocess_text(text))
            intent, confidence = self.get_intent(processed_text)
            
            # Extract entities
            entities = self.extract_entities(text)
            
            # Update context with new information
            self.update_context(context, intent, entities)
            
            # Generate response based on context
            response_data = self.generate_contextual_response(
                intent,
                context,
                entities,
                confidence
            )
            
            # Update session data
            self.update_session_data(user_id, intent, entities)
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return self.get_error_response()

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Initialize or get existing user context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                'current_flow': None,
                'flow_stage': None,
                'collected_info': {},
                'missing_info': [],
                'last_intent': None,
                'turns': 0
            }
        return self.user_contexts[user_id]

    def update_context(self, context: Dict[str, Any], intent: str, entities: Dict[str, Any]) -> None:
        """Update conversation context with new information"""
        # Get flow information for the intent
        intent_flow = self.intents['intents'].get(intent, {}).get('flow', {})
        
        # Update basic context information
        context['last_intent'] = intent
        context['turns'] += 1
        
        # Update collected information
        if entities:
            context['collected_info'].update(entities)
        
        # Update flow information
        if intent_flow and (not context['current_flow'] or intent != context['last_intent']):
            context['current_flow'] = intent
            context['flow_stage'] = 'initial'
            context['missing_info'] = [
                info for info in intent_flow.get('required_info', [])
                if info not in context['collected_info']
            ]

    def generate_contextual_response(self, intent: str, context: Dict[str, Any], 
                                   entities: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Generate appropriate response based on context and intent"""
        # Check confidence threshold
        if confidence < 0.2:
            return self.get_clarification_response()
        
        # Get intent data
        intent_data = self.intents['intents'].get(intent, {})
        flow_data = intent_data.get('flow', {})
        
        # Check for missing required information
        if context['current_flow'] and context['missing_info']:
            missing_item = context['missing_info'][0]
            followup_question = flow_data.get('followup_questions', {}).get(
                missing_item,
                f"Could you please provide your {missing_item}?"
            )
            return {
                'response': followup_question,
=======
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
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
                'intent': intent,
                'entities': entities,
                'confidence': confidence,
                'requires_clarification': False
            }
<<<<<<< HEAD
        
        # Generate base response
        response = random.choice(intent_data.get('responses', ["I'll help you with that."]))
        
        # Add context-specific information
        if context['collected_info']:
            response = self.enhance_response_with_context(
                response, 
                context['collected_info']
            )
        
        return {
            'response': response,
            'intent': intent,
            'entities': entities,
            'confidence': confidence,
            'requires_clarification': False
        }

    def enhance_response_with_context(self, response: str, collected_info: Dict[str, Any]) -> str:
        """Enhance response with collected information"""
        # Replace placeholders with actual values
        replacements = {
            'the amount': f"${collected_info.get('amount', '')}",
            'the duration': collected_info.get('duration', ''),
            'loan': f"{collected_info.get('loan_type', '')} loan",
            'account': f"{collected_info.get('account_type', '')} account"
        }
        
        for placeholder, value in replacements.items():
            if value:
                response = response.replace(placeholder, value)
        
        return response

    def get_clarification_response(self) -> Dict[str, Any]:
        """Generate response when clarification is needed"""
        clarification_responses = [
            "I'm not quite sure what you mean. Could you please rephrase that?",
            "I didn't quite catch that. Can you explain it differently?",
            "Could you provide more details about what you're looking for?"
        ]
        
        return {
            'response': random.choice(clarification_responses),
            'intent': 'clarification',
            'entities': {},
            'confidence': 0.0,
            'requires_clarification': True
        }

    def get_error_response(self) -> Dict[str, Any]:
        """Generate error response"""
        error_responses = [
            "I apologize, but I'm experiencing technical difficulties.",
            "Sorry, something went wrong. Please try again later.",
            "I'm having trouble processing your request. Please try again."
        ]
        
        return {
            'response': random.choice(error_responses),
            'intent': 'error',
            'entities': {},
            'confidence': 0.0,
            'requires_clarification': True
        }

    def update_session_data(self, user_id: str, intent: str, entities: Dict[str, Any]) -> None:
        """Update session data for the user"""
        if user_id not in self.session_data:
            self.session_data[user_id] = {
                'interaction_history': [],
                'last_active': datetime.datetime.now()
            }
        
        self.session_data[user_id]['interaction_history'].append({
            'intent': intent,
            'entities': entities,
            'timestamp': datetime.datetime.now()
        })
        
        self.session_data[user_id]['last_active'] = datetime.datetime.now()
=======
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                'intent': 'error',
                'confidence': 0.0,
                'entities': {},
                'requires_clarification': True
            }
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0

    def model_exists(self) -> bool:
        """Check if trained model files exist"""
        try:
<<<<<<< HEAD
            required_files = ['vectorizer.pkl', 'patterns.pkl', 'intents.pkl']
            return all((self.model_dir / file).exists() for file in required_files)
=======
            vectorizer_path = self.model_dir / 'vectorizer.pkl'
            patterns_path = self.model_dir / 'patterns.pkl'
            intents_path = self.model_dir / 'intents.pkl'
            
            return all([
                vectorizer_path.exists(),
                patterns_path.exists(),
                intents_path.exists()
            ])
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
        except Exception as e:
            self.logger.error(f"Error checking model existence: {str(e)}")
            return False

    def save_model(self) -> None:
        """Save the trained model components"""
        try:
            # Save vectorizer
            with open(self.model_dir / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f, protocol=4)
            
<<<<<<< HEAD
            # Save patterns
=======
            # Save patterns data
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
            patterns_data = {
                'patterns': self.patterns,
                'pattern_classes': self.pattern_classes
            }
            with open(self.model_dir / 'patterns.pkl', 'wb') as f:
                pickle.dump(patterns_data, f, protocol=4)
            
            # Save intents
            with open(self.model_dir / 'intents.pkl', 'wb') as f:
                pickle.dump(self.intents, f, protocol=4)
<<<<<<< HEAD
            
            self.logger.info("Model saved successfully")
            
=======
                
            self.logger.info("Model saved successfully")
>>>>>>> 9fdb6d10d289e5760e15d46ba47a2b1e134d69f0
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
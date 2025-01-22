import json
from pathlib import Path
import os
from datetime import datetime, timedelta
import pickle
import re
from typing import Dict, Any, Optional
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
        """
        Initialize NLP Service with necessary components
        
        Args:
            model_dir (str): Directory containing model files
        """
        # Set up paths
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__))).parent
        self.cache_file = self.base_path / 'data' / 'response_cache.pkl'
        self.model_dir = self.base_path / model_dir
        self.cache_duration = timedelta(hours=24)
        self.conversation_context = {}
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.initialize_components()
        self.setup_directories()
        
        # Initialize basic training data
        self.initialize_training_data()
        
        # Load cache
        self.load_cache()

    def initialize_components(self) -> None:
        """Initialize NLP components"""
        try:
            self.lemmatizer = WordNetLemmatizer()
            # Convert stopwords to list instead of set
            self.stop_words = list(stopwords.words('english'))
            
            # Load spaCy model
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.warning("Downloading spacy model...")
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        try:
            data_dir = self.base_path / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error setting up directories: {str(e)}")
            raise

    def initialize_training_data(self) -> None:
        """Initialize basic training data and vectorizer"""
        try:
            # Basic initial training data
            self.intents = {
                'intents': {
                    'greeting': {
                        'patterns': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
                        'responses': ['Hello! How can I help you today?', 'Hi there! How may I assist you?']
                    },
                    'farewell': {
                        'patterns': ['goodbye', 'bye', 'see you', 'see you later'],
                        'responses': ['Goodbye! Have a great day!', 'Bye! Take care!']
                    },
                    'fallback': {
                        'patterns': [],
                        'responses': ["I'm not quite sure I understood. Could you please rephrase that?"]
                    }
                }
            }

            # Prepare patterns and classes
            self.patterns = []
            self.pattern_classes = []
            for intent, data in self.intents['intents'].items():
                for pattern in data.get('patterns', []):
                    self.patterns.append(pattern)
                    self.pattern_classes.append(intent)

            # Initialize and fit vectorizer with 'english' stopwords
            self.vectorizer = TfidfVectorizer(
                tokenizer=self.preprocess_text,
                stop_words='english'  # Use built-in English stopwords
            )
            if self.patterns:
                self.X = self.vectorizer.fit_transform(self.patterns)
            else:
                self.X = None
                
            self.logger.info("Training data initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing training data: {str(e)}")
            raise

    def load_cache(self) -> None:
        """Load cached responses"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            else:
                self.cache = {}
        except Exception as e:
            self.logger.warning(f"Error loading cache, creating new cache: {str(e)}")
            self.cache = {}

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        try:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {str(e)}")
            return text

    def get_response(self, text: str, user_id: str = 'default') -> Dict[str, Any]:
        """
        Generate response based on user input
        
        Args:
            text (str): User input text
            user_id (str): Unique identifier for the user
            
        Returns:
            Dict[str, Any]: Response containing intent, response text, and confidence
        """
        try:
            if not self.patterns or self.X is None:
                return {
                    'intent': 'fallback',
                    'response': "I'm still learning. Please try again later.",
                    'confidence': 0.0
                }

            # Process input
            processed_text = self.preprocess_text(text)
            input_vector = self.vectorizer.transform([processed_text])
            
            # Calculate similarities
            similarities = cosine_similarity(input_vector, self.X)
            most_similar = np.argmax(similarities)
            confidence = float(similarities[0][most_similar])
            
            # Get intent and response
            intent = self.pattern_classes[most_similar]
            
            if confidence < 0.2:
                return {
                    'intent': 'fallback',
                    'response': "I'm not quite sure I understood. Could you please rephrase that?",
                    'confidence': 0.0
                }
            
            responses = self.intents['intents'][intent]['responses']
            response = np.random.choice(responses)
            
            return {
                'intent': intent,
                'response': response,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'intent': 'error',
                'response': "I apologize, but I'm experiencing technical difficulties.",
                'confidence': 0.0
            }
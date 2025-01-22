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
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic components
        self.initialize_components()
        
        # Create necessary directories
        self.setup_directories()
        
        # Load models and cache
        self.load_model()
        self.initialize_cache()

    def initialize_components(self) -> None:
        """Initialize NLP components"""
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = list(set(stopwords.words('english')))  # Convert set to list
            self.cache_duration = timedelta(hours=24)
            self.conversation_context = {}
            
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

    def load_model(self) -> None:
        """Load the trained model components safely"""
        try:
            # Initialize new vectorizer
            self.vectorizer = TfidfVectorizer(
                tokenizer=self.preprocess_text,
                stop_words=self.stop_words
            )
            
            # Load patterns and intents first
            with open(self.model_dir / 'patterns.pkl', 'rb') as f:
                patterns_data = pickle.load(f)
                self.patterns = patterns_data['patterns']
                self.pattern_classes = patterns_data['pattern_classes']
            
            with open(self.model_dir / 'intents.pkl', 'rb') as f:
                self.intents = pickle.load(f)
            
            # Fit vectorizer on patterns
            self.X = self.vectorizer.fit_transform(self.patterns)
            
            self.logger.info("Model loaded successfully")
            
        except FileNotFoundError:
            self.logger.error("Model files not found. Please ensure model files are present in the models directory")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def initialize_cache(self) -> None:
        """Initialize or load response cache"""
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
                return self.get_fallback_response()
            
            responses = self.intents['intents'][intent]['responses']
            response = np.random.choice(responses)
            
            return {
                'intent': intent,
                'response': response,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return self.get_error_response()

    def get_fallback_response(self) -> Dict[str, Any]:
        """Generate fallback response for low confidence"""
        return {
            'intent': 'fallback',
            'response': "I'm not quite sure I understood. Could you please rephrase that?",
            'confidence': 0.0
        }

    def get_error_response(self) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'intent': 'error',
            'response': "I apologize, but I'm experiencing technical difficulties.",
            'confidence': 0.0
        }

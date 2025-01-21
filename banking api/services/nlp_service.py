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
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__))).parent
        self.cache_file = self.base_path / 'data' / 'response_cache.pkl'
        self.cache_duration = timedelta(hours=24)
        self.conversation_context = {}
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.warning("Downloading spacy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

        # Create data directory if it doesn't exist
        data_dir = self.base_path / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        # Load model and components
        self.model_dir = self.base_path / model_dir
        self.load_model()
        self.load_cache()

    def load_model(self) -> None:
        """Load the trained model components"""
        try:
            # Load vectorizer components
            with open(self.model_dir / 'vectorizer.pkl', 'rb') as f:
                vectorizer_dict = pickle.load(f)
                
            # Initialize and configure vectorizer
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.vocabulary_ = vectorizer_dict['vocabulary_']
            self.vectorizer.idf_ = vectorizer_dict['idf_']
            if 'stop_words_' in vectorizer_dict:
                self.vectorizer.stop_words_ = vectorizer_dict['stop_words_']
            
            # Load patterns
            with open(self.model_dir / 'patterns.pkl', 'rb') as f:
                patterns_data = pickle.load(f)
                self.patterns = patterns_data['patterns']
                self.pattern_classes = patterns_data['pattern_classes']
            
            # Load intents
            with open(self.model_dir / 'intents.pkl', 'rb') as f:
                self.intents = pickle.load(f)
            
            # Transform patterns
            self.X = self.vectorizer.transform(self.patterns)
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def load_cache(self):
        """Load cached responses"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def get_response(self, text: str, user_id: str = 'default') -> Dict[str, Any]:
        """Generate response based on user input"""
        try:
            # Process input
            processed_text = self.preprocess_text(text)
            input_vector = self.vectorizer.transform([processed_text])
            similarities = cosine_similarity(input_vector, self.X)
            
            # Get most similar pattern
            most_similar = np.argmax(similarities)
            intent = self.pattern_classes[most_similar]
            confidence = float(similarities[0][most_similar])
            
            # Get response
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

    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        return ' '.join(tokens)
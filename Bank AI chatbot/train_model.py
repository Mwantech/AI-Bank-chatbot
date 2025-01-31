import json
import numpy as np
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import spacy
import re

class BankingChatbotTrainer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self.classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.label_encoder = LabelEncoder()
        self.intents = None
        self.responses = {}
        self.entities = {}
        self.required_entities = {}
        
    def load_training_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.intents = data['intents']
        self.entities = data['entities']
        return data
    
    def preprocess_text(self, text):
        # Basic text preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = self.nlp(text)
        # Lemmatization
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])
    
    def prepare_training_data(self):
        X_texts = []
        y_intents = []
        
        for intent in self.intents:
            intent_name = intent['intent']
            self.responses[intent_name] = intent['responses']
            if 'required_entities' in intent:
                self.required_entities[intent_name] = intent['required_entities']
            
            for pattern in intent['patterns']:
                processed_text = self.preprocess_text(pattern)
                X_texts.append(processed_text)
                y_intents.append(intent_name)
        
        # Transform text to TF-IDF features
        X = self.vectorizer.fit_transform(X_texts)
        y = self.label_encoder.fit_transform(y_intents)
        
        return X, y
    
    def train(self):
        X, y = self.prepare_training_data()
        self.classifier.fit(X, y)
    
    def save_model(self, model_dir='models'):
        Path(model_dir).mkdir(exist_ok=True)
        
        # Save all components
        with open(f'{model_dir}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(f'{model_dir}/classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(f'{model_dir}/responses.pkl', 'wb') as f:
            pickle.dump(self.responses, f)
        
        with open(f'{model_dir}/entities.pkl', 'wb') as f:
            pickle.dump(self.entities, f)
        
        with open(f'{model_dir}/required_entities.pkl', 'wb') as f:
            pickle.dump(self.required_entities, f)

if __name__ == '__main__':
    trainer = BankingChatbotTrainer()
    trainer.load_training_data('data/intents.json')
    trainer.train()
    trainer.save_model()
    print("Model training completed and saved!")
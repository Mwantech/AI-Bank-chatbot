from pathlib import Path
import os
from banking_chatbot import BankingChatbot
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def train_and_save_model():
    try:
        # Initialize BankingChatbot
        chatbot = BankingChatbot()
        
        base_path = Path(os.path.abspath(os.path.dirname(__file__))).parent
        models_dir = base_path / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save the complete vectorizer object
        with open(models_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(chatbot.vectorizer, f, protocol=4)

        # Save patterns data
        patterns_data = {
            'patterns': chatbot.patterns,
            'pattern_classes': chatbot.pattern_classes
        }
        with open(models_dir / 'patterns.pkl', 'wb') as f:
            pickle.dump(patterns_data, f, protocol=4)

        # Save intents
        with open(models_dir / 'intents.pkl', 'wb') as f:
            pickle.dump(chatbot.intents, f, protocol=4)

        print("Model trained and saved successfully!")
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()
import json
import torch
from pathlib import Path
import pickle
import re
import os
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class NeuralBankingChatbot:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 64
        self.hidden_size = 64
        self.learning_rate = 0.001
        self.batch_size = 8
        self.epochs = 100
        self.word_to_ix = {"<PAD>": 0, "<UNK>": 1}
        self.tag_to_ix = {}
        self.responses = {}
        self.entities = {}
        self.required_entities = {}
        self.max_length = 20
        
    # Define IntentClassifier as an inner class to ensure it's always available
    class IntentClassifier(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
            super(NeuralBankingChatbot.IntentClassifier, self).__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, tagset_size)
            
        def forward(self, x):
            # x shape: [batch_size, seq_len]
            embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
            
            lstm_out, (hidden, cell) = self.lstm(embedded)
            # hidden shape: [1, batch_size, hidden_dim]
            
            # Use the final hidden state
            out = self.fc(hidden[-1])  # [batch_size, tagset_size]
            return out
        
    def load_training_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.intents = data['intents']
        self.entities = data.get('entities', {})
        
        # Build vocabulary
        for intent in self.intents:
            intent_name = intent['intent']
            if intent_name not in self.tag_to_ix:
                self.tag_to_ix[intent_name] = len(self.tag_to_ix)
            
            self.responses[intent_name] = intent['responses']
            if 'required_entities' in intent:
                self.required_entities[intent_name] = intent['required_entities']
            
            for pattern in intent['patterns']:
                for word in self.tokenize(pattern):
                    if word not in self.word_to_ix:
                        self.word_to_ix[word] = len(self.word_to_ix)
        
        return data
    
    def tokenize(self, sentence):
        # Simple tokenization and preprocessing
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        return sentence.split()
    
    def prepare_sequence(self, sentence):
        tokens = self.tokenize(sentence)
        idxs = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in tokens]
        
        # Pad or truncate sequence to fixed length
        if len(idxs) < self.max_length:
            idxs = idxs + [self.word_to_ix["<PAD>"]] * (self.max_length - len(idxs))
        else:
            idxs = idxs[:self.max_length]
            
        return torch.tensor(idxs, dtype=torch.long)
    
    def prepare_training_data(self):
        X = []
        y = []
        
        for intent in self.intents:
            intent_name = intent['intent']
            tag_idx = self.tag_to_ix[intent_name]
            
            for pattern in intent['patterns']:
                sequence = self.prepare_sequence(pattern)
                X.append(sequence)
                y.append(tag_idx)
        
        return X, y
    
    def train(self):
        X, y = self.prepare_training_data()
        
        # Define model
        vocab_size = len(self.word_to_ix)
        tagset_size = len(self.tag_to_ix)
        
        self.model = self.IntentClassifier(vocab_size, self.embedding_dim, self.hidden_size, tagset_size)
        self.model.to(self.device)
        
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert data to tensors
        X_tensor = torch.stack(X).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch_x, batch_y in dataloader:
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Calculate loss
                loss = loss_function(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader)}")
    
        def extract_entities(self, text):
            """Extract entities from user input with improved pattern matching"""
            doc = self.nlp(text)
            extracted_entities = {}
            
            # Use spaCy's NER to identify standard entities
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    extracted_entities["amount"] = ent.text
                elif ent.label_ == "DATE":
                    extracted_entities["date"] = ent.text
                elif ent.label_ == "PERSON":
                    extracted_entities["person"] = ent.text
                elif ent.label_ == "ORG":
                    extracted_entities["organization"] = ent.text
                elif ent.label_ == "GPE" or ent.label_ == "LOC":
                    extracted_entities["location"] = ent.text
            
            # Custom entity extraction for banking-specific entities
            
            # Extract account_type (checking, savings, etc.)
            account_types = ["checking", "savings", "CD", "money market", "IRA", "retirement", "business"]
            for acc_type in account_types:
                if acc_type in text.lower():
                    extracted_entities["account_type"] = acc_type
                    break
            
            # Extract applicant_type (individual, joint, business, etc.)
            applicant_types = ["individual", "personal", "joint", "business", "student", "minor"]
            for app_type in applicant_types:
                if app_type in text.lower():
                    extracted_entities["applicant_type"] = app_type
                    break
            
            # Extract day_of_week
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "weekday", "weekend"]
            for day in days:
                if day in text.lower():
                    extracted_entities["day_of_week"] = day
                    break
            
            # Check for words indicating location
            location_indicators = ["branch", "location", "office", "center", "downtown", "uptown", "west", "east", "north", "south"]
            for loc in location_indicators:
                if loc in text.lower():
                    # Try to extract the specific location
                    doc = self.nlp(text)
                    for token in doc:
                        if token.text.lower() == loc and token.i > 0:
                            extracted_entities["location"] = doc[token.i-1].text + " " + loc
                            break
                    if "location" not in extracted_entities:
                        extracted_entities["location"] = "nearest"
                    break
            
            return extracted_entities

        # Replace the generate_response method for both chatbot classes with this improved version

    def generate_response(self, user_input):
        """Generate a response based on user input with better intent and entity handling"""
        # Extract entities
        entities = self.extract_entities(user_input)
        
        # Get intent (different methods for each class)
        if hasattr(self, 'predict'):  # Neural Chatbot
            intent_name = self.predict(user_input)
        else:  # Rule-based Chatbot
            processed_input = self.preprocess_text(user_input)
            input_vector = self.vectorizer.transform([processed_input])
            intent_id = self.classifier.predict(input_vector)[0]
            intent_name = self.label_encoder.inverse_transform([intent_id])[0]
        
        # Check for required entities and handle missing ones
        if intent_name in self.required_entities:
            missing_entities = []
            for required_entity in self.required_entities[intent_name]:
                if required_entity not in entities:
                    missing_entities.append(required_entity)
            
            
        
        # Get response templates for the predicted intent
        response_templates = self.responses.get(intent_name, ["I'm not sure how to respond to that."])
        
        # Select a random response template
        import random
        response = random.choice(response_templates)
        
        # Fill in entities in the response
        for entity_type, value in entities.items():
            placeholder = f"{{{entity_type}}}"
            if placeholder in response:
                response = response.replace(placeholder, value)
        
        # Handle default values for required placeholders that we couldn't fill
        # This prevents showing raw {placeholders} to users
        default_values = {
            "{location}": "your nearest branch",
            "{day_of_week}": "weekdays",
            "{account_type}": "new account",
            "{applicant_type}": "customer", 
            "{opening_time}": "9:00 AM",
            "{closing_time}": "5:00 PM",
            "{minimum_deposit}": "25"
        }
        
        for placeholder, default in default_values.items():
            if placeholder in response:
                response = response.replace(placeholder, default)
        
        return response

    def save_model(self, model_dir='neural_model'):
        """Save the model for future use"""
        Path(model_dir).mkdir(exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), f'{model_dir}/model.pt')
        
        # Save configuration
        config = {
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'max_length': self.max_length
        }
        
        with open(f'{model_dir}/config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        # Save vocabulary and other data
        with open(f'{model_dir}/word_to_ix.pkl', 'wb') as f:
            pickle.dump(self.word_to_ix, f)
        
        with open(f'{model_dir}/tag_to_ix.pkl', 'wb') as f:
            pickle.dump(self.tag_to_ix, f)
        
        with open(f'{model_dir}/responses.pkl', 'wb') as f:
            pickle.dump(self.responses, f)
        
        with open(f'{model_dir}/entities.pkl', 'wb') as f:
            pickle.dump(self.entities, f)
        
        with open(f'{model_dir}/required_entities.pkl', 'wb') as f:
            pickle.dump(self.required_entities, f)
    
    def load_model(self, model_dir='neural_model'):
        """Load a previously saved model"""
        # Load vocabulary and other data
        with open(f'{model_dir}/word_to_ix.pkl', 'rb') as f:
            self.word_to_ix = pickle.load(f)
        
        with open(f'{model_dir}/tag_to_ix.pkl', 'rb') as f:
            self.tag_to_ix = pickle.load(f)
        
        with open(f'{model_dir}/responses.pkl', 'rb') as f:
            self.responses = pickle.load(f)
        
        with open(f'{model_dir}/entities.pkl', 'rb') as f:
            self.entities = pickle.load(f)
        
        with open(f'{model_dir}/required_entities.pkl', 'rb') as f:
            self.required_entities = pickle.load(f)
        
        # Load configuration
        with open(f'{model_dir}/config.pkl', 'rb') as f:
            config = pickle.load(f)
            self.embedding_dim = config['embedding_dim']
            self.hidden_size = config['hidden_size']
            self.max_length = config['max_length']
        
        # Define and load model
        vocab_size = len(self.word_to_ix)
        tagset_size = len(self.tag_to_ix)
        
        # Create model instance using the inner class
        self.model = self.IntentClassifier(vocab_size, self.embedding_dim, self.hidden_size, tagset_size)
        self.model.load_state_dict(torch.load(f'{model_dir}/model.pt'))
        self.model.to(self.device)
        self.model.eval()


# Create an interactive chat function to test the chatbot
def interactive_chat(chatbot):
    print("Banking Chatbot is ready! (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = chatbot.generate_response(user_input)
        print(f"Chatbot: {response}")
        print("-" * 50)


# Option 2: Rule-based chatbot (simpler alternative)
class RuleBasedBankingChatbot:
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
        self.entities = data.get('entities', {})
        return data
    
    def preprocess_text(self, text):
        # Basic text preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        doc = self.nlp(text)
        # Lemmatization
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])
    
    def extract_entities(self, text):
        """Extract entities from user input"""
        doc = self.nlp(text)
        extracted_entities = {}
        
        # Use spaCy's NER to identify entities
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                extracted_entities["amount"] = ent.text
            elif ent.label_ == "DATE":
                extracted_entities["date"] = ent.text
            elif ent.label_ == "PERSON":
                extracted_entities["person"] = ent.text
            elif ent.label_ == "ORG":
                extracted_entities["organization"] = ent.text
        
        # Custom entity extraction based on patterns
        for entity_type, patterns in self.entities.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        extracted_entities[entity_type] = match.group(0)
        
        return extracted_entities
    
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
    
    def generate_response(self, user_input):
        """Generate a response based on user input"""
        # Extract entities
        entities = self.extract_entities(user_input)
        
        # Preprocess input
        processed_input = self.preprocess_text(user_input)
        
        # Transform input
        input_vector = self.vectorizer.transform([processed_input])
        
        # Predict intent
        intent_id = self.classifier.predict(input_vector)[0]
        intent_name = self.label_encoder.inverse_transform([intent_id])[0]
        
        # Check if all required entities are present
        if intent_name in self.required_entities:
            missing_entities = []
            for required_entity in self.required_entities[intent_name]:
                if required_entity not in entities:
                    missing_entities.append(required_entity)
            
            if missing_entities:
                # Ask for missing entities
                entity_questions = {
                    "account_number": "What is your account number?",
                    "amount": "How much would you like to transfer?",
                    "recipient": "Who would you like to transfer money to?",
                    "date": "What date would you like this transaction to occur?",
                }
                
                for missing in missing_entities:
                    if missing in entity_questions:
                        return entity_questions[missing]
                
                return "I need more information to complete this request."
        
        # Get response templates for the predicted intent
        response_templates = self.responses.get(intent_name, ["I'm not sure how to respond to that."])
        
        # Select a random response template
        import random
        response = random.choice(response_templates)
        
        # Fill in entities in the response
        for entity_type, value in entities.items():
            placeholder = f"{{{entity_type}}}"
            response = response.replace(placeholder, value)
        
        return response
    
    def save_model(self, model_dir='rule_based_model'):
        """Save the model for future use"""
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
    
    def load_model(self, model_dir='rule_based_model'):
        """Load a previously saved model"""
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


if __name__ == '__main__':
    # Choose which implementation to use
    use_neural = True  # Set to False to use the rule-based model instead
    
    if use_neural:
        # Neural implementation (PyTorch only)
        model_dir = 'neural_model'
        if not os.path.exists(model_dir):
            print("Training the neural banking chatbot...")
            chatbot = NeuralBankingChatbot()
            chatbot.load_training_data('data/intents.json')
            chatbot.train()
            chatbot.save_model(model_dir)
            print("Model training completed and saved!")
        
        print("Loading the trained neural model...")
        inference_chatbot = NeuralBankingChatbot()
        inference_chatbot.load_model(model_dir)
    else:
        # Rule-based implementation
        model_dir = 'rule_based_model'
        if not os.path.exists(model_dir):
            print("Training the rule-based banking chatbot...")
            chatbot = RuleBasedBankingChatbot()
            chatbot.load_training_data('data/intents.json')
            chatbot.train()
            chatbot.save_model(model_dir)
            print("Model training completed and saved!")
        
        print("Loading the trained rule-based model...")
        inference_chatbot = RuleBasedBankingChatbot()
        inference_chatbot.load_model(model_dir)
    
    # Start interactive chat
    interactive_chat(inference_chatbot)
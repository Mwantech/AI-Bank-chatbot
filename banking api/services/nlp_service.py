import json
import openai
from pathlib import Path
import os
from datetime import datetime, timedelta
import pickle

class NLPService:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.cache_file = 'response_cache.pkl'
        self.cache_duration = timedelta(hours=24)
        self.load_intents()
        self.load_cache()

    def load_intents(self):
        with open('intent_patterns.json', 'r') as f:
            self.intent_data = json.load(f)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Clean expired cache entries
                    current_time = datetime.now()
                    self.cache = {
                        k: v for k, v in cache_data.items() 
                        if current_time - v['timestamp'] < self.cache_duration
                    }
            except:
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def classify_intent(self, user_input):
        user_input = user_input.lower()
        
        # Check each intent's patterns
        for intent, data in self.intent_data['intents'].items():
            for pattern in data['patterns']:
                if pattern.lower() in user_input:
                    return {
                        'intent': intent,
                        'response': self.get_random_response(intent)
                    }
        
        return {
            'intent': 'general',
            'response': "I'm not sure I understand. Could you please rephrase that?"
        }

    def get_random_response(self, intent):
        import random
        responses = self.intent_data['intents'].get(intent, {}).get('responses', [])
        return random.choice(responses) if responses else "I understand you need help with that."

    def get_response(self, user_input):
        # Check cache first
        if user_input in self.cache:
            cache_entry = self.cache[user_input]
            if datetime.now() - cache_entry['timestamp'] < self.cache_duration:
                print("Cache hit!")
                return cache_entry['response']

        # Classify intent first
        intent_info = self.classify_intent(user_input)
        
        try:
            # Get GPT response
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI banking assistant. Provide concise responses focused on banking services."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7,
            )
            
            gpt_response = response['choices'][0]['message']['content'].strip()
            
            # Combine intent-based response with GPT response
            final_response = {
                'intent': intent_info['intent'],
                'response': f"{intent_info['response']} {gpt_response}"
            }
            
            # Cache the response
            self.cache[user_input] = {
                'response': final_response,
                'timestamp': datetime.now()
            }
            self.save_cache()
            
            return final_response
            
        except Exception as e:
            # If API call fails, return intent-based response
            return intent_info
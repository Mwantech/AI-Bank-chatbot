import json
from openai import OpenAI
from pathlib import Path
import os
from datetime import datetime, timedelta
import pickle
import re

class NLPService:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__)))
        self.cache_file = self.base_path / 'data' / 'response_cache.pkl'
        self.cache_duration = timedelta(hours=24)
        self.conversation_context = {}  # Store context for each user

        # Create data directory if it doesn't exist
        data_dir = self.base_path / 'data'
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        self.load_intents()
        self.load_cache()

    def load_cache(self):
        """Load cached responses from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    current_time = datetime.now()
                    self.cache = {
                        k: v for k, v in cache_data.items()
                        if current_time - v['timestamp'] < self.cache_duration
                    }
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self):
        """Save responses to cache file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def load_intents(self):
        """Load intent patterns from JSON file"""
        intent_file = self.base_path / 'data' / 'intent_patterns.json'

        # Create default intents if file doesn't exist
        if not intent_file.exists():
            default_intents = {
                "intents": {
                    "greeting": {
                        "patterns": ["hello", "hi", "hey", "how are you"],
                        "responses": ["Hello! I'm your banking assistant. How can I help you today?"]
                    },
                    "transfer_money": {
                        "patterns": ["transfer", "send money", "pay someone"],
                        "responses": ["I can help you transfer money. Who would you like to send money to?"]
                    },
                    "loan_inquiry": {
                        "patterns": ["loan", "borrow", "credit"],
                        "responses": ["I can help you with a loan application. What amount are you looking to borrow?"]
                    },
                    "location_search": {
                        "patterns": ["atm", "branch", "location", "find"],
                        "responses": ["I'll locate the closest branch/ATM for you. Where are you currently?"]
                    },
                    "balance_inquiry": {
                        "patterns": ["balance", "how much", "check account"],
                        "responses": ["I'll help you check your balance. Please log in to your account first."]
                    },
                    "general": {
                        "patterns": [],
                        "responses": ["I'll help you with that. Could you provide more details?"]
                    }
                }
            }

            # Create the data directory if it doesn't exist
            intent_file.parent.mkdir(parents=True, exist_ok=True)

            # Save default intents
            with open(intent_file, 'w') as f:
                json.dump(default_intents, f, indent=4)

            self.intent_data = default_intents
        else:
            with open(intent_file, 'r') as f:
                self.intent_data = json.load(f)

    def get_response(self, user_input, user_id='default'):
        # Initialize or get conversation context
        if user_id not in self.conversation_context:
            self.conversation_context[user_id] = {
                'current_intent': None,
                'conversation_history': [],
                'current_state': None
            }

        context = self.conversation_context[user_id]

        # Add user input to conversation history
        context['conversation_history'].append({"role": "user", "content": user_input})

        try:
            # Check cache first
            cache_key = f"{user_id}:{user_input}"
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < self.cache_duration:
                    return cache_entry['response']

            # Build complete conversation for GPT
            system_message = """You are an AI banking assistant. Your role is to:
            1. Help with banking services like transfers, loans, and balance inquiries
            2. Provide clear, concise responses
            3. Ask for necessary information step by step
            4. Remember context from previous messages
            5. Guide users through banking processes safely"""

            messages = [
                {"role": "system", "content": system_message},
                *context['conversation_history'][-5:]  # Include last 5 messages for context
            ]

            # Get GPT response using the new API method
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="gpt-4o",
                max_tokens=150,
                temperature=0.7
            )
            gpt_response = chat_completion.choices[0].message.content.strip()

            # Log the response for debugging
            print("GPT Response:", gpt_response)

            intent_info = self.classify_intent(user_input, context['current_intent'])
            if intent_info['intent'] != 'general':
                context['current_intent'] = intent_info['intent']

            final_response = self.format_response(intent_info, gpt_response, context)
            self.cache[cache_key] = {'response': final_response, 'timestamp': datetime.now()}
            self.save_cache()
            context['conversation_history'].append({"role": "assistant", "content": final_response['response']})

            return final_response

        except Exception as e:
            print(f"Error: {str(e)}")
            return {"intent": "error", "response": "Something went wrong. Please try again."}

    def classify_intent(self, user_input, current_intent=None):
        user_input = user_input.lower()

        # Check each intent's patterns using regex for word boundary matches
        for intent, data in self.intent_data['intents'].items():
            for pattern in data['patterns']:
                if re.search(rf'\b{re.escape(pattern.lower())}\b', user_input):
                    return {
                        'intent': intent,
                        'response': self.get_random_response(intent)
                    }

        # If no new intent is detected and we have a current intent, continue with it
        if current_intent:
            return {
                'intent': current_intent,
                'response': self.get_random_response(current_intent)
            }

        return {
            'intent': 'general',
            'response': self.get_random_response('general')
        }

    def format_response(self, intent_info, gpt_response, context):
        # Handle different intents and their states
        intent = intent_info['intent']

        if intent == 'transfer_money':
            if 'amount' not in gpt_response.lower() and 'recipient' not in context:
                return {
                    'intent': intent,
                    'response': "Please specify who you'd like to send money to and the amount."
                }
        elif intent == 'loan_inquiry':
            if 'amount' in gpt_response.lower():
                context['current_state'] = 'awaiting_amount'
            elif context.get('current_state') == 'awaiting_amount':
                context['current_state'] = 'amount_received'
        elif intent == 'location_search':
            if any(location in gpt_response.lower() for location in ['where', 'location', 'address']):
                return {
                    'intent': intent,
                    'response': "I'll help you find the nearest ATM. Please share your current location or city."
                }

        return {
            'intent': intent,
            'response': gpt_response
        }

    def get_random_response(self, intent):
        import random
        responses = self.intent_data['intents'].get(intent, {}).get('responses', [])
        if not responses:
            responses = ["I understand you need help with that. Let me assist you."]
        return random.choice(responses)
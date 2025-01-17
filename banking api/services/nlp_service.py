import openai
from functools import lru_cache
import json
import redis
from datetime import timedelta

redis_client = redis.Redis(host='localhost', port=6379, db=0)

class NLPService:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.intent_patterns = {
            'balance': ['balance', 'how much money', 'account balance'],
            'transfer': ['transfer', 'send money', 'payment'],
            'loan': ['loan', 'borrow', 'credit'],
            'support': ['help', 'support', 'issue', 'problem'],
            'investment': ['invest', 'mutual fund', 'fixed deposit'],
            'location': ['atm', 'branch', 'location', 'nearest']
        }

    @lru_cache(maxsize=1000)
    def get_cached_response(self, user_input):
        """Cache frequently requested responses"""
        cache_key = f"nlp_response:{user_input}"
        cached_response = redis_client.get(cache_key)
        
        if cached_response:
            return json.loads(cached_response)
            
        response = self.get_nlp_response(user_input)
        redis_client.setex(
            cache_key,
            timedelta(hours=24),
            json.dumps(response)
        )
        return response

    def get_nlp_response(self, user_input):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for banking operations."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return {
                'response': response['choices'][0]['message']['content'].strip(),
                'intent': self.classify_intent(user_input.lower())
            }
        except Exception as e:
            return {'error': str(e)}

    def classify_intent(self, user_input):
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in user_input.lower() for pattern in patterns):
                return intent
        return 'general'

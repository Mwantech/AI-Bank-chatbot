# At the very top of nlp_service.py
import sys
import os
from typing import Dict, Any, Optional, List, Tuple
import json
import pickle
import re
import logging
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BankingNLPService:
    def __init__(self, intents_file: str = 'intent_patterns.json'):
        """Initialize Banking NLP Service"""
        self._setup_paths()
        self._setup_logging()
        self._initialize_nlp_components()
        self.conversation_state = {}
        
    def _setup_paths(self):
        self.base_path = Path(os.path.abspath(os.path.dirname(__file__))).parent
        self.model_dir = self.base_path / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_nlp_components(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.load_model()

    def load_model(self) -> None:
        """Load or create NLP model"""
        try:
            with open(self.model_dir / 'intents.pkl', 'rb') as f:
                self.intents = pickle.load(f)
            
            with open(self.model_dir / 'patterns.pkl', 'rb') as f:
                patterns_data = pickle.load(f)
                self.patterns = patterns_data['patterns']
                self.pattern_classes = patterns_data['pattern_classes']
            
            self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
            self.X = self.vectorizer.fit_transform(self.patterns)
            
        except FileNotFoundError:
            self.logger.error("Model files not found")
            raise

    def preprocess_text(self, text: str) -> list:
        """Preprocess input text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens 
                if token not in self.stop_words]

    def get_intent(self, text: str) -> Tuple[str, float]:
        """Determine intent from input text"""
        processed_text = ' '.join(self.preprocess_text(text))
        input_vector = self.vectorizer.transform([processed_text])
        similarities = cosine_similarity(input_vector, self.X)
        most_similar = np.argmax(similarities)
        
        return (
            self.pattern_classes[most_similar],
            float(similarities[0][most_similar])
        )

    def process_user_request(self, user_id: int, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process user request with conversation context"""
        try:
            # Initialize or get conversation state
            if user_id not in self.conversation_state:
                self.conversation_state[user_id] = {
                    'current_intent': None,
                    'awaiting_response': False,
                    'last_question': None,
                    'loan_details': None
                }
            
            state = self.conversation_state[user_id]
            
           # Check for context reset conditions
            if state['active_process'] and not self._is_same_process(text, state):
                self.reset_user_context(user_id)
                state = self.conversation_state[user_id]

            # Handle follow-up responses first
            if state['awaiting_response']:
                response = self._handle_followup_response(user_id, text, state)
                if response:
                    return response

            # Get new intent if not in active process
            intent, confidence = self.get_intent(text)
            state['current_intent'] = intent

            # Handle low confidence
            if confidence < 0.5:
                return self._generate_response('fallback', context)

            # Process based on intent
            if intent == 'balance_inquiry':
                return self._handle_balance_inquiry(context)
            
            elif intent == 'transfer_funds':
                return self._initiate_transfer(user_id, context)
            
            elif intent == 'loan_inquiry':
                state['active_process'] = 'loan_inquiry'
                return self._handle_loan_inquiry(user_id, text, state)
            
            elif intent == 'investment_inquiry':
                return self._handle_investment_inquiry(user_id, text, state)
            
            elif intent == 'transaction_inquiry':
                return self._handle_transaction_inquiry(context)

            return self._generate_response(intent, context)

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return self._generate_response('error', context)
        
    def _generate_response(self, intent: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a predefined response based on intent and context"""
        user_name = context.get('user_name', 'valued customer') if context else 'valued customer'
        
        responses = {
            'fallback': f"I apologize {user_name}, I'm not sure I understand. Could you please rephrase that?",
            'error': f"I apologize {user_name}, I couldn't process your request. Please try again.",
            'account_not_found': f"I'm sorry {user_name}, I couldn't find your account details.",
            'no_transactions': f"I apologize {user_name}, no recent transactions were found.",
            'insufficient_accounts': f"I'm sorry {user_name}, you need at least two accounts to make a transfer.",
            'no_loans': f"Hello {user_name}, I don't see any active loans in your account.",
            'no_investments': f"Hello {user_name}, I don't see any active investments in your account.",
            'database_error': f"I apologize {user_name}, there was an issue retrieving your information.",
            'transfer_cancelled': f"I've cancelled the transfer. Is there anything else I can help you with?",
            'transfer_successful': f"Great! Your transfer has been processed successfully.",
            'invalid_amount': f"Please enter a valid amount in dollars.",
            'insufficient_funds': f"I'm sorry, but you don't have sufficient funds for this transaction.",
            'invalid_selection': f"I didn't understand your selection. Please try again.",
            'service_unavailable': f"I apologize, but this service is temporarily unavailable. Please try again later."
        }
        
        suggested_actions = {
            'fallback': ['rephrase_question', 'contact_support', 'view_faq'],
            'error': ['try_again', 'contact_support'],
            'account_not_found': ['update_account', 'contact_support'],
            'no_transactions': ['select_different_account', 'view_other_services'],
            'insufficient_accounts': ['open_new_account', 'view_account_types'],
            'no_loans': ['apply_for_loan', 'view_loan_types'],
            'no_investments': ['start_investing', 'view_investment_options'],
            'database_error': ['try_again', 'contact_support'],
            'transfer_cancelled': ['try_again', 'view_other_services'],
            'transfer_successful': ['view_balance', 'make_another_transfer'],
            'invalid_amount': ['enter_new_amount', 'cancel_transaction'],
            'insufficient_funds': ['enter_new_amount', 'view_balance', 'cancel_transaction'],
            'invalid_selection': ['view_options', 'try_again'],
            'service_unavailable': ['try_again_later', 'contact_support']
        }
        
        return {
            'intent': intent,
            'response': responses.get(intent, f"I apologize {user_name}, I can't help with that right now."),
            'confidence': 0.8 if intent in responses else 0.2,
            'suggested_actions': suggested_actions.get(intent, ['contact_support', 'view_faq'])
        }

    def _handle_followup_response(self, user_id: int, text: str, state: Dict) -> Optional[Dict]:
        """Handle follow-up responses to previous questions"""
        last_intent = state['current_intent']
        last_question = state['last_question']

        intent_handlers = {
            'loan_inquiry': self._process_loan_selection,
            'transfer_funds': self._process_transfer_details,
            'investment_inquiry': self._process_investment_selection,
            'transaction_inquiry': self._process_transaction_selection
        }

        handler = intent_handlers.get(last_intent)
        if handler:
            return handler(user_id, text, state)

        return None


    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract amount from text"""
        amount_pattern = r'\$?\d+(?:\.\d{2})?'
        matches = re.findall(amount_pattern, text)
        if matches:
            return float(matches[0].replace('$', ''))
        return None

    def _parse_date(self, text: str) -> Optional[str]:
        """Parse date from text using common formats"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}'   # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    def reset_user_context(self, user_id: int) -> None:
        """Reset user conversation context"""
        if user_id in self.conversation_state:
            self.conversation_state[user_id] = {
                'current_intent': None,
                'awaiting_response': False,
                'context': {},
                'last_question': None
            }

    def update_context(self, user_id: int, new_context: Dict) -> None:
        """Update user context with new information"""
        if user_id not in self.conversation_state:
            self.conversation_state[user_id] = {
                'current_intent': None,
                'awaiting_response': False,
                'context': {},
                'last_question': None
            }
        
        self.conversation_state[user_id]['context'].update(new_context)
    

    def _handle_balance_inquiry(self, context: Dict) -> Dict[str, Any]:
        """Handle balance inquiry with context"""
        try:
            if not context or 'accounts' not in context:
                return self._generate_response('account_not_found', context)

            accounts = context['accounts']
            if not accounts:
                return self._generate_response('account_not_found', context)

            response = "Here are your account balances:\n"
            for acc in accounts:
                response += f"{acc['account_type'].title()} Account: ${acc['balance']:.2f}\n"

            return {
                'intent': 'balance_inquiry',
                'response': response.strip(),
                'accounts': accounts,
                'suggested_actions': ['view_transactions', 'transfer_funds']
            }

        except Exception as e:
            self.logger.error(f"Balance inquiry error: {str(e)}")
            return self._generate_response('error', context)

    def _initiate_transfer(self, user_id: int, context: Dict) -> Dict[str, Any]:
        """Start transfer funds process"""
        state = self.conversation_state[user_id]
        
        if not context or 'accounts' not in context or len(context['accounts']) < 1:
            return self._generate_response('insufficient_accounts', context)
        
        state['awaiting_response'] = True
        state['last_question'] = 'source_account'
        state['transfer_details'] = {'accounts': context['accounts']}
        
        response = "From which account would you like to transfer?\n"
        for idx, acc in enumerate(context['accounts'], 1):
            response += f"{idx}. {acc['account_type'].title()} (Balance: ${acc['balance']:.2f})\n"
        
        return {
            'intent': 'transfer_funds',
            'response': response.strip(),
            'suggested_actions': ['select_account', 'cancel_transfer']
        }

    def _process_transfer_details(self, user_id: int, text: str, state: Dict) -> Dict[str, Any]:
        """Process transfer details step by step"""
        transfer_details = state.get('transfer_details', {})
        last_question = state['last_question']
        
        if last_question == 'source_account':
            # Process source account selection
            try:
                idx = int(text) - 1
                accounts = transfer_details['accounts']
                if 0 <= idx < len(accounts):
                    transfer_details['source_account'] = accounts[idx]
                    state['last_question'] = 'destination_account'
                    
                    response = "To which account would you like to transfer?\n"
                    for i, acc in enumerate(accounts, 1):
                        if i-1 != idx:  # Don't show source account
                            response += f"{i}. {acc['account_type'].title()} (Balance: ${acc['balance']:.2f})\n"
                    
                    return {
                        'intent': 'transfer_funds',
                        'response': response.strip(),
                        'suggested_actions': ['select_account', 'cancel_transfer']
                    }
            except (ValueError, IndexError):
                return {
                    'intent': 'transfer_funds',
                    'response': "Please select a valid account number.",
                    'suggested_actions': ['select_account', 'cancel_transfer']
                }
                
        elif last_question == 'destination_account':
            # Process destination account selection
            try:
                idx = int(text) - 1
                accounts = transfer_details['accounts']
                if 0 <= idx < len(accounts):
                    transfer_details['destination_account'] = accounts[idx]
                    state['last_question'] = 'amount'
                    return {
                        'intent': 'transfer_funds',
                        'response': "How much would you like to transfer?",
                        'suggested_actions': ['enter_amount', 'cancel_transfer']
                    }
            except (ValueError, IndexError):
                return {
                    'intent': 'transfer_funds',
                    'response': "Please select a valid account number.",
                    'suggested_actions': ['select_account', 'cancel_transfer']
                }
                
        elif last_question == 'amount':
            # Process transfer amount
            try:
                amount = float(re.sub(r'[^\d.]', '', text))
                source_account = transfer_details['source_account']
                
                if amount <= 0:
                    return {
                        'intent': 'transfer_funds',
                        'response': "Please enter a valid amount greater than 0.",
                        'suggested_actions': ['enter_amount', 'cancel_transfer']
                    }
                    
                if amount > source_account['balance']:
                    return {
                        'intent': 'transfer_funds',
                        'response': "Insufficient funds. Please enter a smaller amount.",
                        'suggested_actions': ['enter_amount', 'cancel_transfer']
                    }
                    
                # Reset state
                state['awaiting_response'] = False
                state['last_question'] = None
                
                return {
                    'intent': 'transfer_funds',
                    'response': f"Great! I'll transfer ${amount:.2f} from your {source_account['account_type'].title()} account to your {transfer_details['destination_account']['account_type'].title()} account. Would you like to proceed?",
                    'transfer_details': transfer_details,
                    'suggested_actions': ['confirm_transfer', 'cancel_transfer']
                }
                
            except ValueError:
                return {
                    'intent': 'transfer_funds',
                    'response': "Please enter a valid amount.",
                    'suggested_actions': ['enter_amount', 'cancel_transfer']
                }

        return self._generate_response('error', None)
    
    def _handle_loan_inquiry(self, user_id: int, text: str, state: Dict) -> Dict[str, Any]:
        """Handle loan inquiry with improved state management"""
        # If already in loan process, continue
        if state.get('loan_details'):
            return self._process_loan_selection(user_id, text, state)
            
        # Initialize loan process
        loan_types = {
            'personal': {
                'name': 'Personal Loans',
                'min_amount': 1000,
                'max_amount': 50000,
                'rate_range': '8.99% - 15.99%'
            },
            'home': {
                'name': 'Home Loans',
                'min_amount': 50000,
                'max_amount': 1000000,
                'rate_range': '4.99% - 6.99%'
            },
            'auto': {
                'name': 'Auto Loans',
                'min_amount': 5000,
                'max_amount': 100000,
                'rate_range': '5.99% - 9.99%'
            },
            'business': {
                'name': 'Business Loans',
                'min_amount': 10000,
                'max_amount': 500000,
                'rate_range': '7.99% - 12.99%'
            }
        }

        # Initialize loan details
        state.update({
            'awaiting_response': True,
            'last_question': 'loan_type',
            'loan_details': {
                'available_types': loan_types,
                'step': 'type_selection'
            },
            'active_process': 'loan_inquiry'
        })

        # Build response message
        response = "What type of loan are you interested in? We offer:\n"
        for idx, (_, loan_info) in enumerate(loan_types.items(), 1):
            response += f"{idx}. {loan_info['name']} (Rates: {loan_info['rate_range']})\n"

        return {
            'intent': 'loan_inquiry',
            'response': response.strip(),
            'suggested_actions': list(loan_types.keys()) + ['cancel'],
            'state': state
        }

    def _process_loan_selection(self, user_id: int, text: str, state: Dict) -> Dict[str, Any]:
        """Process loan selection with enhanced validation"""
        loan_details = state['loan_details']
        current_step = loan_details.get('step', 'type_selection')
        
        try:
            if current_step == 'type_selection':
                return self._handle_loan_type_selection(text, state)
            elif current_step == 'amount_selection':
                return self._handle_loan_amount_selection(text, state)
            elif current_step == 'term_selection':
                return self._handle_loan_term_selection(text, state)
            else:
                return self._generate_loan_error(state, "Unknown step in loan process")
        except Exception as e:
            self.logger.error(f"Loan processing error: {str(e)}")
            return self._generate_loan_error(state, "Error processing loan request")

    def _handle_loan_type_selection(self, text: str, state: Dict) -> Dict[str, Any]:
        """Handle loan type selection with improved matching"""
        loan_types = state['loan_details']['available_types']
        text = text.lower().strip()
        
        # Try direct matches first
        for key, details in loan_types.items():
            if key in text or details['name'].lower() in text:
                return self._update_loan_type(state, key)
        
        # Try numeric selection
        try:
            idx = int(text) - 1
            if 0 <= idx < len(loan_types):
                key = list(loan_types.keys())[idx]
                return self._update_loan_type(state, key)
        except ValueError:
            pass
        
        # No match found
        return {
            'intent': 'loan_inquiry',
            'response': "Please select a valid loan type by number or name:",
            'suggested_actions': list(loan_types.keys()) + ['cancel'],
            'state': state
        }

    def _update_loan_type(self, state: Dict, loan_key: str) -> Dict[str, Any]:
        """Update state with selected loan type"""
        loan_details = state['loan_details']
        loan_info = loan_details['available_types'][loan_key]
        
        loan_details.update({
            'selected_type': loan_key,
            'selected_info': loan_info,
            'step': 'amount_selection'
        })
        
        response = (
            f"Great choice! For {loan_info['name']}:\n"
            f"- Amount Range: ${loan_info['min_amount']:,} to ${loan_info['max_amount']:,}\n"
            f"- Rate Range: {loan_info['rate_range']}\n"
            f"How much would you like to borrow?"
        )
        
        return {
            'intent': 'loan_inquiry',
            'response': response,
            'suggested_actions': ['enter_amount', 'view_details', 'cancel'],
            'state': state
        }

    def _handle_loan_amount_selection(self, text: str, state: Dict) -> Dict[str, Any]:
        """Handle amount selection with better parsing"""
        loan_info = state['loan_details']['selected_info']
        
        try:
            amount = self._extract_currency_value(text)
            if not amount:
                raise ValueError("No valid amount found")
                
            if amount < loan_info['min_amount']:
                return self._generate_amount_error(state, "too_low")
                
            if amount > loan_info['max_amount']:
                return self._generate_amount_error(state, "too_high")
                
            return self._update_loan_amount(state, amount)
            
        except ValueError:
            return self._generate_amount_error(state, "invalid")

    def _update_loan_amount(self, state: Dict, amount: float) -> Dict[str, Any]:
        """Update state with validated loan amount"""
        state['loan_details']['amount'] = amount
        state['loan_details']['step'] = 'term_selection'
        
        terms = self._get_available_terms(state['loan_details']['selected_type'])
        response = "Please select a loan term:\n" + "\n".join(
            f"{idx}. {term}" for idx, term in enumerate(terms, 1)
        )
        
        return {
            'intent': 'loan_inquiry',
            'response': response,
            'suggested_actions': ['select_term', 'view_details', 'cancel'],
            'state': state
        }

    def _handle_loan_term_selection(self, text: str, state: Dict) -> Dict[str, Any]:
        """Handle term selection with confirmation"""
        terms = self._get_available_terms(state['loan_details']['selected_type'])
        
        try:
            # Try numeric selection
            idx = int(text.strip()) - 1
            if 0 <= idx < len(terms):
                term = terms[idx]
        except ValueError:
            # Try text match
            term = next((t for t in terms if t.lower() in text.lower()), None)
        
        if not term:
            return {
                'intent': 'loan_inquiry',
                'response': "Please select a valid term from the options:",
                'suggested_actions': ['select_term', 'cancel'],
                'state': state
            }
            
        return self._finalize_loan_details(state, term)

    def _finalize_loan_details(self, state: Dict, term: str) -> Dict[str, Any]:
        """Finalize loan details and show summary"""
        loan_details = state['loan_details']
        monthly_payment = self._calculate_monthly_payment(
            loan_details['amount'],
            float(loan_details['selected_info']['rate_range'].split('-')[0].strip().replace('%', '')),
            self._term_to_months(term)
        )
        
        loan_details.update({
            'term': term,
            'monthly_payment': monthly_payment,
            'step': 'confirmation'
        })
        
        response = (
            "Loan Summary:\n"
            f"- Type: {loan_details['selected_info']['name']}\n"
            f"- Amount: ${loan_details['amount']:,.2f}\n"
            f"- Term: {term}\n"
            f"- Estimated Monthly Payment: ${monthly_payment:,.2f}\n\n"
            "Would you like to proceed with this application?"
        )
        
        state['awaiting_response'] = False
        state['last_question'] = None
        
        return {
            'intent': 'loan_inquiry',
            'response': response,
            'suggested_actions': ['confirm', 'modify', 'cancel'],
            'state': state
        }

    # Helper methods
    def _is_same_process(self, text: str, state: Dict) -> bool:
        """Check if user is continuing current process"""
        if 'cancel' in text.lower():
            return False
        return any(keyword in text.lower() for keyword in ['loan', 'apply', 'continue'])

    def _extract_currency_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
        if match:
            return float(match.group(1).replace(',', ''))
        return None

    def _term_to_months(self, term: str) -> int:
        """Convert term string to months"""
        parts = term.split()
        value = int(parts[0])
        return value * 12 if 'year' in parts[1] else value

    def _generate_loan_error(self, state: Dict, error_type: str) -> Dict[str, Any]:
        """Generate error response for loan process"""
        messages = {
            'invalid': "Please enter a valid amount (e.g., $5000 or 5000)",
            'too_low': f"Minimum amount is ${state['loan_details']['selected_info']['min_amount']:,}",
            'too_high': f"Maximum amount is ${state['loan_details']['selected_info']['max_amount']:,}",
            'default': "Unable to process loan request. Please try again."
        }
        
        return {
            'intent': 'loan_inquiry',
            'response': messages.get(error_type, messages['default']),
            'suggested_actions': ['retry', 'modify', 'cancel'],
            'state': state
        }



    def _handle_investment_inquiry(self, user_id: int, text: str, state: Dict) -> Dict[str, Any]:
        """Handle investment inquiry with context"""
        if state['awaiting_response']:
            return self._process_investment_selection(user_id, text, state)

        investment_types = {
            'fixed_deposit': 'Fixed Deposits',
            'mutual_funds': 'Mutual Funds',
            'stocks': 'Stocks',
            'bonds': 'Bonds'
        }

        state['awaiting_response'] = True
        state['last_question'] = 'investment_type'
        state['available_options'] = investment_types

        response = "What type of investment are you interested in? We offer:\n"
        for idx, (_, inv_type) in enumerate(investment_types.items(), 1):
            response += f"{idx}. {inv_type}\n"

        return {
            'intent': 'investment_inquiry',
            'response': response.strip(),
            'suggested_actions': ['select_investment', 'schedule_advisor']
        }

    def _process_investment_selection(self, user_id: int, text: str, state: Dict) -> Dict[str, Any]:
        """Process investment type selection"""
        text = text.lower().strip()
        investment_types = state['available_options']
        
        selected_investment = None
        for inv_key, inv_name in investment_types.items():
            if inv_key.replace('_', ' ') in text or inv_name.lower() in text:
                selected_investment = inv_name
                break

        if not selected_investment:
            try:
                idx = int(text) - 1
                if 0 <= idx < len(investment_types):
                    selected_investment = list(investment_types.values())[idx]
            except ValueError:
                pass

        if not selected_investment:
            return {
                'intent': 'investment_inquiry',
                'response': "I didn't catch which investment type you're interested in. Please select a number or type the investment name.",
                'suggested_actions': ['select_investment_type']
            }

        # Reset state
        state['awaiting_response'] = False
        state['last_question'] = None

        return {
            'intent': 'investment_inquiry',
            'response': f"I'll help you learn more about {selected_investment}. Would you like to:\n1. View current rates\n2. Schedule a consultation with an advisor\n3. Start investing now",
            'suggested_actions': ['view_rates', 'schedule_advisor', 'start_investing']
        }

    def _handle_transaction_inquiry(self, context: Dict) -> Dict[str, Any]:
        """Handle transaction inquiry with context"""
        try:
            if not context or 'accounts' not in context:
                return self._generate_response('account_not_found', context)

            accounts = context['accounts']
            if not accounts:
                return self._generate_response('no_transactions', context)

            response = "Which account would you like to see transactions for?\n"
            for idx, acc in enumerate(accounts, 1):
                response += f"{idx}. {acc['account_type'].title()} Account\n"

            return {
                'intent': 'transaction_inquiry',
                'response': response.strip(),
                'accounts': accounts,
                'suggested_actions': ['select_account', 'view_all_transactions']
            }

        except Exception as e:
            self.logger.error(f"Transaction inquiry error: {str(e)}")
            return self._generate_response('error', context)
    
    def _process_transaction_selection(self, user_id: int, text: str, state: Dict) -> Dict[str, Any]:
        """Process transaction history selection"""
        try:
            accounts = state['context'].get('accounts', [])
            idx = int(text) - 1
            
            if 0 <= idx < len(accounts):
                selected_account = accounts[idx]
                state['awaiting_response'] = False
                state['last_question'] = None
                
                # Simulate getting recent transactions
                response = f"Recent transactions for your {selected_account['account_type'].title()} Account:\n"
                response += "1. No actual transaction data available in this example\n"
                response += "Would you like to:\n"
                response += "1. View more transactions\n"
                response += "2. Download statement\n"
                response += "3. Search transactions"
                
                return {
                    'intent': 'transaction_inquiry',
                    'response': response,
                    'account': selected_account,
                    'suggested_actions': ['view_more', 'download_statement', 'search_transactions']
                }
            else:
                return {
                    'intent': 'transaction_inquiry',
                    'response': "Please select a valid account number.",
                    'suggested_actions': ['select_account', 'view_all_transactions']
                }
                
        except ValueError:
            return {
                'intent': 'transaction_inquiry',
                'response': "Please select a valid account number.",
                'suggested_actions': ['select_account', 'view_all_transactions']
            }

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
            
            # Check if we're awaiting a specific response
            if state['awaiting_response'] and state['last_question']:
                response = self._handle_followup_response(user_id, text, state)
                if response:
                    return response

            # Get new intent
            intent, confidence = self.get_intent(text)
            state['current_intent'] = intent

            # Handle low confidence
            if confidence < 0.2:
                return self._generate_response('fallback', context)

            # Process based on intent
            if intent == 'balance_inquiry':
                return self._handle_balance_inquiry(context)
            
            elif intent == 'transfer_funds':
                return self._initiate_transfer(user_id, context)
            
            elif intent == 'loan_inquiry':
                return self._handle_loan_inquiry(user_id, text, state)
            
            elif intent == 'investment_inquiry':
                return self._handle_investment_inquiry(user_id, text, state)
            
            elif intent == 'transaction_inquiry':
                return self._handle_transaction_inquiry(context)

            return self._generate_response(intent, context)

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return self._generate_response('error', context)

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
        """Handle loan inquiry with full context management"""
        # Check if we're in an active loan conversation and have received initial input
        if text.lower() != "i need a loan" and state.get('awaiting_response') and state.get('loan_details'):
            return self._process_loan_selection(user_id, text, state)

        # Initialize loan types and details only for new conversations
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

        # Initialize fresh conversation state only for new conversations
        if not state.get('awaiting_response'):
            state.clear()  # Clear any previous state
            state['awaiting_response'] = True
            state['last_question'] = 'loan_type'
            state['loan_details'] = {
                'available_types': loan_types,
                'step': 'type_selection'
            }

            # Generate initial response
            response = "What type of loan are you interested in? We offer:\n"
            for idx, (_, loan_info) in enumerate(loan_types.items(), 1):
                response += f"{idx}. {loan_info['name']} (Rates: {loan_info['rate_range']})\n"

            return {
                'intent': 'loan_inquiry',
                'response': response.strip(),
                'suggested_actions': ['select_loan_type', 'view_rates', 'calculate_emi'],
                'state': state
            }
        
        # If we get here, something went wrong with the state
        return self._process_loan_selection(user_id, text, state)

    def _process_loan_selection(self, user_id: int, text: str, state: Dict) -> Dict[str, Any]:
        """Process loan selection and follow-up questions"""
        loan_details = state.get('loan_details', {})
        current_step = loan_details.get('step', 'type_selection')
        
        # Handle each step of the loan process
        try:
            if current_step == 'type_selection':
                response_dict = self._handle_loan_type_selection(text, state)
            elif current_step == 'amount_selection':
                response_dict = self._handle_loan_amount_selection(text, state)
            elif current_step == 'term_selection':
                response_dict = self._handle_loan_term_selection(text, state)
            else:
                response_dict = {
                    'intent': 'loan_inquiry',
                    'response': "I apologize, but I've lost track of our conversation. Let's start over with your loan inquiry.",
                    'suggested_actions': ['restart_loan_inquiry'],
                    'state': {}
                }
                state.clear()
        except Exception as e:
            response_dict = {
                'intent': 'error',
                'response': "I encountered an error processing your request. Let's start over.",
                'suggested_actions': ['restart_loan_inquiry'],
                'state': {}
            }
            state.clear()
        
        # Ensure state is included in response
        response_dict['state'] = state
        return response_dict

    def _handle_loan_type_selection(self, text: str, state: Dict) -> Dict[str, Any]:
        """Handle loan type selection"""
        text = text.lower().strip()
        loan_types = state['loan_details']['available_types']
        
        # Try to match by name or number
        selected_loan_type = None
        selected_loan_info = None
        
        # Try to match by text
        for loan_key, loan_info in loan_types.items():
            if loan_key in text or loan_info['name'].lower().replace(' loans', '') in text:
                selected_loan_type = loan_key
                selected_loan_info = loan_info
                break
        
        # Try to match by number if no text match
        if not selected_loan_type:
            try:
                idx = int(text) - 1
                if 0 <= idx < len(loan_types):
                    selected_loan_type = list(loan_types.keys())[idx]
                    selected_loan_info = list(loan_types.values())[idx]
            except ValueError:
                pass
        
        if not selected_loan_type:
            return {
                'intent': 'loan_inquiry',
                'response': "I didn't catch which loan type you're interested in. Please select a number or type the loan name.",
                'suggested_actions': ['select_loan_type', 'view_rates']
            }

        # Update loan details with selection
        state['loan_details'].update({
            'selected_type': selected_loan_type,
            'selected_info': selected_loan_info,
            'step': 'amount_selection'
        })

        response = (
            f"Great! For {selected_loan_info['name']}, we offer loans between "
            f"${selected_loan_info['min_amount']:,} and ${selected_loan_info['max_amount']:,} "
            f"with rates from {selected_loan_info['rate_range']}.\n"
            f"How much would you like to borrow?"
        )

        return {
            'intent': 'loan_inquiry',
            'response': response,
            'suggested_actions': ['enter_amount', 'calculate_emi', 'view_rates']
        }

    def _handle_loan_amount_selection(self, text: str, state: Dict) -> Dict[str, Any]:
        """Handle loan amount selection"""
        loan_details = state['loan_details']
        selected_info = loan_details['selected_info']
        
        # Extract amount from text
        try:
            # Remove currency symbols and commas, then convert to float
            amount_str = text.replace('$', '').replace(',', '').strip()
            # Find first sequence of numbers (with possible decimal point)
            import re
            amount_match = re.search(r'\d+(\.\d+)?', amount_str)
            if amount_match:
                amount = float(amount_match.group())
            else:
                raise ValueError("No valid number found")
        except (ValueError, AttributeError):
            return {
                'intent': 'loan_inquiry',
                'response': "Please enter a valid loan amount in dollars (for example: $5000 or 5000).",
                'suggested_actions': ['enter_amount', 'view_rates']
            }
        
        # Validate amount against loan limits
        if amount < selected_info['min_amount']:
            return {
                'intent': 'loan_inquiry',
                'response': (
                    f"The minimum loan amount for {selected_info['name']} is ${selected_info['min_amount']:,}. "
                    f"Please enter a larger amount or select a different loan type."
                ),
                'suggested_actions': ['enter_amount', 'select_different_loan']
            }
        
        if amount > selected_info['max_amount']:
            return {
                'intent': 'loan_inquiry',
                'response': (
                    f"The maximum loan amount for {selected_info['name']} is ${selected_info['max_amount']:,}. "
                    f"Please enter a smaller amount or select a different loan type."
                ),
                'suggested_actions': ['enter_amount', 'select_different_loan']
            }
        
        # Update loan details
        loan_details.update({
            'amount': amount,
            'step': 'term_selection'
        })

        # Get available terms based on loan type
        terms = self._get_available_terms(loan_details['selected_type'])
        response = (
            f"For a ${amount:,.2f} {selected_info['name']}, we offer the following terms:\n"
            + "\n".join(f"{idx}. {term}" for idx, term in enumerate(terms, 1))
            + "\n\nPlease select a term by entering the number."
        )

        return {
            'intent': 'loan_inquiry',
            'response': response,
            'suggested_actions': ['select_term', 'calculate_emi', 'view_rates']
        }

    def _handle_loan_term_selection(self, text: str, state: Dict) -> Dict[str, Any]:
        """Handle loan term selection"""
        loan_details = state['loan_details']
        terms = self._get_available_terms(loan_details['selected_type'])
        
        # Try to match term selection
        selected_term = None
        try:
            # Try to match by number
            idx = int(text.strip()) - 1
            if 0 <= idx < len(terms):
                selected_term = terms[idx]
        except ValueError:
            # Try to match by text
            text_lower = text.lower().strip()
            for term in terms:
                if term.lower() in text_lower:
                    selected_term = term
                    break
        
        if not selected_term:
            return {
                'intent': 'loan_inquiry',
                'response': (
                    "Please select a valid term by entering the number from the options provided.\n"
                    + "\n".join(f"{idx}. {term}" for idx, term in enumerate(terms, 1))
                ),
                'suggested_actions': ['select_term', 'view_rates']
            }
                
        # Calculate estimated monthly payment
        term_months = int(selected_term.split()[0])
        if 'years' in selected_term:
            term_months *= 12
            
        estimated_rate = float(loan_details['selected_info']['rate_range'].split(' - ')[0].replace('%', ''))
        monthly_payment = self._calculate_monthly_payment(
            loan_details['amount'],
            estimated_rate,
            term_months
        )

        # Reset conversation state
        state['awaiting_response'] = False
        state['last_question'] = None
        loan_details['selected_term'] = selected_term
        loan_details['monthly_payment'] = monthly_payment
        
        response = (
            f"Based on your selections:\n"
            f"- Loan Type: {loan_details['selected_info']['name']}\n"
            f"- Amount: ${loan_details['amount']:,.2f}\n"
            f"- Term: {selected_term}\n"
            f"- Estimated Rate: {estimated_rate}%\n"
            f"- Estimated Monthly Payment: ${monthly_payment:,.2f}\n\n"
            f"Would you like to proceed with the loan application?"
        )

        return {
            'intent': 'loan_inquiry',
            'response': response,
            'loan_details': loan_details,
            'suggested_actions': ['apply_now', 'adjust_terms', 'speak_to_advisor']
        }

    def _get_available_terms(self, loan_type: str) -> List[str]:
        """Get available terms based on loan type"""
        terms = {
            'personal': ['12 months', '24 months', '36 months', '48 months', '60 months'],
            'home': ['15 years', '20 years', '30 years'],
            'auto': ['36 months', '48 months', '60 months', '72 months'],
            'business': ['12 months', '24 months', '36 months', '48 months', '60 months']
        }
        return terms.get(loan_type, ['12 months', '24 months', '36 months'])

    def _calculate_monthly_payment(self, principal: float, annual_rate: float, term_months: int) -> float:
        """Calculate estimated monthly loan payment"""
        monthly_rate = (annual_rate / 100) / 12
        if monthly_rate == 0:
            return principal / term_months
        return principal * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)



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
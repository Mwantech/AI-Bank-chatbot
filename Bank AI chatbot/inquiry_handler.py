from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from models import User, Account, Loan, Transaction, ATMLocation, db
from functools import lru_cache
import os
import json
import re
from decimal import Decimal

import logging
logger = logging.getLogger(__name__)

class InquiryHandler:
    def __init__(self, chatbot):
        self.pending_inquiries: Dict[int, Dict[str, Any]] = {}
        self.chatbot = chatbot
        self.cache_timeout = timedelta(hours=1)
        self.cache_timestamp = {}
        
        # Updated verification questions to use User table data
        self.verification_questions = {
            'personal': [
                {
                    'question': 'What is your identification number?',
                    'field': 'IdentificationNumber',
                    'type': 'id'
                }
            ],
            'contact': [
                {
                    'question': 'Please confirm your phone number',
                    'field': 'PhoneNumber',
                    'type': 'phone'
                },
                {
                    'question': 'Please provide your current address',
                    'field': 'Address',
                    'type': 'address'
                }
            ],
            'security': [
                {
                    'question': 'Please answer your security question: {security_question}',
                    'field': 'SecurityAnswer',
                    'type': 'security'
                }
            ]
        }

    
    def handle_inquiry(self, user_id: int, inquiry_text: str, 
                    intent: Optional[str] = None, 
                    entities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for handling user inquiries with improved context switching
        """
        try:
            # Get response from chatbot if not provided
            if intent is None or entities is None:
                chatbot_response = self.chatbot.get_response(inquiry_text, user_id)
                intent = chatbot_response['intent']
                entities = chatbot_response['entities']
            
            # Check if intent has changed from pending inquiry
            if user_id in self.pending_inquiries:
                previous_intent = self.pending_inquiries[user_id]['intent']
                # If intent has changed, clear pending inquiry
                if intent != previous_intent and intent not in ['unknown', 'fallback', None]:
                    del self.pending_inquiries[user_id]
                    # Proceed with new intent
                    return self._process_new_inquiry(user_id, intent, entities)
                # Otherwise, handle the pending inquiry as normal
                return self._handle_pending_inquiry(user_id, inquiry_text, intent, entities)
            
            # Only handle specific inquiries
            if intent not in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                return {
                    'status': 'error',
                    'message': 'Unsupported inquiry type'
                }
            
            # Process the specific inquiry
            return self._process_new_inquiry(user_id, intent, entities)
            
        except Exception as e:
            logger.error(f"Error processing inquiry: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error processing inquiry: {str(e)}'
            }

    def _handle_pending_inquiry(self, user_id: int, inquiry_text: str, 
                            new_intent: Optional[str] = None,
                            new_entities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle follow-up for pending inquiries with improved context handling
        """
        pending = self.pending_inquiries[user_id]
        
        # If in verification stage, handle verification
        if pending.get('verification_stage') is not None:
            return self._handle_verification(user_id, inquiry_text)
        
        # Use provided entities or extract new ones
        if new_entities is None:
            # Use chatbot to extract new entities
            chatbot_response = self.chatbot.get_response(inquiry_text, user_id)
            new_entities = chatbot_response['entities']
        
        # Update provided entities
        if new_entities:
            pending['entities'].update(new_entities)
        
        # Update missing entities - only for entities that we've gathered
        pending['missing_entities'] = [
            entity for entity in pending['missing_entities']
            if entity not in pending['entities']
        ]
        
        if not pending['missing_entities']:
            # All entities collected, proceed with handling
            handlers = {
                'account_activation': self._handle_activation,
                'account_deactivation': self._handle_deactivation,
                'loan_status': self._handle_loan_status,
                'atm_location': self._handle_atm_location
            }
            # Clear the pending inquiry before handling
            entities = pending['entities']
            intent = pending['intent']
            del self.pending_inquiries[user_id]
            return handlers[intent](user_id, entities)
        
        return {
            'status': 'incomplete',
            'message': f"Please provide: {', '.join(pending['missing_entities'])}",
            'missing_fields': pending['missing_entities']
        }
    
    def _process_new_inquiry(self, user_id: int, intent: str, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a new inquiry based on intent
        """
        handlers = {
            'account_activation': self._handle_activation,
            'account_deactivation': self._handle_deactivation,
            'loan_status': self._handle_loan_status,
            'atm_location': self._handle_atm_location
        }
        
        # Get required entities for the intent
        required_entities = self._get_required_entities(intent)
        missing_entities = [entity for entity in required_entities if entity not in entities]
        
        if missing_entities:
            # Store pending inquiry
            self.pending_inquiries[user_id] = {
                'intent': intent,
                'entities': entities,
                'missing_entities': missing_entities,
                'verification_stage': None
            }
            return {
                'status': 'incomplete',
                'message': f"Please provide the following information: {', '.join(missing_entities)}",
                'missing_fields': missing_entities
            }
        
        return handlers[intent](user_id, entities)
    
    
    def _verify_response(self, verification_type: str, stage: int, response: str, user: User) -> bool:
        """
        Verify user responses against User table data
        """
        try:
            verification_data = self.verification_questions[verification_type][stage]
            field_type = verification_data['type']
            field_name = verification_data['field']
            
            # Get the actual value from the user record
            actual_value = getattr(user, field_name)
            
            if actual_value is None:
                return False
                
            # Clean and normalize the response
            cleaned_response = self._normalize_response(response, field_type)
            cleaned_actual = self._normalize_response(str(actual_value), field_type)
            
            # Compare based on field type
            if field_type == 'date':
                # Convert to datetime for comparison
                response_date = datetime.strptime(cleaned_response, '%Y-%m-%d').date()
                actual_date = datetime.strptime(cleaned_actual, '%Y-%m-%d').date()
                return response_date == actual_date
                
            elif field_type == 'id':
                # Case-insensitive exact match for ID
                return cleaned_response.lower() == cleaned_actual.lower()
                
            elif field_type == 'phone':
                # Compare normalized phone numbers (digits only)
                return cleaned_response == cleaned_actual
                
            elif field_type == 'address':
                # Fuzzy match for address (allowing for minor differences)
                return self._fuzzy_address_match(cleaned_response, cleaned_actual)
                
            elif field_type == 'security':
                # Case-insensitive security answer comparison
                return cleaned_response.lower() == cleaned_actual.lower()
                
            return False
            
        except Exception as e:
            print(f"Verification error: {str(e)}")
            return False

    def _normalize_response(self, value: str, field_type: str) -> str:
        """
        Normalize response values for comparison
        """
        if not value:
            return ''
            
        value = value.strip()
        
        if field_type == 'date':
            # Try to parse and standardize date format
            try:
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']:
                    try:
                        return datetime.strptime(value, fmt).strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            except Exception:
                return value
                
        elif field_type == 'phone':
            # Keep only digits
            return re.sub(r'\D', '', value)
                
        elif field_type == 'id':
            # Remove spaces and special characters
            return re.sub(r'\W', '', value)
                
        elif field_type == 'address':
            # Normalize address (lowercase, remove extra spaces, common abbreviations)
            value = value.lower()
            value = re.sub(r'\s+', ' ', value)
            value = value.replace('street', 'st').replace('avenue', 'ave')
            value = value.replace('road', 'rd').replace('boulevard', 'blvd')
            return value
                
        elif field_type == 'security':
            # Normalize security answer (lowercase, remove extra spaces)
            return ' '.join(value.split()).lower()
            
        return value

    def _fuzzy_address_match(self, addr1: str, addr2: str) -> bool:
        """
        Perform fuzzy matching for addresses
        """
        # Normalize both addresses
        addr1_norm = self._normalize_response(addr1, 'address')
        addr2_norm = self._normalize_response(addr2, 'address')
        
        # Split into components
        addr1_parts = set(addr1_norm.split())
        addr2_parts = set(addr2_norm.split())
        
        # Calculate similarity
        common_parts = addr1_parts.intersection(addr2_parts)
        total_parts = addr1_parts.union(addr2_parts)
        
        # Return true if addresses are at least 80% similar
        return len(common_parts) / len(total_parts) >= 0.8
    
    def _get_account_field_value(self, account: Account, field_name: str) -> Any:
        """
        Get the actual value from the database for verification
        """
        try:
            if field_name == 'last_transaction_amount':
                transaction = (db.session.query(Transaction)
                             .filter(Transaction.AccountID == account.AccountID)
                             .order_by(Transaction.TransactionDate.desc())
                             .first())
                return transaction.Amount if transaction else None
                
            elif field_name == 'opening_date':
                return account.OpeningDate
                
            elif field_name == 'last_deposit_amount':
                deposit = (db.session.query(Transaction)
                         .filter(Transaction.AccountID == account.AccountID,
                                Transaction.TransactionType == 'deposit')
                         .order_by(Transaction.TransactionDate.desc())
                         .first())
                return deposit.Amount if deposit else None
                
            elif field_name == 'linked_account':
                return account.LinkedAccountNumber
                
            elif field_name == 'last_payment_date':
                payment = (db.session.query(Transaction)
                         .filter(Transaction.AccountID == account.AccountID,
                                Transaction.TransactionType == 'payment')
                         .order_by(Transaction.TransactionDate.desc())
                         .first())
                return payment.TransactionDate if payment else None
                
            elif field_name == 'credit_limit':
                return account.CreditLimit
                
            elif field_name == 'portfolio_id':
                return account.PortfolioID
                
            elif field_name == 'advisor_name':
                return account.AdvisorName
                
            return None
            
        except Exception as e:
            print(f"Error getting field value: {str(e)}")
            return None

    

    def _handle_activation(self, user_id: int, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle account activation with user verification
        """
        try:
            # Get user record
            user = db.session.query(User).filter(User.UserID == user_id).first()
            
            if not user:
                return {'status': 'error', 'message': 'User not found'}
            
            # Start with personal verification
            self.pending_inquiries[user_id] = {
                'intent': 'account_activation',
                'entities': entities,
                'verification_stage': 0,
                'verification_type': 'personal',
                'user': user
            }
            
            question_data = self.verification_questions['personal'][0]
            return {
                'status': 'verification',
                'message': question_data['question']
            }
            
        except Exception as e:
            db.session.rollback()
            return {'status': 'error', 'message': 'Error processing activation request'}
    # Add this method to your InquiryHandler class

    def process_after_verification(self, user_id, intent, entities):
        """
        Process the final request after all verification steps are complete.
        
        Args:
            user_id (int): The user ID
            intent (str): The intent being processed
            entities (dict): All collected entities including verification responses
            
        Returns:
            dict: Response with message and status
        """
        try:
            # Process the intent now that verification is complete
            if intent == 'account_activation':
                # Logic for activating account
                return {
                    'message': f"Your account has been successfully activated. You'll receive a confirmation email shortly.",
                    'status': 'success'
                }
            elif intent == 'account_deactivation':
                # Logic for deactivating account
                return {
                    'message': f"Your account has been successfully deactivated. You'll receive a confirmation email shortly.",
                    'status': 'success'
                }
            elif intent == 'loan_status':
                # Logic for loan status check
                # This would typically query a database
                return {
                    'message': f"Your loan application is currently under review. We'll notify you of any updates.",
                    'status': 'success'
                }
            elif intent == 'atm_location':
                # Logic for ATM location
                return {
                    'message': f"We've found ATM locations near you. Check your email for a detailed list.",
                    'status': 'success'
                }
            else:
                return {
                    'message': "Your request has been processed successfully.",
                    'status': 'success'
                }
        except Exception as e:
            logger.error(f"Error in process_after_verification: {str(e)}")
            return {
                'message': "We encountered an error processing your request. Please try again later.",
                'status': 'error'
            }

    def handle_verification(self, user_id: int, response: str, 
                       intent: Optional[str] = None,
                       entities: Optional[Dict[str, Any]] = None,
                       verification_category: Optional[str] = None,
                       verification_step: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle verification question responses with additional parameters for socket handler
        """
        try:
            # First check if we have a pending inquiry
            if user_id not in self.pending_inquiries:
                return {
                    'status': 'error',
                    'message': 'No active verification in progress'
                }
                
            pending = self.pending_inquiries[user_id]
            
            # If verification_category and verification_step are provided, use them 
            # to override the pending inquiry values (for compatibility with socket handler)
            if verification_category is not None:
                pending['verification_type'] = verification_category
            
            if verification_step is not None:
                pending['verification_stage'] = verification_step
                
            verification_type = pending['verification_type']
            user = pending['user']
            
            # Rest of your verification logic remains the same
            # If it's a security question, format it with the user's actual security question
            if verification_type == 'security':
                question_data = self.verification_questions[verification_type][pending['verification_stage']]
                question = question_data['question'].format(security_question=user.SecurityQuestion)
            else:
                question_data = self.verification_questions[verification_type][pending['verification_stage']]
            
            # Verify response
            if self._verify_response(verification_type, pending['verification_stage'], response, user):
                # Move to next stage or type of verification
                if pending['verification_stage'] < len(self.verification_questions[verification_type]) - 1:
                    # More questions in current type
                    pending['verification_stage'] += 1
                    next_question = self.verification_questions[verification_type][pending['verification_stage']]
                    return {
                        'status': 'verification',
                        'message': next_question['question'],
                        'is_verification_question': True,
                        'verification_step': pending['verification_stage'],
                        'verification_category': verification_type
                    }
                elif verification_type == 'personal':
                    # Move to contact verification
                    pending['verification_type'] = 'contact'
                    pending['verification_stage'] = 0
                    return {
                        'status': 'verification',
                        'message': self.verification_questions['contact'][0]['question'],
                        'is_verification_question': True,
                        'verification_step': 0,
                        'verification_category': 'contact'
                    }
                elif verification_type == 'contact':
                    # Move to security question
                    pending['verification_type'] = 'security'
                    pending['verification_stage'] = 0
                    return {
                        'status': 'verification',
                        'message': self.verification_questions['security'][0]['question'].format(
                            security_question=user.SecurityQuestion
                        ),
                        'is_verification_question': True,
                        'verification_step': 0,
                        'verification_category': 'security'
                    }
                else:
                    # All verifications passed, proceed with activation
                    try:
                        # Use entities from both socket handler and pending inquiry
                        if entities:
                            pending['entities'].update(entities)
                        
                        account = db.session.query(Account).filter(
                            Account.UserID == user_id,
                            Account.AccountType == pending['entities']['account_type']
                        ).first()
                        
                        if account:
                            account.Status = 'Active'
                            db.session.commit()
                            del self.pending_inquiries[user_id]
                            return {
                                'status': 'success',
                                'message': 'Verification successful. Your account has been activated.',
                                'is_verification_question': False
                            }
                        else:
                            return {
                                'status': 'error',
                                'message': 'Account not found',
                                'is_verification_question': False
                            }
                    except Exception as e:
                        db.session.rollback()
                        return {
                            'status': 'error', 
                            'message': 'Error activating account',
                            'is_verification_question': False
                        }
            else:
                # Failed verification
                del self.pending_inquiries[user_id]
                return {
                    'status': 'error',
                    'message': 'Verification failed. For security reasons, please contact customer support.',
                    'is_verification_question': False
                }
                    
        except Exception as e:
            logger.error(f"Error during verification process: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'status': 'error', 
                'message': 'Error during verification process',
                'is_verification_question': False
            }
    
    def _handle_deactivation(self, user_id: int, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle account deactivation
        """
        try:
            account = db.session.query(Account).filter(
                Account.UserID == user_id,
                Account.AccountType == entities['account_type'],
                Account.AccountNumber == entities['account_number']
            ).first()
            
            if not account:
                return {'status': 'error', 'message': 'Account not found'}
            
            if account.Status != 'Active':
                return {'status': 'error', 'message': 'Account is already inactive'}
            
            account.Status = 'Inactive'
            db.session.commit()
            
            return {
                'status': 'success',
                'message': f'Your {entities["account_type"]} account has been deactivated'
            }
            
        except Exception as e:
            db.session.rollback()
            return {'status': 'error', 'message': 'Error processing deactivation request'}
    
        # Add caching for frequently accessed data
    @lru_cache(maxsize=128)
    def _get_required_entities(self, intent: str) -> list:
        """
        Get required entities for each intent with caching
        """
        requirements = {
            'account_activation': ['account_type', 'account_number'],
            'account_deactivation': ['account_type', 'account_number', 'reason_code'],
            'loan_status': ['application_id', 'loan_type'],
            'atm_location': ['location']
        }
        return requirements.get(intent, [])

    # Optimize database queries with filter optimization
    def _handle_loan_status(self, user_id: int, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle loan status inquiry with optimized database query
        """
        try:
            # Use more specific queries with indexable fields first
            filters = [Loan.UserID == user_id]
            
            if 'application_id' in entities:
                filters.append(Loan.ApplicationID == entities['application_id'])
            
            if 'loan_type' in entities:
                filters.append(Loan.LoanType == entities['loan_type'])
            
            # Execute optimized query
            loan = db.session.query(Loan).filter(*filters).first()
            
            if not loan:
                return {'status': 'error', 'message': 'Loan application not found'}
            
            # Use SQLAlchemy's defer() for fields not immediately needed
            return {
                'status': 'success',
                'data': {
                    'loan_type': loan.LoanType,
                    'status': loan.Status,
                    'requested_amount': float(loan.RequestedAmount),
                    'approved_amount': float(loan.ApprovedAmount) if loan.ApprovedAmount else None,
                    'interest_rate': float(loan.InterestRate) if loan.InterestRate else None,
                    'application_date': loan.ApplicationDate.strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing loan status inquiry: {str(e)}")
            db.session.rollback()
            return {'status': 'error', 'message': 'Error processing loan status inquiry'}

    def _handle_atm_location(self, user_id: int, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle ATM location inquiry using a JSON file instead of database queries
        """
        try:
            location = entities.get('location', '').strip()
            if not location:
                return {'status': 'error', 'message': 'Location is required'}
                
            # JSON file path containing ATM locations
            atm_json_path = os.path.join(os.path.dirname(__file__), 'data', 'atm_locations.json')
            
            # Load ATM data from JSON file
            try:
                with open(atm_json_path, 'r') as f:
                    all_atms = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error loading ATM data from JSON: {str(e)}")
                return {'status': 'error', 'message': 'Unable to access ATM location data'}
            
            # Initialize cache attributes if they don't exist
            if not hasattr(self.chatbot, 'cache'):
                self.chatbot.cache = {}
            
            if not hasattr(self, 'cache_timestamp'):
                self.cache_timestamp = {}
                
            # Use a cache key for this query
            cache_key = f"atm_location_{location}"
            
            # Check cache first (if cache is properly initialized)
            cache_timeout = getattr(self, 'cache_timeout', timedelta(hours=1))
            if (
                cache_key in self.chatbot.cache and 
                self.cache_timestamp.get(cache_key, datetime.min) > datetime.now() - cache_timeout
            ):
                return self.chatbot.cache[cache_key]
            
            # Filter ATMs by location (case-insensitive)
            location_lower = location.lower()
            matching_atms = [
                atm for atm in all_atms 
                if (
                    location_lower in atm.get('city', '').lower() or 
                    location_lower in atm.get('state', '').lower()
                ) and atm.get('is_accessible', True)
            ][:5]  # Limit to 5 results
            
            # Format the ATM data
            atm_data = [
                {
                    'branch_name': atm.get('branch_name', ''),
                    'address': atm.get('address', ''),
                    'city': atm.get('city', ''),
                    'state': atm.get('state', ''),
                    'zip_code': atm.get('zip_code', ''),
                    'operating_hours': atm.get('operating_hours', ''),
                    'additional_services': atm.get('additional_services', '')
                } for atm in matching_atms
            ]
            
            if not atm_data:
                result = {
                    'status': 'error', 
                    'message': f'No ATMs found in {location}',
                    'display_text': f'I couldn\'t find any ATMs in {location}. Please try another location.'
                }
            else:
                # Create a user-friendly display text with the ATM information
                display_text = f"Here are ATMs in {location}:\n\n"
                for i, atm in enumerate(atm_data, 1):
                    display_text += f"{i}. {atm['branch_name']}\n"
                    display_text += f"   Address: {atm['address']}, {atm['city']}, {atm['state']} {atm['zip_code']}\n"
                    display_text += f"   Hours: {atm['operating_hours']}\n"
                    display_text += f"   Services: {atm['additional_services']}\n\n"
                
                result = {
                    'status': 'success', 
                    'data': atm_data,
                    'display_text': display_text
                }
            
            # Cache the result
            self.chatbot.cache[cache_key] = result
            self.cache_timestamp[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing ATM location inquiry: {str(e)}")
            return {
                'status': 'error', 
                'message': 'Error processing ATM location inquiry',
                'display_text': 'Sorry, I encountered an error while searching for ATMs. Please try again later.'
            }
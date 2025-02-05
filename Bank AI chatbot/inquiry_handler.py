from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from models import User, Account, Loan, Transaction, ATMLocation, db
from functools import lru_cache
import re
from decimal import Decimal

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
                    'question': 'Please provide your date of birth (YYYY-MM-DD)',
                    'field': 'DateOfBirth',
                    'type': 'date'
                },
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
        Main entry point for handling user inquiries
        """
        try:
            # First check pending inquiries
            if user_id in self.pending_inquiries:
                return self._handle_pending_inquiry(user_id, inquiry_text)
            
            # Use provided intent and entities if available
            if intent is None or entities is None:
                chatbot_response = self.chatbot.get_response(inquiry_text, user_id)
                intent = chatbot_response['intent']
                entities = chatbot_response['entities']
            
            # Only handle specific inquiries
            if intent not in ['account_activation', 'account_deactivation', 'loan_status', 'atm_location']:
                return {
                    'status': 'error',
                    'message': 'Unsupported inquiry type'
                }
            
            # Process the specific inquiry
            return self._process_new_inquiry(user_id, intent, entities)
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing inquiry: {str(e)}'
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
    
    def _handle_pending_inquiry(self, user_id: int, inquiry_text: str) -> Dict[str, Any]:
        """
        Handle follow-up for pending inquiries including verification questions
        """
        pending = self.pending_inquiries[user_id]
        
        # If in verification stage, handle verification
        if pending.get('verification_stage') is not None:
            return self._handle_verification(user_id, inquiry_text)
        
        # Use chatbot to extract new entities
        chatbot_response = self.chatbot.get_response(inquiry_text, user_id)
        new_entities = chatbot_response['entities']
        
        # Update provided entities
        pending['entities'].update(new_entities)
        
        # Update missing entities
        pending['missing_entities'] = [
            entity for entity in pending['missing_entities']
            if entity not in new_entities
        ]
        
        if not pending['missing_entities']:
            # All entities collected, proceed with handling
            handlers = {
                'account_activation': self._handle_activation,
                'account_deactivation': self._handle_deactivation,
                'loan_status': self._handle_loan_status,
                'atm_location': self._handle_atm_location
            }
            return handlers[pending['intent']](user_id, pending['entities'])
        
        return {
            'status': 'incomplete',
            'message': f"Please provide: {', '.join(pending['missing_entities'])}",
            'missing_fields': pending['missing_entities']
        }
    
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

    def _handle_verification(self, user_id: int, response: str) -> Dict[str, Any]:
        """
        Handle verification question responses
        """
        try:
            pending = self.pending_inquiries[user_id]
            verification_type = pending['verification_type']
            user = pending['user']
            
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
                        'message': next_question['question']
                    }
                elif verification_type == 'personal':
                    # Move to contact verification
                    pending['verification_type'] = 'contact'
                    pending['verification_stage'] = 0
                    return {
                        'status': 'verification',
                        'message': self.verification_questions['contact'][0]['question']
                    }
                elif verification_type == 'contact':
                    # Move to security question
                    pending['verification_type'] = 'security'
                    pending['verification_stage'] = 0
                    return {
                        'status': 'verification',
                        'message': self.verification_questions['security'][0]['question'].format(
                            security_question=user.SecurityQuestion
                        )
                    }
                else:
                    # All verifications passed, proceed with activation
                    try:
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
                                'message': 'Verification successful. Your account has been activated.'
                            }
                    except Exception as e:
                        db.session.rollback()
                        return {'status': 'error', 'message': 'Error activating account'}
            else:
                # Failed verification
                del self.pending_inquiries[user_id]
                return {
                    'status': 'error',
                    'message': 'Verification failed. For security reasons, please contact customer support.'
                }
                
        except Exception as e:
            return {'status': 'error', 'message': 'Error during verification process'}
    
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
    
    def _handle_loan_status(self, user_id: int, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle loan status inquiry
        """
        try:
            loan = db.session.query(Loan).filter(
                Loan.UserID == user_id,
                Loan.ApplicationID == entities['application_id']
            ).first()
            
            if not loan:
                return {'status': 'error', 'message': 'Loan application not found'}
            
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
            db.session.rollback()
            return {'status': 'error', 'message': 'Error processing loan status inquiry'}
    
    def _handle_atm_location(self, user_id: int, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Handle ATM location inquiry
        """
        try:
            atms = db.session.query(ATMLocation).filter(
                ATMLocation.City == entities['location'],
                ATMLocation.IsAccessible == True
            ).limit(5).all()
            
            if not atms:
                return {'status': 'error', 'message': 'No ATMs found in the specified location'}
            
            return {
                'status': 'success',
                'data': [{
                    'branch_name': atm.BranchName,
                    'address': atm.Address,
                    'city': atm.City,
                    'state': atm.State,
                    'zip_code': atm.ZipCode,
                    'operating_hours': atm.OperatingHours,
                    'additional_services': atm.AdditionalServices
                } for atm in atms]
            }
            
        except Exception as e:
            db.session.rollback()
            return {'status': 'error', 'message': 'Error processing ATM location inquiry'}
    
    def _get_required_entities(self, intent: str) -> list:
        """
        Get required entities for each intent
        """
        requirements = {
            'account_activation': ['account_type', 'account_number'],
            'account_deactivation': ['account_type', 'account_number', 'reason_code'],
            'loan_status': ['application_id', 'loan_type'],
            'atm_location': ['location']
        }
        return requirements.get(intent, [])
    

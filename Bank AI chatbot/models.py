from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    UserID = db.Column(db.Integer, primary_key=True)
    FirstName = db.Column(db.String(50), nullable=False)
    LastName = db.Column(db.String(50), nullable=False)
    Email = db.Column(db.String(100), unique=True, nullable=False)
    PhoneNumber = db.Column(db.String(20), unique=True, nullable=False)
    PasswordHash = db.Column(db.String(200), nullable=False)
    DateOfBirth = db.Column(db.Date)
    Address = db.Column(db.String(200))
    IdentificationNumber = db.Column(db.String(50))
    SecurityQuestion = db.Column(db.String(200))
    SecurityAnswer = db.Column(db.String(200))
    AccountStatus = db.Column(db.String(20), default='Active')
    CreatedAt = db.Column(db.DateTime, default=datetime.utcnow)
    UpdatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    accounts = db.relationship('Account', backref='user', lazy=True)
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True)
    loans = db.relationship('Loan', backref='user', lazy=True)

class Account(db.Model):
    __tablename__ = 'accounts'
    
    AccountID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('users.UserID'), nullable=False)
    AccountType = db.Column(db.String(50), nullable=False)
    AccountNumber = db.Column(db.String(20), unique=True, nullable=False)
    Balance = db.Column(db.Numeric(15, 2), default=0.00)
    Currency = db.Column(db.String(3), default='USD')
    Status = db.Column(db.String(20), default='Active')
    CreatedAt = db.Column(db.DateTime, default=datetime.utcnow)
    UpdatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    TransactionID = db.Column(db.Integer, primary_key=True)
    AccountID = db.Column(db.Integer, db.ForeignKey('accounts.AccountID'), nullable=False)
    Type = db.Column(db.String(50), nullable=False)
    Amount = db.Column(db.Numeric(15, 2), nullable=False)
    Description = db.Column(db.String(200))
    Status = db.Column(db.String(20), default='Completed')
    TransactionDate = db.Column(db.DateTime, default=datetime.utcnow)

class Loan(db.Model):
    __tablename__ = 'loans'
    
    LoanID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    UserID = db.Column(db.Integer, db.ForeignKey('users.UserID'), nullable=False)
    ApplicationID = db.Column(db.String(50), unique=True, nullable=False)
    LoanType = db.Column(db.String(50), nullable=False)  # Personal, Home, Auto, Business
    RequestedAmount = db.Column(db.Numeric(15, 2), nullable=False)
    ApprovedAmount = db.Column(db.Numeric(15, 2), nullable=True)
    InterestRate = db.Column(db.Numeric(5, 2), nullable=True)
    TermMonths = db.Column(db.Integer, nullable=True)
    Status = db.Column(db.String(20), default='Pending')
    ApplicationDate = db.Column(db.DateTime, default=datetime.utcnow)
    ApprovalDate = db.Column(db.DateTime, nullable=True)
    Purpose = db.Column(db.String(200), nullable=True)
    CollateralDetails = db.Column(db.String(200), nullable=True)
    CreditScore = db.Column(db.Integer, nullable=True)
    StartDate = db.Column(db.DateTime, nullable=True)
    EndDate = db.Column(db.DateTime, nullable=True)
    RemainingBalance = db.Column(db.Numeric(15, 2), nullable=True)
    

class ATMLocation(db.Model):
    __tablename__ = 'atm_locations'
    
    LocationID = db.Column(db.Integer, primary_key=True)
    BranchName = db.Column(db.String(100), nullable=False)
    Address = db.Column(db.String(200), nullable=False)
    City = db.Column(db.String(100), nullable=False)
    State = db.Column(db.String(50), nullable=False)
    ZipCode = db.Column(db.String(20), nullable=False)
    Latitude = db.Column(db.Numeric(10, 7))
    Longitude = db.Column(db.Numeric(10, 7))
    IsAccessible = db.Column(db.Boolean, default=True)
    
    # Additional ATM details
    WithdrawalLimit = db.Column(db.Numeric(10, 2))
    OperatingHours = db.Column(db.String(100))
    AdditionalServices = db.Column(db.String(200))  # e.g., Deposit, Check Balance
    
    # Geographical categorization
    LocationCategory = db.Column(db.String(50))  # Downtown, City Center, Suburban, etc.

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    
    SessionID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('users.UserID'), nullable=False)
    StartTime = db.Column(db.DateTime, default=datetime.utcnow)
    EndTime = db.Column(db.DateTime)
    Status = db.Column(db.String(20), default='Active')

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    MessageID = db.Column(db.Integer, primary_key=True)
    SessionID = db.Column(db.Integer, db.ForeignKey('chat_sessions.SessionID'), nullable=False)
    Message = db.Column(db.Text, nullable=False)
    Response = db.Column(db.Text, nullable=False)
    Intent = db.Column(db.String(100))
    Timestamp = db.Column(db.DateTime, default=datetime.utcnow)
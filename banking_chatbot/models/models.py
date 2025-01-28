from config import db
from datetime import datetime
from decimal import Decimal

class User(db.Model):
    __tablename__ = 'Users'
    __table_args__ = {'extend_existing': True}
    UserID = db.Column(db.Integer, primary_key=True)
    FirstName = db.Column(db.String(50))
    LastName = db.Column(db.String(50))
    Email = db.Column(db.String(100), unique=True, nullable=False)
    PhoneNumber = db.Column(db.String(15), unique=True, nullable=False)
    PasswordHash = db.Column(db.String(255), nullable=False)  # Added password hash field
    DateOfBirth = db.Column(db.Date)
    Address = db.Column(db.Text)
    IdentificationNumber = db.Column(db.String(50))
    SecurityQuestion = db.Column(db.String(255))
    SecurityAnswer = db.Column(db.String(255))
    AccountStatus = db.Column(db.Enum('Active', 'Inactive'), default='Active')
    CreatedAt = db.Column(db.TIMESTAMP, default=datetime.utcnow)

class Account(db.Model):
    __tablename__ = 'Accounts'
    __table_args__ = {'extend_existing': True}
    AccountID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('Users.UserID'), nullable=False)
    AccountNumber = db.Column(db.String(20), unique=True, nullable=False)
    AccountType = db.Column(db.Enum('Savings', 'Current', 'Fixed Deposit'))
    Balance = db.Column(db.DECIMAL(15, 2), default=0.00)
    CreatedAt = db.Column(db.TIMESTAMP, default=datetime.utcnow)

class Transaction(db.Model):
    __tablename__ = 'Transactions'
    __table_args__ = {'extend_existing': True}
    TransactionID = db.Column(db.Integer, primary_key=True)
    AccountID = db.Column(db.Integer, db.ForeignKey('Accounts.AccountID'), nullable=False)
    TransactionType = db.Column(db.Enum('Deposit', 'Withdrawal', 'Transfer', 'Bill Payment'))
    Amount = db.Column(db.DECIMAL(15, 2))
    TransactionDate = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    Description = db.Column(db.Text)

class CustomerSupport(db.Model):
    __tablename__ = 'CustomerSupport'
    __table_args__ = {'extend_existing': True}
    SupportID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('Users.UserID'))
    RequestType = db.Column(db.Enum('FAQ', 'Card Blocking', 'Account Activation', 'Account Deactivation'))
    RequestDetails = db.Column(db.Text)
    Status = db.Column(db.Enum('Pending', 'Resolved'), default='Pending')
    CreatedAt = db.Column(db.TIMESTAMP, default=datetime.utcnow)

class Loan(db.Model):
    __tablename__ = 'Loans'
    __table_args__ = {'extend_existing': True}
    LoanID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('Users.UserID'), nullable=False)
    LoanType = db.Column(db.Enum('Home', 'Personal', 'Auto', 'Education'))
    LoanAmount = db.Column(db.DECIMAL(15, 2))
    InterestRate = db.Column(db.DECIMAL(5, 2))
    LoanStatus = db.Column(db.Enum('Pending', 'Approved', 'Rejected'), default='Pending')
    AppliedAt = db.Column(db.TIMESTAMP, default=datetime.utcnow)

class BranchAndATM(db.Model):
    __tablename__ = 'BranchesAndATMs'
    __table_args__ = {'extend_existing': True}
    LocationID = db.Column(db.Integer, primary_key=True)
    LocationName = db.Column(db.String(100))
    Address = db.Column(db.Text)
    Type = db.Column(db.Enum('Branch', 'ATM'))
    Latitude = db.Column(db.DECIMAL(10, 8))
    Longitude = db.Column(db.DECIMAL(11, 8))

class Investment(db.Model):
    __tablename__ = 'Investments'
    __table_args__ = {'extend_existing': True}
    InvestmentID = db.Column(db.Integer, primary_key=True)
    UserID = db.Column(db.Integer, db.ForeignKey('Users.UserID'), nullable=False)
    InvestmentType = db.Column(db.Enum('Mutual Fund', 'Fixed Deposit'))
    InvestmentAmount = db.Column(db.DECIMAL(15, 2))
    InterestRate = db.Column(db.DECIMAL(5, 2))
    StartDate = db.Column(db.Date)
    MaturityDate = db.Column(db.Date)
    Status = db.Column(db.Enum('Active', 'Matured'), default='Active')
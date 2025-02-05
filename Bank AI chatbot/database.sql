-- Users Table
CREATE TABLE Users (
    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    Email VARCHAR(100) UNIQUE NOT NULL,
    PhoneNumber VARCHAR(20) NOT NULL,
    PasswordHash VARCHAR(255) NOT NULL,
    DateOfBirth DATE NOT NULL,
    Address TEXT NOT NULL,
    IdentificationNumber VARCHAR(50) UNIQUE NOT NULL,
    SecurityQuestion TEXT NOT NULL,
    SecurityAnswer VARCHAR(255) NOT NULL,
    AccountStatus VARCHAR(20) DEFAULT 'Active',
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Accounts Table
CREATE TABLE Accounts (
    AccountID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID INTEGER NOT NULL,
    AccountType VARCHAR(50) NOT NULL,
    AccountNumber VARCHAR(50) UNIQUE NOT NULL,
    Balance DECIMAL(15, 2) DEFAULT 0.00,
    Status VARCHAR(20) DEFAULT 'Active',
    OpenedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    LastTransactionDate TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

-- Transactions Table
CREATE TABLE Transactions (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    AccountID INTEGER NOT NULL,
    TransactionType VARCHAR(50) NOT NULL,
    Amount DECIMAL(15, 2) NOT NULL,
    Description TEXT,
    TransactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Balance DECIMAL(15, 2) NOT NULL,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID)
);

-- Loans Table
CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID INTEGER NOT NULL,
    LoanType VARCHAR(50) NOT NULL,
    LoanAmount DECIMAL(15, 2) NOT NULL,
    InterestRate DECIMAL(5, 2) NOT NULL,
    LoanTerm INTEGER NOT NULL,
    ApplicationDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ApprovalStatus VARCHAR(20) DEFAULT 'Pending',
    StartDate TIMESTAMP,
    EndDate TIMESTAMP,
    RemainingBalance DECIMAL(15, 2),
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

-- ATM Locations Table
CREATE TABLE ATMLocations (
    ATMID INTEGER PRIMARY KEY AUTOINCREMENT,
    LocationName VARCHAR(100) NOT NULL,
    Address TEXT NOT NULL,
    City VARCHAR(50) NOT NULL,
    State VARCHAR(50) NOT NULL,
    PostalCode VARCHAR(20) NOT NULL,
    Latitude DECIMAL(10, 8),
    Longitude DECIMAL(11, 8),
    Status VARCHAR(20) DEFAULT 'Active'
);

-- Chat Sessions Table
CREATE TABLE ChatSessions (
    SessionID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID INTEGER NOT NULL,
    StartTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    EndTime TIMESTAMP,
    Intent VARCHAR(50),
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

-- Chat Messages Table
CREATE TABLE ChatMessages (
    MessageID INTEGER PRIMARY KEY AUTOINCREMENT,
    SessionID INTEGER NOT NULL,
    MessageText TEXT NOT NULL,
    SentBy VARCHAR(20) NOT NULL,
    Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (SessionID) REFERENCES ChatSessions(SessionID)
);



-- Insert Sample Accounts
INSERT INTO Accounts (
    UserID, AccountType, AccountNumber, Balance, Status
) VALUES 
(1, 'Checking', '1234567890', 5000.00, 'Active'),
(1, 'Savings', '0987654321', 10000.00, 'Active');

-- Insert Sample Loans
INSERT INTO Loans (
    UserID, LoanType, LoanAmount, InterestRate, LoanTerm, 
    ApprovalStatus, StartDate, EndDate, RemainingBalance
) VALUES 
(1, 'Personal', 20000.00, 5.5, 36, 'Approved', 
 '2024-02-01', '2027-02-01', 20000.00),
(1, 'Home', 250000.00, 4.25, 360, 'Pending', 
 NULL, NULL, NULL);

-- Insert Sample ATM Locations
INSERT INTO ATMLocations (
    LocationName, Address, City, State, PostalCode, 
    Latitude, Longitude
) VALUES 
('Downtown Branch', '100 Financial District', 'New York', 'NY', '10001', 
 40.7128, -74.0060),
('City Center', '50 Main Street', 'New York', 'NY', '10002', 
 40.7282, -73.9942),
('Community Mall', '200 Shopping Avenue', 'New York', 'NY', '10003', 
 40.7489, -73.9680);

-- Insert Sample Transaction
INSERT INTO Transactions (
    AccountID, TransactionType, Amount, Description, Balance
) VALUES 
(1, 'Deposit', 1000.00, 'Initial Deposit', 1000.00),
(1, 'Transfer', 500.00, 'Transfer to Savings', 500.00),
(2, 'Deposit', 500.00, 'Transfer from Checking', 500.00);
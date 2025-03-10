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

INSERT INTO Accounts (UserID, AccountType, AccountNumber, Balance)
VALUES
(2, 'Checking', 'ACC-2025-201', 5000.00),
(2, 'Savings', 'ACC-2025-202', 15000.00),
(2, 'Investment', 'ACC-2025-203', 25000.00);

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

CREATE TABLE Loans (
    LoanID INTEGER PRIMARY KEY AUTO_INCREMENT,
    UserID INTEGER NOT NULL,
    ApplicationID VARCHAR(50) UNIQUE NOT NULL,
    LoanType VARCHAR(50) NOT NULL,
    RequestedAmount DECIMAL(15, 2) NOT NULL,
    ApprovedAmount DECIMAL(15, 2),
    InterestRate DECIMAL(5, 2),
    TermMonths INTEGER,
    Status VARCHAR(20) DEFAULT 'Pending',
    ApplicationDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ApprovalDate TIMESTAMP NULL DEFAULT NULL,
    Purpose VARCHAR(200),
    CollateralDetails VARCHAR(200),
    CreditScore INTEGER,
    StartDate TIMESTAMP NULL DEFAULT NULL,
    EndDate TIMESTAMP NULL DEFAULT NULL,
    RemainingBalance DECIMAL(15, 2),
    FOREIGN KEY (UserID) REFERENCES Users(UserID)
);

INSERT INTO Loans (
    UserID, ApplicationID, LoanType, RequestedAmount, ApprovedAmount, InterestRate, TermMonths, 
    Status, ApplicationDate, ApprovalDate, Purpose, CollateralDetails, CreditScore, StartDate, EndDate, RemainingBalance
)
VALUES 
(1, 'APP-2025-101', 'Personal', 12000.00, 11500.00, 6.50, 48, 'Approved', '2025-03-10 09:00:00', '2025-03-15 14:00:00', 'Medical expenses', NULL, 730, '2025-04-01 00:00:00', '2029-04-01 00:00:00', 11500.00),
(1, 'APP-2025-102', 'Auto', 22000.00, 21000.00, 4.90, 60, 'Approved', '2025-03-11 10:15:00', '2025-03-16 11:30:00', 'Car repair and upgrade', 'Vehicle: 2019 Ford Focus', 690, '2025-04-05 00:00:00', '2030-04-05 00:00:00', 21000.00),
(2, 'APP-2025-103', 'Home', 350000.00, 340000.00, 3.85, 360, 'Approved', '2025-03-12 14:30:00', '2025-03-20 12:45:00', 'Refinancing home', 'Property: 456 Oak St, Hometown', 780, '2025-04-10 00:00:00', '2055-04-10 00:00:00', 340000.00),
(2, 'APP-2025-104', 'Business', 90000.00, NULL, NULL, NULL, 'Pending', '2025-03-13 11:00:00', NULL, 'Expansion of operations', 'Equipment and inventory', 710, NULL, NULL, NULL),
(1, 'APP-2025-105', 'Personal', 18000.00, 17000.00, 5.95, 60, 'Approved', '2025-03-14 16:45:00', '2025-03-19 10:20:00', 'Vacation loan', NULL, 705, '2025-04-15 00:00:00', '2029-04-15 00:00:00', 17000.00),
(2, 'APP-2025-106', 'Auto', 27000.00, NULL, NULL, NULL, 'Rejected', '2025-03-15 09:20:00', '2025-03-20 15:30:00', 'Upgrade to electric vehicle', 'Vehicle: 2023 Nissan Leaf', 650, NULL, NULL, NULL);


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
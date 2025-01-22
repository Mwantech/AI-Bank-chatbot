CREATE TABLE Users (
    UserID INT AUTO_INCREMENT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100) UNIQUE NOT NULL,
    PhoneNumber VARCHAR(15) UNIQUE NOT NULL,
    DateOfBirth DATE,
    Address TEXT,
    IdentificationNumber VARCHAR(50),
    SecurityQuestion VARCHAR(255),
    SecurityAnswer VARCHAR(255),
    AccountStatus ENUM('Active', 'Inactive') DEFAULT 'Active',
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE Users 
ADD COLUMN PasswordHash VARCHAR(255) NOT NULL;

CREATE TABLE Accounts (
    AccountID INT AUTO_INCREMENT PRIMARY KEY,
    UserID INT NOT NULL,
    AccountNumber VARCHAR(20) UNIQUE NOT NULL,
    AccountType ENUM('Savings', 'Current', 'Fixed Deposit'),
    Balance DECIMAL(15, 2) DEFAULT 0.00,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID) ON DELETE CASCADE
);
CREATE TABLE Transactions (
    TransactionID INT AUTO_INCREMENT PRIMARY KEY,
    AccountID INT NOT NULL,
    TransactionType ENUM('Deposit', 'Withdrawal', 'Transfer', 'Bill Payment'),
    Amount DECIMAL(15, 2),
    TransactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Description TEXT,
    FOREIGN KEY (AccountID) REFERENCES Accounts(AccountID) ON DELETE CASCADE
);
CREATE TABLE CustomerSupport (
    SupportID INT AUTO_INCREMENT PRIMARY KEY,
    UserID INT,
    RequestType ENUM('FAQ', 'Card Blocking', 'Account Activation', 'Account Deactivation'),
    RequestDetails TEXT,
    Status ENUM('Pending', 'Resolved') DEFAULT 'Pending',
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID) ON DELETE CASCADE
);
CREATE TABLE Loans (
    LoanID INT AUTO_INCREMENT PRIMARY KEY,
    UserID INT NOT NULL,
    LoanType ENUM('Home', 'Personal', 'Auto', 'Education'),
    LoanAmount DECIMAL(15, 2),
    InterestRate DECIMAL(5, 2),
    LoanStatus ENUM('Pending', 'Approved', 'Rejected') DEFAULT 'Pending',
    AppliedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID) ON DELETE CASCADE
);
CREATE TABLE BranchesAndATMs (
    LocationID INT AUTO_INCREMENT PRIMARY KEY,
    LocationName VARCHAR(100),
    Address TEXT,
    Type ENUM('Branch', 'ATM'),
    Latitude DECIMAL(10, 8),
    Longitude DECIMAL(11, 8)
);
CREATE TABLE Investments (
    InvestmentID INT AUTO_INCREMENT PRIMARY KEY,
    UserID INT NOT NULL,
    InvestmentType ENUM('Mutual Fund', 'Fixed Deposit'),
    InvestmentAmount DECIMAL(15, 2),
    InterestRate DECIMAL(5, 2),
    StartDate DATE,
    MaturityDate DATE,
    Status ENUM('Active', 'Matured') DEFAULT 'Active',
    FOREIGN KEY (UserID) REFERENCES Users(UserID) ON DELETE CASCADE
);

INSERT INTO Users (FirstName, LastName, Email, PhoneNumber, DateOfBirth, Address, IdentificationNumber, SecurityQuestion, SecurityAnswer, AccountStatus) 
VALUES
('John', 'Doe', 'john.doe@example.com', '1234567890', '1990-01-01', '123 Main Street, Cityville', 'ID123456', 'What is your mother\'s maiden name?', 'Smith', 'Active'),
('Jane', 'Smith', 'jane.smith@example.com', '0987654321', '1985-05-15', '456 Elm Street, Townville', 'ID987654', 'What was the name of your first pet?', 'Buddy', 'Active');
INSERT INTO Accounts (UserID, AccountNumber, AccountType, Balance) 
VALUES
(1, 'ACC123456789', 'Savings', 5000.00),
(2, 'ACC987654321', 'Current', 12000.00);

INSERT INTO Transactions (AccountID, TransactionType, Amount, Description) 
VALUES
(1, 'Deposit', 1000.00, 'Salary deposit'),
(1, 'Withdrawal', 200.00, 'ATM withdrawal'),
(2, 'Transfer', 500.00, 'Transfer to John Doe'),
(2, 'Bill Payment', 100.00, 'Electricity bill payment');

INSERT INTO CustomerSupport (UserID, RequestType, RequestDetails) 
VALUES
(1, 'Card Blocking', 'Requested to block a lost card'),
(2, 'FAQ', 'How to apply for a loan?');
INSERT INTO Loans (UserID, LoanType, LoanAmount, InterestRate, LoanStatus) 
VALUES
(1, 'Home', 200000.00, 6.5, 'Pending'),
(2, 'Auto', 15000.00, 7.0, 'Approved');

INSERT INTO BranchesAndATMs (LocationName, Address, Type, Latitude, Longitude) 
VALUES
('City Center Branch', '789 Pine Street, Cityville', 'Branch', 40.712776, -74.005974),
('Downtown ATM', '101 Maple Avenue, Townville', 'ATM', 34.052235, -118.243683);

INSERT INTO Investments (UserID, InvestmentType, InvestmentAmount, InterestRate, StartDate, MaturityDate) 
VALUES
(1, 'Mutual Fund', 10000.00, 8.5, '2024-01-01', '2025-01-01'),
(2, 'Fixed Deposit', 5000.00, 6.0, '2023-06-01', '2024-06-01');

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

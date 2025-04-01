import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import styles from './Dashboard.module.css';
import api from '../api/api';

const Dashboard = ({ user }) => {
  const [balance, setBalance] = useState(0);
  const [transactions, setTransactions] = useState([]);
  const [amount, setAmount] = useState('');
  const [description, setDescription] = useState('');
  const [receiptData, setReceiptData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('authToken');
        
        if (!token) {
          navigate('/login');
          return;
        }

        // Fetch account data
        const accountResponse = await api.get('/accounts', {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        if (accountResponse.data.accounts && accountResponse.data.accounts.length > 0) {
          const primaryAccount = accountResponse.data.accounts[0];
          setBalance(parseFloat(primaryAccount.Balance));
          
          // Fetch transactions for the primary account
          const transactionsResponse = await api.get(`/accounts/${primaryAccount.AccountID}/transactions`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          
          // Map backend transactions to the format expected by the component
          const formattedTransactions = transactionsResponse.data.transactions.map(t => ({
            id: t.TransactionID,
            type: t.Type.toLowerCase(),
            amount: parseFloat(t.Amount),
            date: new Date(t.TransactionDate).toISOString().split('T')[0],
            description: t.Description
          }));
          
          setTransactions(formattedTransactions);
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load dashboard data');
        setLoading(false);
        
        if (err.response && err.response.status === 401) {
          handleLogout();
        }
      }
    };

    fetchUserData();
  }, [navigate]);

  const handleLogout = () => {
    try {
      const token = localStorage.getItem('authToken');
      
      // Call the logout endpoint
      api.post('/auth/logout', {}, {
        headers: { Authorization: `Bearer ${token}` }
      }).finally(() => {
        // Clear token and user data, then redirect
        localStorage.removeItem('authToken');
        localStorage.removeItem('user');
        navigate('/login');
      });
    } catch (err) {
      console.error('Logout error:', err);
      // Still remove token and redirect on error
      localStorage.removeItem('authToken');
      localStorage.removeItem('user');
      navigate('/login');
    }
  };

// Transaction-related functions with better error handling
const handleDeposit = async () => {
  if (!amount || isNaN(amount) || Number(amount) <= 0) {
    alert('Please enter a valid amount');
    return;
  }

  try {
    setLoading(true);
    const depositAmount = Number(amount);
    
    // Get the primary account ID
    const accountResponse = await api.get('/accounts');
    
    if (!accountResponse.data.accounts || accountResponse.data.accounts.length === 0) {
      alert('No account found');
      return;
    }
    
    const accountId = accountResponse.data.accounts[0].AccountID;
    
    console.log('Making deposit request:', {
      type: 'deposit',
      amount: depositAmount,
      description: description || 'Deposit'
    });
    
    // Create a new transaction
    const response = await api.post(`/accounts/${accountId}/transactions`, {
      type: 'deposit',
      amount: depositAmount,
      description: description || 'Deposit'
    });

    console.log('Deposit response:', response.data);

    // Update the UI with the new transaction
    const newTransaction = {
      id: response.data.transaction.TransactionID,
      type: 'deposit',
      amount: depositAmount,
      date: new Date().toISOString().split('T')[0],
      description: description || 'Deposit'
    };

    setTransactions([newTransaction, ...transactions]);
    // Use the returned balance from the server if available
    if (response.data.newBalance) {
      setBalance(parseFloat(response.data.newBalance));
    } else {
      setBalance(prevBalance => prevBalance + depositAmount);
    }
    
    setAmount('');
    setDescription('');
    
  } catch (err) {
    console.error('Deposit error:', err);
    if (err.response && err.response.data && err.response.data.error) {
      alert(`Failed to process deposit: ${err.response.data.error}`);
    } else {
      alert('Failed to process deposit. Please try again later.');
    }
    
    if (err.response && err.response.status === 401) {
      handleLogout();
    }
  } finally {
    setLoading(false);
  }
};

const handleWithdrawal = async () => {
  if (!amount || isNaN(amount) || Number(amount) <= 0) {
    alert('Please enter a valid amount');
    return;
  }

  const withdrawalAmount = Number(amount);
  if (withdrawalAmount > balance) {
    alert('Insufficient funds');
    return;
  }

  try {
    setLoading(true);
    
    // Get the primary account ID
    const accountResponse = await api.get('/accounts');
    
    if (!accountResponse.data.accounts || accountResponse.data.accounts.length === 0) {
      alert('No account found');
      return;
    }
    
    const accountId = accountResponse.data.accounts[0].AccountID;
    
    console.log('Making withdrawal request:', {
      type: 'withdrawal',
      amount: withdrawalAmount,
      description: description || 'Withdrawal'
    });
    
    // Create a new transaction
    const response = await api.post(`/accounts/${accountId}/transactions`, {
      type: 'withdrawal',
      amount: withdrawalAmount,
      description: description || 'Withdrawal'
    });

    console.log('Withdrawal response:', response.data);

    // Update the UI with the new transaction
    const newTransaction = {
      id: response.data.transaction.TransactionID,
      type: 'withdrawal',
      amount: withdrawalAmount,
      date: new Date().toISOString().split('T')[0],
      description: description || 'Withdrawal'
    };

    setTransactions([newTransaction, ...transactions]);
    // Use the returned balance from the server if available
    if (response.data.newBalance) {
      setBalance(parseFloat(response.data.newBalance));
    } else {
      setBalance(prevBalance => prevBalance - withdrawalAmount);
    }
    
    setAmount('');
    setDescription('');
    
  } catch (err) {
    console.error('Withdrawal error:', err);
    if (err.response && err.response.data && err.response.data.error) {
      alert(`Failed to process withdrawal: ${err.response.data.error}`);
    } else {
      alert('Failed to process withdrawal. Please try again later.');
    }
    
    if (err.response && err.response.status === 401) {
      handleLogout();
    }
  } finally {
    setLoading(false);
  }
};

  const printReceipt = (transaction) => {
    setReceiptData(transaction);
  };

  const closeReceipt = () => {
    setReceiptData(null);
  };

  if (loading) {
    return <div className={styles.loading}>Loading dashboard...</div>;
  }

  if (error) {
    return <div className={styles.error}>{error}</div>;
  }

  return (
    <div className={styles.dashboardContainer}>
      <header className={styles.header}>
        <h1>MyBank Dashboard</h1>
        <div className={styles.userInfo}>
          <span>Welcome, {user ? `${user.firstName} ${user.lastName}` : 'User'}</span>
          <div className={styles.userActions}>
            <Link to="/chat" className={styles.chatLink}>Chat Support</Link>
            <button onClick={handleLogout} className={styles.logoutButton}>Logout</button>
          </div>
        </div>
      </header>

      <div className={styles.dashboardContent}>
        <div className={styles.balanceCard}>
          <h2>Current Balance</h2>
          <div className={styles.balanceAmount}>${balance.toFixed(2)}</div>
        </div>

        <div className={styles.transactionForm}>
          <h2>Perform Transaction</h2>
          <div className={styles.formGroup}>
            <label htmlFor="amount">Amount ($)</label>
            <input
              type="number"
              id="amount"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className={styles.input}
              min="0"
              step="0.01"
            />
          </div>
          <div className={styles.formGroup}>
            <label htmlFor="description">Description</label>
            <input
              type="text"
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className={styles.input}
            />
          </div>
          <div className={styles.buttonGroup}>
            <button onClick={handleDeposit} className={styles.depositButton}>Deposit</button>
            <button onClick={handleWithdrawal} className={styles.withdrawButton}>Withdraw</button>
          </div>
        </div>

        <div className={styles.transactionHistory}>
          <h2>Transaction History</h2>
          {transactions.length === 0 ? (
            <p className={styles.noTransactions}>No transactions found</p>
          ) : (
            <div className={styles.transactionList}>
              {transactions.map(transaction => (
                <div key={transaction.id} className={`${styles.transactionItem} ${transaction.type === 'deposit' ? styles.deposit : styles.withdrawal}`}>
                  <div className={styles.transactionDetails}>
                    <div className={styles.transactionType}>{transaction.type === 'deposit' ? 'Deposit' : 'Withdrawal'}</div>
                    <div className={styles.transactionDesc}>{transaction.description}</div>
                    <div className={styles.transactionDate}>{transaction.date}</div>
                  </div>
                  <div className={styles.transactionAmount}>
                    ${transaction.amount.toFixed(2)}
                  </div>
                  <button 
                    onClick={() => printReceipt(transaction)} 
                    className={styles.receiptButton}
                  >
                    Print Receipt
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {receiptData && (
        <div className={styles.receiptOverlay}>
          <div className={styles.receiptModal}>
            <h3>Transaction Receipt</h3>
            <div className={styles.receiptContent}>
              <p><strong>Transaction ID:</strong> {receiptData.id}</p>
              <p><strong>Date:</strong> {receiptData.date}</p>
              <p><strong>Type:</strong> {receiptData.type === 'deposit' ? 'Deposit' : 'Withdrawal'}</p>
              <p><strong>Amount:</strong> ${receiptData.amount.toFixed(2)}</p>
              <p><strong>Description:</strong> {receiptData.description}</p>
              <p><strong>Balance After Transaction:</strong> ${balance.toFixed(2)}</p>
            </div>
            <div className={styles.receiptActions}>
              <button onClick={() => window.print()} className={styles.printButton}>Print</button>
              <button onClick={closeReceipt} className={styles.closeButton}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
// Login.jsx 
import React, { useState } from "react";
import axios from "axios";
import styles from '../styles.module.css';

const Login = ({ onLoginSuccess }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const response = await axios.post("http://localhost:5000/api/auth/login", {
        email,
        password,
      });

      if (response.status === 200) {
        const { token, user } = response.data;
        
        // Store token and user info in localStorage
        localStorage.setItem('authToken', token);
        localStorage.setItem('userId', user.id);
        localStorage.setItem('userEmail', user.email);
        
        // Configure axios defaults for future requests
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        
        // Pass the user data to parent component
        onLoginSuccess({
          token,
          userId: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName
        });
      }
    } catch (err) {
      if (err.response?.status === 401) {
        setError("Invalid email or password. Please try again.");
      } else if (err.response?.status === 500) {
        setError("Server error. Please try again later.");
      } else {
        setError("An unexpected error occurred. Please check your connection.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles['login-container']}>
      <div className={styles['login-card']}>
        <div className={styles['login-header']}>
          <h2>Welcome Back</h2>
          <p>Please login to your account</p>
        </div>
        
        <form onSubmit={handleSubmit} className={styles['login-form']}>
          <div className={styles['form-group']}>
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="Enter your email"
            />
          </div>

          <div className={styles['form-group']}>
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="Enter your password"
            />
          </div>

          {error && <div className={styles['error-message']}>{error}</div>}

          <button type="submit" className={styles['login-button']} disabled={isLoading}>
            {isLoading ? (
              <div className={styles['loading-spinner']}></div>
            ) : (
              "Login"
            )}
          </button>
        </form>
      </div>
    </div>
  );
};



export default Login;
import React, { useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import styles from "./Auth.module.css";

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

      if (response.data?.token && response.data?.user) {
        const { token, user } = response.data;
        
        // Store token in localStorage
        localStorage.setItem('authToken', `Bearer ${token}`);
        
        // Configure axios defaults for future requests
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        
        // Pass complete user data to parent
        onLoginSuccess({
          token,
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName
        });
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err) {
      if (err.response?.status === 401) {
        setError("Invalid email or password");
      } else if (err.response?.status === 403) {
        setError("Account is inactive");
      } else if (err.response?.status === 500) {
        setError("Server error");
      } else {
        setError(err.message || "Login failed");
      }
      console.error('Login error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.authContainer}>
      <div className={styles.authCard}>
        <div className={styles.authHeader}>
          <h2>Welcome Back</h2>
          <p>Please login to your account</p>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className={styles.formGroup}>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="Enter your email"
              disabled={isLoading}
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="Enter your password"
              disabled={isLoading}
            />
          </div>

          {error && (
            <div className={styles.errorMessage}>
              {error}
            </div>
          )}

          <button 
            type="submit" 
            disabled={isLoading || !email || !password}
            className={styles.submitButton}
          >
            {isLoading ? (
              <div className={styles.spinner}>
                <div></div>
                <div></div>
                <div></div>
                <div></div>
              </div>
            ) : (
              "Login"
            )}
          </button>
          
          <div className={styles.authFooter}>
            <p>Don't have an account? <Link to="/signup" className={styles.link}>Sign up</Link></p>
          </div>
        </form>
      </div>
      
      <div className={styles.absaBranding}>
        <div className={styles.logo}>ABSA</div>
        <h3>Banking Assistant</h3>
        <p>Your personal financial guide</p>
      </div>
    </div>
  );
};

export default Login;
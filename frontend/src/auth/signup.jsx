import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import styles from "./Auth.module.css";

const Signup = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phoneNumber: "",
    password: "",
    confirmPassword: "",
    dateOfBirth: "",
    address: "",
    identificationNumber: "",
    securityQuestion: "",
    securityAnswer: ""
  });
  
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    
    // Basic validation
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    
    setIsLoading(true);

    try {
      const response = await axios.post("http://localhost:5000/api/auth/register", {
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        phoneNumber: formData.phoneNumber,
        password: formData.password,
        dateOfBirth: formData.dateOfBirth || "2000-01-01",
        address: formData.address,
        identificationNumber: formData.identificationNumber,
        securityQuestion: formData.securityQuestion,
        securityAnswer: formData.securityAnswer
      });

      if (response.data?.message === "User registered successfully") {
        // Registration successful - store token if provided
        if (response.data.token) {
          localStorage.setItem('authToken', `Bearer ${response.data.token}`);
          axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.token}`;
        }
        
        // Navigate to login or dashboard
        navigate("/login", { state: { message: "Registration successful! Please login." } });
      }
    } catch (err) {
      if (err.response?.status === 409) {
        setError("Email or phone number already registered");
      } else if (err.response?.status === 400) {
        setError("Missing required fields");
      } else if (err.response?.status === 500) {
        setError("Server error");
      } else {
        setError(err.message || "Registration failed");
      }
      console.error('Registration error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.authContainer}>
      <div className={`${styles.authCard} ${styles.signupCard}`}>
        <div className={styles.authHeader}>
          <h2>Create an Account</h2>
          <p>Join ABSA Banking Assistant</p>
        </div>
        
        <form onSubmit={handleSubmit}>
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="firstName">First Name</label>
              <input
                id="firstName"
                name="firstName"
                type="text"
                value={formData.firstName}
                onChange={handleChange}
                required
                placeholder="Enter your first name"
                disabled={isLoading}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="lastName">Last Name</label>
              <input
                id="lastName"
                name="lastName"
                type="text"
                value={formData.lastName}
                onChange={handleChange}
                required
                placeholder="Enter your last name"
                disabled={isLoading}
              />
            </div>
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="email">Email</label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="Enter your email"
                disabled={isLoading}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="phoneNumber">Phone Number</label>
              <input
                id="phoneNumber"
                name="phoneNumber"
                type="tel"
                value={formData.phoneNumber}
                onChange={handleChange}
                required
                placeholder="Enter your phone number"
                disabled={isLoading}
              />
            </div>
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="password">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Create a password"
                disabled={isLoading}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                placeholder="Confirm your password"
                disabled={isLoading}
              />
            </div>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="dateOfBirth">Date of Birth</label>
            <input
              id="dateOfBirth"
              name="dateOfBirth"
              type="date"
              value={formData.dateOfBirth}
              onChange={handleChange}
              disabled={isLoading}
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="address">Address</label>
            <input
              id="address"
              name="address"
              type="text"
              value={formData.address}
              onChange={handleChange}
              placeholder="Enter your address"
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
            disabled={isLoading}
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
              "Sign Up"
            )}
          </button>
          
          <div className={styles.authFooter}>
            <p>Already have an account? <Link to="/login" className={styles.link}>Login</Link></p>
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

export default Signup;
// LandingPage.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import styles from './LandingPage.module.css';
import bankingImage from '../assets/react.svg'; // You'll need to add this image to your assets folder

const LandingPage = () => {
  return (
    <div className={styles.landingPage}>
      <header className={styles.header}>
        <div className={styles.logo}>
          <h2>ABSA</h2>
        </div>
        <nav className={styles.nav}>
          <ul>
            <li><a href="#features">Features</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#contact">Contact</a></li>
            <li><Link to="/login" className={styles.loginBtn}>Login</Link></li>
          </ul>
        </nav>
      </header>

      <main className={styles.main}>
        <div className={styles.heroSection}>
          <div className={styles.heroContent}>
            <h1>Banking Made Simple with ABSA Chatbot</h1>
            <p>Get instant answers, manage your finances, and access banking services 24/7 with our intelligent banking assistant.</p>
            <Link to="/signup" className={styles.getStartedBtn}>Get Started</Link>
          </div>
          <div className={styles.heroImage}>
            {/* You can replace this with an actual image */}
            <div className={styles.imagePlaceholder}>
              <div className={styles.chatbotPreview}>
                <div className={styles.chatHeader}>ABSA Assistant</div>
                <div className={styles.chatMessages}>
                  <div className={styles.botMessage}>Hello! How can I help with your banking today?</div>
                  <div className={styles.userMessage}>I'd like to check my account balance.</div>
                  <div className={styles.botMessage}>I'd be happy to help you check your balance. Please log in to continue.</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <section id="features" className={styles.featuresSection}>
          <h2>Why Choose ABSA Chatbot?</h2>
          <div className={styles.featureCards}>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>🔒</div>
              <h3>Secure Banking</h3>
              <p>Bank with confidence knowing your data is protected with enterprise-grade security.</p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>⚡</div>
              <h3>Instant Support</h3>
              <p>Get immediate answers to your banking questions anytime, anywhere.</p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>💼</div>
              <h3>Smart Transactions</h3>
              <p>Transfer funds, pay bills, and manage your finances with simple conversations.</p>
            </div>
          </div>
        </section>
      </main>

      <footer className={styles.footer}>
        <p>&copy; 2025 ABSA Bank. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default LandingPage;
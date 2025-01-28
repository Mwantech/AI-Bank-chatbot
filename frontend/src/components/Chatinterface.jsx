
// ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import styles from '../styles.module.css';

const ChatInterface = ({ user }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Set up axios interceptor for authentication
    const interceptor = axios.interceptors.request.use(
      config => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
      },
      error => {
        return Promise.reject(error);
      }
    );

    return () => {
      // Remove interceptor on cleanup
      axios.interceptors.request.eject(interceptor);
    };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setError('');
    setIsLoading(true);

    // Add user message to chat
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);

    try {
      const response = await axios.post('http://localhost:5000/api/chatbot/process', {
        input: userMessage
      });

      if (response.data) {
        setMessages(prev => [...prev, {
          type: 'bot',
          content: response.data.response,
          context: response.data.context
        }]);
      }
    } catch (err) {
      console.error('Chat error:', err);
      
      if (err.response?.status === 401) {
        setError('Session expired. Please login again.');
        // Redirect to login or handle re-authentication
        window.location.href = '/login';
      } else {
        setError('Failed to get response. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles['chat-container']}>
      <div className={styles['chat-header']}>
        <h3>Banking Assistant</h3>
        <p>Welcome, {user.firstName}</p>
      </div>

      <div className={styles['messages-container']}>
        {messages.map((message, index) => (
          <div
            key={index}
            className={`${styles['message']} ${styles[message.type === 'user' ? 'user-message' : 'bot-message']}`}
          >
            <div className={styles['message-content']}>
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className={styles['bot-message']}>
            <div className={styles['typing-indicator']}>...</div>
          </div>
        )}
        {error && (
          <div className={styles['error-message']}>
            {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className={styles['chat-input-form']}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          className={styles['chat-input']}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className={styles['send-button']}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
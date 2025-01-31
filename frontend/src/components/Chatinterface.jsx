import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import axios from 'axios';
import styles from '../styles.module.css';

const ChatInterface = ({ user }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const messagesEndRef = useRef(null);
  const messageReceived = useRef(false);

  // Initialize WebSocket connection
  useEffect(() => {
    const initializeSocket = () => {
      const storedToken = localStorage.getItem('authToken');
      if (!storedToken) {
        setError('Authentication token not found');
        return null;
      }

      const token = storedToken.replace('Bearer ', '');

      const newSocket = io('http://localhost:5000', {
        auth: { token },
        query: { token },
        transportOptions: {
          polling: {
            extraHeaders: {
              'Authorization': `Bearer ${token}`
            }
          }
        },
        transports: ['websocket'],
        withCredentials: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError('');
        setIsReconnecting(false);
      });

      newSocket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        setIsConnected(false);
        if (error.message.includes('auth')) {
          setError('Authentication failed. Please log in again.');
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        } else {
          setError('Failed to connect to server. Retrying...');
        }
      });

      newSocket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setIsConnected(false);
        if (reason === 'io server disconnect') {
          setIsReconnecting(true);
          newSocket.connect();
        }
        setError('Connection lost. Attempting to reconnect...');
      });

      newSocket.on('error', (data) => {
        console.error('Socket error:', data);
        if (data.message.includes('token')) {
          setError(data.message);
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
      });

      newSocket.on('response', (data) => {
        setIsLoading(false);
        messageReceived.current = true;
        
        if (data.error) {
          setError(data.error);
          return;
        }

        setError(''); // Clear any existing errors when response is received
        
        const newMessage = {
          type: 'bot',
          content: data.response,
          intent: data.intent,
          entities: data.entities,
          missing_entities: data.missing_entities,
          messageId: data.message_id
        };

        setMessages(prev => [...prev, newMessage]);

        if (data.conversation_active && data.missing_entities?.length > 0) {
          const promptMessage = {
            type: 'bot',
            content: `Please provide: ${data.missing_entities.join(', ')}`,
            isPrompt: true
          };
          setMessages(prev => [...prev, promptMessage]);
        }
      });

      setSocket(newSocket);
      return newSocket;
    };

    const newSocket = initializeSocket();
    
    return () => {
      if (newSocket) {
        newSocket.off('connect');
        newSocket.off('disconnect');
        newSocket.off('error');
        newSocket.off('response');
        newSocket.close();
      }
    };
  }, []);

  useEffect(() => {
    const startSession = async () => {
      if (isConnected && !sessionId) {
        try {
          const storedToken = localStorage.getItem('authToken');
          if (!storedToken) {
            setError('Authentication token not found');
            return;
          }

          const response = await axios.post('http://localhost:5000/api/chat/start', null, {
            headers: {
              'Authorization': storedToken
            }
          });
          
          setSessionId(response.data.session_id);
          setMessages([{
            type: 'bot',
            content: `Hello ${user.firstName}! How can I assist you today?`
          }]);
          setError('');
        } catch (err) {
          console.error('Failed to start chat session:', err);
          const errorMessage = err.response?.data?.error || 'Failed to start chat session';
          setError(`${errorMessage}. Please refresh the page.`);
        }
      }
    };

    startSession();
  }, [isConnected, user.firstName]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || !sessionId || !isConnected || !socket) return;

    const userMessage = input.trim();
    setInput('');
    setError('');
    setIsLoading(true);
    messageReceived.current = false;

    const storedToken = localStorage.getItem('authToken');
    if (!storedToken) {
      setError('Authentication token not found');
      setIsLoading(false);
      return;
    }

    const token = storedToken.replace('Bearer ', '');

    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }]);

    try {
      socket.emit('message', {
        session_id: sessionId,
        message: userMessage,
        token: token
      }, (acknowledgement) => {
        // Only show error if no response was received within 5 seconds
        setTimeout(() => {
          if (!messageReceived.current) {
            setError('Message may not have been received. Please try again.');
            setIsLoading(false);
          }
        }, 5000);
      });
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message. Please try again.');
      setIsLoading(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const isInputDisabled = !isConnected || !sessionId || isLoading;

  return (
    <div className={styles['chat-container']}>
      <div className={styles['chat-header']}>
        <h3>Banking Assistant</h3>
        <p>Welcome, {user.firstName}</p>
        <div className={styles['connection-status']}>
          {isConnected ? (
            <span className={styles['connected']}>●Connected</span>
          ) : isReconnecting ? (
            <span className={styles['reconnecting']}>●Reconnecting...</span>
          ) : (
            <span className={styles['disconnected']}>●Disconnected</span>
          )}
        </div>
      </div>

      <div className={styles['messages-container']}>
        {messages.map((message, index) => (
          <div
            key={index}
            className={`${styles['message']} ${
              styles[message.type === 'user' ? 'user-message' : 'bot-message']
            } ${message.isPrompt ? styles['prompt-message'] : ''}`}
          >
            <div className={styles['message-content']}>
              {message.content}
            </div>
            {message.type === 'bot' && message.intent && (
              <div className={styles['message-intent']}>
                Intent: {message.intent}
              </div>
            )}
            {message.missing_entities && message.missing_entities.length > 0 && (
              <div className={styles['missing-entities']}>
                Missing: {message.missing_entities.join(', ')}
              </div>
            )}
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
          onChange={handleInputChange}
          placeholder={isConnected ? "Type your message..." : "Connecting..."}
          disabled={isInputDisabled}
          className={styles['chat-input']}
        />
        <button
          type="submit"
          disabled={isInputDisabled || !input.trim()}
          className={styles['send-button']}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
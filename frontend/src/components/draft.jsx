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
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  // Initialize WebSocket connection
  useEffect(() => {
    const initializeSocket = () => {
      const token = localStorage.getItem('authToken');
      if (!token) {
        setError('Authentication token not found');
        return;
      }

      const newSocket = io('http://localhost:5000', {
        extraHeaders: {
          Authorization: `Bearer ${token}`
        },
        auth: {
          token: token
        },
        transports: ['websocket'],
        withCredentials: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      // Socket event handlers
      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError('');
        setIsReconnecting(false);
        reconnectAttempts.current = 0;
      });

      newSocket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        setIsConnected(false);
        if (error.message.includes('auth')) {
          setError('Authentication failed. Please log in again.');
        } else {
          setError('Failed to connect to server. Retrying...');
        }
      });

      newSocket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setIsConnected(false);
        if (reason === 'io server disconnect') {
          // Server disconnected us, attempt reconnect
          setIsReconnecting(true);
          newSocket.connect();
        }
        setError('Connection lost. Attempting to reconnect...');
      });

      newSocket.on('error', (data) => {
        console.error('Socket error:', data);
        setError(data.message);
        if (data.message.includes('token')) {
          // Handle token-related errors
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
      });

      newSocket.on('response', (data) => {
        setIsLoading(false);
        if (data.error) {
          setError(data.error);
          return;
        }

        setMessages(prev => [...prev, {
          type: 'bot',
          content: data.response,
          intent: data.intent,
          entities: data.entities,
          missing_entities: data.missing_entities,
          messageId: data.message_id
        }]);

        // Handle continuation of conversation if needed
        if (data.conversation_active && data.missing_entities) {
          setMessages(prev => [...prev, {
            type: 'bot',
            content: `Please provide: ${data.missing_entities.join(', ')}`,
            isPrompt: true
          }]);
        }
      });

      newSocket.on('warning', (data) => {
        if (window.confirm(data.message)) {
          newSocket.emit('force_end_session', { session_id: sessionId });
        }
      });

      newSocket.on('session_ended', (data) => {
        console.log('Session ended:', data.message);
        setSessionId(null);
        setMessages(prev => [...prev, {
          type: 'bot',
          content: 'Chat session ended. Starting new session...'
        }]);
        // Automatically start a new session
        startChatSession();
      });

      setSocket(newSocket);

      return newSocket;
    };

    const newSocket = initializeSocket();
    return () => newSocket?.close();
  }, []);
  
    // Cleanup on component unmount
    return () => {
      if (newSocket) {
        newSocket.off('connect');
        newSocket.off('disconnect');
        newSocket.off('error');
        newSocket.off('response');
        newSocket.off('warning');
        newSocket.off('session_ended');
        newSocket.close();
      }
    };
  }, []);

  // Start chat session
  useEffect(() => {
    if (isConnected && !sessionId) {
      startChatSession();
    }
  }, [isConnected]);

  const startChatSession = async () => {
    try {
      const token = localStorage.getItem('authToken');
      if (!token) {
        setError('Authentication token not found');
        return;
      }

      const response = await axios.post('http://localhost:5000/api/chat/start', null, {
        headers: {
          'Authorization': `Bearer ${token}`
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
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || !sessionId || !isConnected) return;

    const userMessage = input.trim();
    setInput('');
    setError('');
    setIsLoading(true);

    // Add user message to chat
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }]);

    try {
      // Send message through WebSocket
      socket.emit('message', {
        session_id: sessionId,
        message: userMessage
      }, (acknowledgement) => {
        if (!acknowledgement?.received) {
          setError('Message may not have been received. Please try again.');
          setIsLoading(false);
        }
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
            key={`${message.timestamp || index}-${message.messageId || index}`}
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
            {message.missing_entities && (
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
          onChange={(e) => setInput(e.target.value)}
          placeholder={isConnected ? "Type your message..." : "Connecting..."}
          disabled={isLoading || !sessionId || !isConnected}
          className={styles['chat-input']}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim() || !sessionId || !isConnected}
          className={styles['send-button']}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
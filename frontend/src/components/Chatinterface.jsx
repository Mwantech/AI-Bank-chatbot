import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import axios from 'axios';

const ChatInterface = ({ user, setUser }) => {
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

  const handleLogout = () => {
    // Clear all auth data
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    if (socket) {
      socket.disconnect();
    }
    setUser(null);
    window.location.href = '/login';
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const isInputDisabled = !isConnected || !sessionId || isLoading;

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="bg-blue-600 text-white p-4">
        <div className="flex justify-between items-center">
          <h3 className="text-xl font-semibold">Banking Assistant</h3>
          <button 
            onClick={handleLogout}
            className="px-3 py-1 bg-red-500 text-white text-sm rounded hover:bg-red-600 transition-colors"
          >
            Logout
          </button>
        </div>
        <div className="flex justify-between items-center mt-2">
          <p className="text-sm">Welcome, {user.firstName}</p>
          <div className="text-sm">
            {isConnected ? (
              <span className="flex items-center">
                <span className="w-2 h-2 rounded-full bg-green-400 mr-2"></span>
                Connected
              </span>
            ) : isReconnecting ? (
              <span className="flex items-center">
                <span className="w-2 h-2 rounded-full bg-yellow-400 mr-2 animate-pulse"></span>
                Reconnecting...
              </span>
            ) : (
              <span className="flex items-center">
                <span className="w-2 h-2 rounded-full bg-red-500 mr-2"></span>
                Disconnected
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 overflow-y-auto bg-gray-50">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`mb-4 max-w-[80%] ${
              message.type === 'user' ? 'ml-auto' : 'mr-auto'
            }`}
          >
            <div 
              className={`p-3 rounded-lg ${
                message.type === 'user' 
                  ? 'bg-blue-500 text-white rounded-br-none' 
                  : message.isPrompt 
                    ? 'bg-yellow-100 border border-yellow-300 text-yellow-800'
                    : 'bg-gray-200 text-gray-800 rounded-bl-none'
              }`}
            >
              {message.content}
            </div>
            {message.type === 'bot' && message.intent && (
              <div className="text-xs text-gray-500 mt-1 pl-1">
                Intent: {message.intent}
              </div>
            )}
            {message.missing_entities && message.missing_entities.length > 0 && (
              <div className="text-xs text-gray-500 mt-1 pl-1">
                Missing: {message.missing_entities.join(', ')}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="mb-4 max-w-[80%] mr-auto">
            <div className="p-3 rounded-lg bg-gray-200 text-gray-800 rounded-bl-none flex items-center">
              <div className="flex space-x-1">
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}
        {error && (
          <div className="my-2 p-3 rounded bg-red-100 border border-red-300 text-red-800 text-sm">
            {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="p-3 bg-gray-100 border-t border-gray-200">
        <div className="flex">
          <input
            type="text"
            value={input}
            onChange={handleInputChange}
            placeholder={isConnected ? "Type your message..." : "Connecting..."}
            disabled={isInputDisabled}
            className="flex-1 p-3 border border-gray-300 rounded-l-lg focus:outline-none focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={isInputDisabled || !input.trim()}
            className="bg-blue-600 text-white px-4 py-3 rounded-r-lg hover:bg-blue-700 transition-colors disabled:bg-blue-300 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
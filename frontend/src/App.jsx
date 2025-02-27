import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './auth/Login';
import ChatInterface from './components/Chatinterface';

const App = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for existing auth token and user data
    const checkAuth = () => {
      const token = localStorage.getItem('authToken');
      const storedUser = localStorage.getItem('user');

      if (token && storedUser) {
        try {
          setUser(JSON.parse(storedUser));
        } catch (error) {
          console.error('Error parsing stored user data:', error);
          // Clear invalid data
          localStorage.removeItem('authToken');
          localStorage.removeItem('user');
        }
      }
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const handleLoginSuccess = (userData) => {
    // Store user data in localStorage
    localStorage.setItem('user', JSON.stringify(userData));
    setUser(userData);
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
        <p className="mt-4 text-gray-700">Loading...</p>
      </div>
    );
  }

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-6">
          <Routes>
            <Route 
              path="/login" 
              element={
                user ? (
                  <Navigate to="/" replace />
                ) : (
                  <Login onLoginSuccess={handleLoginSuccess} />
                )
              } 
            />
            <Route 
              path="/" 
              element={
                user ? (
                  <ChatInterface 
                    user={user} 
                    setUser={setUser} 
                  />
                ) : (
                  <Navigate to="/login" replace />
                )
              } 
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;
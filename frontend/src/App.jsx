import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./auth/Login";
import ChatInterface from "./components/Chatinterface";
import './app.css';

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false); // Authentication state

  const handleLoginSuccess = () => {
    setIsAuthenticated(true); // Update state after successful login
  };

  return (
    <Router>
      <Routes>
        {/* Login Route */}
        <Route
          path="/"
          element={
            isAuthenticated ? (
              <Navigate to="/chat" replace />
            ) : (
              <Login onLoginSuccess={handleLoginSuccess} />
            )
          }
        />

        {/* Chat Interface Route */}
        <Route
          path="/chat"
          element={
            isAuthenticated ? (
              <ChatInterface />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />

        {/* Catch-All Route (Optional) */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default App;

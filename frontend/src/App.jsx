// App.jsx or parent component
import React, { useState } from 'react';
import Login from './auth/Login';
import ChatInterface from './components/Chatinterface';

const App = () => {
  const [user, setUser] = useState(null);

  const handleLoginSuccess = (userData) => {
    setUser(userData);
  };

  return (
    <div>
      {!user ? (
        <Login onLoginSuccess={handleLoginSuccess} />
      ) : (
        <ChatInterface user={user} />
      )}
    </div>
  );
};

export default App;
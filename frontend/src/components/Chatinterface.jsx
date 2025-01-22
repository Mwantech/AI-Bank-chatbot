import React, { useState } from "react";
import axios from "axios";
import "./ChatInterface.css";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [userMessage, setUserMessage] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    setError("");

    if (!userMessage.trim()) return;

    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setUserMessage("");
    setIsLoading(true);

    try {
      const response = await axios.post("http://localhost:5000/api/chatbot/chat", {
        message: userMessage,
      });

      const botMessage = response.data.response || response.data.reply || "I couldn't process that request.";
      setMessages((prev) => [...prev, { sender: "bot", text: botMessage }]);
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setError("Unable to connect to the chatbot. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-box">
        <div className="chat-header">
          <h2>Bank Chatbot</h2>
        </div>
        
        <div className="messages-container">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              <div className="message-content">
                {message.text}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-content typing">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
        </div>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleSendMessage} className="chat-input-form">
          <input
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            placeholder="Type your message..."
            className="chat-input"
          />
          <button type="submit" className="send-button" disabled={isLoading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
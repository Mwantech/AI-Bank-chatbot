import React, { useState } from "react";
import axios from "axios";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [userMessage, setUserMessage] = useState("");
  const [error, setError] = useState("");

  const handleSendMessage = async (e) => {
    e.preventDefault();
    setError("");

    if (!userMessage.trim()) return;

    // Add user message to the chat
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setUserMessage("");

    try {
      const response = await axios.post("http://your-backend-url/api/chat", {
        message: userMessage,
      });

      // Add bot response to the chat
      setMessages((prev) => [...prev, { sender: "bot", text: response.data.reply }]);
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setError("Unable to connect to the chatbot. Please try again later.");
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.chatBox}>
        <h2 style={styles.title}>Bank Chatbot</h2>
        <div style={styles.messagesContainer}>
          {messages.map((message, index) => (
            <div
              key={index}
              style={{
                ...styles.message,
                alignSelf: message.sender === "user" ? "flex-end" : "flex-start",
                backgroundColor: message.sender === "user" ? "#007bff" : "#f1f1f1",
                color: message.sender === "user" ? "#ffffff" : "#000000",
              }}
            >
              {message.text}
            </div>
          ))}
        </div>
        {error && <p style={styles.error}>{error}</p>}
        <form onSubmit={handleSendMessage} style={styles.inputForm}>
          <input
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            placeholder="Type your message..."
            style={styles.input}
          />
          <button type="submit" style={styles.button}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "#f4f6f9",
  },
  chatBox: {
    width: "600px",
    height: "80vh",
    borderRadius: "10px",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#ffffff",
    display: "flex",
    flexDirection: "column",
  },
  title: {
    textAlign: "center",
    padding: "15px",
    borderBottom: "1px solid #cccccc",
    fontSize: "18px",
    fontWeight: "bold",
    backgroundColor: "#007bff",
    color: "#ffffff",
  },
  messagesContainer: {
    flex: 1,
    padding: "15px",
    overflowY: "auto",
  },
  message: {
    maxWidth: "70%",
    padding: "10px",
    borderRadius: "10px",
    marginBottom: "10px",
    fontSize: "14px",
  },
  inputForm: {
    display: "flex",
    padding: "10px",
    borderTop: "1px solid #cccccc",
  },
  input: {
    flex: 1,
    padding: "10px",
    borderRadius: "5px",
    border: "1px solid #cccccc",
    fontSize: "14px",
  },
  button: {
    marginLeft: "10px",
    padding: "10px 20px",
    border: "none",
    borderRadius: "5px",
    backgroundColor: "#007bff",
    color: "#ffffff",
    cursor: "pointer",
    fontSize: "14px",
  },
  error: {
    color: "#ff4d4d",
    padding: "10px",
    textAlign: "center",
  },
};

export default ChatInterface;

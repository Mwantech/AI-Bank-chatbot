from model import BankingChatbot

# Initialize the chatbot with saved model
chatbot = BankingChatbot()

# Use the chatbot
response = chatbot.get_response("user_id", "your text here")
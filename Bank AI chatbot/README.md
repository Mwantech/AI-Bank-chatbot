# Banking Chatbot

This project is a banking chatbot application built using Flask, Flask-SocketIO, and various machine learning libraries. The chatbot can handle user authentication, chat sessions, and respond to various banking-related queries.

## Project Structure
. ├── pycache/ ├── .env ├── app.py ├── auth.py ├── chat.py ├── chatbot.py ├── data/ │ └── intents.json ├── generate_secret_key.py ├── instance/ │ └── models/ │ ├── classifier.pkl │ ├── entities.pkl │ ├── label_encoder.pkl │ ├── required_entities.pkl │ ├── responses.pkl │ └── vectorizer.pkl ├── models.py ├── requirements.txt ├── train_model.py


## Setup

### Prerequisites

- Python 3.7+
- MySQL

### Installation

1. Clone the repository:

```sh
git clone https://github.com/yourusername/banking-chatbot.git
cd banking-chatbot

Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Set up the environment variables:
Create a .env file in the root directory with the following content:

DATABASE_URI=mysql+pymysql://root:@localhost/banking_chatbot
JWT_SECRET_KEY=your_jwt_secret_key
FLASK_DEBUG=True  # for development
SECRET_KEY=your_flask_secret_key

Initialize the database:
flask db init
flask db migrate
flask db upgrade

Train the chatbot model:
python train_model.py


Running the Application
Start the Flask application:
The application will be available at http://localhost:5000.
API Endpoints
Authentication
POST /api/auth/register: Register a new user.
POST /api/auth/login: Log in an existing user.
GET /api/auth/user: Get the current user's information (requires token).
POST /api/auth/logout: Log out the current user (requires token).
Chat
POST /api/chat/start: Start a new chat session (requires token).
GET /api/chat/history: Get chat history (requires token).
WebSocket Events
connect: Connect to the chat server (requires token).
disconnect: Disconnect from the chat server.
message: Send a message to the chatbot (requires token).
end_session: End the current chat session (requires token).
force_end_session: Forcefully end the current chat session (requires token).
License
This project is licensed under the MIT License. See the LICENSE file for details.

# Banking Chatbot

A conversational AI chatbot for banking services built with Python. The chatbot can handle customer inquiries, provide ATM locations, and process various banking-related requests.

## Features

- Natural language processing for customer inquiries
- ATM location service
- Secure authentication system
- Intent classification and entity recognition
- Database integration for persistent storage

## Project Structure

- `app.py` - Main application entry point
- `auth.py` - Authentication and user management
- `chat.py` & `chat2.py` - Chatbot conversation handling
- `chatbot.py` - Core chatbot logic
- `inquiry_handler.py` - Customer inquiry processing
- `models.py` - Database models and schemas

## Setup

1. Install dependencies:
```sh
pip install -r requirements.txt
```

2. Set up environment variables:
```sh
cp .env.example .env
python generate_secret_key.py
```

3. Initialize the database:
```sh
sqlite3 instance/banking_chatbot.db < database.sql
```

## Training the Model

To train or retrain the chatbot model:

```sh
python train_model.py
```

This will generate model files in the `models/` directory including:
- Intent classifier
- Entity recognition model
- Label encoders
- Vectorizers

## Data Files

- `data/intents.json` - Training data for chatbot responses
- `data/atm_locations.json` - ATM location information
- `database.sql` - Database schema

## Running the Application

```sh
python app.py
```

The chatbot will start and be ready to process user inquiries.

## Model Components

The system uses several trained models stored in the `models/` directory:
- `classifier.pkl` - Main intent classification model
- `entities.pkl` - Entity recognition model
- `label_encoder.pkl` - Encodes intent labels
- `vectorizer.pkl` - Text vectorization model
- `responses.pkl` - Response template storage
- `required_entities.pkl` - Required entities for each intent

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO
from dotenv import load_dotenv
from models import db
from auth import auth_bp
from accounts_routes import account_bp
from chat import chat_bp, init_app as init_chat
import os

load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Configure CORS for both REST and WebSocket
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:5173"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        },
        r"/socket.io/*": {
            "origins": ["http://localhost:5173"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URI', 'mysql+pymysql://root:@localhost/banking_chatbot'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Secret key for sessions
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

    # Initialize extensions
    db.init_app(app)
    
    # Create SocketIO instance with CORS configuration
    socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5173"])
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(account_bp)
    
    # Initialize chat with the socketio instance
    init_chat(app, socketio)
    
    return app, socketio

if __name__ == '__main__':
    app, socketio = create_app()
    
    # Run with WebSocket support
    socketio.run(
        app,
        debug=os.getenv('FLASK_DEBUG', False),
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True
    )
# app.py
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, send_file
from config import app, db
from routes.account_routes import account_bp
from flask_cors import CORS
from routes.customer_support_routes import support_bp
from routes.loan_routes import loan_bp
from routes.location_routes import location_bp
from routes.investment_routes import investment_bp
from routes.chatbot_routes import chatbot_bp
from routes.auth_routes import auth_bp

# Configure CORS
# Configure CORS - Updated configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 600  # Cache preflight requests for 10 minutes
    }
})

# Also add CORS error handling
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# Register blueprints
app.register_blueprint(account_bp, url_prefix='/api/accounts')
app.register_blueprint(support_bp, url_prefix='/api/support')
app.register_blueprint(loan_bp, url_prefix='/api/loans')
app.register_blueprint(location_bp, url_prefix='/api/locations')
app.register_blueprint(investment_bp, url_prefix='/api/investments')
app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
app.register_blueprint(auth_bp, url_prefix='/api/auth')

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_file(f'static/{path}')

if __name__ == '__main__':
    app.run(debug=True)
# app.py
from flask import Flask, send_file
from config import app, db
from routes.account_routes import account_bp
from routes.customer_support_routes import support_bp
from routes.loan_routes import loan_bp
from routes.location_routes import location_bp
from routes.investment_routes import investment_bp
from routes.chatbot_routes import chatbot_bp

# Register blueprints
app.register_blueprint(account_bp, url_prefix='/api/accounts')
app.register_blueprint(support_bp, url_prefix='/api/support')
app.register_blueprint(loan_bp, url_prefix='/api/loans')
app.register_blueprint(location_bp, url_prefix='/api/locations')
app.register_blueprint(investment_bp, url_prefix='/api/investments')
app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_file(f'static/{path}')

if __name__ == '__main__':
    app.run(debug=True)
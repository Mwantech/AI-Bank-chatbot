from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import jwt
from datetime import datetime, timedelta
import os
import sys

# Blueprint definitions
auth_bp = Blueprint('auth', __name__)
chatbot_bp = Blueprint('chatbot', __name__)

# Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY must be set in environment variables")

JWT_EXPIRATION_DELTA = timedelta(hours=24)

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for token in headers
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        try:
            # Extract token (handle both "Bearer <token>" and plain token formats)
            token = auth_header.split(" ")[-1]
            # Decode token and extract user_id
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(payload['user_id'])
            
            if not current_user:
                return jsonify({'error': 'User not found'}), 404
            
            # Add user to kwargs so the route can access it
            kwargs['current_user'] = current_user
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
    return decorated

# Authentication routes
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    required_fields = ['email', 'password', 'firstName', 'lastName', 'phoneNumber']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if user already exists
    existing_user = User.query.filter(
        (User.Email == data['email']) |
        (User.PhoneNumber == data['phoneNumber'])
    ).first()
    
    if existing_user:
        return jsonify({'error': 'Email or phone number already registered'}), 409
    
    try:
        hashed_password = generate_password_hash(data['password'])
        new_user = User(
            FirstName=data['firstName'],
            LastName=data['lastName'],
            Email=data['email'],
            PhoneNumber=data['phoneNumber'],
            PasswordHash=hashed_password,
            DateOfBirth=datetime.strptime(data.get('dateOfBirth', '2000-01-01'), '%Y-%m-%d').date(),
            Address=data.get('address', ''),
            IdentificationNumber=data.get('identificationNumber', ''),
            SecurityQuestion=data.get('securityQuestion', ''),
            SecurityAnswer=data.get('securityAnswer', ''),
            AccountStatus='Active'
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'message': 'User registered successfully',
            'userId': new_user.UserID
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    user = User.query.filter_by(Email=email).first()
    
    if not user or not check_password_hash(user.PasswordHash, password):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    if user.AccountStatus != 'Active':
        return jsonify({'error': 'Account is inactive'}), 403
    
    # Generate JWT token with user information
    token = jwt.encode({
        'user_id': user.UserID,
        'email': user.Email,
        'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA
    }, JWT_SECRET_KEY, algorithm='HS256')
    
    return jsonify({
        'token': token,
        'user': {
            'id': user.UserID,
            'email': user.Email,
            'firstName': user.FirstName,
            'lastName': user.LastName
        }
    })

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import jwt
from datetime import datetime, timedelta
import os
from models import db, User

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default-dev-key')
JWT_EXPIRATION_DELTA = timedelta(hours=24)

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

def verify_token(token):
    """
    Verify JWT token and return user object
    Used by both REST and WebSocket authentication
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        current_user = User.query.get(payload['user_id'])
        
        if not current_user:
            raise Exception('User not found')
            
        return current_user
        
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError('Token has expired')
    except jwt.InvalidTokenError:
        raise jwt.InvalidTokenError('Invalid token')
    except Exception as e:
        raise Exception(str(e))

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        try:
            token = auth_header.split(" ")[-1]
            current_user = verify_token(token)
            kwargs['current_user'] = current_user
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            return jsonify({'error': str(e)}), 401
        
    return decorated

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    required_fields = ['email', 'password', 'firstName', 'lastName', 'phoneNumber']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    existing_user = User.query.filter(
        (User.Email == data['email']) |
        (User.PhoneNumber == data['phoneNumber'])
    ).first()
    
    if existing_user:
        return jsonify({'error': 'Email or phone number already registered'}), 409
    
    try:
        new_user = User(
            FirstName=data['firstName'],
            LastName=data['lastName'],
            Email=data['email'],
            PhoneNumber=data['phoneNumber'],
            PasswordHash=generate_password_hash(data['password']),
            DateOfBirth=datetime.strptime(data.get('dateOfBirth', '2000-01-01'), '%Y-%m-%d').date(),
            Address=data.get('address', ''),
            IdentificationNumber=data.get('identificationNumber', ''),
            SecurityQuestion=data.get('securityQuestion', ''),
            SecurityAnswer=data.get('securityAnswer', ''),
            AccountStatus='Active'
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        token = jwt.encode({
            'user_id': new_user.UserID,
            'email': new_user.Email,
            'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA
        }, JWT_SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': new_user.UserID,
                'email': new_user.Email,
                'firstName': new_user.FirstName,
                'lastName': new_user.LastName
            }
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
    
    try:
        user = User.query.filter_by(Email=email).first()
        
        if not user or not check_password_hash(user.PasswordHash, password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if user.AccountStatus != 'Active':
            return jsonify({'error': 'Account is inactive'}), 403
        
        # Generate token
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/user', methods=['GET'])
@token_required
def get_user(current_user):
    return jsonify({
        'user': {
            'id': current_user.UserID,
            'email': current_user.Email,
            'firstName': current_user.FirstName,
            'lastName': current_user.LastName,
            'phoneNumber': current_user.PhoneNumber,
            'address': current_user.Address,
            'accountStatus': current_user.AccountStatus
        }
    })

@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    # In a more complete implementation, you might want to blacklist the token
    return jsonify({'message': 'Logged out successfully'})
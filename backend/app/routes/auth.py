from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register new user (now accepts subscription plan)"""
    from app.models.user import User
    from app import db
    
    data = request.get_json()
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    subscription = data.get('subscription', 'free')   # <-- NEW: default free
    
    if not username or not email or not password:
        return jsonify({'error': 'Username, email and password required'}), 400
    
    # Validate subscription plan
    if subscription not in ('free', 'pro', 'enterprise'):
        return jsonify({'error': 'Invalid subscription plan. Choose free, pro, or enterprise.'}), 400
    
    # Check if user exists
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Create user with subscription
    user = User(username=username, email=email, subscription=subscription)
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    
    # Create token with additional claims (so frontend can read subscription easily)
    access_token = create_access_token(
        identity=str(user.id),
        expires_delta=timedelta(hours=24),
        additional_claims={'subscription': subscription, 'username': username}
    )
    
    return jsonify({
        'success': True,
        'message': 'User created successfully',
        'access_token': access_token,
        'user': user.to_dict()
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user (subscription included in token & response)"""
    from app.models.user import User
    
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid username or password'}), 401
    
    # Create token with additional claims
    access_token = create_access_token(
        identity=str(user.id),
        expires_delta=timedelta(hours=24),
        additional_claims={
            'subscription': user.subscription,
            'username': user.username
        }
    )
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'access_token': access_token,
        'user': user.to_dict()
    }), 200


@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    """Get user profile"""
    from app.models.user import User
    
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    }), 200
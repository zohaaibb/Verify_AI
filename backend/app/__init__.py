import os
from dotenv import load_dotenv
load_dotenv(override=True)

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

db = SQLAlchemy()
cors = CORS()
jwt = JWTManager()

@jwt.unauthorized_loader
def unauthorized_response(callback):
    from flask import jsonify
    return jsonify({'msg': 'Missing Authorization Header'}), 401

@jwt.invalid_token_loader
def invalid_token_response(callback):
    from flask import jsonify
    return jsonify({'msg': 'Invalid token'}), 401

@jwt.expired_token_loader
def expired_token_response(callback):
    from flask import jsonify
    return jsonify({'msg': 'Token has expired'}), 401

@jwt.revoked_token_loader
def revoked_token_response(callback):
    from flask import jsonify
    return jsonify({'msg': 'Token has been revoked'}), 401

def create_app(config_name='default'):
    app = Flask(__name__)
    from .config import config
    app.config.from_object(config[config_name])

    if app.debug:
        vt_key = os.environ.get('VIRUSTOTAL_API_KEY', 'NOT SET')
        logger.info(f"🔑 VirusTotal API Key loaded: {vt_key[:15] if vt_key != 'NOT SET' else 'NOT SET'}...")

    db.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})
    jwt.init_app(app)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    from .routes.auth import auth_bp
    from .routes.analysis import analysis_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')

    from .models.user import User
    with app.app_context():
        db.create_all()
        logger.info("✅ Database tables created")

    # Optional: pre-load AI model at startup (comment out if memory heavy)
    with app.app_context():
        try:
            from app.services.text_processor import TextProcessor
            processor = TextProcessor()
            if processor.load_model():
                logger.info("✅✅✅ AI MODEL LOADED SUCCESSFULLY AT STARTUP! ✅✅✅")
        except Exception as e:
            logger.error(f"❌ AI model load failed: {e}")

    return app
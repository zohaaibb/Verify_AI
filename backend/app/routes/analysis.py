from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.orchestrator import AnalysisOrchestrator
import logging
import os
import json
import time
import numpy as np
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__)
orchestrator = AnalysisOrchestrator()


def _convert_numpy(obj):
    """Recursively convert numpy types to native Python types so jsonify can handle them."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


@analysis_bp.route('/modules', methods=['GET'])
@jwt_required()
def get_modules():
    modules = orchestrator.registry.all_modules
    return jsonify({'success': True, 'modules': modules}), 200


@analysis_bp.route('/submit', methods=['POST'])
@jwt_required()
def submit_analysis():
    user_id = get_jwt_identity()
    try:
        input_type = None
        modules = []
        content = None
        user_edited_text = request.form.get('user_edited_text', '').strip() or None

        if request.is_json:
            data = request.get_json()
            input_type = data.get('input_type')
            modules = data.get('modules', [])
            text = data.get('text')
            if input_type in ('text', 'url'):
                content = text
            else:
                return jsonify({'error': 'JSON not supported for this input type'}), 400
        else:
            input_type = request.form.get('input_type')
            modules = json.loads(request.form.get('modules', '[]'))
            text = request.form.get('text')
            if input_type in ('text', 'url'):
                content = text

        if not input_type:
            return jsonify({'error': 'input_type required'}), 400

        if input_type == 'text':
            if not content:
                return jsonify({'error': 'Text content required'}), 400
        elif input_type == 'url':
            if not content:
                return jsonify({'error': 'URL required'}), 400
        elif input_type == 'image':
            file = request.files.get('file')
            if not file:
                return jsonify({'error': 'Image file required'}), 400
            filename = secure_filename(file.filename)
            temp_path = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(temp_path)
            content = temp_path
        elif input_type == 'file':
            file = request.files.get('file')
            if not file:
                return jsonify({'error': 'File required'}), 400
            filename = secure_filename(file.filename)
            temp_path = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(temp_path)
            content = temp_path
        else:
            return jsonify({'error': f'Invalid input_type: {input_type}'}), 400

        result = orchestrator.analyze(input_type, content, modules, user_edited_text=user_edited_text)

        # Convert any numpy/bool_ types to native Python before JSON serialization
        result = _convert_numpy(result)

        # Clean up temporary files
        if input_type in ('image', 'file') and content and os.path.exists(content):
            try:
                os.remove(content)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

        return jsonify({
            'success': True,
            'analysis_id': f"analysis_{user_id}_{int(time.time())}",
            'results': result
        }), 200

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
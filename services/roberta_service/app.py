from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "E:/Verify_AI/models/fake-news-roberta-5M"
app = Flask(__name__)
classifier = None

def load_model():
    """Load the RoBERTa model (99.28% accuracy)"""
    global classifier
    logger.info("🔄 Loading RoBERTa model (this takes 10-20 seconds)...")
    
    try:
        # Method 1: Try pipeline (simplest)
        classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("✅✅✅ RoBERTa loaded successfully on port 5001!")
        
        # Test the model
        test = classifier("This is a test news article")[0]
        logger.info(f"🧪 Test prediction: {test}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline loading failed: {e}")
        
        try:
            # Method 2: Manual loading
            logger.info("Attempting manual loading...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            
            # Wrap in pipeline for consistent interface
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("✅✅✅ RoBERTa loaded manually!")
            return True
            
        except Exception as e2:
            logger.error(f"❌ Manual loading failed: {e2}")
            return False

@app.route('/predict', methods=['POST'])
def predict():
    """Analyze text for fake news"""
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        result = classifier(text)[0]
        label = result['label']
        score = float(result['score'])
        
        # Confidence-based classification (same as your original)
        if score > 0.8:
            is_fake = False
            confidence = score
            reason = "High confidence indicates REAL news"
        elif score < 0.5:
            is_fake = True
            confidence = 1 - score
            reason = "Low confidence indicates FAKE news"
        else:
            is_fake = True
            confidence = score
            reason = "Medium confidence indicates FAKE news (uncertain)"
        
        return jsonify({
            'success': True,
            'is_fake': is_fake,
            'confidence': round(confidence, 4),
            'raw_label': label,
            'raw_score': score,
            'reason': reason,
            'model': 'Arko007/fake-news-roberta-5M',
            'accuracy': '99.28%'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Arko007/fake-news-roberta-5M',
        'loaded': classifier is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Analyze multiple texts at once"""
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    results = []
    for text in texts:
        result = classifier(text)[0]
        results.append({
            'text': text[:50] + '...',
            'label': result['label'],
            'score': float(result['score'])
        })
    
    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        logger.error("❌ Failed to load model. Exiting.")
from app import create_app

app = create_app('development')

# Add health check endpoint
@app.route('/health')
def health():
    """Simple health check endpoint for monitoring"""
    return {
        'status': 'healthy', 
        'service': 'fake-news-detector',
        'version': '1.0.0'
    }

# Add root endpoint for basic info
@app.route('/')
def index():
    """Root endpoint with API information"""
    return {
        'name': 'Fake News Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'auth': '/api/auth',
            'analysis': '/api/analysis',
            'modules': '/api/analysis/modules'
        }
    }

if __name__ == '__main__':
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )
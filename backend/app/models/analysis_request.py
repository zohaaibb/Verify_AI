# app/models/analysis_request.py
from app import db
from datetime import datetime
import json

class AnalysisRequest(db.Model):
    """Stores user analysis requests with selected modules"""
    __tablename__ = 'analysis_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)  # REMOVED ForeignKey
    input_type = db.Column(db.String(20), nullable=False)
    input_content = db.Column(db.Text, nullable=False)
    
    # Selected modules
    selected_modules = db.Column(db.Text, default='{}')
    
    # Results
    ai_result = db.Column(db.Text, default='{}')
    osint_result = db.Column(db.Text, default='{}')
    virustotal_result = db.Column(db.Text, default='{}')
    forensics_result = db.Column(db.Text, default='{}')
    sandbox_result = db.Column(db.Text, default='{}')
    reverse_result = db.Column(db.Text, default='{}')
    trusted_result = db.Column(db.Text, default='{}')
    scraper_result = db.Column(db.Text, default='{}')
    
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'input_type': self.input_type,
            'input_content': self.input_content[:200] + '...' if len(self.input_content) > 200 else self.input_content,
            'selected_modules': json.loads(self.selected_modules) if self.selected_modules else {},
            'ai_result': json.loads(self.ai_result) if self.ai_result else {},
            'osint_result': json.loads(self.osint_result) if self.osint_result else {},
            'virustotal_result': json.loads(self.virustotal_result) if self.virustotal_result else {},
            'forensics_result': json.loads(self.forensics_result) if self.forensics_result else {},
            'sandbox_result': json.loads(self.sandbox_result) if self.sandbox_result else {},
            'reverse_result': json.loads(self.reverse_result) if self.reverse_result else {},
            'trusted_result': json.loads(self.trusted_result) if self.trusted_result else {},
            'scraper_result': json.loads(self.scraper_result) if self.scraper_result else {},
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
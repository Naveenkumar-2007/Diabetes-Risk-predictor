"""
MLOps Dashboard API Endpoints
Flask Blueprint for monitoring and management
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify, request
from datetime import datetime
import json

from mlops_config import *
from mlops.monitoring.monitor import ModelMonitor
from mlops.retraining.auto_retrain import AutoRetrainer

# Create Blueprint
mlops_bp = Blueprint('mlops', __name__, url_prefix='/mlops/api')

# Initialize components
monitor = ModelMonitor()
retrainer = AutoRetrainer()


@mlops_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'MLOps Dashboard'
    })


@mlops_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Get current model information"""
    try:
        if not METADATA_PATH.exists():
            return jsonify({'error': 'Model metadata not found'}), 404
        
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        return jsonify({
            'success': True,
            'model': metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/monitoring/predictions', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions"""
    try:
        limit = request.args.get('limit', default=100, type=int)
        predictions_df = monitor.load_predictions(limit=limit)
        
        if predictions_df.empty:
            return jsonify({
                'success': True,
                'predictions': [],
                'count': 0
            })
        
        # Convert to dict
        predictions = predictions_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/monitoring/drift', methods=['GET'])
def check_drift():
    """Check for data drift"""
    try:
        drift_result = monitor.detect_data_drift()
        return jsonify({
            'success': True,
            'drift': drift_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/monitoring/performance', methods=['GET'])
def check_performance():
    """Check model performance"""
    try:
        days = request.args.get('days', default=7, type=int)
        perf_result = monitor.detect_performance_degradation(days=days)
        
        return jsonify({
            'success': True,
            'performance': perf_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/monitoring/report', methods=['GET'])
def generate_report():
    """Generate comprehensive monitoring report"""
    try:
        report = monitor.generate_monitoring_report()
        return jsonify({
            'success': True,
            'report': report
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/retrain/check', methods=['GET'])
def check_retrain_conditions():
    """Check if retraining is recommended"""
    try:
        should_retrain, reasons = monitor.should_retrain()
        
        return jsonify({
            'success': True,
            'should_retrain': should_retrain,
            'reasons': reasons
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/retrain/trigger', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining"""
    try:
        data = request.get_json() or {}
        force = data.get('force', False)
        
        # This would typically be done async via task queue
        # For now, return acknowledgment
        return jsonify({
            'success': True,
            'message': 'Retraining job queued',
            'force': force,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mlops_bp.route('/stats/summary', methods=['GET'])
def get_stats_summary():
    """Get overall statistics summary"""
    try:
        predictions_df = monitor.load_predictions()
        
        if predictions_df.empty:
            return jsonify({
                'success': True,
                'stats': {
                    'total_predictions': 0,
                    'predictions_today': 0,
                    'predictions_this_week': 0,
                    'avg_daily_predictions': 0
                }
            })
        
        from datetime import timedelta
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        
        stats = {
            'total_predictions': len(predictions_df),
            'predictions_today': len(predictions_df[predictions_df['timestamp'] >= today_start]),
            'predictions_this_week': len(predictions_df[predictions_df['timestamp'] >= week_start]),
            'avg_prediction_probability': float(predictions_df['probability'].mean()),
            'positive_prediction_rate': float((predictions_df['prediction'] == 1).mean())
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Error handlers
@mlops_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@mlops_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

"""
MLOps Dashboard - Model Performance & Monitoring Overview
API endpoints for monitoring dashboard
"""
from flask import Blueprint, jsonify, render_template
from datetime import datetime, timedelta
import json
from pathlib import Path

import sys
sys.path.append('..')
from mlops_config import *
from mlops.model_monitor import ModelMonitor

mlops_bp = Blueprint('mlops', __name__, url_prefix='/mlops')
monitor = ModelMonitor()

@mlops_bp.route('/dashboard')
def dashboard():
    """Render MLOps dashboard"""
    return render_template('mlops_dashboard.html')

@mlops_bp.route('/api/model/info')
def get_model_info():
    """Get current model information"""
    try:
        with open(ARTIFACT_PATH / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return jsonify({
            'success': True,
            'model': metadata
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mlops_bp.route('/api/monitoring/performance/<int:days>')
def get_performance_metrics(days):
    """Get model performance for last N days"""
    try:
        performance = monitor.get_model_performance_summary(days=days)
        return jsonify({
            'success': True,
            'performance': performance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mlops_bp.route('/api/monitoring/drift/check')
def check_drift():
    """Check for data and concept drift"""
    try:
        data_drift = monitor.detect_data_drift()
        concept_drift = monitor.detect_concept_drift()
        should_retrain, reason = monitor.should_trigger_retraining()
        
        return jsonify({
            'success': True,
            'data_drift': data_drift,
            'concept_drift': concept_drift,
            'retraining_recommended': should_retrain,
            'retraining_reason': reason
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mlops_bp.route('/api/monitoring/predictions/recent/<int:limit>')
def get_recent_predictions(limit=100):
    """Get recent predictions"""
    try:
        predictions = []
        log_file = monitor.prediction_log_file
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        predictions.append(json.loads(line))
                    except:
                        continue
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mlops_bp.route('/api/monitoring/alerts/recent')
def get_recent_alerts():
    """Get recent drift alerts"""
    try:
        alerts = []
        alert_file = monitor.drift_log_file
        
        if alert_file.exists():
            with open(alert_file, 'r') as f:
                for line in f:
                    try:
                        alerts.append(json.loads(line))
                    except:
                        continue
        
        return jsonify({
            'success': True,
            'alerts': alerts[-20:],  # Last 20 alerts
            'count': len(alerts)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mlops_bp.route('/api/retrain/history')
def get_retrain_history():
    """Get model retraining history"""
    try:
        retrain_log = ARTIFACT_PATH / "retrain_history.jsonl"
        history = []
        
        if retrain_log.exists():
            with open(retrain_log, 'r') as f:
                for line in f:
                    try:
                        history.append(json.loads(line))
                    except:
                        continue
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mlops_bp.route('/api/retrain/trigger', methods=['POST'])
def trigger_retraining():
    """Manually trigger model retraining"""
    try:
        from mlops.auto_retrain import AutoRetrainer
        retrainer = AutoRetrainer()
        
        # Run in background (for production, use Celery or similar)
        import threading
        thread = threading.Thread(target=retrainer.run_automated_retraining)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model retraining started in background'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

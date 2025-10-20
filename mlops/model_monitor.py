"""
Model Monitoring and Drift Detection System
Tracks model performance and detects data/concept drift
"""
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from mlops_config import *

class ModelMonitor:
    """
    Monitors model performance and detects drift:
    - Prediction logging
    - Performance tracking
    - Data drift detection (PSI - Population Stability Index)
    - Concept drift detection
    - Automated retraining triggers
    """
    
    def __init__(self):
        self.predictions_buffer = deque(maxlen=DRIFT_DETECTION_WINDOW)
        self.prediction_log_file = PREDICTION_LOG_PATH / f"predictions_{datetime.now().strftime('%Y%m')}.jsonl"
        self.drift_log_file = PREDICTION_LOG_PATH / "drift_alerts.jsonl"
        self.performance_log_file = PREDICTION_LOG_PATH / "performance_metrics.jsonl"
        
        # Load reference distribution (from training data)
        self.reference_dist = self._load_reference_distribution()
        
    def _load_reference_distribution(self):
        """Load reference feature distributions from training data"""
        try:
            ref_file = PROCESSED_DATA_PATH / "reference_distribution.pkl"
            if ref_file.exists():
                with open(ref_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            print(f"⚠️  Could not load reference distribution: {e}")
            return None
    
    def log_prediction(self, features, prediction, probability=None, actual_outcome=None, user_id=None):
        """Log a single prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'features': features if isinstance(features, dict) else dict(zip(FEATURE_NAMES, features)),
            'prediction': int(prediction),
            'probability': float(probability) if probability is not None else None,
            'actual_outcome': int(actual_outcome) if actual_outcome is not None else None
        }
        
        # Append to log file
        with open(self.prediction_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Add to buffer for drift detection
        self.predictions_buffer.append(log_entry)
        
        return log_entry
    
    def calculate_psi(self, expected, actual, buckets=10):
        """
        Calculate Population Stability Index (PSI) for drift detection
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change
        PSI > 0.25: Significant change (drift detected)
        """
        def scale_range(values):
            return (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else values
        
        # Create buckets
        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        expected_scaled = scale_range(np.array(expected))
        actual_scaled = scale_range(np.array(actual))
        
        expected_percents = np.percentile(expected_scaled, breakpoints)
        actual_percents = np.percentile(actual_scaled, breakpoints)
        
        # Calculate PSI
        psi_values = []
        for i in range(len(expected_percents) - 1):
            exp_count = np.sum((expected_scaled >= expected_percents[i]) & (expected_scaled < expected_percents[i+1]))
            act_count = np.sum((actual_scaled >= actual_percents[i]) & (actual_scaled < actual_percents[i+1]))
            
            exp_pct = exp_count / len(expected) if len(expected) > 0 else 0.0001
            act_pct = act_count / len(actual) if len(actual) > 0 else 0.0001
            
            # Avoid log(0)
            exp_pct = max(exp_pct, 0.0001)
            act_pct = max(act_pct, 0.0001)
            
            psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
            psi_values.append(psi)
        
        return sum(psi_values)
    
    def detect_data_drift(self):
        """Detect data drift using PSI"""
        if not self.reference_dist or len(self.predictions_buffer) < 100:
            return {
                'drift_detected': False,
                'reason': 'Insufficient data or no reference distribution'
            }
        
        drift_results = {}
        drift_detected = False
        
        # Extract current features
        current_features = {}
        for feature in FEATURE_NAMES:
            current_features[feature] = [
                pred['features'].get(feature, 0) 
                for pred in self.predictions_buffer 
                if 'features' in pred
            ]
        
        # Calculate PSI for each feature
        for feature in FEATURE_NAMES:
            if feature in self.reference_dist and current_features[feature]:
                psi = self.calculate_psi(
                    self.reference_dist[feature],
                    current_features[feature]
                )
                
                drift_results[feature] = {
                    'psi': psi,
                    'drift': psi > DRIFT_THRESHOLD
                }
                
                if psi > DRIFT_THRESHOLD:
                    drift_detected = True
        
        # Log drift alert
        if drift_detected:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'data_drift',
                'drift_features': [f for f, r in drift_results.items() if r.get('drift', False)],
                'psi_scores': {f: r['psi'] for f, r in drift_results.items()},
                'action_required': 'model_retraining_recommended'
            }
            
            with open(self.drift_log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
            
            print(f"🚨 DATA DRIFT DETECTED!")
            print(f"   Affected features: {alert['drift_features']}")
        
        return {
            'drift_detected': drift_detected,
            'feature_psi': drift_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_concept_drift(self, window_size=100):
        """Detect concept drift by monitoring prediction accuracy over time"""
        if len(self.predictions_buffer) < window_size:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Get predictions with actual outcomes
        recent_preds = [
            p for p in list(self.predictions_buffer)[-window_size:]
            if p.get('actual_outcome') is not None
        ]
        
        if len(recent_preds) < 50:
            return {'drift_detected': False, 'reason': 'Insufficient labeled data'}
        
        # Calculate recent accuracy
        correct = sum(1 for p in recent_preds if p['prediction'] == p['actual_outcome'])
        recent_accuracy = correct / len(recent_preds)
        
        # Compare with expected accuracy (from model metadata)
        try:
            with open(ARTIFACT_PATH / "model_metadata.json", 'r') as f:
                metadata = json.load(f)
                expected_accuracy = metadata['metrics']['test_accuracy']
        except:
            expected_accuracy = 0.75  # Default threshold
        
        # Drift if accuracy drops significantly
        accuracy_drop = expected_accuracy - recent_accuracy
        drift_detected = accuracy_drop > 0.10  # 10% drop threshold
        
        if drift_detected:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'concept_drift',
                'recent_accuracy': recent_accuracy,
                'expected_accuracy': expected_accuracy,
                'accuracy_drop': accuracy_drop,
                'action_required': 'immediate_model_retraining'
            }
            
            with open(self.drift_log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
            
            print(f"🚨 CONCEPT DRIFT DETECTED!")
            print(f"   Recent accuracy: {recent_accuracy:.4f}")
            print(f"   Expected: {expected_accuracy:.4f}")
            print(f"   Drop: {accuracy_drop:.4f}")
        
        return {
            'drift_detected': drift_detected,
            'recent_accuracy': recent_accuracy,
            'expected_accuracy': expected_accuracy,
            'accuracy_drop': accuracy_drop
        }
    
    def get_model_performance_summary(self, days=7):
        """Get model performance summary for the last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Read recent predictions
        predictions = []
        if self.prediction_log_file.exists():
            with open(self.prediction_log_file, 'r') as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                        pred_time = datetime.fromisoformat(pred['timestamp'])
                        if pred_time >= cutoff:
                            predictions.append(pred)
                    except:
                        continue
        
        if not predictions:
            return {'message': 'No predictions in the specified period'}
        
        # Calculate metrics
        total_predictions = len(predictions)
        predictions_with_actuals = [p for p in predictions if p.get('actual_outcome') is not None]
        
        summary = {
            'period_days': days,
            'total_predictions': total_predictions,
            'labeled_predictions': len(predictions_with_actuals),
            'average_probability': np.mean([p.get('probability', 0.5) for p in predictions]),
            'prediction_distribution': {
                'positive': sum(1 for p in predictions if p['prediction'] == 1),
                'negative': sum(1 for p in predictions if p['prediction'] == 0)
            }
        }
        
        if predictions_with_actuals:
            correct = sum(1 for p in predictions_with_actuals if p['prediction'] == p['actual_outcome'])
            summary['accuracy'] = correct / len(predictions_with_actuals)
            summary['accuracy_count'] = f"{correct}/{len(predictions_with_actuals)}"
        
        return summary
    
    def should_trigger_retraining(self):
        """Determine if model retraining should be triggered"""
        if not AUTO_RETRAIN_ENABLED:
            return False, "Auto-retraining is disabled"
        
        # Check for drift
        data_drift = self.detect_data_drift()
        concept_drift = self.detect_concept_drift()
        
        # Check for sufficient new data
        new_data_count = len(self.predictions_buffer)
        
        reasons = []
        if data_drift['drift_detected']:
            reasons.append("Data drift detected")
        if concept_drift['drift_detected']:
            reasons.append("Concept drift detected")
        if new_data_count >= MIN_NEW_DATA_THRESHOLD:
            reasons.append(f"Sufficient new data ({new_data_count} predictions)")
        
        should_retrain = len(reasons) > 0
        
        return should_retrain, "; ".join(reasons) if reasons else "No retraining triggers"
    
    def create_reference_distribution(self, data_path):
        """Create reference distribution from training data"""
        df = pd.read_csv(data_path)
        
        reference = {}
        for feature in FEATURE_NAMES:
            if feature in df.columns:
                reference[feature] = df[feature].values.tolist()
        
        ref_file = PROCESSED_DATA_PATH / "reference_distribution.pkl"
        with open(ref_file, 'wb') as f:
            pickle.dump(reference, f)
        
        print(f"✅ Reference distribution saved: {ref_file}")
        self.reference_dist = reference

if __name__ == "__main__":
    monitor = ModelMonitor()
    
    # Create reference distribution if needed
    data_file = Path("../data/raw/diabetes.csv")
    if data_file.exists() and not monitor.reference_dist:
        monitor.create_reference_distribution(data_file)
    
    # Example: Simulate monitoring
    print("\n" + "="*70)
    print("MODEL MONITORING SYSTEM")
    print("="*70)
    
    # Check drift
    drift_status = monitor.detect_data_drift()
    print(f"\nData Drift Status: {drift_status}")
    
    # Check performance
    performance = monitor.get_model_performance_summary(days=30)
    print(f"\nPerformance Summary:")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    
    # Check retraining trigger
    should_retrain, reason = monitor.should_trigger_retraining()
    print(f"\nRetrain Required: {should_retrain}")
    print(f"Reason: {reason}")

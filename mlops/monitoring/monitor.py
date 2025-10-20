"""
Model Monitoring System
Tracks predictions, detects drift, and triggers retraining
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from mlops_config import *


class ModelMonitor:
    """Monitor model performance and data drift"""
    
    def __init__(self):
        self.prediction_log_path = PREDICTION_LOG_PATH
        self.reference_dist_path = REFERENCE_DISTRIBUTION_PATH
        
        # Ensure directories exist
        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        DRIFT_REPORT_PATH.mkdir(parents=True, exist_ok=True)
        PERFORMANCE_REPORT_PATH.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(self, features: dict, prediction: int, probability: float, 
                      user_id: str = None, actual: int = None):
        """
        Log a prediction for monitoring
        
        Args:
            features: Input features dictionary
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            user_id: Optional user identifier
            actual: Optional actual outcome (for performance tracking)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': int(prediction),
            'probability': float(probability),
            'user_id': user_id,
            'actual': actual
        }
        
        # Append to JSONL file
        with open(self.prediction_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def load_predictions(self, limit: int = None) -> pd.DataFrame:
        """Load recent predictions from log"""
        if not self.prediction_log_path.exists():
            return pd.DataFrame()
        
        predictions = []
        with open(self.prediction_log_path, 'r') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        if limit:
            predictions = predictions[-limit:]
        
        if not predictions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def load_reference_distribution(self) -> Dict:
        """Load reference distribution for drift detection"""
        if not self.reference_dist_path.exists():
            return None
        
        with open(self.reference_dist_path, 'rb') as f:
            return pickle.load(f)
    
    def save_reference_distribution(self, data: pd.DataFrame):
        """Save reference distribution from training data"""
        reference = {}
        
        for column in data.columns:
            reference[column] = {
                'mean': float(data[column].mean()),
                'std': float(data[column].std()),
                'min': float(data[column].min()),
                'max': float(data[column].max()),
                'percentiles': {
                    '25': float(data[column].quantile(0.25)),
                    '50': float(data[column].quantile(0.50)),
                    '75': float(data[column].quantile(0.75))
                },
                'values': data[column].tolist()
            }
        
        self.reference_dist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reference_dist_path, 'wb') as f:
            pickle.dump(reference, f)
        
        print(f"✅ Reference distribution saved: {self.reference_dist_path}")
    
    def calculate_psi(self, expected: np.array, actual: np.array, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            expected: Reference distribution
            actual: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value (>0.05 indicates drift)
        """
        def scale_range(data, bins):
            min_val, max_val = np.min(data), np.max(data)
            return np.linspace(min_val, max_val, bins + 1)
        
        # Create bins
        breakpoints = scale_range(expected, bins)
        
        # Calculate distributions
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)
        
        return psi
    
    def detect_data_drift(self) -> Dict:
        """
        Detect data drift using PSI
        
        Returns:
            Dictionary with drift detection results
        """
        print("\n" + "="*60)
        print("🔍 DRIFT DETECTION")
        print("="*60)
        
        # Load reference
        reference = self.load_reference_distribution()
        if not reference:
            print("⚠️  No reference distribution found. Run training first.")
            return {'drift_detected': False, 'reason': 'No reference data'}
        
        # Load recent predictions
        predictions_df = self.load_predictions(limit=DRIFT_DETECTION_WINDOW)
        if predictions_df.empty or len(predictions_df) < 50:
            print(f"⚠️  Insufficient data: {len(predictions_df)} predictions (need 50+)")
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Extract features from predictions
        features_list = predictions_df['features'].apply(pd.Series)
        
        # Calculate PSI for each feature
        drift_results = {}
        drift_detected = False
        
        print(f"\n📊 Analyzing {len(predictions_df)} recent predictions...")
        print(f"\nFeature                    PSI      Status")
        print("-" * 50)
        
        for feature in reference.keys():
            if feature not in features_list.columns:
                continue
            
            # Get distributions
            expected = np.array(reference[feature]['values'])
            actual = features_list[feature].dropna().values
            
            if len(actual) < 10:
                continue
            
            # Calculate PSI
            psi = self.calculate_psi(expected, actual)
            drift_results[feature] = {
                'psi': psi,
                'drift': psi > DRIFT_THRESHOLD_PSI
            }
            
            if psi > DRIFT_THRESHOLD_PSI:
                drift_detected = True
                status = "🔴 DRIFT"
            else:
                status = "🟢 OK"
            
            print(f"{feature:25s} {psi:6.4f}   {status}")
        
        result = {
            'drift_detected': drift_detected,
            'features': drift_results,
            'timestamp': datetime.now().isoformat(),
            'samples_analyzed': len(predictions_df)
        }
        
        # Save drift report
        report_file = DRIFT_REPORT_PATH / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        if drift_detected:
            print(f"\n🔴 DRIFT DETECTED! Retraining recommended.")
        else:
            print(f"\n🟢 No drift detected")
        
        return result
    
    def detect_performance_degradation(self, days: int = PERFORMANCE_WINDOW_DAYS) -> Dict:
        """
        Detect performance degradation
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance metrics and degradation status
        """
        print("\n" + "="*60)
        print("📈 PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Load predictions with actual outcomes
        predictions_df = self.load_predictions()
        
        if predictions_df.empty:
            print("⚠️  No predictions logged yet")
            return {'degradation_detected': False, 'reason': 'No data'}
        
        # Filter predictions with actual outcomes
        with_actual = predictions_df[predictions_df['actual'].notna()].copy()
        
        if len(with_actual) < 10:
            print(f"⚠️  Insufficient labeled data: {len(with_actual)} samples (need 10+)")
            return {'degradation_detected': False, 'reason': 'Insufficient labeled data'}
        
        # Filter by time window
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = with_actual[with_actual['timestamp'] >= cutoff_date]
        
        if len(recent) < 5:
            print(f"⚠️  Insufficient recent data: {len(recent)} samples in last {days} days")
            return {'degradation_detected': False, 'reason': 'Insufficient recent data'}
        
        # Calculate metrics
        y_true = recent['actual'].values
        y_pred = recent['prediction'].values
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        current_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'sample_count': len(recent)
        }
        
        print(f"\n📊 Performance (last {days} days, {len(recent)} samples):")
        print(f"   Accuracy:  {current_metrics['accuracy']:.4f}")
        print(f"   Precision: {current_metrics['precision']:.4f}")
        print(f"   Recall:    {current_metrics['recall']:.4f}")
        print(f"   F1-Score:  {current_metrics['f1_score']:.4f}")
        
        # Check degradation
        degradation_detected = (
            current_metrics['accuracy'] < MIN_ACCURACY - PERFORMANCE_DROP_THRESHOLD or
            current_metrics['f1_score'] < MIN_F1_SCORE - PERFORMANCE_DROP_THRESHOLD
        )
        
        result = {
            'degradation_detected': degradation_detected,
            'metrics': current_metrics,
            'timestamp': datetime.now().isoformat(),
            'analysis_period_days': days
        }
        
        if degradation_detected:
            print(f"\n🔴 PERFORMANCE DEGRADATION DETECTED!")
        else:
            print(f"\n🟢 Performance stable")
        
        return result
    
    def should_retrain(self) -> Tuple[bool, List[str]]:
        """
        Determine if model should be retrained
        
        Returns:
            Tuple of (should_retrain, reasons)
        """
        reasons = []
        
        # Check drift
        drift_result = self.detect_data_drift()
        if drift_result.get('drift_detected'):
            reasons.append("Data drift detected")
        
        # Check performance
        perf_result = self.detect_performance_degradation()
        if perf_result.get('degradation_detected'):
            reasons.append("Performance degradation detected")
        
        # Check sample count
        predictions_df = self.load_predictions()
        if not predictions_df.empty and 'actual' in predictions_df.columns:
            new_samples = len(predictions_df[predictions_df['actual'].notna()])
            if new_samples >= MIN_NEW_SAMPLES_FOR_RETRAIN:
                reasons.append(f"{new_samples} new labeled samples available")
        
        should_retrain = len(reasons) > 0
        
        print("\n" + "="*60)
        print("🔄 RETRAINING RECOMMENDATION")
        print("="*60)
        if should_retrain:
            print("✅ Retraining recommended:")
            for reason in reasons:
                print(f"   • {reason}")
        else:
            print("❌ Retraining not needed")
        
        return should_retrain, reasons
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        print("\n" + "="*60)
        print("📋 MONITORING REPORT")
        print("="*60)
        
        predictions_df = self.load_predictions()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_predictions': len(predictions_df),
            'predictions_last_7_days': len(predictions_df[
                predictions_df['timestamp'] >= datetime.now() - timedelta(days=7)
            ]) if not predictions_df.empty else 0,
            'drift_status': self.detect_data_drift(),
            'performance_status': self.detect_performance_degradation(),
        }
        
        should_retrain, reasons = self.should_retrain()
        report['retrain_recommended'] = should_retrain
        report['retrain_reasons'] = reasons
        
        # Save report
        report_file = PERFORMANCE_REPORT_PATH / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_file}")
        
        return report


if __name__ == "__main__":
    monitor = ModelMonitor()
    report = monitor.generate_monitoring_report()

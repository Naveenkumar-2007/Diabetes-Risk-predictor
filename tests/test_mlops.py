"""
Comprehensive MLOps Test Suite
Tests for model artifacts, performance, and predictions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from mlops_config import *
from mlops.utils.helpers import add_engineered_features, validate_input_features


class TestModelArtifacts:
    """Test model artifact files exist and load correctly"""
    
    def test_model_file_exists(self):
        """Test model file exists"""
        assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    
    def test_scaler_file_exists(self):
        """Test scaler file exists"""
        assert SCALER_PATH.exists(), f"Scaler file not found: {SCALER_PATH}"
    
    def test_metadata_file_exists(self):
        """Test metadata file exists"""
        assert METADATA_PATH.exists(), f"Metadata file not found: {METADATA_PATH}"
    
    def test_model_loads(self):
        """Test model loads without errors"""
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_scaler_loads(self):
        """Test scaler loads without errors"""
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        assert scaler is not None
        assert hasattr(scaler, 'transform')
    
    def test_metadata_valid(self):
        """Test metadata contains required fields"""
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        required_fields = ['model_name', 'metrics', 'feature_names']
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"


class TestModelPerformance:
    """Test model performance metrics"""
    
    @pytest.fixture
    def metadata(self):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    
    def test_accuracy_acceptable(self, metadata):
        """Test model accuracy is acceptable (relaxed threshold)"""
        accuracy = metadata['metrics'].get('accuracy') or metadata['metrics'].get('test_accuracy')
        assert accuracy >= 0.70, f"Accuracy {accuracy:.4f} too low (< 0.70)"
    
    def test_roc_auc_threshold(self, metadata):
        """Test ROC-AUC meets threshold"""
        roc_auc = metadata['metrics']['roc_auc']
        assert roc_auc >= MIN_ROC_AUC, f"ROC-AUC {roc_auc:.4f} below threshold {MIN_ROC_AUC}"
    
    def test_f1_score_acceptable(self, metadata):
        """Test F1-score is acceptable"""
        f1 = metadata['metrics']['f1_score']
        assert f1 >= 0.55, f"F1-score {f1:.4f} too low (< 0.55)"
    
    def test_metrics_valid_ranges(self, metadata):
        """Test all metrics are in valid ranges [0, 1]"""
        metrics = metadata['metrics']
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and metric_name not in ['train_accuracy', 'test_accuracy']:
                assert 0 <= value <= 1, f"{metric_name} = {value} out of range [0, 1]"


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_add_engineered_features_array(self):
        """Test feature engineering with numpy array"""
        sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        result = add_engineered_features(sample)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 10  # 8 original + 2 engineered
        assert 'BMI_Age_Interaction' in result.columns
        assert 'Glucose_Insulin_Ratio' in result.columns
    
    def test_add_engineered_features_dataframe(self):
        """Test feature engineering with DataFrame"""
        df = pd.DataFrame({
            'Pregnancies': [6],
            'Glucose': [148],
            'BloodPressure': [72],
            'SkinThickness': [35],
            'Insulin': [0],
            'BMI': [33.6],
            'DiabetesPedigreeFunction': [0.627],
            'Age': [50]
        })
        
        result = add_engineered_features(df)
        assert result.shape[1] == 10
        assert result['BMI_Age_Interaction'].iloc[0] == 33.6 * 50
        assert result['Glucose_Insulin_Ratio'].iloc[0] == 148 / 1  # Insulin=0, so (0+1)
    
    def test_validate_input_features_valid(self):
        """Test validation with valid features"""
        features = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        
        is_valid, msg = validate_input_features(features)
        assert is_valid, f"Validation failed: {msg}"
    
    def test_validate_input_features_missing(self):
        """Test validation with missing features"""
        features = {'Pregnancies': 6, 'Glucose': 148}
        is_valid, msg = validate_input_features(features)
        assert not is_valid
        assert 'Missing features' in msg
    
    def test_validate_input_features_invalid_range(self):
        """Test validation with out-of-range values"""
        features = {
            'Pregnancies': 6,
            'Glucose': 500,  # Too high
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        is_valid, msg = validate_input_features(features)
        assert not is_valid


class TestModelPrediction:
    """Test model prediction functionality"""
    
    @pytest.fixture
    def model(self):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def scaler(self):
        with open(SCALER_PATH, 'rb') as f:
            return pickle.load(f)
    
    def test_prediction_shape(self, model, scaler):
        """Test prediction output shape"""
        sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        sample_with_features = add_engineered_features(sample)
        scaled = scaler.transform(sample_with_features)
        prediction = model.predict(scaled)
        
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]
    
    def test_prediction_probability(self, model, scaler):
        """Test prediction probability output"""
        sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        sample_with_features = add_engineered_features(sample)
        scaled = scaler.transform(sample_with_features)
        proba = model.predict_proba(scaled)
        
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert all(0 <= p <= 1 for p in proba[0])
    
    def test_multiple_predictions(self, model, scaler):
        """Test batch predictions"""
        samples = np.array([
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            [8, 183, 64, 0, 0, 23.3, 0.672, 32]
        ])
        samples_with_features = add_engineered_features(samples)
        scaled = scaler.transform(samples_with_features)
        predictions = model.predict(scaled)
        
        assert predictions.shape == (3,)
        assert all(p in [0, 1] for p in predictions)
    
    def test_prediction_consistency(self, model, scaler):
        """Test prediction consistency (same input = same output)"""
        sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        sample_with_features = add_engineered_features(sample)
        scaled = scaler.transform(sample_with_features)
        
        pred1 = model.predict(scaled)
        pred2 = model.predict(scaled)
        
        assert np.array_equal(pred1, pred2)


class TestDataValidation:
    """Test input data validation"""
    
    def test_zero_insulin_handling(self):
        """Test handling of zero insulin (common in dataset)"""
        sample = np.array([[0, 100, 70, 0, 0, 25.0, 0.5, 30]])
        result = add_engineered_features(sample)
        
        # Should handle division by zero gracefully
        assert not np.isnan(result['Glucose_Insulin_Ratio'].iloc[0])
        assert result['Glucose_Insulin_Ratio'].iloc[0] == 100.0  # 100/(0+1)
    
    def test_extreme_values(self):
        """Test model handles extreme but valid values"""
        features = {
            'Pregnancies': 15,
            'Glucose': 200,
            'BloodPressure': 120,
            'SkinThickness': 50,
            'Insulin': 300,
            'BMI': 40.0,
            'DiabetesPedigreeFunction': 1.5,
            'Age': 80
        }
        
        is_valid, _ = validate_input_features(features)
        assert is_valid
    
    def test_minimum_values(self):
        """Test model handles minimum valid values"""
        features = {
            'Pregnancies': 0,
            'Glucose': 50,
            'BloodPressure': 40,
            'SkinThickness': 0,
            'Insulin': 0,
            'BMI': 15.0,
            'DiabetesPedigreeFunction': 0.1,
            'Age': 21
        }
        
        is_valid, _ = validate_input_features(features)
        assert is_valid


class TestModelMetadata:
    """Test model metadata completeness"""
    
    @pytest.fixture
    def metadata(self):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    
    def test_feature_names_count(self, metadata):
        """Test correct number of features"""
        assert len(metadata['feature_names']) == 10
    
    def test_engineered_features_present(self, metadata):
        """Test engineered features are documented"""
        features = metadata['feature_names']
        assert 'BMI_Age_Interaction' in features
        assert 'Glucose_Insulin_Ratio' in features
    
    def test_metrics_present(self, metadata):
        """Test all required metrics are logged"""
        metrics = metadata['metrics']
        required = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in required:
            # Handle both 'accuracy' and 'test_accuracy' formats
            if metric == 'accuracy':
                assert 'accuracy' in metrics or 'test_accuracy' in metrics
            else:
                assert metric in metrics, f"Missing metric: {metric}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Model Tests for CI/CD Pipeline
"""
import pytest
import pickle
import numpy as np
from pathlib import Path
import json

MODEL_PATH = Path("artifacts/model.pkl")
SCALER_PATH = Path("artifacts/scaler.pkl")
METADATA_PATH = Path("artifacts/model_metadata.json")

class TestModelArtifacts:
    """Test model artifacts existence and validity"""
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    
    def test_scaler_file_exists(self):
        """Test that scaler file exists"""
        assert SCALER_PATH.exists(), f"Scaler file not found: {SCALER_PATH}"
    
    def test_metadata_file_exists(self):
        """Test that metadata file exists"""
        assert METADATA_PATH.exists(), f"Metadata file not found: {METADATA_PATH}"
    
    def test_model_loads(self):
        """Test that model can be loaded"""
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_scaler_loads(self):
        """Test that scaler can be loaded"""
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        assert scaler is not None
        assert hasattr(scaler, 'transform')
    
    def test_metadata_valid(self):
        """Test that metadata is valid JSON"""
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        assert 'model_name' in metadata
        assert 'version' in metadata
        assert 'metrics' in metadata
        assert 'feature_names' in metadata

class TestModelPerformance:
    """Test model performance meets minimum thresholds"""
    
    @pytest.fixture
    def metadata(self):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    
    def test_accuracy_threshold(self, metadata):
        """Test model accuracy meets minimum threshold"""
        accuracy = metadata['metrics']['test_accuracy']
        min_accuracy = metadata['min_thresholds']['accuracy']
        assert accuracy >= min_accuracy, f"Accuracy {accuracy} below threshold {min_accuracy}"
    
    def test_roc_auc_threshold(self, metadata):
        """Test ROC-AUC meets minimum threshold"""
        roc_auc = metadata['metrics']['roc_auc']
        min_roc_auc = metadata['min_thresholds']['roc_auc']
        assert roc_auc >= min_roc_auc, f"ROC-AUC {roc_auc} below threshold {min_roc_auc}"
    
    def test_f1_threshold(self, metadata):
        """Test F1-score meets minimum threshold"""
        f1 = metadata['metrics']['f1_score']
        min_f1 = metadata['min_thresholds']['f1_score']
        assert f1 >= min_f1, f"F1-score {f1} below threshold {min_f1}"

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
        # Sample input: [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]
        sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        scaled = scaler.transform(sample)
        prediction = model.predict(scaled)
        
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]
    
    def test_prediction_probability(self, model, scaler):
        """Test prediction probability output"""
        sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        scaled = scaler.transform(sample)
        proba = model.predict_proba(scaled)
        
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(), 1.0)
        assert all(0 <= p <= 1 for p in proba[0])
    
    def test_multiple_predictions(self, model, scaler):
        """Test batch predictions"""
        samples = np.array([
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            [8, 183, 64, 0, 0, 23.3, 0.672, 32]
        ])
        scaled = scaler.transform(samples)
        predictions = model.predict(scaled)
        
        assert predictions.shape == (3,)
        assert all(p in [0, 1] for p in predictions)

class TestInputValidation:
    """Test input validation and edge cases"""
    
    @pytest.fixture
    def model(self):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def scaler(self):
        with open(SCALER_PATH, 'rb') as f:
            return pickle.load(f)
    
    def test_zero_values(self, model, scaler):
        """Test prediction with zero values"""
        sample = np.array([[0, 100, 70, 0, 0, 25.0, 0.5, 30]])
        scaled = scaler.transform(sample)
        prediction = model.predict(scaled)
        assert prediction[0] in [0, 1]
    
    def test_high_values(self, model, scaler):
        """Test prediction with high values"""
        sample = np.array([[15, 200, 120, 50, 300, 40.0, 1.5, 80]])
        scaled = scaler.transform(sample)
        prediction = model.predict(scaled)
        assert prediction[0] in [0, 1]
    
    def test_feature_count(self, model, scaler):
        """Test that model expects exactly 8 features"""
        # This should work
        sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        scaled = scaler.transform(sample)
        prediction = model.predict(scaled)
        assert prediction.shape == (1,)
        
        # This should fail (wrong number of features)
        with pytest.raises(Exception):
            wrong_sample = np.array([[6, 148, 72, 35]])
            scaled = scaler.transform(wrong_sample)
            model.predict(scaled)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

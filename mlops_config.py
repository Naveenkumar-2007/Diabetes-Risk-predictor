"""
MLOps Configuration for Diabetes Prediction Model
Centralized configuration for model tracking, versioning, and deployment
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
MLFLOW_EXPERIMENT_NAME = "diabetes-prediction-production"
MLFLOW_MODEL_NAME = "diabetes-risk-predictor"

# Model Versioning
MODEL_REGISTRY_PATH = Path("model_registry")
ARTIFACT_PATH = Path("artifacts")
MODEL_VERSION_FILE = ARTIFACT_PATH / "model_version.txt"

# Data Versioning
DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
DATA_VERSION_FILE = DATA_PATH / "data_version.txt"

# Model Performance Thresholds
MIN_ACCURACY = float(os.getenv('MIN_MODEL_ACCURACY', '0.75'))
MIN_ROC_AUC = float(os.getenv('MIN_MODEL_ROC_AUC', '0.80'))
MIN_F1_SCORE = float(os.getenv('MIN_MODEL_F1', '0.70'))

# Model Monitoring
PREDICTION_LOG_PATH = Path("logs/predictions")
DRIFT_DETECTION_WINDOW = int(os.getenv('DRIFT_DETECTION_WINDOW', '1000'))  # Number of predictions
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', '0.05'))  # PSI threshold

# Automated Retraining
AUTO_RETRAIN_ENABLED = os.getenv('AUTO_RETRAIN_ENABLED', 'true').lower() == 'true'
RETRAIN_SCHEDULE = os.getenv('RETRAIN_SCHEDULE', '0 0 * * 0')  # Weekly on Sunday
MIN_NEW_DATA_THRESHOLD = int(os.getenv('MIN_NEW_DATA_THRESHOLD', '100'))  # Min new predictions for retrain

# Feature Store
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Model Metadata
MODEL_DESCRIPTION = "Diabetes Risk Prediction ML Model with XGBoost"
MODEL_TAGS = {
    "domain": "healthcare",
    "task": "classification",
    "algorithm": "ensemble",
    "framework": "scikit-learn+xgboost"
}

# Create necessary directories
MODEL_REGISTRY_PATH.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

print("✅ MLOps Configuration Loaded")

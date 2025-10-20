"""
MLOps Configuration
Central configuration for all MLOps components
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODEL_REGISTRY_DIR = BASE_DIR / "model_registry"

# MLflow configuration
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "diabetes-prediction-production"
MLFLOW_MODEL_NAME = "diabetes-risk-predictor"

# Model paths
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "raw" / "diabetes.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DISTRIBUTION_PATH = PROCESSED_DATA_DIR / "reference_distribution.pkl"

# Monitoring paths
PREDICTION_LOG_PATH = LOGS_DIR / "predictions" / "predictions.jsonl"
DRIFT_REPORT_PATH = LOGS_DIR / "drift_reports"
PERFORMANCE_REPORT_PATH = LOGS_DIR / "performance_reports"

# Performance thresholds
MIN_ACCURACY = 0.72  # Realistic threshold for diabetes prediction
MIN_ROC_AUC = 0.80
MIN_F1_SCORE = 0.65
MIN_PRECISION = 0.60
MIN_RECALL = 0.60

# Drift detection thresholds
DRIFT_THRESHOLD_PSI = 0.05  # Population Stability Index
DRIFT_THRESHOLD_KS = 0.10   # Kolmogorov-Smirnov test
PERFORMANCE_DROP_THRESHOLD = 0.10  # 10% drop triggers alert

# Monitoring windows
DRIFT_DETECTION_WINDOW = 1000  # Number of predictions to analyze
PERFORMANCE_WINDOW_DAYS = 7

# Retraining triggers
MIN_NEW_SAMPLES_FOR_RETRAIN = 100
RETRAIN_ON_DRIFT = True
RETRAIN_ON_PERFORMANCE_DROP = True

# Feature engineering
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

ENGINEERED_FEATURES = [
    'BMI_Age_Interaction',
    'Glucose_Insulin_Ratio'
]

ALL_FEATURES = FEATURE_NAMES + ENGINEERED_FEATURES

# Model training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODEL_CONFIGS = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'random_forest': {
        'n_estimators': 150,
        'max_depth': 5,
        'random_state': RANDOM_STATE
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 4,
        'random_state': RANDOM_STATE
    }
}

# Ensure directories exist
def setup_directories():
    """Create necessary directories for MLOps"""
    dirs = [
        ARTIFACTS_DIR,
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        LOGS_DIR / "predictions",
        LOGS_DIR / "drift_reports",
        LOGS_DIR / "performance_reports",
        MODEL_REGISTRY_DIR,
        BASE_DIR / "tests"
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ MLOps directories created successfully")

if __name__ == "__main__":
    setup_directories()

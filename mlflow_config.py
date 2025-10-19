import os
from pathlib import Path

# Default MLflow tracking URI (file-based by default in ./mlruns)
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:' + str(Path.cwd() / 'mlruns'))

def configure_mlflow():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow

def print_config():
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

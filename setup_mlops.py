"""
Quick Setup Script for MLOps
Run this to initialize MLOps components
"""
import os
import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("🚀 MLOPS SETUP - Diabetes Prediction System")
print("=" * 70)

# Check Python version
python_version = sys.version_info
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("❌ Python 3.8+ required")
    sys.exit(1)

print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

# Create necessary directories
print("\n📁 Creating directories...")
directories = [
    "mlops",
    "logs/predictions",
    "model_registry",
    "data/processed",
    "tests"
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"   ✅ {directory}")

# Initialize DVC
print("\n📦 Initializing DVC...")
try:
    subprocess.run(["dvc", "init"], check=True, capture_output=True)
    print("   ✅ DVC initialized")
except subprocess.CalledProcessError:
    print("   ⚠️  DVC already initialized or not installed")
except FileNotFoundError:
    print("   ⚠️  DVC not found. Install with: pip install dvc")

# Initialize MLflow
print("\n📊 Setting up MLflow...")
mlflow_db = Path("mlflow.db")
if not mlflow_db.exists():
    print("   ✅ MLflow database will be created on first run")
else:
    print("   ✅ MLflow database exists")

# Create reference distribution
print("\n📈 Creating reference distribution for drift detection...")
data_file = Path("data/raw/diabetes.csv")
if data_file.exists():
    try:
        from mlops.model_monitor import ModelMonitor
        monitor = ModelMonitor()
        monitor.create_reference_distribution(data_file)
        print("   ✅ Reference distribution created")
    except Exception as e:
        print(f"   ⚠️  Could not create reference distribution: {e}")
else:
    print(f"   ⚠️  Data file not found: {data_file}")

# Create sample .env if not exists
env_file = Path(".env")
if not env_file.exists():
    print("\n⚙️  Creating sample .env file...")
    sample_env = """
# MLOps Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Model Performance Thresholds
MIN_MODEL_ACCURACY=0.75
MIN_MODEL_ROC_AUC=0.80
MIN_MODEL_F1=0.70

# Drift Detection
DRIFT_DETECTION_WINDOW=1000
DRIFT_THRESHOLD=0.05

# Automated Retraining
AUTO_RETRAIN_ENABLED=true
MIN_NEW_DATA_THRESHOLD=100

# Add your existing environment variables below
"""
    with open(env_file, 'w') as f:
        f.write(sample_env)
    print("   ✅ Sample .env created (update with your values)")
else:
    print("\n⚙️  .env file exists")

# Test imports
print("\n🧪 Testing MLOps imports...")
try:
    import mlflow
    print("   ✅ mlflow")
except ImportError:
    print("   ❌ mlflow - install with: pip install mlflow")

try:
    import scipy
    print("   ✅ scipy")
except ImportError:
    print("   ❌ scipy - install with: pip install scipy")

try:
    import pytest
    print("   ✅ pytest")
except ImportError:
    print("   ❌ pytest - install with: pip install pytest pytest-cov")

print("\n" + "=" * 70)
print("✅ MLOPS SETUP COMPLETE!")
print("=" * 70)

print("\n📋 Next Steps:")
print("1. Update .env with your configuration")
print("2. Train first model: python mlops/model_trainer_mlops.py")
print("3. Start MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("4. View dashboard at: http://localhost:5000")
print("\n📚 Read MLOPS_README.md for detailed documentation")

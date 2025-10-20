"""
MLOps Setup Script
One-command initialization of MLOps infrastructure
"""
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   MLOps Setup for Diabetes Risk Predictor                   ║
║   Production-Ready ML Operations Infrastructure              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"⚠️  {description} warning: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"⚠️  {description} error: {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("STEP 1: Creating Directory Structure")
    print("="*60)
    
    from mlops_config import setup_directories
    setup_directories()
    
    print("\n" + "="*60)
    print("STEP 2: Checking Python Environment")
    print("="*60)
    
    python_version = sys.version.split()[0]
    print(f"✅ Python version: {python_version}")
    
    print("\n" + "="*60)
    print("STEP 3: Installing MLOps Dependencies")
    print("="*60)
    
    print("\n📦 Checking required packages...")
    required_packages = [
        ('mlflow', 'MLflow'),
        ('scipy', 'SciPy'),
        ('pytest', 'Pytest'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('xgboost', 'XGBoost')
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} not found")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
    
    print("\n" + "="*60)
    print("STEP 4: Initializing DVC")
    print("="*60)
    
    if (Path.cwd() / '.dvc').exists():
        print("✅ DVC already initialized")
    else:
        run_command("dvc init", "Initialize DVC")
    
    print("\n" + "="*60)
    print("STEP 5: Creating Reference Distribution")
    print("="*60)
    
    try:
        import pandas as pd
        from mlops_config import TRAIN_DATA_PATH, PROCESSED_DATA_DIR
        from mlops.monitoring.monitor import ModelMonitor
        
        if TRAIN_DATA_PATH.exists():
            df = pd.read_csv(TRAIN_DATA_PATH)
            X = df.drop('Outcome', axis=1)
            
            from mlops.utils.helpers import add_engineered_features
            X_engineered = add_engineered_features(X)
            
            monitor = ModelMonitor()
            monitor.save_reference_distribution(X_engineered)
        else:
            print(f"⚠️  Training data not found: {TRAIN_DATA_PATH}")
    except Exception as e:
        print(f"⚠️  Error creating reference distribution: {str(e)}")
    
    print("\n" + "="*60)
    print("STEP 6: Setting Up MLflow")
    print("="*60)
    
    from mlops_config import MLFLOW_TRACKING_URI
    print(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"   Experiments will be logged to: mlflow.db")
    
    print("\n" + "="*60)
    print("STEP 7: Creating Sample .env File")
    print("="*60)
    
    env_file = Path('.env.mlops.example')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("""# MLOps Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=diabetes-prediction-production
MIN_ACCURACY=0.72
MIN_ROC_AUC=0.80
MIN_F1_SCORE=0.65
DRIFT_THRESHOLD_PSI=0.05
""")
        print(f"✅ Sample .env file created: {env_file}")
    else:
        print(f"✅ Sample .env file exists: {env_file}")
    
    print("\n" + "="*60)
    print("STEP 8: Verification")
    print("="*60)
    
    checks = [
        (Path('mlops/training/trainer.py').exists(), "Training pipeline"),
        (Path('mlops/monitoring/monitor.py').exists(), "Monitoring system"),
        (Path('mlops/retraining/auto_retrain.py').exists(), "Auto-retraining"),
        (Path('mlops/api/endpoints.py').exists(), "API endpoints"),
        (Path('mlops/utils/helpers.py').exists(), "Utilities"),
        (Path('tests/test_mlops.py').exists(), "Test suite"),
        (Path('mlops_config.py').exists(), "Configuration"),
    ]
    
    all_passed = True
    for check, description in checks:
        if check:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            all_passed = False
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    print("""
📋 Next Steps:

1️⃣  Train initial model:
   python mlops/training/trainer.py

2️⃣  Run monitoring check:
   python mlops/monitoring/monitor.py

3️⃣  Run tests:
   pytest tests/test_mlops.py -v

4️⃣  Start MLflow UI:
   mlflow ui --backend-store-uri sqlite:///mlflow.db

5️⃣  Integrate with Flask:
   # Add to flask_app.py:
   from mlops.api.endpoints import mlops_bp
   from mlops.monitoring.monitor import ModelMonitor
   app.register_blueprint(mlops_bp)

📚 Documentation:
   - MLOps folder structure in: mlops/
   - Configuration: mlops_config.py
   - Tests: tests/test_mlops.py
   - API docs: http://localhost:5000/mlops/api/health

Happy MLOps! 🚀
""")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

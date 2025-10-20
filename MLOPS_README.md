# 🚀 MLOps Integration - Complete Guide

## Overview
This diabetes prediction system now includes a **full MLOps pipeline** with automated training, monitoring, drift detection, and CI/CD.

## 🎯 MLOps Features

### 1. **Experiment Tracking with MLflow**
- ✅ Track all model training experiments
- ✅ Compare model performance metrics
- ✅ Version control for models
- ✅ Model registry for production deployment

### 2. **Automated Model Training**
- ✅ Multi-model comparison (XGBoost, Random Forest, Gradient Boosting)
- ✅ Hyperparameter optimization
- ✅ Cross-validation
- ✅ Performance validation against thresholds

### 3. **Model Monitoring & Drift Detection**
- ✅ Real-time prediction logging
- ✅ Data drift detection using PSI (Population Stability Index)
- ✅ Concept drift detection (performance degradation)
- ✅ Automated retraining triggers

### 4. **CI/CD Pipeline with GitHub Actions**
- ✅ Automated model training on data updates
- ✅ Model testing and validation
- ✅ Performance threshold checks
- ✅ Automated deployment to production

### 5. **Data Versioning with DVC**
- ✅ Track data changes
- ✅ Reproducible experiments
- ✅ Data lineage

---

## 📁 Project Structure

```
Diabetics-Agent/
├── mlops/                          # MLOps components
│   ├── model_trainer_mlops.py     # Training pipeline with MLflow
│   ├── model_monitor.py            # Monitoring & drift detection
│   ├── auto_retrain.py             # Automated retraining
│   └── dashboard_api.py            # MLOps dashboard API
├── mlops_config.py                 # MLOps configuration
├── .github/workflows/              # CI/CD pipelines
│   ├── mlops-pipeline.yml         # Model training workflow
│   └── model-monitoring.yml       # Drift detection workflow
├── tests/
│   └── test_model.py              # Model tests for CI
├── artifacts/                      # Model artifacts
│   ├── model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── logs/predictions/               # Prediction logs for monitoring
├── mlflow.db                       # MLflow tracking database
└── .dvc/                          # DVC configuration
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install dvc mlflow pytest
```

### 2. Initialize MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Access at: http://localhost:5000
```

### 3. Train Model with MLOps

```bash
cd mlops
python model_trainer_mlops.py
```

This will:
- Track experiment in MLflow
- Train multiple models
- Select best model
- Save to model registry
- Deploy to production

### 4. Monitor Model Performance

```bash
python mlops/model_monitor.py
```

### 5. Trigger Automated Retraining

```bash
python mlops/auto_retrain.py
```

---

## 📊 MLOps Dashboard

### Start the Flask App with MLOps

```python
# In flask_app.py, add:
from mlops.dashboard_api import mlops_bp
app.register_blueprint(mlops_bp)
```

### Access MLOps APIs

- **Model Info**: `GET /mlops/api/model/info`
- **Performance Metrics**: `GET /mlops/api/monitoring/performance/<days>`
- **Drift Check**: `GET /mlops/api/monitoring/drift/check`
- **Recent Predictions**: `GET /mlops/api/monitoring/predictions/recent/<limit>`
- **Drift Alerts**: `GET /mlops/api/monitoring/alerts/recent`
- **Retrain History**: `GET /mlops/api/retrain/history`
- **Trigger Retrain**: `POST /mlops/api/retrain/trigger`

---

## 🔄 Automated Workflows

### GitHub Actions Workflows

#### 1. **Model Training Pipeline** (`.github/workflows/mlops-pipeline.yml`)

Triggers:
- Weekly (Sunday midnight)
- Manual dispatch
- When data or MLOps code changes

Actions:
- Validate data
- Train models with MLflow
- Run tests
- Check performance thresholds
- Deploy if successful

#### 2. **Model Monitoring** (`.github/workflows/model-monitoring.yml`)

Triggers:
- Daily at 2 AM
- Manual dispatch

Actions:
- Check for data drift
- Check for concept drift
- Trigger retraining if needed

---

## 📈 Model Monitoring

### Data Drift Detection

Uses **PSI (Population Stability Index)**:

```python
from mlops.model_monitor import ModelMonitor

monitor = ModelMonitor()
drift_status = monitor.detect_data_drift()

if drift_status['drift_detected']:
    print("⚠️ Data drift detected!")
    print(f"Affected features: {drift_status['feature_psi']}")
```

**PSI Interpretation**:
- PSI < 0.1: No significant change
- PSI 0.1-0.25: Moderate change
- PSI > 0.25: Significant drift ⚠️

### Concept Drift Detection

Monitors model accuracy over time:

```python
concept_drift = monitor.detect_concept_drift()

if concept_drift['drift_detected']:
    print(f"⚠️ Performance drop: {concept_drift['accuracy_drop']:.2%}")
```

### Automated Retraining Triggers

Retraining is triggered when:
1. **Data drift** detected (PSI > 0.05)
2. **Concept drift** detected (accuracy drop > 10%)
3. **Sufficient new data** (>100 labeled predictions)

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Model Thresholds
MIN_MODEL_ACCURACY=0.75
MIN_MODEL_ROC_AUC=0.80
MIN_MODEL_F1=0.70

# Drift Detection
DRIFT_DETECTION_WINDOW=1000
DRIFT_THRESHOLD=0.05

# Auto Retraining
AUTO_RETRAIN_ENABLED=true
RETRAIN_SCHEDULE="0 0 * * 0"  # Weekly on Sunday
MIN_NEW_DATA_THRESHOLD=100
```

### MLOps Config (`mlops_config.py`)

Centralized configuration for:
- MLflow settings
- Model thresholds
- Drift parameters
- File paths

---

## 🧪 Testing

### Run Model Tests

```bash
pytest tests/test_model.py -v
```

Tests include:
- ✅ Model artifact existence
- ✅ Model loading
- ✅ Performance thresholds
- ✅ Prediction functionality
- ✅ Input validation

### Test Coverage

```bash
pytest tests/ --cov=mlops --cov=src --cov-report=html
```

---

## 📦 Data Versioning with DVC

### Initialize DVC

```bash
dvc init
```

### Track Data

```bash
# Track training data
dvc add data/raw/diabetes.csv

# Track model artifacts
dvc add artifacts/model.pkl
dvc add artifacts/scaler.pkl

# Commit changes
git add data/.gitignore artifacts/.gitignore data/raw/diabetes.csv.dvc
git commit -m "Track data and models with DVC"
```

### Push to Remote Storage

```bash
# Configure remote (Google Drive example)
dvc remote add -d storage gdrive://YOUR_FOLDER_ID

# Push data and models
dvc push
```

### Pull Latest Data/Models

```bash
dvc pull
```

---

## 🔍 MLflow UI

### Start MLflow Server

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

### Access at `http://localhost:5001`

Features:
- 📊 Compare experiments
- 📈 View metrics (accuracy, ROC-AUC, F1)
- 🏷️ Model registry
- 📝 Parameters and artifacts

---

## 🚨 Monitoring Alerts

### Drift Alerts

Logged in `logs/predictions/drift_alerts.jsonl`:

```json
{
  "timestamp": "2025-10-20T12:00:00",
  "alert_type": "data_drift",
  "drift_features": ["Glucose", "BMI"],
  "psi_scores": {"Glucose": 0.15, "BMI": 0.12},
  "action_required": "model_retraining_recommended"
}
```

### Performance Alerts

```json
{
  "timestamp": "2025-10-20T12:00:00",
  "alert_type": "concept_drift",
  "recent_accuracy": 0.72,
  "expected_accuracy": 0.85,
  "accuracy_drop": 0.13,
  "action_required": "immediate_model_retraining"
}
```

---

## 🔄 Production Deployment

### 1. Update Model in Production

```bash
# Train new model
python mlops/model_trainer_mlops.py

# Model automatically saved to artifacts/
# Flask app automatically loads latest model on restart
```

### 2. Restart Flask App

```bash
python flask_app.py
```

### 3. Verify Deployment

```bash
curl http://localhost:5000/mlops/api/model/info
```

---

## 📊 Metrics & KPIs

### Model Performance Metrics

- **Accuracy**: Classification accuracy
- **ROC-AUC**: Area under ROC curve
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Operational Metrics

- **Prediction Volume**: Daily/weekly predictions
- **Prediction Latency**: Response time
- **Model Uptime**: Availability
- **Drift Score**: PSI values per feature

---

## 🐛 Troubleshooting

### MLflow Not Starting

```bash
# Check if port is in use
lsof -i :5000  # or :5001

# Use different port
mlflow ui --port 5002
```

### Drift Detection Failing

```bash
# Create reference distribution
python -c "
from mlops.model_monitor import ModelMonitor
monitor = ModelMonitor()
monitor.create_reference_distribution('data/raw/diabetes.csv')
"
```

### Model Not Loading

```bash
# Check model file
ls -lh artifacts/model.pkl

# Re-train if corrupted
python mlops/model_trainer_mlops.py
```

---

## 📚 Best Practices

1. **Always version your data** with DVC before training
2. **Track all experiments** in MLflow
3. **Set performance thresholds** in mlops_config.py
4. **Monitor drift daily** with GitHub Actions
5. **Test models** before deploying to production
6. **Log all predictions** for monitoring
7. **Retrain regularly** based on drift detection

---

## 🎯 Next Steps

1. ✅ Set up MLflow tracking
2. ✅ Configure DVC for data versioning
3. ✅ Enable GitHub Actions workflows
4. ✅ Set up monitoring dashboard
5. ✅ Configure automated retraining
6. ✅ Deploy to production

---

## 📞 Support

For issues or questions:
- Check logs in `logs/predictions/`
- Review MLflow experiments
- Check GitHub Actions runs
- Review drift alerts

---

## 🏆 MLOps Maturity Level: **Level 3 - Automated**

- ✅ Version Control (Code, Data, Models)
- ✅ Automated Testing
- ✅ CI/CD Pipeline
- ✅ Automated Monitoring
- ✅ Drift Detection
- ✅ Automated Retraining
- ✅ Model Registry
- ✅ Experiment Tracking

**Status**: Production-Ready! 🚀

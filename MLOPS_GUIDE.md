# MLOps Implementation Guide

## 🎯 Overview

This is a production-ready MLOps implementation for the Diabetes Risk Predictor application with:

- ✅ Automated model training with experiment tracking (MLflow)
- ✅ Real-time drift detection and monitoring
- ✅ Automated retraining triggers
- ✅ Comprehensive testing suite
- ✅ RESTful API for monitoring
- ✅ CI/CD pipelines with GitHub Actions

## 📁 Folder Structure

```
Diabetics-Agent/
├── mlops_config.py              # Central configuration
├── setup_mlops.py               # One-command setup script
│
├── mlops/
│   ├── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Model training pipeline
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── monitor.py           # Drift & performance monitoring
│   ├── retraining/
│   │   ├── __init__.py
│   │   └── auto_retrain.py      # Automated retraining
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py         # Flask API endpoints
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Shared utilities
│
├── tests/
│   └── test_mlops.py            # Comprehensive test suite
│
├── .github/workflows/
│   ├── mlops-training.yml       # Training pipeline
│   └── model-monitoring.yml     # Daily monitoring
│
├── artifacts/                   # Model artifacts
│   ├── model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
│
├── logs/
│   ├── predictions/             # Prediction logs
│   ├── drift_reports/           # Drift analysis reports
│   └── performance_reports/     # Performance reports
│
└── data/
    ├── raw/                     # Original data
    └── processed/               # Processed & reference data
```

## 🚀 Quick Start

### 1. Setup MLOps Infrastructure

```bash
python setup_mlops.py
```

This creates all necessary directories, initializes DVC, and sets up the reference distribution.

### 2. Train Initial Model

```bash
python mlops/training/trainer.py
```

Trains multiple models (XGBoost, Random Forest, Gradient Boosting), selects the best, and registers it to MLflow.

### 3. Run Monitoring

```bash
python mlops/monitoring/monitor.py
```

Checks for data drift and performance degradation.

### 4. Run Tests

```bash
pytest tests/test_mlops.py -v
```

Validates model artifacts, performance, and predictions.

### 5. Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Access at: http://localhost:5001

## 🔧 Configuration

Edit `mlops_config.py` to customize:

```python
# Performance thresholds
MIN_ACCURACY = 0.72
MIN_ROC_AUC = 0.80
MIN_F1_SCORE = 0.65

# Drift thresholds
DRIFT_THRESHOLD_PSI = 0.05

# Retraining triggers
MIN_NEW_SAMPLES_FOR_RETRAIN = 100
```

## 📊 Monitoring System

### Drift Detection

Uses **Population Stability Index (PSI)** to detect data drift:

- PSI < 0.05: No drift (🟢)
- PSI 0.05-0.10: Moderate drift (🟡)
- PSI > 0.10: Significant drift (🔴)

### Performance Tracking

Monitors:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Triggers retraining if performance drops > 10%.

### Automated Retraining

Retraining is triggered when:
1. Data drift detected (PSI > 0.05)
2. Performance degradation (> 10% drop)
3. Sufficient new labeled data (> 100 samples)

## 🌐 API Endpoints

Integrate monitoring into your Flask app:

```python
from mlops.api.endpoints import mlops_bp
from mlops.monitoring.monitor import ModelMonitor

# Register blueprint
app.register_blueprint(mlops_bp)

# Initialize monitor
model_monitor = ModelMonitor()

# Log predictions
@app.route('/predict', methods=['POST'])
def predict():
    # ... your prediction code ...
    model_monitor.log_prediction(
        features=features_dict,
        prediction=prediction,
        probability=probability,
        user_id=user_id
    )
```

### Available Endpoints

- `GET /mlops/api/health` - Health check
- `GET /mlops/api/model/info` - Model metadata
- `GET /mlops/api/monitoring/predictions?limit=100` - Recent predictions
- `GET /mlops/api/monitoring/drift` - Drift analysis
- `GET /mlops/api/monitoring/performance?days=7` - Performance metrics
- `GET /mlops/api/monitoring/report` - Full monitoring report
- `GET /mlops/api/retrain/check` - Check retrain conditions
- `POST /mlops/api/retrain/trigger` - Trigger retraining
- `GET /mlops/api/stats/summary` - Statistics summary

## 🧪 Testing

### Run All Tests

```bash
pytest tests/test_mlops.py -v
```

### Test Categories

1. **Artifact Tests**: Verify model files exist and load
2. **Performance Tests**: Check metrics meet thresholds
3. **Feature Engineering Tests**: Validate feature creation
4. **Prediction Tests**: Test model predictions
5. **Validation Tests**: Check input validation

### Coverage Report

```bash
pytest tests/test_mlops.py --cov=mlops --cov-report=html
```

## 🔄 CI/CD Pipelines

### Training Pipeline (Weekly)

Runs every Sunday or on manual trigger:
1. Validates data
2. Trains models
3. Runs tests
4. Checks performance
5. Commits artifacts
6. Pushes to GitHub

### Monitoring Pipeline (Daily)

Runs daily at 2 AM:
1. Checks for drift
2. Analyzes performance
3. Triggers retraining if needed
4. Uploads reports

## 📈 Feature Engineering

The system automatically adds engineered features:

1. **BMI_Age_Interaction**: `BMI × Age`
2. **Glucose_Insulin_Ratio**: `Glucose / (Insulin + 1)`

These features improve model performance and are automatically applied during training and prediction.

## 🔐 Best Practices

### 1. Logging Predictions

Always log predictions with actual outcomes when available:

```python
model_monitor.log_prediction(
    features=features_dict,
    prediction=prediction,
    probability=probability,
    user_id=user_id,
    actual=actual_outcome  # Add when known
)
```

### 2. Regular Monitoring

Check monitoring reports weekly:

```bash
python mlops/monitoring/monitor.py
```

### 3. Model Versioning

MLflow automatically versions models. Access versions:

```bash
mlflow ui
# Navigate to Models > diabetes-risk-predictor
```

### 4. Backup Before Retraining

The system automatically backs up training data before retraining.

### 5. Test After Training

Always run tests after training:

```bash
pytest tests/test_mlops.py -v
```

## 🐛 Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:

```bash
# Ensure you're in the project root
cd /path/to/Diabetics-Agent

# Run scripts from root
python mlops/training/trainer.py
```

### MLflow Database Locked

If MLflow database is locked:

```bash
# Stop MLflow UI
# Delete lock file
rm mlflow.db-shm mlflow.db-wal
```

### No Reference Distribution

If monitoring fails with "No reference distribution":

```bash
python setup_mlops.py
```

### Test Failures

If tests fail due to feature mismatch, ensure model was trained with feature engineering:

```bash
python mlops/training/trainer.py
```

## 📚 Additional Resources

- **MLflow Docs**: https://mlflow.org/docs/latest/
- **DVC Docs**: https://dvc.org/doc
- **Pytest Docs**: https://docs.pytest.org/

## 🤝 Contributing

When adding new features:

1. Update configuration in `mlops_config.py`
2. Add tests in `tests/test_mlops.py`
3. Update this documentation
4. Run full test suite
5. Commit with descriptive message

## 📝 License

This MLOps implementation is part of the Diabetes Risk Predictor project.

---

**Last Updated**: October 20, 2025  
**Version**: 1.0.0  
**Maintainer**: MLOps Team

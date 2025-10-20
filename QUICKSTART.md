# 🚀 MLOps Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Initialize MLOps
```bash
python setup_mlops.py
```

### Step 3: Train Your First Model
```bash
python mlops/model_trainer_mlops.py
```

### Step 4: Start Monitoring
```bash
# Terminal 1: Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# Terminal 2: Start Flask app
python flask_app.py
```

### Step 5: Access Dashboards
- **MLflow UI**: http://localhost:5001
- **Main App**: http://localhost:5000
- **MLOps APIs**: http://localhost:5000/mlops/api/*

---

## 🎯 Common Tasks

### Train New Model
```bash
python mlops/model_trainer_mlops.py
```

### Check for Drift
```bash
python mlops/model_monitor.py
```

### Trigger Retraining
```bash
python mlops/auto_retrain.py
```

### Run Tests
```bash
pytest tests/test_model.py -v
```

### View MLflow Experiments
```bash
mlflow ui
# Access at http://localhost:5000
```

---

## 📊 MLOps APIs

### Get Model Info
```bash
curl http://localhost:5000/mlops/api/model/info
```

### Check Drift
```bash
curl http://localhost:5000/mlops/api/monitoring/drift/check
```

### Get Performance (Last 7 Days)
```bash
curl http://localhost:5000/mlops/api/monitoring/performance/7
```

### Trigger Retraining
```bash
curl -X POST http://localhost:5000/mlops/api/retrain/trigger
```

---

## 🔧 Configuration

Edit `mlops_config.py` or `.env`:

```python
# Model Thresholds
MIN_ACCURACY = 0.75
MIN_ROC_AUC = 0.80
MIN_F1_SCORE = 0.70

# Drift Detection
DRIFT_THRESHOLD = 0.05
DRIFT_DETECTION_WINDOW = 1000

# Auto Retraining
AUTO_RETRAIN_ENABLED = True
MIN_NEW_DATA_THRESHOLD = 100
```

---

## 🐛 Troubleshooting

### MLflow UI won't start
```bash
# Try different port
mlflow ui --port 5002
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Monitoring not working
```bash
# Create reference distribution
python -c "from mlops.model_monitor import ModelMonitor; ModelMonitor().create_reference_distribution('data/raw/diabetes.csv')"
```

---

## 📚 Full Documentation

Read **MLOPS_README.md** for complete documentation.

---

## ✅ Success Checklist

- [x] Dependencies installed
- [x] MLOps initialized (setup_mlops.py)
- [x] Model trained successfully
- [x] MLflow UI accessible
- [x] Flask app running
- [x] Monitoring enabled
- [x] GitHub Actions configured

**Status**: Ready for Production! 🎉

"""
Retrain the diabetes prediction model with proper preprocessing and best algorithms
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn
from pathlib import Path

print("="*70)
print("DIABETES PREDICTION MODEL - RETRAINING WITH BEST PRACTICES")
print("="*70)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv(r"data\raw\diabetes.csv")
print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   ✓ Target distribution: {df['Outcome'].value_counts().to_dict()}")

# Handle missing values (zeros in some columns are actually missing)
print("\n2. Handling missing values...")
# In this dataset, 0 values in these columns are biologically impossible (missing data)
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        print(f"   - {col}: {zero_count} zeros found, replacing with median")
        df[col] = df[col].replace(0, df[col].median())

print("   ✓ Missing values handled")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
print("\n3. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# Feature scaling
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Features scaled (StandardScaler)")

# Ensure artifacts dir exists
Path("artifacts").mkdir(parents=True, exist_ok=True)

# Start MLflow run
mlflow.set_experiment("diabetes-prediction")
with mlflow.start_run(run_name="retrain-model") as run:
    mlflow.log_param("n_samples", df.shape[0])
    mlflow.log_param("n_features", X.shape[1])

# Train multiple models and find the best one
print("\n5. Training and evaluating multiple models...")
print("-"*70)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
}

# Hyperparameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

best_model = None
best_score = 0
best_model_name = ""
results = {}

for name, model in models.items():
    print(f"\n{name}:")
    print(f"   - Grid search with cross-validation...")
    
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_estimator = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_estimator.predict(X_test_scaled)
    y_pred_proba = best_estimator.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': best_estimator,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'best_params': grid_search.best_params_
    }
    
    print(f"   ✓ Best params: {grid_search.best_params_}")
    print(f"   ✓ Accuracy: {accuracy:.4f}")
    print(f"   ✓ ROC-AUC: {roc_auc:.4f}")
    
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = best_estimator
        best_model_name = name

    # Log model cross-validation results for this candidate
    mlflow.log_metric(f"{name}_accuracy", accuracy)
    mlflow.log_metric(f"{name}_roc_auc", roc_auc)

print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name} (ROC-AUC: {best_score:.4f})")
print("="*70)

# Log best model info
mlflow.log_param("best_model", best_model_name)
mlflow.log_metric("best_roc_auc", best_score)

# Detailed evaluation of best model
print("\n6. Detailed evaluation of best model...")
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives:  {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives:  {cm[1][1]}")

# Test with sample cases
print("\n7. Testing with sample cases...")
print("-"*70)

test_cases = [
    {
        'name': 'Healthy Person',
        'data': [0, 85, 60, 20, 80, 22.0, 0.200, 25],
        'expected': 'Low Risk'
    },
    {
        'name': 'Low Risk (from dataset)',
        'data': [1, 85, 66, 29, 0, 26.6, 0.351, 31],
        'expected': 'Low Risk'
    },
    {
        'name': 'High Risk (from dataset)',
        'data': [6, 148, 72, 35, 0, 33.6, 0.627, 50],
        'expected': 'High Risk'
    },
    {
        'name': 'Very High Risk',
        'data': [8, 183, 64, 0, 0, 23.3, 0.672, 32],
        'expected': 'High Risk'
    },
    {
        'name': 'Moderate Case',
        'data': [3, 120, 70, 30, 100, 28.5, 0.400, 35],
        'expected': 'Moderate'
    }
]

for case in test_cases:
    test_input = np.array([case['data']]).reshape(1, -1)
    test_scaled = scaler.transform(test_input)
    
    prediction = best_model.predict(test_scaled)[0]
    probability = best_model.predict_proba(test_scaled)[0]
    
    risk = "High Risk" if prediction == 1 else "Low Risk"
    confidence = max(probability) * 100
    
    print(f"\n{case['name']} (Expected: {case['expected']}):")
    print(f"   Prediction: {risk}")
    print(f"   Confidence: {confidence:.2f}%")
    print(f"   Probabilities: [No Diabetes: {probability[0]:.3f}, Diabetes: {probability[1]:.3f}]")

# Save the best model and scaler
print("\n8. Saving model and scaler...")

with open(r'artifacts\model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"   ✓ Model saved: artifacts/model.pkl")

with open(r'artifacts\scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Scaler saved: artifacts/scaler.pkl")

# Log artifacts and register model with MLflow
try:
    mlflow.log_artifact("artifacts/model_info.txt")
    mlflow.log_artifact("artifacts/model.pkl")
    mlflow.log_artifact("artifacts/scaler.pkl")
    # Save sklearn model using MLflow for easy loading and registry operations
    mlflow.sklearn.log_model(best_model, artifact_path="sklearn-model", registered_model_name="diabetes-prediction-model")
    print("   ✓ Model and artifacts logged to MLflow")
except Exception as e:
    print(f"⚠️ Warning: failed to log model to MLflow: {e}")

# Save model info
with open(r'artifacts\model_info.txt', 'w') as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]):.4f}\n")
    f.write(f"\nBest Parameters:\n")
    f.write(f"{results[best_model_name]['best_params']}\n")
    f.write(f"\nFeature Names:\n")
    f.write(f"{list(X.columns)}\n")
print(f"   ✓ Model info saved: artifacts/model_info.txt")

print("\n" + "="*70)
print("✓ MODEL RETRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nNext steps:")
print("1. The Flask app will now use the new trained model")
print("2. Restart the Flask server if it's running")
print("3. Test predictions with various input values")

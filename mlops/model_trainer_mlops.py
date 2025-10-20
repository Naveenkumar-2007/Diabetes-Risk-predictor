"""
MLOps-Enhanced Model Training Pipeline
Includes experiment tracking, model versioning, and automated deployment
"""
import pandas as pd
import numpy as np
import pickle
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from mlops_config import *

class MLOpsModelTrainer:
    """
    Model training with full MLOps integration:
    - Experiment tracking with MLflow
    - Model versioning and registry
    - Automated model selection
    - Performance validation
    - Automated deployment
    """
    
    def __init__(self, experiment_name=None):
        self.experiment_name = experiment_name or MLFLOW_EXPERIMENT_NAME
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
        print(f"📊 MLflow Experiment: {self.experiment_name}")
        print(f"📍 Tracking URI: {MLFLOW_TRACKING_URI}")
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess training data"""
        print("\n" + "="*70)
        print("DATA LOADING & PREPROCESSING")
        print("="*70)
        
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Log data version
        data_hash = pd.util.hash_pandas_object(df).sum()
        with open(DATA_VERSION_FILE, 'w') as f:
            f.write(f"{data_hash}\n{datetime.now().isoformat()}")
        
        # Handle missing values (zeros are missing in this dataset)
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_columns:
            df[col] = df[col].replace(0, df[col].median())
        
        # Feature engineering
        df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
        df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
        
        print("✅ Data preprocessing completed")
        
        # Save processed data
        processed_file = PROCESSED_DATA_PATH / f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(processed_file, index=False)
        
        return df
    
    def split_and_scale_data(self, df):
        """Split data and apply scaling"""
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
    
    def train_model(self, X_train, y_train, X_test, y_test, model_name, model, params):
        """Train a single model with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("training_date", datetime.now().isoformat())
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_test_pred
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1_score': f1_score(y_test, y_test_pred),
                'roc_auc': roc_auc_score(y_test, y_test_proba)
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            mlflow.log_dict({
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(y_test, y_test_pred, output_dict=True)
            }, "evaluation_results.json")
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"\n📈 {model_name} Results:")
            print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
            
            return model, metrics, mlflow.active_run().info.run_id
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""
        print("\n" + "="*70)
        print("MODEL TRAINING & EVALUATION")
        print("="*70)
        
        models_config = {
            'XGBoost': {
                'model': XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 5
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    class_weight='balanced'
                ),
                'params': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 5
                }
            }
        }
        
        results = {}
        for model_name, config in models_config.items():
            model, metrics, run_id = self.train_model(
                X_train, y_train, X_test, y_test,
                model_name, config['model'], config['params']
            )
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'run_id': run_id
            }
        
        return results
    
    def select_best_model(self, results):
        """Select best model based on performance criteria"""
        print("\n" + "="*70)
        print("MODEL SELECTION")
        print("="*70)
        
        # Rank by F1-score (balanced metric for classification)
        best_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
        best_result = results[best_name]
        
        print(f"🏆 Best Model: {best_name}")
        print(f"   F1-Score: {best_result['metrics']['f1_score']:.4f}")
        print(f"   Accuracy: {best_result['metrics']['test_accuracy']:.4f}")
        print(f"   ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")
        
        # Validate against thresholds
        if best_result['metrics']['test_accuracy'] < MIN_ACCURACY:
            print(f"⚠️  Warning: Accuracy {best_result['metrics']['test_accuracy']:.4f} below threshold {MIN_ACCURACY}")
        if best_result['metrics']['roc_auc'] < MIN_ROC_AUC:
            print(f"⚠️  Warning: ROC-AUC {best_result['metrics']['roc_auc']:.4f} below threshold {MIN_ROC_AUC}")
        if best_result['metrics']['f1_score'] < MIN_F1_SCORE:
            print(f"⚠️  Warning: F1-Score {best_result['metrics']['f1_score']:.4f} below threshold {MIN_F1_SCORE}")
        
        return best_name, best_result
    
    def register_model(self, model_name, run_id, metrics):
        """Register model in MLflow Model Registry"""
        print("\n" + "="*70)
        print("MODEL REGISTRATION")
        print("="*70)
        
        try:
            # Register model
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
            
            # Add version tags
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                MLFLOW_MODEL_NAME,
                model_version.version,
                "validation_status",
                "approved" if metrics['test_accuracy'] >= MIN_ACCURACY else "pending"
            )
            
            print(f"✅ Model registered: {MLFLOW_MODEL_NAME}")
            print(f"   Version: {model_version.version}")
            print(f"   Run ID: {run_id}")
            
            return model_version.version
        except Exception as e:
            print(f"⚠️  Model registration failed: {e}")
            print("   Continuing with local model save...")
            return None
    
    def save_production_model(self, model, scaler, feature_names, model_name, metrics, version=None):
        """Save model and artifacts for production"""
        print("\n" + "="*70)
        print("PRODUCTION MODEL DEPLOYMENT")
        print("="*70)
        
        # Save model
        model_path = ARTIFACT_PATH / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = ARTIFACT_PATH / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'version': version or datetime.now().strftime('%Y%m%d_%H%M%S'),
            'training_date': datetime.now().isoformat(),
            'feature_names': feature_names,
            'metrics': metrics,
            'mlflow_model_name': MLFLOW_MODEL_NAME,
            'min_thresholds': {
                'accuracy': MIN_ACCURACY,
                'roc_auc': MIN_ROC_AUC,
                'f1_score': MIN_F1_SCORE
            }
        }
        
        metadata_path = ARTIFACT_PATH / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
        
        # Update model version file
        with open(MODEL_VERSION_FILE, 'w') as f:
            f.write(f"{metadata['version']}\n{datetime.now().isoformat()}")
        
        print("\n🚀 Model ready for production deployment!")
        
    def run_full_pipeline(self, data_path):
        """Execute complete MLOps training pipeline"""
        print("\n" + "="*70)
        print("🚀 MLOPS TRAINING PIPELINE STARTED")
        print("="*70)
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Load and preprocess data
        df = self.load_and_preprocess_data(data_path)
        
        # 2. Split and scale
        X_train, X_test, y_train, y_test, scaler, feature_names = self.split_and_scale_data(df)
        
        # 3. Train all models
        results = self.train_all_models(X_train, y_train, X_test, y_test)
        
        # 4. Select best model
        best_name, best_result = self.select_best_model(results)
        
        # 5. Register model
        version = self.register_model(best_name, best_result['run_id'], best_result['metrics'])
        
        # 6. Save for production
        self.save_production_model(
            best_result['model'],
            scaler,
            feature_names,
            best_name,
            best_result['metrics'],
            version
        )
        
        print("\n" + "="*70)
        print("✅ MLOPS TRAINING PIPELINE COMPLETED")
        print("="*70)
        print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return best_result

if __name__ == "__main__":
    trainer = MLOpsModelTrainer()
    data_file = "../data/raw/diabetes.csv"
    
    if Path(data_file).exists():
        results = trainer.run_full_pipeline(data_file)
    else:
        print(f"❌ Data file not found: {data_file}")

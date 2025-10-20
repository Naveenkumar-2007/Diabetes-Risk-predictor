"""
Model Training Pipeline with MLflow
Production-ready training with experiment tracking and model registry
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

from mlops_config import *
from mlops.utils.helpers import add_engineered_features, get_feature_importance


class ModelTrainer:
    """Production model training with MLflow tracking"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
        # Setup MLflow
        mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
    def load_and_prepare_data(self):
        """Load data and apply feature engineering"""
        print("📁 Loading data...")
        df = pd.read_csv(TRAIN_DATA_PATH)
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Add engineered features
        print("🔧 Engineering features...")
        X_engineered = add_engineered_features(X)
        
        print(f"✅ Data loaded: {len(df)} samples, {X_engineered.shape[1]} features")
        return X_engineered, y
    
    def preprocess_data(self, X, y):
        """Split and scale data"""
        print("📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, name, model, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model"""
        print(f"\n🔄 Training {name}...")
        
        with mlflow.start_run(run_name=name):
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            
            return metrics
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare"""
        print("\n" + "="*60)
        print("🚀 TRAINING MULTIPLE MODELS")
        print("="*60)
        
        # XGBoost
        xgb = XGBClassifier(**MODEL_CONFIGS['xgboost'])
        self.train_model('XGBoost', xgb, X_train, X_test, y_train, y_test)
        
        # Random Forest
        rf = RandomForestClassifier(**MODEL_CONFIGS['random_forest'])
        self.train_model('Random Forest', rf, X_train, X_test, y_train, y_test)
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(**MODEL_CONFIGS['gradient_boosting'])
        self.train_model('Gradient Boosting', gb, X_train, X_test, y_train, y_test)
    
    def select_best_model(self):
        """Select best model based on F1-score (balanced metric)"""
        print("\n" + "="*60)
        print("🏆 MODEL SELECTION")
        print("="*60)
        
        # Compare models
        comparison = pd.DataFrame(self.results).T
        print(comparison.round(4))
        
        # Select best by F1-score (handles class imbalance better)
        best_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n✅ Best Model: {best_name}")
        print(f"   F1-Score: {self.results[best_name]['f1_score']:.4f}")
        print(f"   ROC-AUC: {self.results[best_name]['roc_auc']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def validate_model(self):
        """Validate model against thresholds"""
        print("\n" + "="*60)
        print("✅ VALIDATION")
        print("="*60)
        
        metrics = self.results[self.best_model_name]
        
        checks = [
            (metrics['accuracy'] >= MIN_ACCURACY, f"Accuracy: {metrics['accuracy']:.4f} (min: {MIN_ACCURACY})"),
            (metrics['roc_auc'] >= MIN_ROC_AUC, f"ROC-AUC: {metrics['roc_auc']:.4f} (min: {MIN_ROC_AUC})"),
            (metrics['f1_score'] >= MIN_F1_SCORE, f"F1-Score: {metrics['f1_score']:.4f} (min: {MIN_F1_SCORE})"),
        ]
        
        all_passed = True
        for passed, msg in checks:
            status = "✅" if passed else "⚠️"
            print(f"{status} {msg}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def save_model(self, X_train):
        """Save model artifacts"""
        print("\n📦 Saving model artifacts...")
        
        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'version': 1,
            'training_date': datetime.now().isoformat(),
            'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else ALL_FEATURES,
            'metrics': self.results[self.best_model_name],
            'thresholds': {
                'min_accuracy': MIN_ACCURACY,
                'min_roc_auc': MIN_ROC_AUC,
                'min_f1_score': MIN_F1_SCORE
            }
        }
        
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to: {MODEL_PATH}")
        print(f"✅ Scaler saved to: {SCALER_PATH}")
        print(f"✅ Metadata saved to: {METADATA_PATH}")
    
    def register_to_mlflow(self):
        """Register best model to MLflow Model Registry"""
        print("\n📝 Registering model to MLflow...")
        
        # Get the latest run
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            filter_string=f"tags.mlflow.runName = '{self.best_model_name}'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs.empty:
            run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{run_id}/model"
            
            # Register model
            mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
            print(f"✅ Model registered: {MLFLOW_MODEL_NAME}")
            print(f"   Run ID: {run_id}")
    
    def run_pipeline(self):
        """Execute full training pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING MLOPS TRAINING PIPELINE")
        print("="*60)
        
        try:
            # Load data
            X, y = self.load_and_prepare_data()
            
            # Preprocess
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
            
            # Train models
            self.train_all_models(X_train, X_test, y_train, y_test)
            
            # Select best
            self.select_best_model()
            
            # Validate
            is_valid = self.validate_model()
            
            # Save
            self.save_model(X)
            
            # Register to MLflow
            self.register_to_mlflow()
            
            print("\n" + "="*60)
            print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_pipeline()

"""
Automated Model Retraining Pipeline
Triggered by drift detection or schedule
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import subprocess

sys.path.append('..')
from mlops_config import *
from mlops.model_monitor import ModelMonitor
from mlops.model_trainer_mlops import MLOpsModelTrainer

class AutoRetrainer:
    """
    Automated model retraining system:
    - Triggered by drift detection
    - Scheduled retraining
    - Data validation before retraining
    - Automated deployment after successful training
    """
    
    def __init__(self):
        self.monitor = ModelMonitor()
        self.trainer = MLOpsModelTrainer()
        self.retrain_log = ARTIFACT_PATH / "retrain_history.jsonl"
        
    def validate_retraining_conditions(self):
        """Validate if retraining should proceed"""
        print("\n" + "="*70)
        print("VALIDATING RETRAINING CONDITIONS")
        print("="*70)
        
        # Check if data file exists
        data_file = DATA_PATH / "diabetes.csv"
        if not data_file.exists():
            return False, f"Data file not found: {data_file}"
        
        # Check for drift or schedule
        should_retrain, reason = self.monitor.should_trigger_retraining()
        
        if not should_retrain:
            return False, reason
        
        print(f"✅ Retraining validated")
        print(f"   Reason: {reason}")
        
        return True, reason
    
    def collect_new_training_data(self):
        """Collect new data from production predictions for retraining"""
        print("\n" + "="*70)
        print("COLLECTING NEW TRAINING DATA")
        print("="*70)
        
        # Read predictions with actual outcomes
        new_data = []
        log_file = self.monitor.prediction_log_file
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                        if pred.get('actual_outcome') is not None:
                            # Add to new training data
                            features = pred['features']
                            features['Outcome'] = pred['actual_outcome']
                            new_data.append(features)
                    except:
                        continue
        
        if new_data:
            # Save new data
            import pandas as pd
            df_new = pd.DataFrame(new_data)
            new_data_file = PROCESSED_DATA_PATH / f"new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_new.to_csv(new_data_file, index=False)
            
            print(f"✅ Collected {len(new_data)} new labeled samples")
            print(f"   Saved to: {new_data_file}")
            
            # Merge with original data
            df_original = pd.read_csv(DATA_PATH / "diabetes.csv")
            df_combined = pd.concat([df_original, df_new], ignore_index=True)
            
            combined_file = PROCESSED_DATA_PATH / f"combined_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_combined.to_csv(combined_file, index=False)
            
            print(f"✅ Combined dataset: {len(df_combined)} total samples")
            
            return combined_file
        else:
            print(f"⚠️  No new labeled data available, using original dataset")
            return DATA_PATH / "diabetes.csv"
    
    def execute_retraining(self, data_file):
        """Execute the retraining pipeline"""
        print("\n" + "="*70)
        print("🚀 EXECUTING AUTOMATED RETRAINING")
        print("="*70)
        
        start_time = datetime.now()
        
        try:
            # Run training pipeline
            result = self.trainer.run_full_pipeline(data_file)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log retraining event
            log_entry = {
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration,
                'data_file': str(data_file),
                'model_name': result['metrics'],
                'success': True,
                'metrics': result['metrics']
            }
            
            with open(self.retrain_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            print(f"\n✅ Retraining completed successfully in {duration:.2f}s")
            
            return True, result
            
        except Exception as e:
            print(f"\n❌ Retraining failed: {e}")
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            
            with open(self.retrain_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            return False, str(e)
    
    def run_automated_retraining(self):
        """Main method to run automated retraining"""
        print("\n" + "="*70)
        print("AUTOMATED MODEL RETRAINING PIPELINE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Validate conditions
        valid, reason = self.validate_retraining_conditions()
        if not valid:
            print(f"\n⏸️  Retraining skipped: {reason}")
            return False
        
        # 2. Collect new data
        data_file = self.collect_new_training_data()
        
        # 3. Execute retraining
        success, result = self.execute_retraining(data_file)
        
        if success:
            print("\n" + "="*70)
            print("✅ AUTOMATED RETRAINING COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"New model deployed and ready for production")
        else:
            print("\n" + "="*70)
            print("❌ AUTOMATED RETRAINING FAILED")
            print("="*70)
            print(f"Error: {result}")
        
        return success

if __name__ == "__main__":
    retrainer = AutoRetrainer()
    retrainer.run_automated_retraining()

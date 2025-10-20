"""
Automated Retraining System
Triggers retraining based on monitoring signals
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from mlops_config import *
from mlops.monitoring.monitor import ModelMonitor
from mlops.training.trainer import ModelTrainer


class AutoRetrainer:
    """Automated model retraining orchestrator"""
    
    def __init__(self):
        self.monitor = ModelMonitor()
        self.trainer = ModelTrainer()
    
    def check_retraining_conditions(self):
        """Check if retraining is needed"""
        print("\n" + "="*60)
        print("🔍 CHECKING RETRAINING CONDITIONS")
        print("="*60)
        
        should_retrain, reasons = self.monitor.should_retrain()
        
        return should_retrain, reasons
    
    def collect_new_training_data(self) -> pd.DataFrame:
        """
        Collect new training data from predictions with actual outcomes
        
        Returns:
            DataFrame with new training samples
        """
        print("\n📊 Collecting new training data...")
        
        # Load predictions with actual outcomes
        predictions_df = self.monitor.load_predictions()
        
        if predictions_df.empty:
            print("⚠️  No predictions available")
            return pd.DataFrame()
        
        # Filter predictions with actual outcomes
        with_actual = predictions_df[predictions_df['actual'].notna()].copy()
        
        if with_actual.empty:
            print("⚠️  No labeled predictions available")
            return pd.DataFrame()
        
        # Extract features and outcomes
        features_df = with_actual['features'].apply(pd.Series)
        features_df['Outcome'] = with_actual['actual'].values
        
        print(f"✅ Collected {len(features_df)} new labeled samples")
        
        return features_df
    
    def merge_with_existing_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing training data"""
        print("\n🔗 Merging with existing training data...")
        
        # Load existing data
        existing_data = pd.read_csv(TRAIN_DATA_PATH)
        print(f"   Existing samples: {len(existing_data)}")
        print(f"   New samples: {len(new_data)}")
        
        # Merge
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates()
        
        print(f"✅ Combined dataset: {len(combined_data)} samples")
        
        return combined_data
    
    def execute_retraining(self, use_new_data: bool = True):
        """
        Execute model retraining
        
        Args:
            use_new_data: Whether to include new labeled predictions
        """
        print("\n" + "="*60)
        print("🔄 STARTING AUTOMATED RETRAINING")
        print("="*60)
        
        try:
            if use_new_data:
                # Collect and merge new data
                new_data = self.collect_new_training_data()
                
                if not new_data.empty and len(new_data) >= 10:
                    combined_data = self.merge_with_existing_data(new_data)
                    
                    # Save updated training data
                    backup_path = TRAIN_DATA_PATH.parent / f"diabetes_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    TRAIN_DATA_PATH.rename(backup_path)
                    print(f"📦 Backup saved: {backup_path}")
                    
                    combined_data.to_csv(TRAIN_DATA_PATH, index=False)
                    print(f"✅ Updated training data saved")
                else:
                    print("⚠️  Insufficient new data, using existing dataset")
            
            # Run training pipeline
            success = self.trainer.run_pipeline()
            
            if success:
                print("\n" + "="*60)
                print("✅ AUTOMATED RETRAINING COMPLETED!")
                print("="*60)
            else:
                print("\n❌ Retraining failed")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Retraining error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, force: bool = False):
        """
        Run auto-retraining workflow
        
        Args:
            force: Force retraining even if conditions not met
        """
        if force:
            print("🔧 FORCED RETRAINING MODE")
            return self.execute_retraining()
        
        # Check conditions
        should_retrain, reasons = self.check_retraining_conditions()
        
        if should_retrain:
            print(f"\n✅ Retraining triggered: {', '.join(reasons)}")
            return self.execute_retraining()
        else:
            print("\n❌ Retraining conditions not met")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Model Retraining')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    args = parser.parse_args()
    
    retrainer = AutoRetrainer()
    retrainer.run(force=args.force)

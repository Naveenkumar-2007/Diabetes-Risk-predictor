"""
MLOps Utilities
Shared helper functions for feature engineering and data processing
"""
import numpy as np
import pandas as pd
from typing import Union

def add_engineered_features(data: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Add engineered features to the dataset
    
    Args:
        data: Input data (numpy array or pandas DataFrame)
        
    Returns:
        DataFrame with original + engineered features
    """
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])
    else:
        df = data.copy()
    
    # Add engineered features
    df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
    
    return df

def prepare_features_for_prediction(features: dict) -> np.ndarray:
    """
    Prepare features from dictionary for model prediction
    
    Args:
        features: Dictionary with feature names as keys
        
    Returns:
        Numpy array with features in correct order including engineered features
    """
    # Extract base features in order
    base_features = [
        features.get('Pregnancies', 0),
        features.get('Glucose', 0),
        features.get('BloodPressure', 0),
        features.get('SkinThickness', 0),
        features.get('Insulin', 0),
        features.get('BMI', 0),
        features.get('DiabetesPedigreeFunction', 0),
        features.get('Age', 0)
    ]
    
    # Create array and add engineered features
    arr = np.array([base_features])
    df = add_engineered_features(arr)
    
    return df.values

def validate_input_features(features: dict) -> tuple[bool, str]:
    """
    Validate input features
    
    Args:
        features: Dictionary with feature names and values
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    # Check all required features present
    missing = [f for f in required_features if f not in features]
    if missing:
        return False, f"Missing features: {', '.join(missing)}"
    
    # Check value ranges
    validations = {
        'Pregnancies': (0, 20, "Pregnancies must be between 0-20"),
        'Glucose': (0, 300, "Glucose must be between 0-300"),
        'BloodPressure': (0, 200, "Blood Pressure must be between 0-200"),
        'SkinThickness': (0, 100, "Skin Thickness must be between 0-100"),
        'Insulin': (0, 900, "Insulin must be between 0-900"),
        'BMI': (0, 70, "BMI must be between 0-70"),
        'DiabetesPedigreeFunction': (0, 3, "DPF must be between 0-3"),
        'Age': (0, 120, "Age must be between 0-120")
    }
    
    for feature, (min_val, max_val, msg) in validations.items():
        value = features.get(feature, 0)
        try:
            value = float(value)
            if not min_val <= value <= max_val:
                return False, msg
        except (ValueError, TypeError):
            return False, f"{feature} must be a number"
    
    return True, ""

def load_latest_model():
    """Load the latest trained model and scaler"""
    import pickle
    from mlops_config import MODEL_PATH, SCALER_PATH
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    return {}

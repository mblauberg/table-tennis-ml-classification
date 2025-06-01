#!/usr/bin/env python3
"""
Generate Partial Dependence Plots for existing LightGBM model
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
from sklearn.inspection import PartialDependenceDisplay

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_data():
    """Load the existing LightGBM model and validation data"""
    
    # Load processed data
    df = pd.read_csv('data/processed/processed_dataset.csv')
    logger.info(f"Loaded {len(df)} rows from data")
    
    # Load validation indices
    with open('splits/val.json', 'r') as f:
        val_indices = json.load(f)
    
    # Identify feature columns (same as training)
    exclude_cols = ['id', 'testmode', 'age', 'playYears', 'height', 'weight']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    
    # Get validation data
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    
    # Load scaler and scale features
    scaler = joblib.load('models/scaler.pkl')
    X_val_scaled = scaler.transform(X_val)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
    
    # Load LightGBM model
    model = lgb.Booster(model_file='models/lgbm.pkl')
    
    logger.info(f"Validation set: {X_val_scaled.shape}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    return model, X_val_scaled, y_val, feature_cols

def create_lgbm_wrapper(model):
    """Create a scikit-learn compatible wrapper for LightGBM"""
    class LGBMWrapper:
        def __init__(self, lgb_model):
            self.model = lgb_model
            
        def predict_proba(self, X):
            return self.model.predict(X, num_iteration=self.model.best_iteration)
            
        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
    return LGBMWrapper(model)

def generate_partial_dependence_plots(model, X_val, feature_names, output_dir):
    """Generate partial dependence plots for top features"""
    logger.info("Generating partial dependence plots...")
    
    # Load feature importance to get top features
    importance_df = pd.read_csv('models/lgbm_feature_importance.csv')
    top_features = importance_df['feature'].head(6).tolist()
    
    logger.info(f"Top features for PDP: {top_features}")
    
    # Create sklearn-compatible wrapper
    wrapped_model = create_lgbm_wrapper(model)
    
    # Sample data for efficiency
    sample_size = min(2000, len(X_val))
    sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
    X_sample = X_val.iloc[sample_indices]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate PDPs for top 6 features
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        try:
            feature_idx = feature_names.index(feature)
            
            logger.info(f"Generating PDP for feature: {feature}")
            
            # Create partial dependence plot
            display = PartialDependenceDisplay.from_estimator(
                wrapped_model,
                X_sample,
                [feature_idx],
                feature_names=feature_names,
                ax=axes[i],
                kind="average"
            )
            axes[i].set_title(f'Partial Dependence: {feature}')
            
        except Exception as e:
            logger.warning(f"Could not create PDP for {feature}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {feature}', ha='center', va='center')
            axes[i].set_title(f'PDP Error: {feature}')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'lgbm_partial_dependence.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Partial dependence plots saved to {plot_path}")

if __name__ == "__main__":
    logger.info("Starting partial dependence plot generation...")
    
    try:
        # Load model and data
        model, X_val_scaled, y_val, feature_names = load_model_and_data()
        
        # Generate PDP plots
        generate_partial_dependence_plots(model, X_val_scaled, feature_names, 'models')
        
        logger.info("Partial dependence plot generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in PDP generation: {e}")
        raise 
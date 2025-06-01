#!/usr/bin/env python3
"""
Generate SHAP visualizations for existing LightGBM model
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import lightgbm as lgb

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

def generate_shap_plots(model, X_val, feature_names, output_dir):
    """Generate SHAP analysis plots"""
    logger.info("Generating SHAP analysis...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Sample for efficiency
    sample_size = min(1000, len(X_val))
    sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
    X_sample = X_val.iloc[sample_indices]
    
    logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    logger.info(f"SHAP values shape: {np.array(shap_values).shape}")
    logger.info(f"Sample data shape: {X_sample.shape}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert SHAP values to numpy arrays for proper handling
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    # For multiclass, shap_values should be (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
    if len(shap_values.shape) == 3:
        if shap_values.shape[0] == 3:  # (n_classes, n_samples, n_features)
            shap_values_list = [shap_values[i] for i in range(3)]
        else:  # (n_samples, n_features, n_classes)
            shap_values_list = [shap_values[:, :, i] for i in range(3)]
    else:
        # Handle as list of arrays
        shap_values_list = shap_values
    
    try:
        # SHAP summary plot for each class
        class_names = ['air swing', 'full power', 'stable']
        for class_idx, class_name in enumerate(class_names):
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_list[class_idx], 
                X_sample, 
                feature_names=feature_names,
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Summary Plot - {class_name}')
            
            plot_path = output_dir / f'shap_summary_class_{class_idx}_{class_name.replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary plot for {class_name} saved to {plot_path}")
        
        # Overall SHAP summary plot (using all classes)
        logger.info("Generating overall SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_list, 
            X_sample, 
            feature_names=feature_names,
            show=False,
            max_display=20,
            class_names=class_names
        )
        plt.title('SHAP Summary Plot - All Classes')
        
        plot_path = output_dir / 'shap_summary_all_classes.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Overall SHAP summary plot saved to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {e}")
        logger.info("Trying simplified approach...")
        
        # Simplified: Just generate bar plots for feature importance
        try:
            for class_idx, class_name in enumerate(class_names):
                plt.figure(figsize=(12, 8))
                
                # Calculate mean absolute SHAP values for this class
                mean_shap = np.abs(shap_values_list[class_idx]).mean(axis=0)
                
                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': mean_shap
                }).sort_values('importance', ascending=False)
                
                # Plot top 20 features
                top_20 = importance_df.head(20)
                plt.barh(range(len(top_20)), top_20['importance'][::-1], alpha=0.7)
                plt.yticks(range(len(top_20)), top_20['feature'][::-1])
                plt.xlabel('Mean |SHAP value|')
                plt.title(f'SHAP Feature Importance - {class_name}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_path = output_dir / f'shap_importance_class_{class_idx}_{class_name.replace(" ", "_")}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP importance plot for {class_name} saved to {plot_path}")
                
        except Exception as e2:
            logger.error(f"Error generating simplified SHAP plots: {e2}")

if __name__ == "__main__":
    logger.info("Starting SHAP visualization generation...")
    
    try:
        # Load model and data
        model, X_val_scaled, y_val, feature_names = load_model_and_data()
        
        # Generate SHAP plots
        generate_shap_plots(model, X_val_scaled, feature_names, 'models')
        
        logger.info("SHAP visualization generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in SHAP generation: {e}")
        raise 
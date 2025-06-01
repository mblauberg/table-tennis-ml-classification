#!/usr/bin/env python3
"""
Generate plots for trained Random Forest model

This script loads a saved Random Forest model and generates:
- Feature importance plot (Gini importance)
- Optuna optimization history
- Confusion matrix
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_and_model(data_path, model_path, scaler_path, train_indices_path, val_indices_path):
    """
    Load data, trained model, and scaler
    
    Args:
        data_path: Path to processed CSV file
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        train_indices_path: Path to training indices JSON
        val_indices_path: Path to validation indices JSON
        
    Returns:
        tuple: (model, scaler, X_train, y_train, X_val, y_val, feature_names)
    """
    logger.info("Loading data, model, and scaler...")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load split indices
    with open(train_indices_path, 'r') as f:
        train_indices = json.load(f)
    with open(val_indices_path, 'r') as f:
        val_indices = json.load(f)
    
    # Identify feature columns
    exclude_cols = ['id', 'date', 'testmode', 'teststage', 'fileindex', 'age', 'playYears', 'height', 'weight']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    
    # Split data
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Scaler loaded from {scaler_path}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    return model, scaler, X_train, y_train, X_val, y_val, feature_cols

def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot feature importance from Random Forest
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    logger.info("Generating feature importance plot...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Select top 20 features
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_importance[::-1], alpha=0.7)
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel('Feature Importance (Gini)')
    plt.title('Random Forest Feature Importance (Top 20)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'rf_feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to {plot_path}")
    
    # Save feature importance data
    importance_data = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    importance_path = Path(output_dir) / 'rf_feature_importance.csv'
    importance_data.to_csv(importance_path, index=False)
    logger.info(f"Feature importance data saved to {importance_path}")

def plot_optuna_optimization_from_study(study_path, output_dir):
    """
    Plot Optuna optimization history from saved study
    
    Args:
        study_path: Path to saved Optuna study
        output_dir: Directory to save plots
    """
    try:
        # Try to load study from file
        study = joblib.load(study_path)
        logger.info(f"Loaded Optuna study from {study_path}")
        
        # Plot optimization history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        trials = study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        plt.plot(values)
        plt.xlabel('Trial')
        plt.ylabel('F1-macro score')
        plt.title('Optimization History')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        best_values = []
        best_so_far = -np.inf
        for value in values:
            if value > best_so_far:
                best_so_far = value
            best_values.append(best_so_far)
        plt.plot(best_values)
        plt.xlabel('Trial')
        plt.ylabel('Best F1-macro score')
        plt.title('Best Score History')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_dir) / 'rf_optuna_optimization.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Optuna optimization plot saved to {plot_path}")
        
    except FileNotFoundError:
        logger.warning(f"Study file not found at {study_path}. Skipping optimization plot.")
    except Exception as e:
        logger.warning(f"Could not load study: {e}. Skipping optimization plot.")

def plot_confusion_matrix(model, scaler, X_val, y_val, output_dir):
    """
    Plot confusion matrix for validation set
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save plots
    """
    logger.info("Generating confusion matrix...")
    
    # Scale features and predict
    X_val_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_val_scaled)
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['air swing', 'full power', 'stable'],
                yticklabels=['air swing', 'full power', 'stable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest - Confusion Matrix')
    
    # Save plot
    plot_path = Path(output_dir) / 'rf_confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")

def plot_oob_score_analysis(model, output_dir):
    """
    Plot out-of-bag score analysis if available
    
    Args:
        model: Trained Random Forest model
        output_dir: Directory to save plots
    """
    if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
        logger.info("Generating OOB score analysis...")
        
        # Create a simple visualization of OOB score
        plt.figure(figsize=(8, 6))
        oob_error = 1 - model.oob_score_
        
        plt.bar(['OOB Score', 'OOB Error'], [model.oob_score_, oob_error], 
                color=['green', 'red'], alpha=0.7)
        plt.ylabel('Score')
        plt.title(f'Random Forest Out-of-Bag Analysis\nOOB Score: {model.oob_score_:.4f}')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = Path(output_dir) / 'rf_oob_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"OOB analysis plot saved to {plot_path}")
    else:
        logger.info("OOB score not available for this model.")

def main():
    parser = argparse.ArgumentParser(description='Generate plots for trained Random Forest model')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--model', required=True, help='Path to saved model (rf.pkl)')
    parser.add_argument('--scaler', required=True, help='Path to saved scaler (scaler.pkl)')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    parser.add_argument('--study', help='Path to saved Optuna study (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting plot generation for Random Forest...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Model file: {args.model}")
    logger.info(f"Scaler file: {args.scaler}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data and model
    model, scaler, X_train, y_train, X_val, y_val, feature_names = load_data_and_model(
        args.data, args.model, args.scaler, args.splits[0], args.splits[1]
    )
    
    # Generate plots
    plot_feature_importance(model, feature_names, output_dir)
    plot_confusion_matrix(model, scaler, X_val, y_val, output_dir)
    plot_oob_score_analysis(model, output_dir)
    
    # Plot Optuna optimization if study path provided
    if args.study:
        plot_optuna_optimization_from_study(args.study, output_dir)
    
    logger.info("Plot generation completed successfully!")

if __name__ == "__main__":
    main() 
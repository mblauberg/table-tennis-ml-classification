#!/usr/bin/env python3
"""
Generate plots for trained Logistic Regression model

This script loads a saved logistic regression model and generates:
- Validation curve for regularization parameter C
- Feature importance plot (coefficients for each class)
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, validation_curve
import joblib

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
        tuple: (model, scaler, X_train, y_train, groups_train, X_val, y_val, feature_names)
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
    groups = df['id']
    
    # Split data
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    groups_train = groups.iloc[train_indices]
    
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Scaler loaded from {scaler_path}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    return model, scaler, X_train, y_train, groups_train, X_val, y_val, feature_cols

def plot_validation_curve(X_train, y_train, groups_train, output_dir):
    """
    Plot validation curve for regularization parameter C
    
    Args:
        X_train: Training features (scaled)
        y_train: Training labels
        groups_train: Group labels for GroupKFold
        output_dir: Directory to save plots
    """
    logger.info("Generating validation curve...")
    
    # Parameter range
    param_range = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    
    # Configure model
    lr = LogisticRegression(
        multi_class='multinomial',
        penalty='l2',
        solver='saga',
        class_weight='balanced',
        max_iter=5000,
        tol=1e-4,
        random_state=42
    )
    
    # Configure GroupKFold
    group_kfold = GroupKFold(n_splits=5)
    
    # Validation curve
    train_scores, val_scores = validation_curve(
        lr, X_train, y_train, 
        param_name='C', param_range=param_range,
        cv=group_kfold, scoring='f1_macro',
        groups=groups_train, n_jobs=-1
    )
    
    # Calculate means and stds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.semilogx(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Regularization parameter C')
    plt.ylabel('F1-macro score')
    plt.title('Validation Curve for Logistic Regression')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = Path(output_dir) / 'lr_validation_curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Validation curve saved to {plot_path}")

def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot feature importance (coefficients) for each class
    
    Args:
        model: Trained logistic regression model
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    logger.info("Generating feature importance plots...")
    
    # Get coefficients
    coef = model.coef_  # Shape: (n_classes, n_features)
    class_names = ['air swing', 'full power', 'stable']
    
    # Plot for each class
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (class_name, ax) in enumerate(zip(class_names, axes)):
        # Get top 15 features by absolute coefficient value
        coef_abs = np.abs(coef[i])
        top_indices = np.argsort(coef_abs)[-15:]
        
        top_features = [feature_names[j] for j in top_indices]
        top_coefs = coef[i][top_indices]
        
        # Plot
        colors = ['red' if c < 0 else 'blue' for c in top_coefs]
        ax.barh(range(len(top_features)), top_coefs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Coefficient value')
        ax.set_title(f'Top Features for {class_name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'lr_feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to {plot_path}")
    
    # Save coefficients data
    coef_data = pd.DataFrame(coef.T, columns=class_names, index=feature_names)
    coef_path = Path(output_dir) / 'lr_coefficients.csv'
    coef_data.to_csv(coef_path)
    logger.info(f"Coefficients data saved to {coef_path}")

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
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['air swing', 'full power', 'stable'],
                yticklabels=['air swing', 'full power', 'stable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Logistic Regression - Confusion Matrix')
    
    # Save plot
    plot_path = Path(output_dir) / 'lr_confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots for trained logistic regression model')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--model', required=True, help='Path to saved model (lr.pkl)')
    parser.add_argument('--scaler', required=True, help='Path to saved scaler (scaler.pkl)')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting plot generation for logistic regression...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Model file: {args.model}")
    logger.info(f"Scaler file: {args.scaler}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data and model
    model, scaler, X_train, y_train, groups_train, X_val, y_val, feature_names = load_data_and_model(
        args.data, args.model, args.scaler, args.splits[0], args.splits[1]
    )
    
    # Scale training data for validation curve
    X_train_scaled = scaler.transform(X_train)
    
    # Generate plots
    plot_validation_curve(X_train_scaled, y_train, groups_train, output_dir)
    plot_feature_importance(model, feature_names, output_dir)
    plot_confusion_matrix(model, scaler, X_val, y_val, output_dir)
    
    logger.info("Plot generation completed successfully!")

if __name__ == "__main__":
    main() 
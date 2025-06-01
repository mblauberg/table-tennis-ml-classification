#!/usr/bin/env python3
"""
Logistic Regression Baseline Model for Table Tennis Swing Classification

This module implements a multinomial logistic regression model with:
- StandardScaler for feature normalization
- Grid search with GroupKFold cross-validation
- Macro-F1 score optimization
- Balanced class weights
- Visualization of validation curves and feature importance
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
from sklearn.model_selection import GridSearchCV, GroupKFold, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_splits(data_path, train_indices_path, val_indices_path):
    """
    Load data and split indices
    
    Args:
        data_path: Path to processed CSV file
        train_indices_path: Path to training indices JSON
        val_indices_path: Path to validation indices JSON
        
    Returns:
        tuple: (X_train, y_train, groups_train, X_val, y_val, groups_val)
    """
    logger.info("Loading data and splits...")
    
    # Load processed data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load split indices
    with open(train_indices_path, 'r') as f:
        train_indices = json.load(f)
    with open(val_indices_path, 'r') as f:
        val_indices = json.load(f)
    
    logger.info(f"Train indices: {len(train_indices)}")
    logger.info(f"Val indices: {len(val_indices)}")
    
    # Identify numeric columns only (exclude categorical and non-feature columns)
    exclude_cols = ['id', 'testmode']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols[:10]}...")  # Show first 10 features
    logger.info(f"Excluded columns: {exclude_cols}")
    
    # Split data
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    groups_train = groups.iloc[train_indices]
    
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    groups_val = groups.iloc[val_indices]
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    
    return X_train, y_train, groups_train, X_val, y_val, groups_val

def prepare_features(X_train, X_val, scaler_path):
    """
    Scale features using StandardScaler
    
    Args:
        X_train: Training features
        X_val: Validation features
        scaler_path: Path to save the scaler
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, scaler)
    """
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    logger.info(f"Feature scaling completed")
    logger.info(f"Training features mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_val_scaled, scaler

def perform_grid_search(X_train, y_train, groups_train):
    """
    Perform grid search with GroupKFold cross-validation
    
    Args:
        X_train: Training features (scaled)
        y_train: Training labels
        groups_train: Group labels for GroupKFold
        
    Returns:
        GridSearchCV: Fitted grid search object
    """
    logger.info("Performing grid search...")
    
    # Define parameter grid
    param_grid = {
        'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]
    }
    
    # Configure logistic regression
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
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=group_kfold,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info(f"Grid search parameters: {param_grid}")
    logger.info("Using 5-fold GroupKFold cross-validation")
    logger.info("Optimizing for macro-F1 score")
    
    # Fit grid search
    grid_search.fit(X_train, y_train, groups=groups_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search

def train_final_model(X_train, y_train, best_params, model_path):
    """
    Train final model with best parameters
    
    Args:
        X_train: Training features (scaled)
        y_train: Training labels
        best_params: Best parameters from grid search
        model_path: Path to save the model
        
    Returns:
        LogisticRegression: Trained model
    """
    logger.info("Training final model...")
    
    # Configure final model
    final_model = LogisticRegression(
        multi_class='multinomial',
        penalty='l2',
        solver='saga',
        class_weight='balanced',
        max_iter=5000,
        tol=1e-4,
        random_state=42,
        **best_params
    )
    
    # Train model
    final_model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return final_model

def evaluate_model(model, X_val, y_val):
    """
    Evaluate model on validation set
    
    Args:
        model: Trained model
        X_val: Validation features (scaled)
        y_val: Validation labels
    """
    logger.info("Evaluating model on validation set...")
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    # Metrics
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_micro = f1_score(y_val, y_pred, average='micro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    
    logger.info(f"Validation F1-macro: {f1_macro:.4f}")
    logger.info(f"Validation F1-micro: {f1_micro:.4f}")
    logger.info(f"Validation F1-weighted: {f1_weighted:.4f}")
    
    # Classification report
    logger.info("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['air swing', 'full power', 'stable']))
    
    return y_pred, y_pred_proba

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

def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model for table tennis swing classification')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save model and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting logistic regression training...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Train split: {args.splits[0]}")
    logger.info(f"Val split: {args.splits[1]}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    X_train, y_train, groups_train, X_val, y_val, groups_val = load_data_splits(
        args.data, args.splits[0], args.splits[1]
    )
    
    # Prepare features
    scaler_path = output_dir / 'scaler.pkl'
    X_train_scaled, X_val_scaled, scaler = prepare_features(X_train, X_val, scaler_path)
    
    # Grid search
    grid_search = perform_grid_search(X_train_scaled, y_train, groups_train)
    
    # Train final model
    model_path = output_dir / 'lr.pkl'
    final_model = train_final_model(X_train_scaled, y_train, grid_search.best_params_, model_path)
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(final_model, X_val_scaled, y_val)
    
    # Generate plots
    plot_validation_curve(X_train_scaled, y_train, groups_train, output_dir)
    plot_feature_importance(final_model, X_train.columns.tolist(), output_dir)
    
    logger.info("Logistic regression training completed successfully!")

if __name__ == "__main__":
    main() 
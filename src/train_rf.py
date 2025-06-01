#!/usr/bin/env python3
"""
Random Forest Training Module for COMP4702 Assignment

Implements Random Forest classifier with hyperparameter optimization
using Optuna and GroupKFold cross-validation.

Week 9 Concepts:
- Ensemble methods and bagging
- Random Forest algorithm
- Hyperparameter optimization
- Feature importance analysis
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import optuna

# Random seed for reproducibility
SEED = 123

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
    
    # Split data
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    groups_train = groups.iloc[train_indices]
    
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    groups_val = groups.iloc[val_indices]
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    
    return X_train, y_train, groups_train, X_val, y_val, groups_val, feature_cols

def prepare_features(X_train, X_val, scaler_path):
    """
    Load existing scaler or create new one if needed
    
    Args:
        X_train: Training features
        X_val: Validation features
        scaler_path: Path to existing scaler
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, scaler)
    """
    logger.info("Preparing features...")
    
    # Try to load existing scaler from logistic regression
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded existing scaler from {scaler_path}")
    except FileNotFoundError:
        logger.info("Creating new scaler...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, scaler_path)
        logger.info(f"New scaler saved to {scaler_path}")
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    logger.info("Feature scaling completed")
    
    return X_train_scaled, X_val_scaled, scaler

def create_rf_objective(X_train, y_train, groups_train):
    """Create Optuna objective function for Random Forest hyperparameter optimization"""
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_categorical('max_depth', [None] + list(range(5, 16))),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.5, 0.7, 1.0]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': SEED,
            'n_jobs': -1
        }
        
        # Create Random Forest model
        rf = RandomForestClassifier(**params)
        
        # 5-fold GroupKFold cross-validation
        group_kfold = GroupKFold(n_splits=5)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(
            rf, X_train, y_train, 
            cv=group_kfold, 
            groups=groups_train,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        # Return mean macro F1 score
        mean_score = cv_scores.mean()
        
        # Log trial results
        trial.set_user_attr('cv_scores', cv_scores.tolist())
        trial.set_user_attr('cv_std', cv_scores.std())
        
        return mean_score
        
    return objective

def train_final_model(X_train, y_train, best_params, model_path):
    """
    Train final Random Forest model with best hyperparameters
    
    Args:
        X_train: Training features (scaled)
        y_train: Training labels
        best_params: Best parameters from Optuna
        model_path: Path to save the model
        
    Returns:
        RandomForestClassifier: Trained model
    """
    logger.info("Training final Random Forest model...")
    
    # Configure final model
    final_model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
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

def plot_optuna_optimization(study, output_dir):
    """
    Plot Optuna optimization history
    
    Args:
        study: Optuna study object
        output_dir: Directory to save plots
    """
    logger.info("Generating Optuna optimization plots...")
    
    # Optimization history
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

def main():
    parser = argparse.ArgumentParser(description='Random Forest training for table tennis classification')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save model and plots')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(SEED)
    
    logger.info("Starting Random Forest training...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Train split: {args.splits[0]}")
    logger.info(f"Val split: {args.splits[1]}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {SEED}")
    logger.info(f"Optuna trials: {args.n_trials}")
    
    # Load data
    X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names = load_data_splits(
        args.data, args.splits[0], args.splits[1]
    )
    
    # Prepare features (reuse scaler from logistic regression)
    scaler_path = output_dir / 'scaler.pkl'
    X_train_scaled, X_val_scaled, scaler = prepare_features(X_train, X_val, scaler_path)
    
    # Hyperparameter optimization with Optuna
    logger.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize', study_name='rf_optimization')
    objective = create_rf_objective(X_train_scaled, y_train, groups_train)
    
    # Optimize
    study.optimize(objective, n_trials=args.n_trials)
    
    # Log best results
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best cross-validation score: {study.best_value:.4f}")
    
    # Train final model
    model_path = output_dir / 'rf.pkl'
    final_model = train_final_model(X_train_scaled, y_train, study.best_params, model_path)
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(final_model, X_val_scaled, y_val)
    
    # Generate plots
    plot_feature_importance(final_model, feature_names, output_dir)
    plot_optuna_optimization(study, output_dir)
    
    logger.info("Random Forest training completed successfully!")

if __name__ == "__main__":
    main() 
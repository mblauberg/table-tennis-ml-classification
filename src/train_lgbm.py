#!/usr/bin/env python3
"""
LightGBM Training Module for COMP4702 Assignment

Implements LightGBM classifier with hyperparameter optimization using Optuna,
SHAP-based interpretability, and partial dependence plots.

Week 10 Concepts:
- Gradient boosting
- LightGBM algorithm
- SHAP interpretability
- Partial dependence plots
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
import lightgbm as lgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.inspection import PartialDependenceDisplay
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
        tuple: (X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names)
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
    
    # Identify feature columns (exclude id, testmode, and categorical columns)
    exclude_cols = ['id', 'testmode']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    
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
    Load existing scaler and scale features
    
    Args:
        X_train: Training features
        X_val: Validation features
        scaler_path: Path to existing scaler
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, scaler)
    """
    logger.info("Preparing features...")
    
    # Load existing scaler
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded existing scaler from {scaler_path}")
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert back to DataFrame for LightGBM
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    logger.info("Feature scaling completed")
    
    return X_train_scaled, X_val_scaled, scaler

def create_lgbm_objective(X_train, y_train, groups_train):
    """Create Optuna objective function for LightGBM hyperparameter optimization"""
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_categorical('max_depth', [-1] + list(range(4, 13))),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'random_state': SEED,
            'verbose': -1,
            'force_col_wise': True
        }
        
        # 5-fold GroupKFold cross-validation
        group_kfold = GroupKFold(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in group_kfold.split(X_train, y_train, groups_train):
            # Split data
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_vl, label=y_vl, reference=train_data)
            
            # Train model with early stopping
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Predict and calculate F1-macro
            y_pred = model.predict(X_vl, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_vl, y_pred_class, average='macro')
            cv_scores.append(f1)
        
        # Return mean macro F1 score
        mean_score = np.mean(cv_scores)
        
        # Log trial results
        trial.set_user_attr('cv_scores', cv_scores)
        trial.set_user_attr('cv_std', np.std(cv_scores))
        
        return mean_score
        
    return objective

def train_final_model(X_train, y_train, X_val, y_val, best_params, model_path):
    """
    Train final LightGBM model with best hyperparameters
    
    Args:
        X_train: Training features (scaled)
        y_train: Training labels
        X_val: Validation features (for early stopping)
        y_val: Validation labels
        best_params: Best parameters from Optuna
        model_path: Path to save the model
        
    Returns:
        LightGBM model: Trained model
    """
    logger.info("Training final LightGBM model...")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Configure final model parameters
    final_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'verbose': -1,
        'force_col_wise': True,
        **best_params
    }
    
    # Train model with early stopping
    model = lgb.train(
        final_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Save model
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluate model on validation set
    
    Args:
        model: Trained LightGBM model
        X_val: Validation features (scaled)
        y_val: Validation labels
    """
    logger.info("Evaluating model on validation set...")
    
    # Predictions
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
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
    Plot LightGBM feature importance
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    logger.info("Generating LightGBM feature importance plot...")
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'][::-1], alpha=0.7)
    plt.yticks(range(len(top_20)), top_20['feature'][::-1])
    plt.xlabel('Feature Importance (Gain)')
    plt.title('LightGBM Feature Importance (Top 20)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'lgbm_feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to {plot_path}")
    
    # Save importance data
    importance_path = Path(output_dir) / 'lgbm_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance data saved to {importance_path}")
    
    return importance_df

def plot_shap_analysis(model, X_val, feature_names, output_dir):
    """
    Generate SHAP analysis plots
    
    Args:
        model: Trained LightGBM model
        X_val: Validation features (scaled)
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    logger.info("Generating SHAP analysis...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (sample for efficiency)
    sample_size = min(1000, len(X_val))
    sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
    X_sample = X_val.iloc[sample_indices]
    
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP summary plot for each class
    for class_idx, class_name in enumerate(['air swing', 'full power', 'stable']):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values[class_idx], 
            X_sample, 
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        plt.title(f'SHAP Summary Plot - {class_name}')
        
        plot_path = Path(output_dir) / f'shap_summary_class_{class_idx}_{class_name.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary plot for {class_name} saved to {plot_path}")
    
    # Overall SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.title('SHAP Summary Plot - All Classes')
    
    plot_path = Path(output_dir) / 'shap_summary_all_classes.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Overall SHAP summary plot saved to {plot_path}")
    
    return shap_values

def plot_partial_dependence(model, X_val, feature_names, top_features, output_dir):
    """
    Generate partial dependence plots for top features
    
    Args:
        model: Trained LightGBM model
        X_val: Validation features
        feature_names: List of feature names
        top_features: List of top feature names
        output_dir: Directory to save plots
    """
    logger.info("Generating partial dependence plots...")
    
    # Create a wrapper for the LightGBM model to be sklearn-compatible
    class LGBMWrapper:
        def __init__(self, model):
            self.model = model
        
        def predict_proba(self, X):
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        
        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
    wrapped_model = LGBMWrapper(model)
    
    # Sample data for efficiency
    sample_size = min(2000, len(X_val))
    sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
    X_sample = X_val.iloc[sample_indices]
    
    # Select top 6 features for PDP
    top_6_features = top_features[:6]
    
    # Create partial dependence plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_6_features):
        try:
            feature_idx = feature_names.index(feature)
            
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
    plot_path = Path(output_dir) / 'lgbm_partial_dependence.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Partial dependence plots saved to {plot_path}")

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
    plot_path = Path(output_dir) / 'lgbm_optuna_optimization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Optuna optimization plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='LightGBM training for table tennis classification')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save model and plots')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(SEED)
    
    logger.info("Starting LightGBM training...")
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
    
    # Prepare features (reuse scaler from previous models)
    scaler_path = output_dir / 'scaler.pkl'
    X_train_scaled, X_val_scaled, scaler = prepare_features(X_train, X_val, scaler_path)
    
    # Hyperparameter optimization with Optuna
    logger.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize', study_name='lgbm_optimization')
    objective = create_lgbm_objective(X_train_scaled, y_train, groups_train)
    
    # Optimize
    study.optimize(objective, n_trials=args.n_trials)
    
    # Log best results
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best cross-validation score: {study.best_value:.4f}")
    
    # Train final model
    model_path = output_dir / 'lgbm.pkl'
    final_model = train_final_model(X_train_scaled, y_train, X_val_scaled, y_val, study.best_params, model_path)
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(final_model, X_val_scaled, y_val)
    
    # Generate feature importance plot
    importance_df = plot_feature_importance(final_model, feature_names, output_dir)
    
    # Generate SHAP analysis
    shap_values = plot_shap_analysis(final_model, X_val_scaled, feature_names, output_dir)
    
    # Generate partial dependence plots for top features
    top_features = importance_df['feature'].head(6).tolist()
    plot_partial_dependence(final_model, X_val_scaled, feature_names, top_features, output_dir)
    
    # Generate Optuna optimization plots
    plot_optuna_optimization(study, output_dir)
    
    logger.info("LightGBM training completed successfully!")

if __name__ == "__main__":
    main() 
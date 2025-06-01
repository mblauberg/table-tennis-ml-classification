#!/usr/bin/env python3
"""
Generate plots for trained LightGBM model

This script loads a saved LightGBM model and generates:
- Feature importance plot (gain-based)
- SHAP analysis plots
- Partial dependence plots
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
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
import joblib
import shap
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

def plot_shap_analysis(model, scaler, X_val, feature_names, output_dir):
    """
    Generate SHAP analysis plots
    
    Args:
        model: Trained LightGBM model
        scaler: Fitted scaler
        X_val: Validation features
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    logger.info("Generating SHAP analysis...")
    
    # Scale features
    X_val_scaled = scaler.transform(X_val)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val.index)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values (sample for efficiency)
    sample_size = min(1000, len(X_val_scaled_df))
    sample_indices = np.random.choice(len(X_val_scaled_df), sample_size, replace=False)
    X_sample = X_val_scaled_df.iloc[sample_indices]
    
    logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
    shap_values = explainer.shap_values(X_sample)
    
    # Handle SHAP values structure for multiclass
    if isinstance(shap_values, list):
        shap_values_array = np.array(shap_values)
    else:
        shap_values_array = shap_values
    
    # Ensure correct shape: (n_classes, n_samples, n_features)
    if len(shap_values_array.shape) == 3:
        if shap_values_array.shape[0] == 3:  # Already correct
            shap_values_list = [shap_values_array[i] for i in range(3)]
        else:  # (n_samples, n_features, n_classes) -> transpose
            shap_values_list = [shap_values_array[:, :, i] for i in range(3)]
    else:
        shap_values_list = shap_values
    
    class_names = ['air swing', 'full power', 'stable']
    
    try:
        # SHAP summary plot for each class
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
            
            plot_path = Path(output_dir) / f'shap_summary_class_{class_idx}_{class_name.replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary plot for {class_name} saved to {plot_path}")
        
        # Overall SHAP summary plot
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
        
        plot_path = Path(output_dir) / 'shap_summary_all_classes.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Overall SHAP summary plot saved to {plot_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate standard SHAP plots: {e}. Generating simplified plots...")
        
        # Generate simplified bar plots for feature importance
        for class_idx, class_name in enumerate(class_names):
            plt.figure(figsize=(12, 8))
            
            # Calculate mean absolute SHAP values for this class
            mean_shap = np.abs(shap_values_list[class_idx]).mean(axis=0)
            
            # Create feature importance DataFrame
            shap_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            top_20 = shap_importance_df.head(20)
            plt.barh(range(len(top_20)), top_20['importance'][::-1], alpha=0.7)
            plt.yticks(range(len(top_20)), top_20['feature'][::-1])
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'SHAP Feature Importance - {class_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = Path(output_dir) / f'shap_importance_class_{class_idx}_{class_name.replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP importance plot for {class_name} saved to {plot_path}")
    
    return shap_values_list

def plot_partial_dependence(model, scaler, X_val, feature_names, top_features, output_dir):
    """
    Generate partial dependence plots for top features
    
    Args:
        model: Trained LightGBM model
        scaler: Fitted scaler
        X_val: Validation features
        feature_names: List of feature names
        top_features: List of top feature names
        output_dir: Directory to save plots
    """
    logger.info("Generating partial dependence plots...")
    
    # Scale features
    X_val_scaled = scaler.transform(X_val)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val.index)
    
    # Create a wrapper for the LightGBM model to be sklearn-compatible
    class LGBMWrapper:
        def __init__(self, model):
            self.model = model
        
        def predict_proba(self, X):
            return self.model.predict(X)
        
        def predict(self, X):
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
    
    wrapped_model = LGBMWrapper(model)
    
    # Sample data for efficiency
    sample_size = min(2000, len(X_val_scaled_df))
    sample_indices = np.random.choice(len(X_val_scaled_df), sample_size, replace=False)
    X_sample = X_val_scaled_df.iloc[sample_indices]
    
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
        plot_path = Path(output_dir) / 'lgbm_optuna_optimization.png'
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
    y_pred_proba = model.predict(X_val_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['air swing', 'full power', 'stable'],
                yticklabels=['air swing', 'full power', 'stable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('LightGBM - Confusion Matrix')
    
    # Save plot
    plot_path = Path(output_dir) / 'lgbm_confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")

def plot_learning_curves(model, output_dir):
    """
    Plot learning curves if available in model
    
    Args:
        model: Trained LightGBM model
        output_dir: Directory to save plots
    """
    try:
        # Check if model has evaluation results
        if hasattr(model, 'evals_result_'):
            evals_result = model.evals_result_
            
            if evals_result:
                logger.info("Generating learning curves...")
                
                plt.figure(figsize=(12, 5))
                
                # Plot training and validation curves for each metric
                metrics = list(evals_result[list(evals_result.keys())[0]].keys())
                
                for i, metric in enumerate(metrics):
                    plt.subplot(1, len(metrics), i+1)
                    
                    for dataset_name, results in evals_result.items():
                        if metric in results:
                            plt.plot(results[metric], label=f'{dataset_name}')
                    
                    plt.xlabel('Boosting Round')
                    plt.ylabel(metric)
                    plt.title(f'Learning Curve - {metric}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = Path(output_dir) / 'lgbm_learning_curves.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Learning curves saved to {plot_path}")
            else:
                logger.info("No evaluation results found in model.")
        else:
            logger.info("Model does not have evaluation results.")
            
    except Exception as e:
        logger.warning(f"Could not generate learning curves: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots for trained LightGBM model')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--model', required=True, help='Path to saved model (lgbm.pkl)')
    parser.add_argument('--scaler', required=True, help='Path to saved scaler (scaler.pkl)')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    parser.add_argument('--study', help='Path to saved Optuna study (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting plot generation for LightGBM...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Model file: {args.model}")
    logger.info(f"Scaler file: {args.scaler}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data and model
    model, scaler, X_train, y_train, X_val, y_val, feature_names = load_data_and_model(
        args.data, args.model, args.scaler, args.splits[0], args.splits[1]
    )
    
    # Generate feature importance plot
    importance_df = plot_feature_importance(model, feature_names, output_dir)
    top_features = importance_df['feature'].head(10).tolist()
    
    # Generate SHAP analysis
    plot_shap_analysis(model, scaler, X_val, feature_names, output_dir)
    
    # Generate partial dependence plots
    plot_partial_dependence(model, scaler, X_val, feature_names, top_features, output_dir)
    
    # Generate confusion matrix
    plot_confusion_matrix(model, scaler, X_val, y_val, output_dir)
    
    # Generate learning curves
    plot_learning_curves(model, output_dir)
    
    # Plot Optuna optimization if study path provided
    if args.study:
        plot_optuna_optimization_from_study(args.study, output_dir)
    
    logger.info("Plot generation completed successfully!")

if __name__ == "__main__":
    main() 
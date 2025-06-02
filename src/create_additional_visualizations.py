#!/usr/bin/env python3
"""
Additional Visualization Creator for COMP4702 Assignment

This script demonstrates how to load the saved training data and model results
to create additional visualizations beyond what's provided in the main training scripts.

Usage:
    python src/create_additional_visualizations.py --model rf|lgbm|lr [--output custom_plots]

Available visualizations:
1. Hyperparameter optimization plots
2. Prediction confidence analysis
3. Class probability distributions
4. Custom learning curves
5. Error analysis plots
6. Model comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_model_data(model_type='rf'):
    """
    Load all saved data for a specific model type
    
    Args:
        model_type: 'rf' for Random Forest, 'lgbm' for LightGBM, or 'lr' for Logistic Regression
        
    Returns:
        dict: All loaded data components
    """
    if model_type == 'rf':
        model_dir = Path('results/random_forest')
    elif model_type == 'lgbm':
        model_dir = Path('results/lightgbm')
    elif model_type == 'lr':
        model_dir = Path('results/logistic_regression')
    else:
        raise ValueError("model_type must be 'rf', 'lgbm', or 'lr'")
    
    print(f"Loading {model_type.upper()} data from {model_dir}")
    
    # Load training data
    training_data = np.load(model_dir / 'training_data.npz', allow_pickle=True)
    print(f"✓ Training data loaded: {training_data.files}")
    
    # Load optimization results
    with open(model_dir / 'optimization_study.json', 'r') as f:
        study_data = json.load(f)
    
    if model_type == 'lr':
        print(f"✓ Grid search results loaded: {len(study_data['cv_results']['param_C'])} parameter values")
    else:
        print(f"✓ Optimization study loaded: {len(study_data['trials'])} trials")
    
    # Load predictions
    predictions = np.load(model_dir / 'predictions.npz')
    print(f"✓ Predictions loaded: {predictions.files}")
    
    # Load model
    if model_type == 'rf':
        model = joblib.load(model_dir / 'random_forest.pkl')
        with open(model_dir / 'tree_details.json', 'r') as f:
            model_details = json.load(f)
    elif model_type == 'lgbm':
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_dir / 'lightgbm.pkl'))
        with open(model_dir / 'boosting_details.json', 'r') as f:
            model_details = json.load(f)
    else:  # lr
        model = joblib.load(model_dir / 'logistic_model.pkl')
        with open(model_dir / 'model_details.json', 'r') as f:
            model_details = json.load(f)
    
    print(f"✓ Model and details loaded")
    
    return {
        'training_data': training_data,
        'study_data': study_data,
        'predictions': predictions,
        'model': model,
        'model_details': model_details,
        'model_type': model_type,
        'model_dir': model_dir
    }

def plot_hyperparameter_optimization(data, output_dir):
    """Create detailed hyperparameter optimization plots"""
    study_data = data['study_data']
    model_type = data['model_type']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_type.upper()} Hyperparameter Optimization Analysis', fontsize=16)
    
    if model_type == 'lr':
        # Handle GridSearchCV results for Logistic Regression
        cv_results = study_data['cv_results']
        param_C = cv_results['param_C']
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        
        # 1. Optimization progress (parameter sweep)
        axes[0, 0].errorbar(range(len(param_C)), mean_scores, yerr=std_scores, 
                           marker='o', capsize=5, alpha=0.7)
        axes[0, 0].axhline(y=study_data['best_score'], color='red', linestyle='--', 
                          label=f'Best: {study_data["best_score"]:.4f}')
        axes[0, 0].set_xlabel('Parameter Index')
        axes[0, 0].set_ylabel('F1-Macro Score')
        axes[0, 0].set_title('Grid Search Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Regularization parameter vs performance
        axes[0, 1].semilogx(param_C, mean_scores, 'o-', alpha=0.7, color='blue')
        axes[0, 1].fill_between(param_C, 
                               np.array(mean_scores) - np.array(std_scores),
                               np.array(mean_scores) + np.array(std_scores), 
                               alpha=0.2, color='blue')
        axes[0, 1].axvline(x=study_data['best_params']['C'], color='red', linestyle='--',
                          label=f'Best C: {study_data["best_params"]["C"]}')
        axes[0, 1].set_xlabel('Regularization Parameter C')
        axes[0, 1].set_ylabel('F1-Macro Score')
        axes[0, 1].set_title('Bias-Variance Tradeoff')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Score distribution
        axes[1, 0].hist(mean_scores, bins=min(10, len(mean_scores)), alpha=0.7, color='green')
        axes[1, 0].axvline(x=study_data['best_score'], color='red', linestyle='--', 
                          label=f'Best: {study_data["best_score"]:.4f}')
        axes[1, 0].set_xlabel('F1-Macro Score')
        axes[1, 0].set_ylabel('Number of Parameters')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Best parameters visualization
        axes[1, 1].text(0.5, 0.5, f"Logistic Regression Optimization:\n\n"
                                 f"• Best C: {study_data['best_params']['C']}\n"
                                 f"• Best Score: {study_data['best_score']:.4f}\n"
                                 f"• CV Std: {cv_results['std_test_score'][study_data['best_index']]:.4f}\n"
                                 f"• Parameters Tested: {len(param_C)}\n"
                                 f"• CV Folds: {study_data['n_splits']}",
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_title('Optimization Summary')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
    else:
        # Handle Optuna trials for RF/LightGBM (existing code)
        trials_df = pd.DataFrame([trial for trial in study_data['trials'] if trial['state'] == 'COMPLETE'])
        
        # 1. Optimization progress
        axes[0, 0].plot(range(len(trials_df)), trials_df['value'], 'o-', alpha=0.7)
        axes[0, 0].axhline(y=study_data['best_value'], color='red', linestyle='--', 
                          label=f'Best: {study_data["best_value"]:.4f}')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('F1-Macro Score')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Parameter importance (if multiple trials)
        if len(trials_df) > 1:
            param_cols = [col for col in trials_df.columns if col.startswith('params.')]
            if param_cols:
                param_data = pd.DataFrame([trial['params'] for trial in study_data['trials']])
                correlations = param_data.corrwith(trials_df['value']).abs().sort_values(ascending=True)
                
                correlations.plot(kind='barh', ax=axes[0, 1], color='skyblue', alpha=0.7)
                axes[0, 1].set_xlabel('Absolute Correlation with Performance')
                axes[0, 1].set_title('Parameter Importance')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Score distribution
        axes[1, 0].hist(trials_df['value'], bins=min(20, len(trials_df)), alpha=0.7, color='green')
        axes[1, 0].axvline(x=study_data['best_value'], color='red', linestyle='--', 
                          label=f'Best: {study_data["best_value"]:.4f}')
        axes[1, 0].set_xlabel('F1-Macro Score')
        axes[1, 0].set_ylabel('Number of Trials')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Best parameters visualization
        best_params = study_data['best_params']
        param_names = list(best_params.keys())[:6]  # Show top 6 parameters
        param_values = [best_params[name] for name in param_names]
        
        # Normalize values for better visualization
        normalized_values = []
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            if isinstance(value, (int, float)):
                normalized_values.append(value)
            else:
                normalized_values.append(i + 1)  # For categorical values
        
        axes[1, 1].barh(range(len(param_names)), normalized_values, alpha=0.7, color='orange')
        axes[1, 1].set_yticks(range(len(param_names)))
        axes[1, 1].set_yticklabels([f"{name}: {best_params[name]}" for name in param_names])
        axes[1, 1].set_xlabel('Parameter Value (normalized)')
        axes[1, 1].set_title('Best Parameters')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'{model_type}_hyperparameter_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Hyperparameter analysis saved: {plot_path}")

def plot_prediction_analysis(data, output_dir):
    """Create detailed prediction analysis plots"""
    predictions = data['predictions']
    model_type = data['model_type']
    
    y_true = predictions['y_true']
    y_pred = predictions['y_pred']
    y_pred_proba = predictions['y_pred_proba']
    
    class_names = ['Air Swing', 'Full Power', 'Stable']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_type.upper()} Prediction Analysis', fontsize=16)
    
    # 1. Confusion matrix with percentages
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title('Confusion Matrix (%)')
    
    # 2. Prediction confidence by class
    max_probs = np.max(y_pred_proba, axis=1)
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_confidences = max_probs[class_mask]
        axes[0, 1].hist(class_confidences, bins=20, alpha=0.7, label=class_name)
    
    axes[0, 1].set_xlabel('Maximum Prediction Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Prediction Confidence by True Class')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Probability distribution for each class
    for i, class_name in enumerate(class_names):
        axes[1, 0].hist(y_pred_proba[:, i], bins=30, alpha=0.6, label=f'{class_name} prob')
    
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Probability Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Correct vs Incorrect predictions by confidence
    correct = (y_true == y_pred)
    
    axes[1, 1].scatter(max_probs[correct], np.ones(np.sum(correct)), 
                      alpha=0.6, label='Correct', color='green')
    axes[1, 1].scatter(max_probs[~correct], np.zeros(np.sum(~correct)), 
                      alpha=0.6, label='Incorrect', color='red')
    
    axes[1, 1].set_xlabel('Maximum Prediction Probability')
    axes[1, 1].set_ylabel('Prediction Correctness')
    axes[1, 1].set_title('Accuracy vs Confidence')
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Incorrect', 'Correct'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'{model_type}_prediction_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Prediction analysis saved: {plot_path}")

def plot_model_specific_analysis(data, output_dir):
    """Create model-specific analysis plots"""
    model_type = data['model_type']
    model_details = data['model_details']
    
    if model_type == 'rf':
        # Random Forest specific plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest Specific Analysis', fontsize=16)
        
        # Tree depth distribution
        tree_depths = model_details['tree_depths']
        axes[0, 0].hist(tree_depths, bins=max(1, len(set(tree_depths))), alpha=0.7, color='forestgreen')
        axes[0, 0].set_xlabel('Tree Depth')
        axes[0, 0].set_ylabel('Number of Trees')
        axes[0, 0].set_title('Tree Depth Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # OOB score visualization
        if model_details['oob_score']:
            axes[0, 1].bar(['OOB Score'], [model_details['oob_score']], color='blue', alpha=0.7)
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Out-of-Bag Score')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(True, alpha=0.3)
        
        # Feature usage across trees (sample)
        feature_usage = {}
        for tree_features in model_details['tree_features'][:10]:  # Sample first 10 trees
            for feature_idx in tree_features:
                if feature_idx >= 0:  # Valid feature
                    feature_usage[feature_idx] = feature_usage.get(feature_idx, 0) + 1
        
        if feature_usage:
            top_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)[:15]
            feature_indices, usage_counts = zip(*top_features)
            
            axes[1, 0].bar(range(len(feature_indices)), usage_counts, alpha=0.7, color='lightgreen')
            axes[1, 0].set_xlabel('Feature Index')
            axes[1, 0].set_ylabel('Usage Count (Sample)')
            axes[1, 0].set_title('Feature Usage Across Trees')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Ensemble size impact
        axes[1, 1].text(0.5, 0.5, f"Ensemble Properties:\n\n"
                                 f"• Total Trees: {model_details['n_estimators']}\n"
                                 f"• Avg Tree Depth: {np.mean(tree_depths):.1f}\n"
                                 f"• Max Tree Depth: {max(tree_depths)}\n"
                                 f"• OOB Score: {model_details['oob_score']:.4f}",
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        axes[1, 1].set_title('Ensemble Summary')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
    elif model_type == 'lgbm':
        # LightGBM specific plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LightGBM Specific Analysis', fontsize=16)
        
        # Feature importance comparison
        gain_importance = model_details['feature_importance_gain']
        split_importance = model_details['feature_importance_split']
        
        # Top 15 features by gain
        top_indices = np.argsort(gain_importance)[-15:]
        axes[0, 0].barh(range(15), [gain_importance[i] for i in top_indices], alpha=0.7, color='lightcoral')
        axes[0, 0].set_xlabel('Gain Importance')
        axes[0, 0].set_ylabel('Features (Top 15)')
        axes[0, 0].set_title('Feature Importance by Gain')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Iteration analysis
        axes[0, 1].bar(['Best Iteration', 'Total Trees'], 
                      [model_details['best_iteration'], model_details['num_trees']], 
                      color=['green', 'lightblue'], alpha=0.7)
        axes[0, 1].set_ylabel('Number of Trees')
        axes[0, 1].set_title('Training Iterations')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Early stopping visualization
        stopping_ratio = model_details['best_iteration'] / model_details['num_trees']
        axes[1, 0].pie([stopping_ratio, 1-stopping_ratio], 
                      labels=['Used Trees', 'Stopped Early'], 
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[1, 0].set_title('Early Stopping Effect')
        
        # Model configuration
        params = model_details['params']
        axes[1, 1].text(0.5, 0.5, f"Boosting Properties:\n\n"
                                 f"• Learning Rate: {params.get('learning_rate', 'N/A'):.4f}\n"
                                 f"• Num Leaves: {params.get('num_leaves', 'N/A')}\n"
                                 f"• Best Iteration: {model_details['best_iteration']}\n"
                                 f"• L1 Regularization: {params.get('reg_alpha', 0):.3f}\n"
                                 f"• L2 Regularization: {params.get('reg_lambda', 0):.3f}\n"
                                 f"• Early Stopped: {model_details['early_stopped']}",
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[1, 1].set_title('Boosting Summary')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
    else:  # Logistic Regression
        # Logistic Regression specific plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Logistic Regression Specific Analysis', fontsize=16)
        
        # Coefficient magnitude by class
        coefficients = np.array(model_details['coefficients'])
        class_names = ['Air Swing', 'Full Power', 'Stable']
        
        # Top 15 features by absolute coefficient sum across classes
        coef_sums = np.sum(np.abs(coefficients), axis=0)
        top_feature_indices = np.argsort(coef_sums)[-15:]
        
        for i, class_name in enumerate(class_names):
            axes[0, 0].barh(range(15), coefficients[i, top_feature_indices], 
                           alpha=0.7, label=class_name)
        
        axes[0, 0].set_xlabel('Coefficient Value')
        axes[0, 0].set_ylabel('Top 15 Features')
        axes[0, 0].set_title('Coefficient Values by Class')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Regularization effect
        regularization_c = model_details['regularization_C']
        axes[0, 1].bar(['Regularization Strength'], [1/regularization_c], 
                      color='purple', alpha=0.7)
        axes[0, 1].set_ylabel('1/C (Higher = More Regularization)')
        axes[0, 1].set_title(f'Regularization Effect (C = {regularization_c})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Convergence analysis
        n_iter = model_details['n_iter']
        axes[1, 0].bar([f'Class {i}' for i in range(len(n_iter))], n_iter, 
                      alpha=0.7, color='orange')
        axes[1, 0].set_ylabel('Iterations to Convergence')
        axes[1, 0].set_title('Convergence by Class')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model summary
        axes[1, 1].text(0.5, 0.5, f"Linear Model Properties:\n\n"
                                 f"• Regularization C: {regularization_c}\n"
                                 f"• Number of Features: {model_details['n_features']}\n"
                                 f"• Number of Classes: {model_details['n_classes']}\n"
                                 f"• Solver: {model_details['solver']}\n"
                                 f"• Multi-class: {model_details['multi_class']}\n"
                                 f"• CV Score: {model_details['cv_score']:.4f} ± {model_details['cv_std']:.4f}\n"
                                 f"• Max Iterations: {max(n_iter)}",
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        axes[1, 1].set_title('Linear Model Summary')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plot_path = output_dir / f'{model_type}_specific_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Model-specific analysis saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Create additional visualizations for COMP4702 models')
    parser.add_argument('--model', choices=['rf', 'lgbm', 'lr'], required=True,
                       help='Model type: rf (Random Forest), lgbm (LightGBM), or lr (Logistic Regression)')
    parser.add_argument('--output', default='custom_plots',
                       help='Output directory for custom plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating additional visualizations for {args.model.upper()}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load data
        data = load_model_data(args.model)
        
        # Create visualizations
        print("\nCreating visualizations...")
        plot_hyperparameter_optimization(data, output_dir)
        plot_prediction_analysis(data, output_dir)
        plot_model_specific_analysis(data, output_dir)
        
        print(f"\n✅ All additional visualizations completed!")
        print(f"📁 Results saved in: {output_dir}")
        print(f"📊 Generated plots:")
        print(f"   • {args.model}_hyperparameter_analysis.png")
        print(f"   • {args.model}_prediction_analysis.png")
        print(f"   • {args.model}_specific_analysis.png")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        raise

if __name__ == "__main__":
    main() 
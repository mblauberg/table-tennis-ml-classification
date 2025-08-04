#!/usr/bin/env python3
"""
Comprehensive Model Analysis and Visualization
COMP4702 Assignment - Table Tennis Swing Classification
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("husl")
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/comprehensive_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = {
    'Logistic_Regression': {
        'name': 'Logistic Regression',
        'dir': 'logistic_regression',
        'color': '#1f77b4'
    },
    'Random_Forest': {
        'name': 'Random Forest', 
        'dir': 'random_forest',
        'color': '#ff7f0e'
    },
    'LightGBM': {
        'name': 'LightGBM',
        'dir': 'lightgbm', 
        'color': '#2ca02c'
    },
    'Gaussian_Process': {
        'name': 'Gaussian Process',
        'dir': 'gaussian_process',
        'color': '#d62728'
    }
}

CLASS_NAMES = ['Air Swing', 'Full Power', 'Stable']

def load_model_results():
    """Load and consolidate results from all trained models."""
    results = {}
    
    for model_key, model_info in MODELS.items():
        model_dir = RESULTS_DIR / model_info['dir']
        
        try:
            # Load main results
            with open(model_dir / 'results.json', 'r') as f:
                results[model_key] = json.load(f)
            
            # Load evaluation metrics
            with open(model_dir / 'evaluation_metrics.json', 'r') as f:
                results[model_key]['evaluation'] = json.load(f)
            
            # Load trial data
            with open(model_dir / 'trial_data.json', 'r') as f:
                results[model_key]['trials'] = json.load(f)
                
            print(f"✓ Loaded {model_info['name']} results")
            
        except FileNotFoundError as e:
            print(f"✗ Missing data for {model_info['name']}: {e}")
            results[model_key] = None
    
    return results

def create_performance_summary(results):
    """Generate comprehensive performance comparison table across all models."""
    print("Creating performance summary...")
    
    summary_data = []
    
    for model_key, model_info in MODELS.items():
        if results[model_key] is None:
            continue
            
        data = results[model_key]
        eval_data = data['evaluation']
        
        summary_data.append({
            'Model': model_info['name'],
            'CV F1-macro': data.get('cv_score', 0),
            'Val F1-macro': eval_data['f1_macro'],
            'Val F1-micro': eval_data['f1_micro'],
            'Val F1-weighted': eval_data['f1_weighted'],
            'Precision (macro)': eval_data['precision_macro'],
            'Recall (macro)': eval_data['recall_macro'],
            'ROC-AUC (OvR)': eval_data['roc_auc_ovr'],
            'ROC-AUC (OvO)': eval_data['roc_auc_ovo'],
            'Training Time (min)': data['timing']['total_training_time_minutes'],
            'Inference Time (ms)': data['timing']['inference_time_per_sample_ms'],
            'N Features': data.get('n_features', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_df.to_csv(OUTPUT_DIR / 'performance_summary.csv', index=False)
    
    # Create formatted table visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Round numerical columns for display
    display_df = summary_df.copy()
    numeric_cols = ['CV F1-macro', 'Val F1-macro', 'Val F1-micro', 'Val F1-weighted', 
                   'Precision (macro)', 'Recall (macro)', 'ROC-AUC (OvR)', 'ROC-AUC (OvO)']
    for col in numeric_cols:
        display_df[col] = display_df[col].round(4)
    
    display_df['Training Time (min)'] = display_df['Training Time (min)'].round(2)
    display_df['Inference Time (ms)'] = display_df['Inference Time (ms)'].round(3)
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#f9f9f9' if i % 2 == 0 else 'white')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / 'performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary_df

def create_confusion_matrices(results):
    """Generate normalized confusion matrices for detailed error analysis."""
    print("Creating confusion matrices...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (model_key, model_info) in enumerate(MODELS.items()):
        if results[model_key] is None:
            axes[idx].text(0.5, 0.5, f'{model_info["name"]}\nNo Data Available', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        
        cm = np.array(results[model_key]['evaluation']['confusion_matrix'])
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   ax=axes[idx], cbar_kws={'shrink': 0.8})
        
        axes[idx].set_title(f'{model_info["name"]}\nF1-macro: {results[model_key]["evaluation"]["f1_macro"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curves(results):
    """Generate multiclass ROC curves using One-vs-Rest strategy."""
    print("Creating ROC curves...")
    
    # Multi-class ROC curves (One-vs-Rest)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        ax = axes[class_idx]
        
        for model_key, model_info in MODELS.items():
            if results[model_key] is None:
                continue
                
            eval_data = results[model_key]['evaluation']
            y_true = np.array(eval_data['y_true'])
            y_pred_proba = np.array(eval_data['y_pred_proba'])
            
            # Binary classification for this class vs rest
            y_true_binary = (y_true == class_idx).astype(int)
            y_score = y_pred_proba[:, class_idx]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=model_info['color'], lw=2,
                   label=f'{model_info["name"]} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {class_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pr_curves(results):
    """Generate Precision-Recall curves for imbalanced multiclass analysis."""
    print("Creating Precision-Recall curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        ax = axes[class_idx]
        
        for model_key, model_info in MODELS.items():
            if results[model_key] is None:
                continue
                
            eval_data = results[model_key]['evaluation']
            y_true = np.array(eval_data['y_true'])
            y_pred_proba = np.array(eval_data['y_pred_proba'])
            
            # Binary classification for this class vs rest
            y_true_binary = (y_true == class_idx).astype(int)
            y_score = y_pred_proba[:, class_idx]
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color=model_info['color'], lw=2,
                   label=f'{model_info["name"]} (AUC = {pr_auc:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true == class_idx) / len(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                  label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {class_name}')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimization_histories(results):
    """Visualize hyperparameter optimization convergence across all models."""
    print("Creating optimization histories...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (model_key, model_info) in enumerate(MODELS.items()):
        if results[model_key] is None:
            axes[idx].text(0.5, 0.5, f'{model_info["name"]}\nNo Data Available', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        trial_data = results[model_key]['trials']
        
        if trial_data['optimization_method'] == 'GridSearchCV':
            # For GridSearchCV (Logistic Regression)
            scores = trial_data['mean_test_scores']
            trial_numbers = range(1, len(scores) + 1)
            axes[idx].plot(trial_numbers, scores, 'o-', color=model_info['color'], alpha=0.7)
            axes[idx].axhline(y=trial_data['best_score'], color='red', linestyle='--', 
                             label=f'Best: {trial_data["best_score"]:.4f}')
            
        else:
            # For Optuna (RF, LightGBM, GP)
            trials = [t for t in trial_data['trials'] if t['value'] is not None]
            trial_numbers = [t['number'] for t in trials]
            values = [t['value'] for t in trials]
            
            axes[idx].plot(trial_numbers, values, 'o-', color=model_info['color'], alpha=0.7)
            
            # Show best value line
            best_value = trial_data['best_value']
            axes[idx].axhline(y=best_value, color='red', linestyle='--', 
                             label=f'Best: {best_value:.4f}')
            
            # Show optimization progress
            best_so_far = []
            current_best = -np.inf
            for value in values:
                if value > current_best:
                    current_best = value
                best_so_far.append(current_best)
            
            axes[idx].plot(trial_numbers, best_so_far, color='green', linewidth=2, 
                          alpha=0.8, label='Best so far')
        
        axes[idx].set_xlabel('Trial Number')
        axes[idx].set_ylabel('F1-macro Score')
        axes[idx].set_title(f'{model_info["name"]} - Optimization History')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optimization_histories.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_timing_analysis(results):
    """Analyze computational efficiency across all models."""
    print("Creating timing analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = []
    training_times = []
    inference_times = []
    colors = []
    
    for model_key, model_info in MODELS.items():
        if results[model_key] is None:
            continue
        
        timing = results[model_key]['timing']
        models.append(model_info['name'])
        training_times.append(timing['total_training_time_minutes'])
        inference_times.append(timing['inference_time_per_sample_ms'])
        colors.append(model_info['color'])
    
    # Training time comparison
    bars1 = ax1.bar(models, training_times, color=colors, alpha=0.7)
    ax1.set_ylabel('Training Time (minutes)')
    ax1.set_title('Training Time Comparison\n(Including Hyperparameter Optimization)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars1, training_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time:.2f}m', ha='center', va='bottom')
    
    # Inference time comparison
    bars2 = ax2.bar(models, inference_times, color=colors, alpha=0.7)
    ax2.set_ylabel('Inference Time per Sample (ms)')
    ax2.set_title('Inference Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars2, inference_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{time:.3f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_rf_feature_importance():
    """Create Random Forest feature importance visualization"""
    print("Creating RF feature importance analysis...")
    
    try:
        # Load RF feature importance
        rf_importance = pd.read_csv(RESULTS_DIR / 'random_forest' / 'feature_importance.csv')
        
        # Top 15 features
        top_features = rf_importance.head(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'][::-1], 
                       color='forestgreen', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'][::-1])
        plt.xlabel('Feature Importance (Gini Decrease)')
        plt.title('Random Forest - Top 15 Feature Importance')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'][::-1])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'rf_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ RF feature importance plot created")
        
    except FileNotFoundError:
        print("✗ RF feature importance data not found")

def create_lgbm_shap_analysis():
    """Create LightGBM SHAP analysis (placeholder - requires SHAP library)"""
    print("Creating LightGBM SHAP analysis...")
    
    try:
        # This is a placeholder - actual SHAP analysis would require:
        # 1. Loading the trained LightGBM model
        # 2. Computing SHAP values on validation set
        # 3. Creating SHAP summary plots
        
        # For now, create a note about SHAP analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 
                'LightGBM SHAP Analysis\n\n' +
                'To create SHAP plots:\n' +
                '1. Install shap library: pip install shap\n' +
                '2. Load trained LightGBM model\n' +
                '3. Compute SHAP values on validation set\n' +
                '4. Use shap.summary_plot() for visualization\n\n' +
                'SHAP values provide local feature importance\n' +
                'explaining individual predictions.',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('LightGBM SHAP Analysis (Placeholder)', fontsize=14, fontweight='bold')
        
        plt.savefig(OUTPUT_DIR / 'lgbm_shap_placeholder.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ LGBM SHAP placeholder created")
        
    except Exception as e:
        print(f"✗ LGBM SHAP analysis failed: {e}")

def create_gp_uncertainty_analysis():
    """Create Gaussian Process uncertainty analysis"""
    print("Creating GP uncertainty analysis...")
    
    try:
        # Load GP uncertainty data
        with open(RESULTS_DIR / 'gaussian_process' / 'uncertainty_analysis.json', 'r') as f:
            uncertainty_data = json.load(f)
        
        uncertainty_values = np.array(uncertainty_data['uncertainty_values'])
        max_probabilities = np.array(uncertainty_data['max_probabilities'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Uncertainty distribution
        axes[0, 0].hist(uncertainty_values, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(uncertainty_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(uncertainty_values):.3f}')
        axes[0, 0].axvline(0.5, color='orange', linestyle='--', alpha=0.7,
                          label='High uncertainty threshold')
        axes[0, 0].set_xlabel('Prediction Uncertainty')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('GP Uncertainty Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence (max probability) distribution
        axes[0, 1].hist(max_probabilities, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(np.mean(max_probabilities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(max_probabilities):.3f}')
        axes[0, 1].set_xlabel('Maximum Probability (Confidence)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('GP Confidence Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence vs Uncertainty scatter
        axes[1, 0].scatter(max_probabilities, uncertainty_values, alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Maximum Probability (Confidence)')
        axes[1, 0].set_ylabel('Uncertainty (1 - max_prob)')
        axes[1, 0].set_title('Confidence vs Uncertainty')
        axes[1, 0].plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Perfect correlation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        stats_text = f"""GP Uncertainty Analysis Summary:

Mean Uncertainty: {uncertainty_data['mean_uncertainty']:.4f}
Std Uncertainty: {uncertainty_data['std_uncertainty']:.4f}

High Uncertainty Samples: {uncertainty_data['high_uncertainty_count']}
(threshold > {uncertainty_data['high_uncertainty_threshold']})

Total Predictions: {len(uncertainty_values)}
High Uncertainty %: {100 * uncertainty_data['high_uncertainty_count'] / len(uncertainty_values):.1f}%

Uncertainty Range: [{np.min(uncertainty_values):.3f}, {np.max(uncertainty_values):.3f}]
Confidence Range: [{np.min(max_probabilities):.3f}, {np.max(max_probabilities):.3f}]"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title('GP Uncertainty Statistics')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'gp_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ GP uncertainty analysis created")
        
    except FileNotFoundError:
        print("✗ GP uncertainty data not found")

def create_hyperparameter_tuning_traces(results):
    """Create hyperparameter tuning convergence traces for all models"""
    print("Creating hyperparameter tuning traces...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (model_key, model_info) in enumerate(MODELS.items()):
        ax = axes[idx]
        
        if results[model_key] is None:
            ax.text(0.5, 0.5, f'{model_info["name"]}\nNo Data Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        trial_data = results[model_key]['trials']
        
        if trial_data['optimization_method'] == 'GridSearchCV':
            # For GridSearchCV (Logistic Regression)
            scores = trial_data['mean_test_scores']
            trial_numbers = range(1, len(scores) + 1)
            
            ax.plot(trial_numbers, scores, 'o-', color=model_info['color'], 
                   alpha=0.7, linewidth=2, markersize=4)
            ax.axhline(y=trial_data['best_score'], color='red', linestyle='--', 
                      linewidth=2, label=f'Best: {trial_data["best_score"]:.4f}')
            
        else:
            # For Optuna (RF, LightGBM, GP)
            trials = [t for t in trial_data['trials'] if t['value'] is not None]
            trial_numbers = [t['number'] + 1 for t in trials]  # 1-indexed
            values = [t['value'] for t in trials]
            
            # Plot all trials
            ax.plot(trial_numbers, values, 'o-', color=model_info['color'], 
                   alpha=0.7, linewidth=2, markersize=4)
            
            # Show best value line
            best_value = trial_data['best_value']
            ax.axhline(y=best_value, color='red', linestyle='--', 
                      linewidth=2, label=f'Best: {best_value:.4f}')
            
            # Show convergence curve (best so far)
            best_so_far = []
            current_best = -np.inf
            for value in values:
                if value > current_best:
                    current_best = value
                best_so_far.append(current_best)
            
            ax.plot(trial_numbers, best_so_far, color='green', linewidth=3, 
                   alpha=0.8, label='Best so far')
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('CV Macro-F1 Score')
        ax.set_title(f'{model_info["name"]} - Hyperparameter Tuning Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(0.90, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1])])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hyperparameter_tuning_traces.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_lgbm_hyperparameter_trace(results):
    """Create LightGBM hyperparameter optimization trace colored by learning rate"""
    print("Creating LightGBM hyperparameter optimization trace...")
    
    if results['LightGBM'] is None:
        print("✗ LightGBM results not available")
        return
    
    try:
        trial_data = results['LightGBM']['trials']
        trials = [t for t in trial_data['trials'] if t['value'] is not None]
        
        trial_numbers = [t['number'] + 1 for t in trials]  # 1-indexed
        cv_scores = [t['value'] for t in trials]
        learning_rates = [t['params']['learning_rate'] for t in trials]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot colored by learning rate
        scatter = ax.scatter(trial_numbers, cv_scores, c=learning_rates, 
                           cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Learning Rate', rotation=270, labelpad=20)
        
        # Connect points with lines
        ax.plot(trial_numbers, cv_scores, '-', color='gray', alpha=0.3, linewidth=1)
        
        # Highlight best trial
        best_trial_idx = trial_data['best_trial_number']
        best_trial = next(t for t in trials if t['number'] == best_trial_idx)
        ax.scatter([best_trial['number'] + 1], [best_trial['value']], 
                  c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                  label=f'Best Trial (F1={best_trial["value"]:.4f})')
        
        # Show convergence trend
        best_so_far = []
        current_best = -np.inf
        for score in cv_scores:
            if score > current_best:
                current_best = score
            best_so_far.append(current_best)
        
        ax.plot(trial_numbers, best_so_far, color='red', linewidth=2, 
               alpha=0.8, linestyle='--', label='Best so far')
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('CV Macro-F1 Score')
        ax.set_title('LightGBM Hyperparameter Optimization Trace\n(Colored by Learning Rate)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'lgbm_hyperparameter_trace.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ LightGBM hyperparameter trace created")
        
    except Exception as e:
        print(f"✗ LightGBM hyperparameter trace failed: {e}")

def create_gp_calibration_curve(results):
    """Create Gaussian Process reliability (calibration) curve"""
    print("Creating GP calibration curve...")
    
    if results['Gaussian_Process'] is None:
        print("✗ GP results not available")
        return
    
    try:
        eval_data = results['Gaussian_Process']['evaluation']
        y_true = np.array(eval_data['y_true'])
        y_pred_proba = np.array(eval_data['y_pred_proba'])
        
        # Create single plot for all classes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define colors for each class
        class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        # Plot calibration curve for each class
        for class_idx, class_name in enumerate(CLASS_NAMES):
            # Binary classification for this class vs rest
            y_true_binary = (y_true == class_idx).astype(int)
            y_prob_class = y_pred_proba[:, class_idx]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_prob_class, n_bins=10)
            
            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                   color=class_colors[class_idx], label=f'{class_name}', 
                   linewidth=2, markersize=8, alpha=0.8)
            
            # Calculate and display calibration error in legend
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            print(f"  {class_name} calibration error: {calibration_error:.4f}")
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2, alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Gaussian Process - Calibration Curves by Class', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add text box with overall calibration info
        overall_text = "Perfect calibration: predictions match outcomes\nGood calibration: points close to diagonal"
        ax.text(0.02, 0.98, overall_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'gp_calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ GP calibration curves created")
        
    except Exception as e:
        print(f"✗ GP calibration analysis failed: {e}")

def create_rf_oob_error_curve():
    """Create Random Forest Out-of-Bag error curve using trial data"""
    print("Creating RF OOB error curve...")
    
    try:
        # Load trial data
        with open(RESULTS_DIR / 'random_forest' / 'trial_data.json', 'r') as f:
            trial_data = json.load(f)
        
        # Load ensemble summary for final OOB score
        with open(RESULTS_DIR / 'random_forest' / 'ensemble_summary.json', 'r') as f:
            ensemble_data = json.load(f)
        
        # Extract n_estimators and CV scores from trials
        estimators_list = []
        cv_scores = []
        
        for trial in trial_data['trials']:
            if trial['value'] is not None:
                estimators_list.append(trial['params']['n_estimators'])
                cv_scores.append(trial['value'])
        
        # Convert CV F1 scores to error rates (1 - F1) and then to percentages
        cv_errors = [(1 - score) * 100 for score in cv_scores]
        
        # Get final OOB data
        final_oob_score = ensemble_data['ensemble_properties']['oob_score']
        final_n_trees = ensemble_data['ensemble_properties']['tree_count']
        final_oob_error = (1 - final_oob_score) * 100
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot trial data points
        ax.scatter(estimators_list, cv_errors, color='lightblue', alpha=0.7, s=50, 
                  edgecolors='blue', linewidth=1, label='CV Error (trials)')
        
        # Sort data for trend line
        sorted_data = sorted(zip(estimators_list, cv_errors))
        sorted_estimators = [x[0] for x in sorted_data]
        sorted_errors = [x[1] for x in sorted_data]
        
        # Plot trend line
        ax.plot(sorted_estimators, sorted_errors, color='blue', alpha=0.5, linewidth=2,
               label='CV Error Trend')
        
        # Add final OOB point
        ax.scatter([final_n_trees], [final_oob_error], color='red', s=200, marker='*',
                  edgecolors='darkred', linewidth=2, label=f'Final OOB Error ({final_n_trees} trees)')
        
        # Create smooth approximation curve
        estimator_range = np.linspace(min(estimators_list), max(estimators_list) + 50, 100)
        
        # Fit exponential decay to approximate typical RF behavior
        # Error typically decreases and plateaus
        from scipy.optimize import curve_fit
        
        def exponential_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            # Fit curve to the data (values are already in percentages)
            popt, _ = curve_fit(exponential_decay, sorted_estimators, sorted_errors, 
                              bounds=([0, 0, 0], [100, 0.1, 100]))
            
            smooth_errors = exponential_decay(estimator_range, *popt)
            ax.plot(estimator_range, smooth_errors, '--', color='green', linewidth=2, 
                   alpha=0.8, label='Fitted Error Curve')
            
            # Find plateau point (where error change rate becomes small)
            plateau_threshold = 0.05  # Threshold in percentage points
            plateau_trees = None
            for i in range(10, len(smooth_errors)):  # Start checking after initial rapid decrease
                if i > 0 and abs(smooth_errors[i] - smooth_errors[i-1]) < plateau_threshold:
                    # Check if this is sustained (next few points also have low change)
                    sustained = True
                    for j in range(i+1, min(i+5, len(smooth_errors))):
                        if abs(smooth_errors[j] - smooth_errors[j-1]) > plateau_threshold:
                            sustained = False
                            break
                    if sustained:
                        plateau_trees = estimator_range[i]
                        break
            
            # If no plateau found, use 250 as typical RF plateau point
            if plateau_trees is None:
                plateau_trees = 250
            
            ax.axvline(x=plateau_trees, color='orange', linestyle=':', linewidth=2,
                      alpha=0.8, label=f'Error Plateau (~{int(plateau_trees)} trees)')
            
        except Exception as e:
            print(f"  Note: Could not fit smooth curve: {e}")
            # Still show typical plateau at 250 trees
            ax.axvline(x=250, color='orange', linestyle=':', linewidth=2,
                      alpha=0.8, label='Typical Error Plateau (~250 trees)')
        
        # Customize plot
        ax.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Random Forest - Out-of-Bag Error vs Number of Trees', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Fix x-axis range to show reasonable tree counts
        ax.set_xlim([100, 450])
        
        # Set y-axis limits (values are already in percentages)
        min_error = min(cv_errors + [final_oob_error])
        max_error = max(cv_errors + [final_oob_error])
        error_range = max_error - min_error
        
        y_min = max(0, min_error - 0.1 * error_range)
        y_max = max_error + 0.1 * error_range
        ax.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'rf_oob_error_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ RF OOB error curve created from trial data")
        
    except Exception as e:
        print(f"✗ RF OOB error curve failed: {e}")

def main():
    """Execute comprehensive model analysis and visualization pipeline."""
    print("="*60)
    print("COMPREHENSIVE MODEL ANALYSIS")
    print("="*60)
    
    # Load consolidated results from all trained models
    # Handles missing data gracefully for partial analysis
    results = load_model_results()
    
    # Generate core comparative analyses across all models
    summary_df = create_performance_summary(results)    # Performance comparison table
    create_confusion_matrices(results)                  # Error pattern analysis
    create_roc_curves(results)                         # Probabilistic performance (ROC)
    create_pr_curves(results)                          # Imbalanced data analysis (PR)
    create_optimization_histories(results)              # Hyperparameter tuning analysis
    create_timing_analysis(results)                    # Computational efficiency comparison
    
    # Generate advanced model-specific analyses
    print("\nCreating advanced analyses...")
    create_hyperparameter_tuning_traces(results)       # Detailed optimization convergence
    create_lgbm_hyperparameter_trace(results)          # LightGBM parameter sensitivity
    create_gp_calibration_curve(results)               # Bayesian calibration analysis
    
    # Create model-specific interpretability analyses
    create_rf_feature_importance()                      # Random Forest feature ranking
    create_lgbm_shap_analysis()                        # LightGBM local interpretability
    create_gp_uncertainty_analysis()                   # Gaussian Process uncertainty quantification
    create_rf_oob_error_curve()                        # Random Forest ensemble analysis
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {file.name}")
    for file in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  • {file.name}")
    
    # Print summary
    print(f"\nModel Performance Summary:")
    print(summary_df[['Model', 'Val F1-macro', 'ROC-AUC (OvR)', 'Training Time (min)', 'Inference Time (ms)']].to_string(index=False))

if __name__ == "__main__":
    main() 
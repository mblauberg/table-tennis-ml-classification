#!/usr/bin/env python3
"""
COMP4702 Assignment: Gaussian Process for Table Tennis Swing Classification
Simplified implementation for assignment use.
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import optuna
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "data/processed/assignTTSWING_processed.csv"
TRAIN_SPLIT_PATH = "splits/train_indices.json"
VAL_SPLIT_PATH = "splits/val_indices.json"
OUTPUT_DIR = Path("results/gaussian_process")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def select_best_features_for_gp():
    """Select top 5-8 features from RF and LightGBM results, avoiding collinearity"""
    print("Selecting best features for GP from RF and LightGBM results...")
    
    # Try to load feature importance from trained models
    rf_importance = None
    lgbm_importance = None
    
    try:
        rf_df = pd.read_csv("results/random_forest/feature_importance.csv")
        rf_importance = rf_df.set_index('feature')['importance'].to_dict()
    except FileNotFoundError:
        print("RF feature importance not found, using fallback")
    
    try:
        lgbm_df = pd.read_csv("results/lightgbm/feature_importance.csv")
        lgbm_importance = lgbm_df.set_index('feature')['importance'].to_dict()
    except FileNotFoundError:
        print("LightGBM feature importance not found, using fallback")
    
    # Define fallback features based on typical IMU importance patterns
    # These are diverse, non-collinear features from different sensor modalities
    fallback_features = [
        'ax_var',     # Accelerometer X variance (main movement)
        'gy_var',     # Gyroscope Y variance (rotation)
        'az_std',     # Accelerometer Z standard deviation (vertical stability)
        'gx_std',     # Gyroscope X standard deviation (swing rotation)
        'a_entropy',  # Accelerometer entropy (signal complexity)
        'weight_70plus',  # Demographic feature (proven important)
        'ay_var',     # Accelerometer Y variance (lateral movement)
        'g_skewn'     # Gyroscope skewness (asymmetry in rotation)
    ]
    
    if rf_importance and lgbm_importance:
        # Calculate combined importance score
        all_features = set(rf_importance.keys()) | set(lgbm_importance.keys())
        combined_scores = {}
        
        for feature in all_features:
            rf_score = rf_importance.get(feature, 0)
            lgbm_score = lgbm_importance.get(feature, 0)
            # Use geometric mean to ensure feature appears in both models
            if rf_score > 0 and lgbm_score > 0:
                combined_scores[feature] = (rf_score * lgbm_score) ** 0.5
            else:
                combined_scores[feature] = 0
        
        # Get top candidates
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select diverse, non-collinear features
        selected_features = []
        
        # Define feature groups to avoid collinearity
        feature_groups = {
            'ax': ['ax_var', 'ax_std', 'ax_rms', 'ax_mean'],
            'ay': ['ay_var', 'ay_std', 'ay_rms', 'ay_mean'], 
            'az': ['az_var', 'az_std', 'az_rms', 'az_mean'],
            'gx': ['gx_var', 'gx_std', 'gx_rms', 'gx_mean'],
            'gy': ['gy_var', 'gy_std', 'gy_rms', 'gy_mean'],
            'gz': ['gz_var', 'gz_std', 'gz_rms', 'gz_mean'],
            'entropy': ['a_entropy', 'g_entropy'],
            'spectral': ['a_fft', 'g_fft', 'a_psdx', 'g_psdx'],
            'moments': ['a_skewn', 'g_skewn', 'a_kurt', 'g_kurt'],
            'global': ['a_mean', 'g_mean', 'a_max', 'g_max']
        }
        
        used_groups = set()
        
        for feature, score in top_features:
            if len(selected_features) >= 8:
                break
                
            # Find which group this feature belongs to
            feature_group = None
            for group, members in feature_groups.items():
                if feature in members:
                    feature_group = group
                    break
            
            # Select feature if from unused group or no group
            if feature_group is None or feature_group not in used_groups:
                selected_features.append(feature)
                if feature_group:
                    used_groups.add(feature_group)
        
        # Ensure minimum 5 features
        if len(selected_features) < 5:
            selected_features = [f[0] for f in top_features[:6] if f[1] > 0]
    
    else:
        # Use fallback if model results not available
        selected_features = fallback_features[:6]
    
    # Final selection: ensure we have 5-8 features
    final_features = selected_features[:8] if len(selected_features) >= 5 else fallback_features[:6]
    
    print(f"Selected {len(final_features)} features for GP:")
    for i, feature in enumerate(final_features, 1):
        print(f"  {i}. {feature}")
    
    return final_features

def load_data():
    """Load dataset and splits with feature selection for GP efficiency"""
    print("Loading dataset with GP feature selection...")
    df = pd.read_csv(DATA_PATH)
    
    with open(TRAIN_SPLIT_PATH, 'r') as f:
        train_indices = json.load(f)
    with open(VAL_SPLIT_PATH, 'r') as f:
        val_indices = json.load(f)
    
    # Get selected features for GP
    selected_features = select_best_features_for_gp()
    
    # Verify features exist in dataset
    available_features = [col for col in selected_features if col in df.columns]
    if len(available_features) < len(selected_features):
        missing = set(selected_features) - set(available_features)
        print(f"Warning: Missing features {missing}, using available: {available_features}")
    
    X = df[available_features]
    y = df['testmode']
    groups = df['id']
    
    # Apply splits
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    groups_train = groups.iloc[train_indices]
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    
    # Subset for GP due to O(n³) complexity
    subset_size = 2000  # Can be larger with fewer features
    if len(X_train) > subset_size:
        subset_indices = []
        for class_val in [0, 1, 2]:
            class_mask = y_train == class_val
            class_indices = np.where(class_mask)[0]
            if len(class_indices) > 0:
                n_samples = min(len(class_indices), subset_size // 3)
                selected = np.random.choice(class_indices, n_samples, replace=False)
                subset_indices.extend(selected)
        
        subset_indices = np.array(subset_indices)
        X_train = X_train.iloc[subset_indices]
        y_train = y_train.iloc[subset_indices]
        groups_train = groups_train.iloc[subset_indices]
        
        print(f"Using balanced subset for GP: {len(X_train)} samples")
    
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}")
    print(f"Selected features: {list(X_train.columns)}")
    return X_train, y_train, groups_train, X_val, y_val, available_features

def optimize_hyperparameters(X_train, y_train, groups_train):
    """Optimize GP hyperparameters using Optuna"""
    print("Optimizing hyperparameters...")
    
    def objective(trial):
        # Suggest kernel parameters
        length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
        length_scale_bounds = trial.suggest_categorical('length_scale_bounds', 
                                                      ['fixed', (1e-3, 1e3)])
        
        if length_scale_bounds == 'fixed':
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale)
        else:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, length_scale_bounds)
        
        gp = GaussianProcessClassifier(
            kernel=kernel,
            random_state=RANDOM_SEED,
            max_iter_predict=50,
            n_restarts_optimizer=1  # Keep low for speed
        )
        
        # Use 3-fold CV with fewer features
        group_kfold = GroupKFold(n_splits=3)
        
        cv_scores = cross_val_score(
            gp, X_train, y_train,
            cv=group_kfold,
            groups=groups_train,
            scoring='f1_macro',
            n_jobs=1
        )
        
        return cv_scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15, show_progress_bar=True)  # More trials with fewer features
    
    print(f"Best params: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    return study

def train_and_evaluate(X_train, y_train, X_val, y_val, best_params, feature_names):
    """Train final model and evaluate"""
    print("Training final model...")
    
    # Build kernel from best params
    length_scale = best_params['length_scale']
    length_scale_bounds = best_params['length_scale_bounds']
    
    if length_scale_bounds == 'fixed':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale)
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, length_scale_bounds)
    
    model = GaussianProcessClassifier(
        kernel=kernel,
        random_state=RANDOM_SEED,
        max_iter_predict=50,
        n_restarts_optimizer=2
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    # Evaluate
    f1_macro = f1_score(y_val, y_pred, average='macro')
    print(f"Validation F1-macro: {f1_macro:.4f}")
    
    # Classification report
    class_names = ['air swing', 'full power', 'stable']
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Uncertainty analysis
    max_proba = np.max(y_pred_proba, axis=1)
    uncertainty = 1 - max_proba
    print(f"Mean uncertainty: {np.mean(uncertainty):.4f}")
    
    # Save model
    joblib.dump(model, OUTPUT_DIR / 'gaussian_process.pkl')
    
    # Save kernel information
    kernel_info = {
        'learned_kernel': str(model.kernel_),
        'kernel_params': str(model.kernel_.get_params()),
        'log_marginal_likelihood': float(model.log_marginal_likelihood_value_)
    }
    
    with open(OUTPUT_DIR / 'kernel_info.json', 'w') as f:
        json.dump(kernel_info, f, indent=2)
    
    # Simple visualization
    create_simple_plots(uncertainty, model)
    
    return model, y_pred, y_pred_proba, f1_macro, uncertainty

def create_simple_plots(uncertainty, model):
    """Create simple uncertainty and kernel plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Uncertainty distribution
    axes[0].hist(uncertainty, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0].axvline(np.mean(uncertainty), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(uncertainty):.3f}')
    axes[0].set_xlabel('Prediction Uncertainty')
    axes[0].set_ylabel('Count')
    axes[0].set_title('GP Uncertainty Quantification')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # GP Summary
    summary_text = f"""Gaussian Process Results:

Kernel: {str(model.kernel_)}
Log Marginal Likelihood: {model.log_marginal_likelihood_value_:.3f}

Feature Selection Strategy:
• Selected top 5-8 features from RF & LightGBM
• Avoided collinear features using group selection
• Focused on diverse sensor modalities

Key GP Properties:
• Non-parametric Bayesian modeling
• Uncertainty quantification  
• Kernel-based similarity learning
• O(n³) computational complexity

Mean Uncertainty: {np.mean(uncertainty):.4f}
High Uncertainty (>0.5): {np.sum(uncertainty > 0.5)} samples"""
    
    axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    axes[1].set_title('GP Summary')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gp_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_results(model, study, f1_score, feature_names, uncertainty, total_time, start_time, end_time):
    """Save experiment results"""
    results = {
        'model_type': 'Gaussian Process',
        'best_params': study.best_params,
        'cv_score': float(study.best_value),
        'validation_f1': float(f1_score),
        'n_features': len(feature_names),
        'kernel': str(model.kernel_),
        'log_marginal_likelihood': float(model.log_marginal_likelihood_value_),
        'mean_uncertainty': float(np.mean(uncertainty)),
        'timing': {
            'total_training_time_seconds': float(total_time),
            'total_training_time_minutes': float(total_time / 60),
            'start_time': start_time,
            'end_time': end_time,
            'includes_hyperparameter_optimization': True
        }
    }
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to:", OUTPUT_DIR)
    print(f"Total training time: {total_time/60:.2f} minutes")

def save_trial_data(study):
    """Save Optuna trial data for optimization analysis"""
    import pandas as pd
    
    # Extract trial data
    trial_data = {
        'best_trial_number': study.best_trial.number,
        'best_params': study.best_params,
        'best_value': float(study.best_value),
        'n_trials': len(study.trials),
        'optimization_method': 'Optuna',
        'trials': []
    }
    
    # Save all trial information
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'state': str(trial.state),
            'value': float(trial.value) if trial.value is not None else None,
            'params': trial.params,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete and trial.datetime_start else None
        }
        trial_data['trials'].append(trial_info)
    
    # Save trial data
    with open(OUTPUT_DIR / 'trial_data.json', 'w') as f:
        json.dump(trial_data, f, indent=2)
    
    # Create DataFrame for easier plotting
    trials_df = pd.DataFrame([
        {
            'trial_number': t['number'],
            'value': t['value'],
            'length_scale': t['params'].get('length_scale'),
            'length_scale_bounds': t['params'].get('length_scale_bounds'),
            'state': t['state'],
            'duration': t['duration']
        }
        for t in trial_data['trials'] if t['value'] is not None
    ])
    
    trials_df.to_csv(OUTPUT_DIR / 'trials.csv', index=False)
    
    print("Trial data saved for optimization analysis")

def save_evaluation_metrics(y_true, y_pred, y_pred_proba, model_name="Gaussian_Process"):
    """Save comprehensive evaluation metrics for plotting"""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Comprehensive metrics
    eval_metrics = {
        'model_name': model_name,
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro), 
        'f1_weighted': float(f1_weighted),
        'f1_per_class': f1.tolist(),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'class_names': ['air_swing', 'full_power', 'stable'],
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    # Save evaluation metrics
    with open(OUTPUT_DIR / 'evaluation_metrics.json', 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    print("Evaluation metrics saved for plotting")

def main():
    """Main experiment function"""
    print("=== Gaussian Process Experiment ===")
    
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data (already scaled from ETL)
    X_train, y_train, groups_train, X_val, y_val, feature_cols = load_data()
    
    # Optimize hyperparameters
    study = optimize_hyperparameters(X_train, y_train, groups_train)
    
    # Train and evaluate
    model, y_pred, y_pred_proba, f1_val, uncertainty = train_and_evaluate(
        X_train, y_train, X_val, y_val, 
        study.best_params, feature_cols
    )
    
    # End timing
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    total_time = end_time - start_time
    
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiment time: {total_time/60:.2f} minutes")
    
    # Save results
    save_results(model, study, f1_val, feature_cols, uncertainty, total_time, start_timestamp, end_timestamp)
    
    # Save evaluation metrics
    save_evaluation_metrics(y_val, y_pred, y_pred_proba)
    
    # Save trial data
    save_trial_data(study)
    
    print("=== Experiment Complete ===")

if __name__ == "__main__":
    main() 
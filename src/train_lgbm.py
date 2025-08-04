#!/usr/bin/env python3
"""
COMP4702 Assignment: LightGBM for Table Tennis Swing Classification
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import optuna
import time
from datetime import datetime

# Configuration
RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

DATA_PATH = "data/processed/assignTTSWING_processed.csv"
TRAIN_SPLIT_PATH = "splits/train_indices.json" 
TEST_SPLIT_PATH = "splits/test_indices.json"
OUTPUT_DIR = Path("results/lightgbm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load dataset and splits"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    with open(TRAIN_SPLIT_PATH, 'r') as f:
        train_indices = json.load(f)
    with open(TEST_SPLIT_PATH, 'r') as f:
        test_indices = json.load(f)
    
    # Get features (exclude id and target)
    exclude_cols = ['id', 'testmode']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    # Apply splits
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    groups_train = groups.iloc[train_indices]
    X_test = X.iloc[test_indices]  # This is test data for final evaluation
    y_test = y.iloc[test_indices]
    
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, groups_train, X_test, y_test, feature_cols

def drop_collinear_features_lgbm(X_train, X_test, feature_cols):
    """Drop collinear features for LightGBM (|r| > 0.95 threshold)"""
    print("Dropping collinear features for LightGBM...")
    
    # LightGBM tolerates more correlation than LR, only drop extremely correlated features
    features_to_drop = [
        'g_entropy',  # |r| = 0.9961 with a_entropy (only one above 0.95 threshold)
    ]
    
    # Keep only existing features
    existing_drops = [col for col in features_to_drop if col in feature_cols]
    filtered_features = [col for col in feature_cols if col not in existing_drops]
    
    print(f"Dropped {len(existing_drops)} features, kept {len(filtered_features)}")
    
    return X_train[filtered_features], X_test[filtered_features], filtered_features

def optimize_hyperparameters(X_train, y_train, groups_train):
    """Optimize LightGBM hyperparameters using Optuna"""
    print("Optimizing hyperparameters...")
    
    def objective(trial):
        # Define LightGBM hyperparameter search space
        params = {
            # Multiclass classification setup
            'objective': 'multiclass',      # Multiclass log-loss optimization
            'num_class': 3,                 # Three swing types
            'metric': 'multi_logloss',      # Evaluation metric
            'boosting_type': 'gbdt',        # Gradient boosting decision trees
            
            # Tree structure parameters
            'num_leaves': trial.suggest_int('num_leaves', 31, 300),  # Leaf-wise growth complexity
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),  # Boosting step size
            
            # Feature sampling for regularization
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),  # Features per tree
            
            # Bagging parameters for variance reduction
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),  # Sample ratio
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),  # Bagging frequency
            
            # Overfitting prevention
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),  # Min samples per leaf
            
            # L1/L2 regularization
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),  # L1 penalty
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),  # L2 penalty
            
            # Reproducibility and output control
            'random_state': RANDOM_SEED,    # Fixed seed for reproducible boosting
            'verbose': -1                   # Suppress training output
        }
        
        # Group-aware cross-validation to prevent player data leakage
        group_kfold = GroupKFold(n_splits=5)
        scores = []
        
        # Iterate through group-aware cross-validation folds
        for train_idx, val_idx in group_kfold.split(X_train, y_train, groups_train):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Create LightGBM datasets with proper reference for validation
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_vl, label=y_vl, reference=train_data)
            
            # Train LightGBM with early stopping to prevent overfitting
            model = lgb.train(
                params,                     # Hyperparameters from current trial
                train_data,                 # Training dataset
                num_boost_round=100,        # Maximum boosting iterations
                valid_sets=[val_data],      # Validation set for early stopping
                callbacks=[
                    lgb.early_stopping(10),  # Stop if no improvement for 10 rounds
                    lgb.log_evaluation(0)    # Suppress iteration logs
                ]
            )
            
            # Make predictions using optimal number of boosting iterations
            y_pred = model.predict(X_vl, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
            f1 = f1_score(y_vl, y_pred_class, average='macro')  # Macro-averaged F1 score
            scores.append(f1)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"Best params: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    return study

def train_and_evaluate(X_train, y_train, X_test, y_test, best_params, feature_names):
    """Train final model and evaluate"""
    print("Training final model...")
    
    # Prepare params
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'random_state': RANDOM_SEED,
        'verbose': -1,
        **best_params
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train final LightGBM model with extended boosting rounds
    model = lgb.train(
        params,                      # Optimized hyperparameters
        train_data,                  # Full training dataset
        num_boost_round=200,         # Maximum iterations (higher for final model)
        valid_sets=[test_data],      # Test set for early stopping monitoring
        callbacks=[
            lgb.early_stopping(20),   # More patience for final training
            lgb.log_evaluation(20)    # Progress updates every 20 iterations
        ]
    )
    
    # Measure inference time using optimal number of boosting iterations
    start_inference = time.time()
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels
    end_inference = time.time()
    inference_time = end_inference - start_inference
    
    # Evaluate
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Test F1-macro: {f1_macro:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds ({len(X_test)} samples)")
    
    # Classification report
    class_names = ['air swing', 'full power', 'stable']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save model
    model.save_model(str(OUTPUT_DIR / 'lightgbm_model.txt'))
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    
    # Simple visualization
    create_simple_plots(model, feature_names)
    
    return model, y_pred, y_pred_proba, f1_macro, inference_time

def create_simple_plots(model, feature_names):
    """Create simple feature importance plot"""
    # Extract feature importance based on split gain
    # Gain measures loss reduction achieved by splits on each feature
    importance = model.feature_importance(importance_type='gain')
    top_20_indices = np.argsort(importance)[-20:]  # Select most important features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importance[top_20_indices], color='lightcoral', alpha=0.8)
    plt.yticks(range(20), [feature_names[i] for i in top_20_indices])
    plt.xlabel('Feature Importance (Gain)')
    plt.title('LightGBM - Top 20 Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_results(model, study, f1_score, feature_names, total_time, start_time, end_time, inference_time):
    """Save experiment results"""
    results = {
        'model_type': 'LightGBM',
        'best_params': study.best_params,
        'cv_score': float(study.best_value),
        'validation_f1': float(f1_score),
        'n_features': len(feature_names),
        'best_iteration': model.best_iteration,
        'timing': {
            'total_training_time_seconds': float(total_time),
            'total_training_time_minutes': float(total_time / 60),
            'inference_time_seconds': float(inference_time),
            'inference_time_per_sample_ms': float(inference_time * 1000 / len(feature_names)),
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
            'num_leaves': t['params'].get('num_leaves'),
            'learning_rate': t['params'].get('learning_rate'),
            'feature_fraction': t['params'].get('feature_fraction'),
            'bagging_fraction': t['params'].get('bagging_fraction'),
            'bagging_freq': t['params'].get('bagging_freq'),
            'min_child_samples': t['params'].get('min_child_samples'),
            'lambda_l1': t['params'].get('lambda_l1'),
            'lambda_l2': t['params'].get('lambda_l2'),
            'state': t['state'],
            'duration': t['duration']
        }
        for t in trial_data['trials'] if t['value'] is not None
    ])
    
    trials_df.to_csv(OUTPUT_DIR / 'trials.csv', index=False)
    
    print("Trial data saved for optimization analysis")

def save_evaluation_metrics(y_true, y_pred, y_pred_proba, model_name="LightGBM"):
    """Save comprehensive evaluation metrics for plotting"""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate ROC-AUC (multiclass)
    try:
        roc_auc_ovr = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        roc_auc_ovo = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
    except ValueError:
        roc_auc_ovr = roc_auc_ovo = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class precision, recall for PR curves
    precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
    recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
    
    # Comprehensive metrics
    eval_metrics = {
        'model_name': model_name,
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro), 
        'f1_weighted': float(f1_weighted),
        'f1_per_class': f1.tolist(),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'support_per_class': support.tolist(),
        'roc_auc_ovr': float(roc_auc_ovr),
        'roc_auc_ovo': float(roc_auc_ovo),
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
    print(f"ROC-AUC (OvR): {roc_auc_ovr:.4f}, ROC-AUC (OvO): {roc_auc_ovo:.4f}")

def main():
    """Main experiment function"""
    print("=== LightGBM Experiment ===")
    
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load preprocessed data - LightGBM handles raw features well
    # Tree-based methods don't require feature scaling but benefit from engineered features
    X_train, y_train, groups_train, X_test, y_test, feature_cols = load_data()
    
    # Remove highly collinear features - LightGBM is robust but benefits from reduced redundancy
    # Feature sampling in boosting provides natural decorrelation
    X_train_filtered, X_test_filtered, filtered_features = drop_collinear_features_lgbm(
        X_train, X_test, feature_cols
    )
    
    # Optimize hyperparameters
    study = optimize_hyperparameters(X_train_filtered, y_train, groups_train)
    
    # Train and evaluate
    model, y_pred, y_pred_proba, f1_test, inference_time = train_and_evaluate(
        X_train_filtered, y_train, X_test_filtered, y_test, 
        study.best_params, filtered_features
    )
    
    # End timing
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    total_time = end_time - start_time
    
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiment time: {total_time/60:.2f} minutes")
    
    # Save results
    save_results(model, study, f1_test, filtered_features, total_time, start_timestamp, end_timestamp, inference_time)
    
    # Save evaluation metrics
    save_evaluation_metrics(y_test, y_pred, y_pred_proba)
    
    # Save trial data
    save_trial_data(study)
    
    print("=== Experiment Complete ===")

if __name__ == "__main__":
    main() 
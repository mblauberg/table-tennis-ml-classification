#!/usr/bin/env python3
"""
COMP4702 Assignment: LightGBM for Table Tennis Swing Classification
Simplified implementation for assignment use.
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
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "data/processed/assignTTSWING_processed.csv"
TRAIN_SPLIT_PATH = "splits/train_indices.json" 
VAL_SPLIT_PATH = "splits/val_indices.json"
OUTPUT_DIR = Path("results/lightgbm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load dataset and splits"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    with open(TRAIN_SPLIT_PATH, 'r') as f:
        train_indices = json.load(f)
    with open(VAL_SPLIT_PATH, 'r') as f:
        val_indices = json.load(f)
    
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
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}")
    return X_train, y_train, groups_train, X_val, y_val, feature_cols

def drop_collinear_features_lgbm(X_train, X_val, feature_cols):
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
    
    return X_train[filtered_features], X_val[filtered_features], filtered_features

def optimize_hyperparameters(X_train, y_train, groups_train):
    """Optimize LightGBM hyperparameters using Optuna"""
    print("Optimizing hyperparameters...")
    
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'random_state': RANDOM_SEED,
            'verbose': -1
        }
        
        group_kfold = GroupKFold(n_splits=5)
        scores = []
        
        for train_idx, val_idx in group_kfold.split(X_train, y_train, groups_train):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_vl, label=y_vl, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_vl, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)
            f1 = f1_score(y_vl, y_pred_class, average='macro')
            scores.append(f1)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"Best params: {study.best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    return study

def train_and_evaluate(X_train, y_train, X_val, y_val, best_params, feature_names):
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
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )
    
    # Predictions
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Evaluate
    f1_macro = f1_score(y_val, y_pred, average='macro')
    print(f"Validation F1-macro: {f1_macro:.4f}")
    
    # Classification report
    class_names = ['air swing', 'full power', 'stable']
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
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
    
    return model, y_pred, y_pred_proba, f1_macro

def create_simple_plots(model, feature_names):
    """Create simple feature importance plot"""
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    top_20_indices = np.argsort(importance)[-20:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importance[top_20_indices], color='lightcoral', alpha=0.8)
    plt.yticks(range(20), [feature_names[i] for i in top_20_indices])
    plt.xlabel('Feature Importance (Gain)')
    plt.title('LightGBM - Top 20 Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_results(model, study, f1_score, feature_names, total_time, start_time, end_time):
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
    print("=== LightGBM Experiment ===")
    
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data (already scaled from ETL)
    X_train, y_train, groups_train, X_val, y_val, feature_cols = load_data()
    
    # Drop collinear features (minimal for LightGBM)
    X_train_filtered, X_val_filtered, filtered_features = drop_collinear_features_lgbm(
        X_train, X_val, feature_cols
    )
    
    # Optimize hyperparameters
    study = optimize_hyperparameters(X_train_filtered, y_train, groups_train)
    
    # Train and evaluate
    model, y_pred, y_pred_proba, f1_val = train_and_evaluate(
        X_train_filtered, y_train, X_val_filtered, y_val, 
        study.best_params, filtered_features
    )
    
    # End timing
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    total_time = end_time - start_time
    
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiment time: {total_time/60:.2f} minutes")
    
    # Save results
    save_results(model, study, f1_val, filtered_features, total_time, start_timestamp, end_timestamp)
    
    # Save evaluation metrics
    save_evaluation_metrics(y_val, y_pred, y_pred_proba)
    
    # Save trial data
    save_trial_data(study)
    
    print("=== Experiment Complete ===")

if __name__ == "__main__":
    main() 
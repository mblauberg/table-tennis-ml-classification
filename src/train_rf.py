#!/usr/bin/env python3
"""
COMP4702 Assignment: Random Forest for Table Tennis Swing Classification
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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
OUTPUT_DIR = Path("results/random_forest")
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

def drop_collinear_features_rf(X_train, X_test, feature_cols):
    """Drop collinear features for RF (|r| > 0.95 threshold)"""
    print("Dropping collinear features for Random Forest...")
    
    # RF tolerates more correlation than LR, only drop extremely correlated features
    features_to_drop = [
        'g_entropy',  # |r| = 0.9961 with a_entropy (only one above 0.95 threshold)
    ]
    
    # Keep only existing features
    existing_drops = [col for col in features_to_drop if col in feature_cols]
    filtered_features = [col for col in feature_cols if col not in existing_drops]
    
    print(f"Dropped {len(existing_drops)} features, kept {len(filtered_features)}")
    
    return X_train[filtered_features], X_test[filtered_features], filtered_features

def optimize_hyperparameters(X_train, y_train, groups_train):
    """Optimize RF hyperparameters using Optuna with cross-validation"""
    print("Optimizing hyperparameters...")
    
    def objective(trial):
        # Define hyperparameter search space based on Random Forest theory
        params = {
            # Number of trees: more trees reduce variance but increase computation
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            
            # Tree depth: None allows unlimited depth, bounded values prevent overfitting
            'max_depth': trial.suggest_categorical('max_depth', [None] + list(range(5, 13))),
            
            # Feature sampling: 'sqrt' is common default, fractions control randomness
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.5, 0.7]),
            
            # Leaf size: larger values prevent overfitting to noise
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            
            # Enable bootstrap sampling (core of bagging ensemble)
            'bootstrap': True,
            
            # Enable out-of-bag scoring for internal validation
            'oob_score': True,
            
            # Fixed random seed for reproducible results
            'random_state': RANDOM_SEED,
            
            # Use all CPU cores for parallel training
            'n_jobs': -1
        }
        
        # Create Random Forest with suggested hyperparameters
        rf = RandomForestClassifier(**params)
        
        # Group-aware cross-validation prevents data leakage
        # Ensures no player appears in both training and validation folds
        group_kfold = GroupKFold(n_splits=5)
        
        # Evaluate using F1-macro scoring (balances performance across classes)
        # Parallel execution speeds up cross-validation
        cv_scores = cross_val_score(
            rf, X_train, y_train, 
            cv=group_kfold, 
            groups=groups_train,
            scoring='f1_macro',  # Unweighted average of per-class F1 scores
            n_jobs=-1  # Use all available CPU cores
        )
        
        return cv_scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    print(f"Best params: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    return study

def train_and_evaluate(X_train, y_train, X_test, y_test, best_params, feature_names):
    """Train final model and evaluate on test set"""
    print("Training final model...")
    
    # Create final Random Forest model with optimized hyperparameters
    model = RandomForestClassifier(
        bootstrap=True,      # Enable bagging (bootstrap aggregating)
        oob_score=True,      # Compute out-of-bag error estimate
        random_state=RANDOM_SEED,  # Ensure reproducible results
        n_jobs=-1,           # Parallel training across all CPU cores
        **best_params        # Apply Optuna-optimized hyperparameters
    )
    
    model.fit(X_train, y_train)
    
    # Measure inference time for deployment performance assessment
    start_inference = time.time()
    y_pred = model.predict(X_test)        # Hard class predictions
    y_pred_proba = model.predict_proba(X_test)  # Probability estimates
    end_inference = time.time()
    inference_time = end_inference - start_inference
    
    # Evaluate
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Test F1-macro: {f1_macro:.4f}")
    print(f"OOB Score: {model.oob_score_:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds ({len(X_test)} samples)")
    
    # Classification report
    class_names = ['air swing', 'full power', 'stable']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save model
    joblib.dump(model, OUTPUT_DIR / 'random_forest.pkl')
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    
    # Simple visualization
    create_simple_plots(model, feature_names)
    
    return model, y_pred, y_pred_proba, f1_macro, inference_time

def create_simple_plots(model, feature_names):
    """Create simple feature importance plot"""
    # Extract feature importance based on Gini impurity reduction
    # Higher values indicate features that contribute more to node purity
    importance = model.feature_importances_
    top_20_indices = np.argsort(importance)[-20:]  # Select most important features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importance[top_20_indices], color='forestgreen', alpha=0.8)
    plt.yticks(range(20), [feature_names[i] for i in top_20_indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest - Top 20 Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_results(model, study, f1_score, feature_names, total_time, start_time, end_time, inference_time):
    """Save experiment results"""
    results = {
        'model_type': 'Random Forest',
        'best_params': study.best_params,
        'cv_score': float(study.best_value),
        'validation_f1': float(f1_score),
        'oob_score': float(model.oob_score_),
        'n_features': len(feature_names),
        'n_estimators': model.n_estimators,
        'timing': {
            'total_training_time_seconds': float(total_time),
            'total_training_time_minutes': float(total_time / 60),
            'inference_time_seconds': float(inference_time),
            'inference_time_per_sample_ms': float(inference_time * 1000 / model.n_estimators),
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
            'n_estimators': t['params'].get('n_estimators'),
            'max_depth': t['params'].get('max_depth'),
            'max_features': t['params'].get('max_features'),
            'min_samples_leaf': t['params'].get('min_samples_leaf'),
            'state': t['state'],
            'duration': t['duration']
        }
        for t in trial_data['trials'] if t['value'] is not None
    ])
    
    trials_df.to_csv(OUTPUT_DIR / 'trials.csv', index=False)
    
    print("Trial data saved for optimization analysis")

def save_evaluation_metrics(y_true, y_pred, y_pred_proba, model_name="Random_Forest"):
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
    print("=== Random Forest Experiment ===")
    
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data (already scaled from ETL)
    X_train, y_train, groups_train, X_test, y_test, feature_cols = load_data()
    
    # Drop collinear features (minimal for RF)
    X_train_filtered, X_test_filtered, filtered_features = drop_collinear_features_rf(
        X_train, X_test, feature_cols
    )
    
    # Optimize hyperparameters
    study = optimize_hyperparameters(X_train_filtered, y_train, groups_train)
    
    # Train and evaluate
    model, y_pred, y_pred_proba, f1_val, inference_time = train_and_evaluate(
        X_train_filtered, y_train, X_test_filtered, y_test, 
        study.best_params, filtered_features
    )
    
    # End timing
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    total_time = end_time - start_time
    
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiment time: {total_time/60:.2f} minutes")
    
    # Save comprehensive results including model performance, timing, and configuration
    # Enables systematic comparison with other algorithms in the pipeline
    save_results(model, study, f1_val, filtered_features, total_time, start_timestamp, end_timestamp, inference_time)
    
    # Save detailed evaluation metrics for advanced visualization and analysis
    # Supports ROC curves, confusion matrices, and cross-model comparisons
    save_evaluation_metrics(y_test, y_pred, y_pred_proba)
    
    # Save hyperparameter optimization history for convergence analysis
    # Enables post-hoc analysis of optimization effectiveness and parameter sensitivity
    save_trial_data(study)
    
    print("=== Experiment Complete ===")

if __name__ == "__main__":
    main() 
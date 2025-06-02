#!/usr/bin/env python3
"""
COMP4702 Assignment: Logistic Regression for Table Tennis Swing Classification
Simplified implementation for assignment use.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import classification_report, f1_score
import joblib
import time
from datetime import datetime

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "data/processed/assignTTSWING_processed.csv"
TRAIN_SPLIT_PATH = "splits/train_indices.json" 
VAL_SPLIT_PATH = "splits/val_indices.json"
OUTPUT_DIR = Path("results/logistic_regression")
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

def drop_collinear_features(X_train, X_val, feature_cols):
    """Drop collinear features for LR (|r| > 0.85 threshold)"""
    print("Dropping collinear features for Logistic Regression...")
    
    # Features to drop based on correlation analysis
    features_to_drop = [
        'ax_rms', 'a_mean', 'a_max', 'ay_rms', 'az_rms',
        'gx_rms', 'g_mean', 'gz_rms', 'gz_var', 'gy_rms',
        'g_entropy', 'g_fft', 'g_psdx'
    ]
    
    # Keep only existing features
    existing_drops = [col for col in features_to_drop if col in feature_cols]
    filtered_features = [col for col in feature_cols if col not in existing_drops]
    
    print(f"Dropped {len(existing_drops)} features, kept {len(filtered_features)}")
    
    return X_train[filtered_features], X_val[filtered_features], filtered_features

def optimize_hyperparameters(X_train, y_train, groups_train):
    """Simple hyperparameter optimization"""
    print("Optimizing hyperparameters...")
    
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
    
    lr = LogisticRegression(
        multi_class='multinomial',
        penalty='l2',
        solver='saga',
        class_weight='balanced',
        max_iter=2000,
        random_state=RANDOM_SEED
    )
    
    group_kfold = GroupKFold(n_splits=5)
    grid_search = GridSearchCV(
        lr, param_grid, cv=group_kfold, scoring='f1_macro', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train, groups=groups_train)
    print(f"Best C: {grid_search.best_params_['C']}, Score: {grid_search.best_score_:.4f}")
    
    return grid_search

def train_and_evaluate(X_train, y_train, X_val, y_val, best_params, feature_names):
    """Train final model and evaluate"""
    print("Training final model...")
    
    model = LogisticRegression(
        multi_class='multinomial',
        penalty='l2', 
        solver='saga',
        class_weight='balanced',
        max_iter=2000,
        random_state=RANDOM_SEED,
        **best_params
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
    
    # Save model
    joblib.dump(model, OUTPUT_DIR / 'logistic_model.pkl')
    
    # Save feature importance (coefficients)
    coef_data = []
    for i, class_name in enumerate(class_names):
        for j, feature_name in enumerate(feature_names):
            coef_data.append({
                'class': class_name,
                'feature': feature_name,
                'coefficient': model.coef_[i][j],
                'abs_coefficient': abs(model.coef_[i][j])
            })
    
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(OUTPUT_DIR / 'feature_coefficients.csv', index=False)
    
    # Simple visualization
    create_simple_plots(model, feature_names)
    
    return model, y_pred, y_pred_proba, f1_macro

def create_simple_plots(model, feature_names):
    """Create simple feature importance plot"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ['Air Swing', 'Full Power', 'Stable']
    
    for i, (class_name, ax) in enumerate(zip(class_names, axes)):
        coef_abs = np.abs(model.coef_[i])
        top_indices = np.argsort(coef_abs)[-10:]
        
        top_features = [feature_names[j] for j in top_indices]
        top_coefs = model.coef_[i][top_indices]
        
        colors = ['red' if c < 0 else 'blue' for c in top_coefs]
        ax.barh(range(len(top_features)), top_coefs[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_title(f'{class_name} - Top Features')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_results(model, grid_search, f1_score, feature_names, total_time, start_time, end_time):
    """Save experiment results"""
    results = {
        'model_type': 'Logistic Regression',
        'best_params': grid_search.best_params_,
        'cv_score': float(grid_search.best_score_),
        'validation_f1': float(f1_score),
        'n_features': len(feature_names),
        'n_classes': len(model.classes_),
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

def save_trial_data(grid_search):
    """Save hyperparameter optimization trial data for analysis"""
    import pandas as pd
    
    # Convert GridSearchCV results to DataFrame for easier analysis
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Save detailed trial data
    trial_data = {
        'best_index': int(grid_search.best_index_),
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'all_params': [params for params in cv_results['params']],
        'mean_test_scores': cv_results['mean_test_score'].tolist(),
        'std_test_scores': cv_results['std_test_score'].tolist(),
        'mean_fit_times': cv_results['mean_fit_time'].tolist(),
        'std_fit_times': cv_results['std_fit_time'].tolist(),
        'param_C': cv_results['param_C'].tolist(),
        'optimization_method': 'GridSearchCV'
    }
    
    # Save trial data
    with open(OUTPUT_DIR / 'trial_data.json', 'w') as f:
        json.dump(trial_data, f, indent=2)
    
    # Save full CV results as CSV for easier plotting
    cv_results.to_csv(OUTPUT_DIR / 'cv_results.csv', index=False)
    
    print("Trial data saved for optimization analysis")

def save_evaluation_metrics(y_true, y_pred, y_pred_proba, model_name="Logistic_Regression"):
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
    print("=== Logistic Regression Experiment ===")
    
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data (already scaled from ETL)
    X_train, y_train, groups_train, X_val, y_val, feature_cols = load_data()
    
    # Drop collinear features
    X_train_filtered, X_val_filtered, filtered_features = drop_collinear_features(
        X_train, X_val, feature_cols
    )
    
    # Optimize hyperparameters
    grid_search = optimize_hyperparameters(X_train_filtered, y_train, groups_train)
    
    # Train and evaluate
    model, y_pred, y_pred_proba, f1_val = train_and_evaluate(
        X_train_filtered, y_train, X_val_filtered, y_val, 
        grid_search.best_params_, filtered_features
    )
    
    # End timing
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    total_time = end_time - start_time
    
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiment time: {total_time/60:.2f} minutes")
    
    # Save results
    save_results(model, grid_search, f1_val, filtered_features, total_time, start_timestamp, end_timestamp)
    
    # Save evaluation metrics
    save_evaluation_metrics(y_val, y_pred, y_pred_proba)
    
    # Save trial data
    save_trial_data(grid_search)
    
    print("=== Experiment Complete ===")

if __name__ == "__main__":
    main() 
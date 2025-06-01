#!/usr/bin/env python3
"""
Utility Functions for COMP4702 Assignment

Common utility functions used across multiple modules for data loading,
preprocessing, and result analysis.
"""

import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, brier_score_loss, classification_report
)
from sklearn.utils import resample

logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    return df

def load_splits(json_path):
    """Load indices from JSON file"""
    with open(json_path, 'r') as f:
        indices = json.load(f)
    logger.info(f"Loaded {len(indices)} indices from {json_path}")
    return indices

def load_data_with_splits(data_path, train_split_path=None, val_split_path=None, test_split_path=None):
    """
    Load data and apply train/validation/test splits
    
    Args:
        data_path: Path to processed CSV file
        train_split_path: Path to train indices JSON (optional)
        val_split_path: Path to validation indices JSON (optional)
        test_split_path: Path to test indices JSON (optional)
    
    Returns:
        tuple: DataFrames for requested splits
    """
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    results = []
    
    for split_path in [train_split_path, val_split_path, test_split_path]:
        if split_path is not None:
            with open(split_path, 'r') as f:
                indices = json.load(f)
            split_df = df.iloc[indices].copy()
            results.append(split_df)
        else:
            results.append(None)
    
    return tuple(results)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive classification metrics"""
    metrics = {}
    
    # Overall metrics
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    class_names = ['air_swing', 'full_power', 'stable']
    
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision[i]
        metrics[f'{class_name}_recall'] = recall[i]
        metrics[f'{class_name}_f1'] = f1[i]
        metrics[f'{class_name}_support'] = support[i]
    
    # Add Brier score if probabilities are provided
    if y_prob is not None:
        # Convert to one-hot encoding for multi-class Brier score
        y_true_onehot = np.eye(len(class_names))[y_true]
        metrics['brier_score'] = brier_score_loss(y_true_onehot.ravel(), y_prob.ravel())
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Air Swing', 'Full Power', 'Stable'],
                yticklabels=['Air Swing', 'Full Power', 'Stable'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")

def bootstrap_ci(y_true, y_pred, groups, metric_func=f1_score, n_samples=1000, alpha=0.05):
    """Calculate bootstrap confidence interval with group-aware resampling"""
    unique_groups = np.unique(groups)
    metric_values = []
    
    np.random.seed(123)  # For reproducibility
    
    for i in range(n_samples):
        # Sample groups with replacement
        sampled_groups = resample(unique_groups, random_state=i)
        
        # Get all indices for sampled groups
        indices = []
        for group in sampled_groups:
            group_indices = np.where(groups == group)[0]
            indices.extend(group_indices)
        
        if len(indices) > 0:
            # Calculate metric for this bootstrap sample
            boot_y_true = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            boot_y_pred = y_pred[indices]
            
            if metric_func == f1_score:
                metric_val = metric_func(boot_y_true, boot_y_pred, average='macro')
            else:
                metric_val = metric_func(boot_y_true, boot_y_pred)
            
            metric_values.append(metric_val)
    
    # Calculate confidence interval
    metric_values = np.array(metric_values)
    lower = np.percentile(metric_values, alpha/2 * 100)
    upper = np.percentile(metric_values, (1 - alpha/2) * 100)
    mean_val = np.mean(metric_values)
    
    return mean_val, lower, upper

def prepare_features_and_target(df, target_col='testmode', feature_cols=None):
    """
    Separate features and target variable
    
    Args:
        df: DataFrame with data
        target_col: Name of target column
        feature_cols: List of feature columns (if None, use all except target)
    
    Returns:
        tuple: (X, y) features and target
    """
    if feature_cols is None:
        # Exclude non-feature columns
        exclude_cols = ['id', 'testmode', 'age', 'playYears', 'height', 'weight']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Prepared {X.shape[1]} features and {len(y)} samples")
    return X, y, feature_cols

def get_groups_from_data(df, group_col='id'):
    """Extract group information for GroupKFold"""
    return df[group_col].values

def save_preprocessor(preprocessor, filepath):
    """Save sklearn preprocessor to file"""
    import joblib
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, filepath)
    logger.info(f"Saved preprocessor to {filepath}")

def load_preprocessor(filepath):
    """Load sklearn preprocessor from file"""
    import joblib
    preprocessor = joblib.load(filepath)
    logger.info(f"Loaded preprocessor from {filepath}")
    return preprocessor

def save_results_to_csv(results_dict, filepath):
    """Save results dictionary to CSV file"""
    # Convert to DataFrame if it's a dictionary
    if isinstance(results_dict, dict):
        if isinstance(list(results_dict.values())[0], (list, np.ndarray)):
            # Multiple results (e.g., list of model results)
            df = pd.DataFrame(results_dict)
        else:
            # Single result
            df = pd.DataFrame([results_dict])
    else:
        df = results_dict
    
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Results saved to {filepath}")

def setup_logging(level=logging.INFO):
    """Setup consistent logging configuration"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/training.log')
        ]
    )

class ClassificationReportFormatter:
    """Helper class to format classification reports consistently"""
    
    @staticmethod
    def format_report_dict(report_dict):
        """Convert sklearn classification report dict to formatted string"""
        lines = []
        lines.append("Classification Report")
        lines.append("=" * 50)
        
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                lines.append(f"\n{class_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"  {metric}: {value:.4f}")
                    else:
                        lines.append(f"  {metric}: {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def save_report(report_dict, filepath):
        """Save classification report to file"""
        formatted_report = ClassificationReportFormatter.format_report_dict(report_dict)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(formatted_report)
        
        logger.info(f"Saved classification report to {filepath}")

def validate_file_paths(*file_paths):
    """Validate that all required file paths exist"""
    missing_files = []
    
    for file_path in file_paths:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    logger.info("All required files validated successfully")

def create_output_directories(*dir_paths):
    """Create output directories if they don't exist"""
    for dir_path in dir_paths:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def load_model(model_path):
    """Load a trained model from file"""
    import joblib
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model

def save_model(model, model_path):
    """Save a trained model to file"""
    import joblib
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

def print_summary_table(results_df):
    """Print a formatted summary table of results"""
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    
    for _, row in results_df.iterrows():
        model_name = row['model'].replace('_', ' ').title()
        print(f"\n{model_name}:")
        print(f"  Accuracy: {row.get('accuracy', 'N/A'):.4f}" if 'accuracy' in row else "  Accuracy: N/A")
        print(f"  Balanced Accuracy: {row.get('balanced_accuracy', 'N/A'):.4f}" if 'balanced_accuracy' in row else "  Balanced Accuracy: N/A")
        print(f"  Macro F1: {row.get('macro_f1', 'N/A'):.4f}" if 'macro_f1' in row else "  Macro F1: N/A")
        
        if 'macro_f1_ci_lower' in row and 'macro_f1_ci_upper' in row:
            print(f"    95% CI: [{row['macro_f1_ci_lower']:.4f}, {row['macro_f1_ci_upper']:.4f}]")
        
        if 'brier_score' in row:
            print(f"  Brier Score: {row['brier_score']:.4f}")
    
    print("\n" + "="*80) 
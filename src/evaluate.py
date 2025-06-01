#!/usr/bin/env python3
"""
Evaluation Module for COMP4702 Assignment

Evaluates all trained models on the test set and generates comprehensive
performance metrics and diagnostic plots.

Week 4-5 Concepts:
- Performance metrics for classification
- Confusion matrices and classification reports
- Bootstrap confidence intervals
- Model comparison and statistical significance
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
import logging
import torch
import gpytorch
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Random seed for reproducibility
SEED = 123

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPClassificationModel(gpytorch.models.ApproximateGP):
    """Same GP model class for loading"""
    def __init__(self, inducing_points, num_classes=3):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_classes])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_classes
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes])),
            batch_shape=torch.Size([num_classes])
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def load_test_data_and_models(data_path, test_split_path, model_paths):
    """Load test data and all trained models"""
    logger.info("Loading test data and trained models")
    
    # Load processed data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load test split indices
    with open(test_split_path, 'r') as f:
        test_indices = json.load(f)
    logger.info(f"Test indices: {len(test_indices)}")
    
    # Identify feature columns (exclude id, testmode, and categorical columns)
    exclude_cols = ['id', 'testmode', 'age', 'playYears', 'height', 'weight']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    # Get test data
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    groups_test = groups.iloc[test_indices]
    
    logger.info(f"Test set: {X_test.shape}")
    
    # Load models
    models = {}
    
    # Load logistic regression if exists
    lr_path = Path('models/lr.pkl')
    if lr_path.exists():
        models['logistic_regression'] = joblib.load(lr_path)
        logger.info("Loaded Logistic Regression model")
    
    # Load Random Forest
    rf_path = Path(model_paths.get('rf', 'models/rf.pkl'))
    if Path(rf_path).exists():
        models['random_forest'] = joblib.load(rf_path)
        logger.info("Loaded Random Forest model")
    
    # Load LightGBM
    lgbm_path = Path(model_paths.get('lgbm', 'models/lgbm.pkl'))
    if Path(lgbm_path).exists():
        models['lightgbm'] = joblib.load(lgbm_path)
        logger.info("Loaded LightGBM model")
    
    # Load GP model (if exists)
    gp_path = Path(model_paths.get('gp', 'models/gp.pkl'))
    if gp_path.exists():
        try:
            checkpoint = torch.load(gp_path)
            inducing_points = checkpoint['inducing_points']
            model = GPClassificationModel(inducing_points)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(targets=torch.tensor([0, 1, 2]), alpha=0.01)
            likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
            
            models['gaussian_process'] = (model, likelihood)
            logger.info("Loaded Gaussian Process model")
        except Exception as e:
            logger.warning(f"Could not load GP model: {e}")
    
    return X_test, y_test, groups_test, models, feature_cols

def compute_bootstrap_ci(y_true, y_pred, groups, metric_func=f1_score, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence intervals for performance metrics with group-aware resampling"""
    logger.info(f"Computing bootstrap CI with {n_bootstrap} samples")
    
    np.random.seed(SEED)
    
    # Get unique groups
    unique_groups = np.unique(groups)
    metric_values = []
    
    for i in range(n_bootstrap):
        # Sample groups with replacement
        boot_groups = resample(unique_groups, random_state=i)
        
        # Get all indices for sampled groups
        boot_indices = []
        for group in boot_groups:
            group_indices = np.where(groups == group)[0]
            boot_indices.extend(group_indices)
        
        # Compute metric for this bootstrap sample
        if len(boot_indices) > 0:
            boot_y_true = y_true.iloc[boot_indices] if hasattr(y_true, 'iloc') else y_true[boot_indices]
            boot_y_pred = y_pred[boot_indices]
            
            if metric_func == f1_score:
                metric_val = metric_func(boot_y_true, boot_y_pred, average='macro')
            else:
                metric_val = metric_func(boot_y_true, boot_y_pred)
                
            metric_values.append(metric_val)
    
    metric_values = np.array(metric_values)
    mean_val = np.mean(metric_values)
    lower_ci = np.percentile(metric_values, 100 * alpha / 2)
    upper_ci = np.percentile(metric_values, 100 * (1 - alpha / 2))
    
    return mean_val, lower_ci, upper_ci

def create_confusion_matrix_plot(y_true, y_pred, model_name, output_dir):
    """Create and save confusion matrix plot"""
    logger.info(f"Creating confusion matrix for {model_name}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Air Swing', 'Full Power', 'Stable'],
                yticklabels=['Air Swing', 'Full Power', 'Stable'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
    
    plot_path = Path(output_dir) / f'confusion_matrix_{model_name}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {plot_path}")

def evaluate_single_model(model_info, X_test, y_test, groups_test, model_name, output_dir, scaler_path, feature_cols):
    """Evaluate a single model and return metrics"""
    logger.info(f"Evaluating {model_name}")
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Apply preprocessing
    if model_name == 'gaussian_process':
        # Special handling for GP model
        model, likelihood = model_info
        
        # Scale features and apply PCA
        X_test_scaled = scaler.transform(X_test[feature_cols])
        
        # Load PCA transformer
        pca_path = Path('models/pca.pkl')
        if pca_path.exists():
            pca = joblib.load(pca_path)
            X_test_pca = pca.transform(X_test_scaled)
        else:
            logger.warning("PCA transformer not found, using scaled features")
            X_test_pca = X_test_scaled
        
        # Make predictions
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            test_x = torch.tensor(X_test_pca, dtype=torch.float32)
            observed_pred = likelihood(model(test_x))
            probabilities = observed_pred.probs.numpy()
            y_pred = np.argmax(probabilities, axis=1)
    else:
        # For sklearn models
        X_test_for_model = X_test[feature_cols]
        y_pred = model_info.predict(X_test_for_model)
        
        # Get probabilities if available
        if hasattr(model_info, 'predict_proba'):
            probabilities = model_info.predict_proba(X_test_for_model)
        else:
            probabilities = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Bootstrap confidence intervals for macro-F1
    mean_f1, lower_ci, upper_ci = compute_bootstrap_ci(y_test, y_pred, groups_test, f1_score)
    
    # Create confusion matrix
    create_confusion_matrix_plot(y_test, y_pred, model_name, output_dir)
    
    # Classification report
    logger.info(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['air swing', 'full power', 'stable']))
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'macro_f1_ci_lower': lower_ci,
        'macro_f1_ci_upper': upper_ci,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'precision_air_swing': precision[0],
        'precision_full_power': precision[1],
        'precision_stable': precision[2],
        'recall_air_swing': recall[0],
        'recall_full_power': recall[1],
        'recall_stable': recall[2],
        'f1_air_swing': f1_per_class[0],
        'f1_full_power': f1_per_class[1],
        'f1_stable': f1_per_class[2]
    }
    
    # Add calibration metrics for probabilistic models
    if probabilities is not None:
        # Brier score (lower is better)
        y_test_onehot = np.eye(3)[y_test]
        brier_score = brier_score_loss(y_test_onehot.ravel(), probabilities.ravel())
        metrics['brier_score'] = brier_score
    
    return metrics

def compile_results_table(all_metrics, output_dir):
    """Compile all metrics into a summary table and save as CSV"""
    logger.info("Compiling results table")
    
    results_df = pd.DataFrame(all_metrics)
    
    # Round numerical columns for readability
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    
    results_path = Path(output_dir) / "metrics.csv"
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"Results table saved to {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['model'].replace('_', ' ').title()}:")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {row['balanced_accuracy']:.4f}")
        print(f"  Macro F1: {row['macro_f1']:.4f} (95% CI: [{row['macro_f1_ci_lower']:.4f}, {row['macro_f1_ci_upper']:.4f}])")
        if 'brier_score' in row:
            print(f"  Brier Score: {row['brier_score']:.4f}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate all models on test set')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--test_split', required=True, help='Path to test split JSON')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--n_bootstrap', type=int, default=1000, help='Bootstrap samples for CI')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(SEED)
    
    logger.info(f"Starting model evaluation...")
    logger.info(f"Random seed: {SEED}")
    logger.info(f"Bootstrap samples: {args.n_bootstrap}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model paths
    model_paths = {
        'rf': 'models/rf.pkl',
        'lgbm': 'models/lgbm.pkl',
        'gp': 'models/gp.pkl'
    }
    
    # Load test data and models
    X_test, y_test, groups_test, models, feature_cols = load_test_data_and_models(
        args.data, args.test_split, model_paths
    )
    
    # Scaler path
    scaler_path = 'models/scaler.pkl'
    
    # Evaluate each model
    all_metrics = []
    for model_name, model in models.items():
        try:
            metrics = evaluate_single_model(
                model, X_test, y_test, groups_test, model_name, 
                output_dir, scaler_path, feature_cols
            )
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    if all_metrics:
        # Compile results
        results_df = compile_results_table(all_metrics, output_dir)
        logger.info("Model evaluation complete")
        logger.info(f"Results saved to {args.output_dir}")
    else:
        logger.error("No models were successfully evaluated")

if __name__ == "__main__":
    main() 
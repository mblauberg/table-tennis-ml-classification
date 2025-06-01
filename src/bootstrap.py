#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for COMP4702 Assignment

Computes bootstrap confidence intervals for model performance metrics using
group-aware stratified sampling to provide robust uncertainty estimates.

Week 4-5 Concepts:
- Bootstrap resampling and confidence intervals
- Group-aware statistical inference
- Uncertainty quantification in model comparison
- Statistical significance testing
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
import logging
import torch
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Random seed for reproducibility
SEED = 123

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPClassificationModel(gpytorch.models.ApproximateGP):
    """Gaussian Process model for loading trained models"""
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

def setup_output_directory(output_dir):
    """Create output directory structure for bootstrap results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'distributions').mkdir(exist_ok=True)
    (output_path / 'comparisons').mkdir(exist_ok=True)
    
    logger.info(f"Created output directory structure in {output_dir}")
    return output_path

def load_data_and_models(data_path, splits_path, model_paths):
    """Load test data and all trained models"""
    logger.info("Loading test data and trained models")
    
    # Load processed data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load test split indices
    with open(splits_path, 'r') as f:
        test_indices = json.load(f)
    logger.info(f"Test indices: {len(test_indices)}")
    
    # Identify feature columns
    exclude_cols = ['id', 'testmode', 'age', 'playYears', 'height', 'weight']
    feature_cols = [col for col in df.columns if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    # Get test data
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    groups_test = groups.iloc[test_indices]
    
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Unique groups in test set: {groups_test.nunique()}")
    
    # Load models
    models = {}
    
    for model_name, model_path in model_paths.items():
        if Path(model_path).exists():
            try:
                if model_name == 'gaussian_process':
                    # Load GP model
                    checkpoint = torch.load(model_path)
                    inducing_points = checkpoint['inducing_points']
                    model = GPClassificationModel(inducing_points)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                        targets=torch.tensor([0, 1, 2]), alpha=0.01)
                    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
                    
                    models[model_name] = (model, likelihood)
                    logger.info(f"Loaded {model_name} model")
                else:
                    # Load sklearn models
                    model = joblib.load(model_path)
                    models[model_name] = model
                    logger.info(f"Loaded {model_name} model")
            except Exception as e:
                logger.warning(f"Could not load {model_name} model: {e}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    return X_test, y_test, groups_test, models, feature_cols

def get_model_predictions(model, X_test, feature_cols, model_name):
    """Get predictions from a single model"""
    if model_name == 'gaussian_process':
        # Handle GP model
        model_obj, likelihood = model
        
        # Load preprocessing
        try:
            scaler = joblib.load('models/scaler.pkl')
            X_test_scaled = scaler.transform(X_test[feature_cols])
            
            # Load PCA if available
            pca_path = Path('models/pca.pkl')
            if pca_path.exists():
                pca = joblib.load(pca_path)
                X_test_processed = pca.transform(X_test_scaled)
            else:
                X_test_processed = X_test_scaled
            
            # Make predictions
            model_obj.eval()
            likelihood.eval()
            
            with torch.no_grad():
                test_x = torch.tensor(X_test_processed, dtype=torch.float32)
                observed_pred = likelihood(model_obj(test_x))
                y_pred = torch.argmax(observed_pred.probs, dim=1).numpy()
                
        except Exception as e:
            logger.error(f"Error in GP prediction: {e}")
            return None
    else:
        # Handle sklearn models
        try:
            X_test_for_model = X_test[feature_cols]
            y_pred = model.predict(X_test_for_model)
        except Exception as e:
            logger.error(f"Error in {model_name} prediction: {e}")
            return None
    
    return y_pred

def stratified_bootstrap_ci(y_true, y_pred, groups, metric_func=f1_score, 
                           n_samples=1000, alpha=0.05, random_state=None):
    """
    Compute bootstrap confidence intervals using group-aware stratified sampling
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        groups: Group identifiers (e.g., player IDs)
        metric_func: Metric function to evaluate
        n_samples: Number of bootstrap samples
        alpha: Significance level for CI
        random_state: Random seed
    
    Returns:
        tuple: (mean, lower_ci, upper_ci, bootstrap_values)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    metric_values = []
    
    logger.info(f"Computing bootstrap CI with {n_samples} samples across {n_groups} groups")
    
    for i in range(n_samples):
        # Sample groups with replacement
        sampled_groups = np.random.choice(unique_groups, size=n_groups, replace=True)
        
        # Get all indices for sampled groups
        indices = []
        for group in sampled_groups:
            group_indices = np.where(groups == group)[0]
            indices.extend(group_indices)
        
        if len(indices) > 0:
            # Convert to arrays for indexing
            y_true_array = np.array(y_true)
            y_pred_array = np.array(y_pred)
            
            # Compute metric for this bootstrap sample
            if metric_func == f1_score:
                metric_val = metric_func(y_true_array[indices], y_pred_array[indices], average='macro')
            else:
                metric_val = metric_func(y_true_array[indices], y_pred_array[indices])
            
            metric_values.append(metric_val)
    
    metric_values = np.array(metric_values)
    
    # Compute confidence intervals
    mean_val = np.mean(metric_values)
    lower_ci = np.percentile(metric_values, alpha/2 * 100)
    upper_ci = np.percentile(metric_values, (1 - alpha/2) * 100)
    
    logger.info(f"Bootstrap CI: {mean_val:.4f} [{lower_ci:.4f}, {upper_ci:.4f}]")
    
    return mean_val, lower_ci, upper_ci, metric_values

def compute_model_bootstrap_cis(models, X_test, y_test, groups_test, feature_cols, 
                               n_samples=1000, alpha=0.05):
    """Compute bootstrap confidence intervals for all models"""
    logger.info("Computing bootstrap confidence intervals for all models")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Processing {model_name}")
        
        # Get predictions
        y_pred = get_model_predictions(model, X_test, feature_cols, model_name)
        
        if y_pred is not None:
            # Compute bootstrap CIs for multiple metrics
            model_results = {}
            
            # Macro F1-score
            mean_f1, lower_f1, upper_f1, bootstrap_f1 = stratified_bootstrap_ci(
                y_test, y_pred, groups_test, f1_score, n_samples, alpha, SEED
            )
            model_results['macro_f1'] = {
                'mean': mean_f1,
                'lower': lower_f1,
                'upper': upper_f1,
                'bootstrap_values': bootstrap_f1
            }
            
            # Accuracy
            mean_acc, lower_acc, upper_acc, bootstrap_acc = stratified_bootstrap_ci(
                y_test, y_pred, groups_test, accuracy_score, n_samples, alpha, SEED + 1
            )
            model_results['accuracy'] = {
                'mean': mean_acc,
                'lower': lower_acc,
                'upper': upper_acc,
                'bootstrap_values': bootstrap_acc
            }
            
            # Balanced Accuracy
            mean_bal_acc, lower_bal_acc, upper_bal_acc, bootstrap_bal_acc = stratified_bootstrap_ci(
                y_test, y_pred, groups_test, balanced_accuracy_score, n_samples, alpha, SEED + 2
            )
            model_results['balanced_accuracy'] = {
                'mean': mean_bal_acc,
                'lower': lower_bal_acc,
                'upper': upper_bal_acc,
                'bootstrap_values': bootstrap_bal_acc
            }
            
            results[model_name] = model_results
            
            logger.info(f"{model_name} - Macro F1: {mean_f1:.4f} [{lower_f1:.4f}, {upper_f1:.4f}]")
        else:
            logger.warning(f"Failed to get predictions for {model_name}")
    
    return results

def plot_bootstrap_distributions(results, output_path):
    """Plot bootstrap distributions for each metric"""
    logger.info("Creating bootstrap distribution plots")
    
    metrics = ['macro_f1', 'accuracy', 'balanced_accuracy']
    metric_labels = ['Macro F1-Score', 'Accuracy', 'Balanced Accuracy']
    
    for metric, label in zip(metrics, metric_labels):
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for i, (model_name, model_results) in enumerate(results.items()):
            if metric in model_results:
                bootstrap_values = model_results[metric]['bootstrap_values']
                mean_val = model_results[metric]['mean']
                lower_ci = model_results[metric]['lower']
                upper_ci = model_results[metric]['upper']
                
                # Histogram of bootstrap values
                axes[i].hist(bootstrap_values, bins=50, alpha=0.7, density=True, edgecolor='black')
                axes[i].axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
                axes[i].axvline(lower_ci, color='orange', linestyle='--', linewidth=2, label=f'95% CI: [{lower_ci:.3f}, {upper_ci:.3f}]')
                axes[i].axvline(upper_ci, color='orange', linestyle='--', linewidth=2)
                
                axes[i].set_title(f'{model_name.replace("_", " ").title()}\n{label}', fontweight='bold')
                axes[i].set_xlabel(label)
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'distributions/bootstrap_distributions_{metric}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_ci_comparison(results, output_path):
    """Plot confidence interval comparison across models and metrics"""
    logger.info("Creating confidence interval comparison plots")
    
    metrics = ['macro_f1', 'accuracy', 'balanced_accuracy']
    metric_labels = ['Macro F1-Score', 'Accuracy', 'Balanced Accuracy']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        model_names = []
        means = []
        errors_lower = []
        errors_upper = []
        
        for model_name, model_results in results.items():
            if metric in model_results:
                model_names.append(model_name.replace('_', ' ').title())
                mean_val = model_results[metric]['mean']
                lower_ci = model_results[metric]['lower']
                upper_ci = model_results[metric]['upper']
                
                means.append(mean_val)
                errors_lower.append(mean_val - lower_ci)
                errors_upper.append(upper_ci - mean_val)
        
        if model_names:
            # Error bar plot
            axes[i].errorbar(range(len(model_names)), means, 
                           yerr=[errors_lower, errors_upper], 
                           fmt='o', capsize=10, elinewidth=3, markersize=8, 
                           linewidth=2, capthick=2)
            
            # Add horizontal line for random guess baseline
            if metric == 'macro_f1' or metric == 'accuracy':
                axes[i].axhline(y=1/3, color='red', linestyle='--', alpha=0.7, 
                              label='Random Guess (33.3%)')
                axes[i].legend()
            
            axes[i].set_xticks(range(len(model_names)))
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label}\n95% Bootstrap Confidence Intervals', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'comparisons/ci_comparison_all_metrics.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

def save_bootstrap_results(results, output_path):
    """Save bootstrap results to CSV files"""
    logger.info("Saving bootstrap results to CSV")
    
    # Create summary table
    summary_data = []
    
    for model_name, model_results in results.items():
        row = {'model': model_name}
        
        for metric in ['macro_f1', 'accuracy', 'balanced_accuracy']:
            if metric in model_results:
                row[f'{metric}_mean'] = model_results[metric]['mean']
                row[f'{metric}_lower'] = model_results[metric]['lower']
                row[f'{metric}_upper'] = model_results[metric]['upper']
                row[f'{metric}_ci_width'] = (model_results[metric]['upper'] - 
                                           model_results[metric]['lower'])
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'bootstrap_ci_summary.csv', index=False)
    
    # Save detailed bootstrap values
    for model_name, model_results in results.items():
        model_data = {}
        for metric in ['macro_f1', 'accuracy', 'balanced_accuracy']:
            if metric in model_results:
                model_data[metric] = model_results[metric]['bootstrap_values']
        
        if model_data:
            model_df = pd.DataFrame(model_data)
            model_df.to_csv(output_path / f'bootstrap_values_{model_name}.csv', index=False)
    
    logger.info("Bootstrap results saved successfully")

def generate_bootstrap_report(results, output_path, n_samples):
    """Generate a comprehensive bootstrap analysis report"""
    logger.info("Generating bootstrap analysis report")
    
    report_lines = [
        "# Bootstrap Confidence Intervals Analysis Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Analysis Configuration",
        f"- **Bootstrap Samples**: {n_samples:,}",
        f"- **Confidence Level**: 95%",
        f"- **Sampling Method**: Group-aware stratified (by player ID)",
        f"- **Random Seed**: {SEED}",
        "",
        "## Bootstrap Results Summary",
        "",
        "### Macro F1-Score",
    ]
    
    # Add results for each metric
    for metric, metric_label in [('macro_f1', 'Macro F1-Score'), 
                                ('accuracy', 'Accuracy'), 
                                ('balanced_accuracy', 'Balanced Accuracy')]:
        
        if metric != 'macro_f1':
            report_lines.extend(["", f"### {metric_label}"])
        
        metric_results = []
        for model_name, model_results in results.items():
            if metric in model_results:
                mean_val = model_results[metric]['mean']
                lower_ci = model_results[metric]['lower']
                upper_ci = model_results[metric]['upper']
                ci_width = upper_ci - lower_ci
                
                metric_results.append({
                    'model': model_name.replace('_', ' ').title(),
                    'mean': mean_val,
                    'lower': lower_ci,
                    'upper': upper_ci,
                    'width': ci_width
                })
        
        # Sort by mean performance
        metric_results.sort(key=lambda x: x['mean'], reverse=True)
        
        for result in metric_results:
            report_lines.append(
                f"- **{result['model']}**: {result['mean']:.4f} "
                f"[{result['lower']:.4f}, {result['upper']:.4f}] "
                f"(CI width: {result['width']:.4f})"
            )
    
    # Add interpretation section
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "### Model Performance Ranking",
        "Based on bootstrap confidence intervals:",
        ""
    ])
    
    # Rank models by macro F1 performance
    f1_rankings = []
    for model_name, model_results in results.items():
        if 'macro_f1' in model_results:
            f1_rankings.append((
                model_name.replace('_', ' ').title(),
                model_results['macro_f1']['mean'],
                model_results['macro_f1']['lower'],
                model_results['macro_f1']['upper']
            ))
    
    f1_rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, mean_f1, lower, upper) in enumerate(f1_rankings, 1):
        report_lines.append(f"{i}. **{model}**: {mean_f1:.4f} [{lower:.4f}, {upper:.4f}]")
    
    report_lines.extend([
        "",
        "### Statistical Significance",
        "Models with non-overlapping confidence intervals show statistically significant differences.",
        "",
        "### Uncertainty Analysis",
        "- Bootstrap confidence intervals provide robust uncertainty estimates",
        "- Group-aware sampling prevents optimistic bias from player-level dependencies",
        "- Narrow confidence intervals indicate stable model performance",
        "",
        "## Recommendations",
        "1. **Model Selection**: Choose models with highest mean performance and narrow CIs",
        "2. **Deployment Confidence**: Consider CI width when assessing deployment readiness",
        "3. **Further Analysis**: Investigate models with overlapping CIs for practical differences",
        "4. **Validation**: Use group-aware validation in future model development",
    ])
    
    # Save report
    with open(output_path / 'bootstrap_analysis_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info("Bootstrap analysis report saved")

def main():
    parser = argparse.ArgumentParser(description='Compute bootstrap confidence intervals for model performance')
    parser.add_argument('--data', required=True, help='Path to processed data CSV')
    parser.add_argument('--splits', required=True, help='Path to test split JSON file')
    parser.add_argument('--output_dir', required=True, help='Directory to save bootstrap results')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of bootstrap samples')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for CI')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(SEED)
    
    logger.info(f"Starting bootstrap confidence interval analysis")
    logger.info(f"Data: {args.data}")
    logger.info(f"Test splits: {args.splits}")
    logger.info(f"Bootstrap samples: {args.n_samples}")
    logger.info(f"Significance level: {args.alpha}")
    logger.info(f"Random seed: {SEED}")
    
    # Setup output directory
    output_path = setup_output_directory(args.output_dir)
    
    # Model paths
    model_paths = {
        'logistic_regression': 'models/lr.pkl',
        'random_forest': 'models/rf.pkl',
        'lightgbm': 'models/lgbm.pkl',
        'gaussian_process': 'models/gp.pkl'
    }
    
    # Load data and models
    X_test, y_test, groups_test, models, feature_cols = load_data_and_models(
        args.data, args.splits, model_paths
    )
    
    if not models:
        logger.error("No models were successfully loaded")
        return
    
    # Compute bootstrap confidence intervals
    results = compute_model_bootstrap_cis(
        models, X_test, y_test, groups_test, feature_cols, 
        args.n_samples, args.alpha
    )
    
    if not results:
        logger.error("No bootstrap results were computed")
        return
    
    # Create visualizations
    plot_bootstrap_distributions(results, output_path)
    plot_ci_comparison(results, output_path)
    
    # Save results
    save_bootstrap_results(results, output_path)
    
    # Generate report
    generate_bootstrap_report(results, output_path, args.n_samples)
    
    logger.info("Bootstrap confidence interval analysis completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 
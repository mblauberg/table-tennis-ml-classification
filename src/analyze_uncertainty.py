#!/usr/bin/env python3
"""
Uncertainty Analysis for COMP4702 Assignment

Performs detailed uncertainty analysis for the Sparse Gaussian Process model
to quantify prediction confidence and analyze model calibration.

Week 11 Concepts:
- Uncertainty quantification in Bayesian models
- Model calibration and reliability
- Expected Calibration Error (ECE)
- Epistemic vs. aleatoric uncertainty
"""

import argparse
import pandas as pd
import numpy as np
import json
import torch
import gpytorch
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss, classification_report
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
    """Create output directory structure for uncertainty analysis"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'calibration').mkdir(exist_ok=True)
    (output_path / 'distributions').mkdir(exist_ok=True)
    (output_path / 'analysis').mkdir(exist_ok=True)
    
    logger.info(f"Created output directory structure in {output_dir}")
    return output_path

def load_data_and_model(data_path, splits_path, model_path):
    """Load test data and trained GP model"""
    logger.info("Loading test data and GP model")
    
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
    logger.info(f"Class distribution in test: {y_test.value_counts().sort_index().tolist()}")
    
    # Load preprocessing components
    scaler = joblib.load('models/scaler.pkl')
    X_test_scaled = scaler.transform(X_test[feature_cols])
    
    # Load PCA if available
    pca_path = Path('models/pca.pkl')
    if pca_path.exists():
        pca = joblib.load(pca_path)
        X_test_processed = pca.transform(X_test_scaled)
        logger.info(f"Applied PCA transformation: {X_test_processed.shape}")
    else:
        X_test_processed = X_test_scaled
        logger.info("PCA not found, using scaled features directly")
    
    # Load GP model
    try:
        checkpoint = torch.load(model_path)
        inducing_points = checkpoint['inducing_points']
        model = GPClassificationModel(inducing_points)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            targets=torch.tensor([0, 1, 2]), alpha=0.01)
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        logger.info(f"Loaded GP model from {model_path}")
        
        return X_test_processed, y_test, groups_test, model, likelihood, feature_cols
        
    except Exception as e:
        logger.error(f"Error loading GP model: {e}")
        return None, None, None, None, None, None

def get_gp_predictions_with_uncertainty(model, likelihood, X_test, n_samples=100):
    """Get GP predictions with uncertainty estimates"""
    logger.info("Computing GP predictions with uncertainty estimates")
    
    model.eval()
    likelihood.eval()
    
    # Convert to tensor
    test_x = torch.tensor(X_test, dtype=torch.float32)
    
    # Monte Carlo sampling for uncertainty estimation
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Get posterior samples
            observed_pred = likelihood(model(test_x))
            probs = observed_pred.probs.numpy()
            predictions.append(np.argmax(probs, axis=1))
            confidences.append(np.max(probs, axis=1))
    
    # Aggregate results
    predictions = np.array(predictions)  # shape: (n_samples, n_test)
    confidences = np.array(confidences)  # shape: (n_samples, n_test)
    
    # Final predictions (mode)
    final_predictions = []
    prediction_uncertainty = []
    confidence_mean = []
    confidence_std = []
    
    for i in range(test_x.shape[0]):
        # Most frequent prediction
        pred_counts = np.bincount(predictions[:, i], minlength=3)
        final_pred = np.argmax(pred_counts)
        final_predictions.append(final_pred)
        
        # Prediction uncertainty (entropy over prediction distribution)
        pred_probs = pred_counts / n_samples
        pred_entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10))
        prediction_uncertainty.append(pred_entropy)
        
        # Confidence statistics
        confidence_mean.append(np.mean(confidences[:, i]))
        confidence_std.append(np.std(confidences[:, i]))
    
    # Get final probabilistic predictions
    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        final_probs = observed_pred.probs.numpy()
    
    logger.info(f"Generated predictions for {len(final_predictions)} test samples")
    
    return (np.array(final_predictions), final_probs, 
            np.array(prediction_uncertainty), np.array(confidence_mean), np.array(confidence_std))

def compute_calibration(y_true, y_prob, n_bins=10):
    """Compute calibration metrics"""
    bin_count = np.zeros(n_bins)
    bin_sum_confidence = np.zeros(n_bins)
    bin_sum_accuracy = np.zeros(n_bins)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    max_probs = np.max(y_prob, axis=1)
    predicted_classes = np.argmax(y_prob, axis=1)
    correct_predictions = (predicted_classes == y_true).astype(float)
    
    for i in range(n_bins):
        bin_indices = np.where((max_probs > bin_boundaries[i]) & 
                              (max_probs <= bin_boundaries[i+1]))[0]
        if len(bin_indices) > 0:
            bin_count[i] = len(bin_indices)
            bin_sum_confidence[i] = np.sum(max_probs[bin_indices])
            bin_sum_accuracy[i] = np.sum(correct_predictions[bin_indices])
    
    # Avoid division by zero
    valid_bins = bin_count > 0
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    
    bin_accuracy[valid_bins] = bin_sum_accuracy[valid_bins] / bin_count[valid_bins]
    bin_confidence[valid_bins] = bin_sum_confidence[valid_bins] / bin_count[valid_bins]
    
    # Calculate ECE
    total_samples = np.sum(bin_count)
    if total_samples > 0:
        ece = np.sum(bin_count * np.abs(bin_accuracy - bin_confidence)) / total_samples
    else:
        ece = 0.0
    
    return bin_boundaries, bin_accuracy, bin_confidence, bin_count, ece

def plot_reliability_diagram(bin_boundaries, bin_accuracy, bin_confidence, bin_count, ece, output_path):
    """Plot reliability diagram"""
    logger.info("Creating reliability diagram")
    
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    valid_bins = bin_count > 0
    
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Reliability bars
    if np.any(valid_bins):
        plt.bar(bin_centers[valid_bins], bin_accuracy[valid_bins], 
               width=0.08, alpha=0.7, edgecolor='black', label='Accuracy')
        
        # Confidence line
        plt.plot(bin_centers[valid_bins], bin_confidence[valid_bins], 
                'ro-', linewidth=2, markersize=8, label='Confidence')
        
        # Add sample counts as text
        for i, (center, accuracy, count) in enumerate(zip(bin_centers[valid_bins], 
                                                          bin_accuracy[valid_bins], 
                                                          bin_count[valid_bins])):
            plt.text(center, accuracy + 0.02, f'{int(count)}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Reliability Diagram\nExpected Calibration Error: {ece:.4f}', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / 'calibration/reliability_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_distributions(prediction_uncertainty, confidence_mean, confidence_std, 
                                  y_true, y_pred, output_path):
    """Plot uncertainty distributions"""
    logger.info("Creating uncertainty distribution plots")
    
    class_labels = {0: 'Air Swing', 1: 'Full Power', 2: 'Stable'}
    
    # Uncertainty by class
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prediction uncertainty histogram
    axes[0, 0].hist(prediction_uncertainty, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Prediction Uncertainty (Entropy)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Prediction Uncertainty', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confidence mean histogram
    axes[0, 1].hist(confidence_mean, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Mean Confidence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Mean Confidence', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty by true class
    for class_id, class_name in class_labels.items():
        class_mask = y_true == class_id
        if np.any(class_mask):
            axes[1, 0].hist(prediction_uncertainty[class_mask], bins=20, alpha=0.6, 
                           label=f'{class_name} (n={np.sum(class_mask)})')
    
    axes[1, 0].set_xlabel('Prediction Uncertainty')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Uncertainty Distribution by True Class', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Misclassification vs uncertainty
    correct = (y_true == y_pred)
    incorrect = ~correct
    
    if np.any(correct):
        axes[1, 1].scatter(prediction_uncertainty[correct], confidence_mean[correct], 
                          alpha=0.6, label=f'Correct ({np.sum(correct)})', s=20)
    if np.any(incorrect):
        axes[1, 1].scatter(prediction_uncertainty[incorrect], confidence_mean[incorrect], 
                          alpha=0.6, label=f'Incorrect ({np.sum(incorrect)})', s=20, color='red')
    
    axes[1, 1].set_xlabel('Prediction Uncertainty')
    axes[1, 1].set_ylabel('Mean Confidence')
    axes[1, 1].set_title('Misclassification vs. Uncertainty', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'distributions/uncertainty_distributions.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

def analyze_high_uncertainty_samples(X_test_processed, y_true, y_pred, prediction_uncertainty, 
                                    confidence_mean, groups_test, output_path, top_n=20):
    """Analyze samples with highest uncertainty"""
    logger.info(f"Analyzing top {top_n} highest uncertainty samples")
    
    # Get indices of highest uncertainty samples
    high_uncertainty_indices = np.argsort(prediction_uncertainty)[-top_n:]
    
    # Create analysis dataframe
    analysis_data = []
    for idx in high_uncertainty_indices:
        analysis_data.append({
            'sample_index': idx,
            'player_id': groups_test.iloc[idx],
            'true_class': y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx],
            'predicted_class': y_pred[idx],
            'prediction_uncertainty': prediction_uncertainty[idx],
            'mean_confidence': confidence_mean[idx],
            'correct': (y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx]) == y_pred[idx]
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.sort_values('prediction_uncertainty', ascending=False)
    
    # Save analysis
    analysis_df.to_csv(output_path / 'analysis/high_uncertainty_samples.csv', index=False)
    
    # Summary statistics
    logger.info(f"High uncertainty samples analysis:")
    logger.info(f"- Mean uncertainty: {analysis_df['prediction_uncertainty'].mean():.4f}")
    logger.info(f"- Mean confidence: {analysis_df['mean_confidence'].mean():.4f}")
    logger.info(f"- Accuracy: {analysis_df['correct'].mean():.4f}")
    
    # Plot high uncertainty samples
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Uncertainty vs confidence for high uncertainty samples
    colors = ['green' if correct else 'red' for correct in analysis_df['correct']]
    scatter = axes[0].scatter(analysis_df['prediction_uncertainty'], analysis_df['mean_confidence'], 
                             c=colors, alpha=0.7, s=50)
    axes[0].set_xlabel('Prediction Uncertainty')
    axes[0].set_ylabel('Mean Confidence')
    axes[0].set_title(f'Top {top_n} Highest Uncertainty Samples', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add legend manually
    axes[0].scatter([], [], c='green', alpha=0.7, s=50, label='Correct')
    axes[0].scatter([], [], c='red', alpha=0.7, s=50, label='Incorrect')
    axes[0].legend()
    
    # Class distribution of high uncertainty samples
    class_counts = analysis_df['true_class'].value_counts().sort_index()
    class_labels = ['Air Swing', 'Full Power', 'Stable']
    
    bars = axes[1].bar(range(len(class_counts)), class_counts.values, 
                      color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('True Class')
    axes[1].set_ylabel('Number of High Uncertainty Samples')
    axes[1].set_title('Class Distribution of High Uncertainty Samples', fontweight='bold')
    axes[1].set_xticks(range(len(class_labels)))
    axes[1].set_xticklabels(class_labels)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'analysis/high_uncertainty_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    return analysis_df

def calculate_additional_metrics(y_true, y_prob, y_pred):
    """Calculate additional uncertainty and performance metrics"""
    logger.info("Calculating additional metrics")
    
    # Brier score for each class
    brier_scores = []
    for class_id in range(3):
        y_true_binary = (y_true == class_id).astype(int)
        y_prob_class = y_prob[:, class_id]
        brier_score = brier_score_loss(y_true_binary, y_prob_class)
        brier_scores.append(brier_score)
    
    # Overall metrics
    max_probs = np.max(y_prob, axis=1)
    entropy = -np.sum(y_prob * np.log(y_prob + 1e-10), axis=1)
    
    metrics = {
        'brier_score_class_0': brier_scores[0],
        'brier_score_class_1': brier_scores[1], 
        'brier_score_class_2': brier_scores[2],
        'mean_brier_score': np.mean(brier_scores),
        'mean_max_probability': np.mean(max_probs),
        'mean_entropy': np.mean(entropy),
        'accuracy': np.mean(y_true == y_pred)
    }
    
    return metrics

def generate_uncertainty_report(metrics, ece, analysis_df, output_path):
    """Generate comprehensive uncertainty analysis report"""
    logger.info("Generating uncertainty analysis report")
    
    report_lines = [
        "# Gaussian Process Uncertainty Analysis Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report provides comprehensive uncertainty analysis for the Sparse Gaussian Process",
        "model used in table tennis swing classification. The analysis evaluates model calibration,",
        "prediction confidence, and identifies high-uncertainty samples for further investigation.",
        "",
        "## Calibration Analysis",
        "",
        f"**Expected Calibration Error (ECE)**: {ece:.4f}",
        "",
        "The ECE measures how well the model's predicted probabilities match actual accuracy.",
        "Lower values indicate better calibration. An ECE < 0.1 is generally considered well-calibrated.",
        "",
        "## Performance Metrics",
        "",
        f"- **Model Accuracy**: {metrics['accuracy']:.4f}",
        f"- **Mean Brier Score**: {metrics['mean_brier_score']:.4f}",
        f"- **Mean Maximum Probability**: {metrics['mean_max_probability']:.4f}",
        f"- **Mean Prediction Entropy**: {metrics['mean_entropy']:.4f}",
        "",
        "### Class-Specific Brier Scores",
        f"- **Air Swing (Class 0)**: {metrics['brier_score_class_0']:.4f}",
        f"- **Full Power (Class 1)**: {metrics['brier_score_class_1']:.4f}",
        f"- **Stable (Class 2)**: {metrics['brier_score_class_2']:.4f}",
        "",
        "## High Uncertainty Analysis",
        "",
        f"**Top 20 Highest Uncertainty Samples:**",
        f"- Mean uncertainty: {analysis_df['prediction_uncertainty'].mean():.4f}",
        f"- Mean confidence: {analysis_df['mean_confidence'].mean():.4f}",
        f"- Accuracy on high uncertainty samples: {analysis_df['correct'].mean():.4f}",
        "",
        "High uncertainty samples often indicate:",
        "1. Ambiguous cases near decision boundaries",
        "2. Outliers or unusual patterns",
        "3. Areas where more training data would be beneficial",
        "",
        "## Key Findings",
        "",
        "### Model Calibration",
    ]
    
    if ece < 0.05:
        calibration_assessment = "excellent (ECE < 0.05)"
    elif ece < 0.1:
        calibration_assessment = "good (ECE < 0.1)"
    elif ece < 0.2:
        calibration_assessment = "fair (ECE < 0.2)"
    else:
        calibration_assessment = "poor (ECE >= 0.2)"
    
    report_lines.extend([
        f"The model shows **{calibration_assessment}** calibration.",
        "",
        "### Uncertainty Patterns",
        f"- Average entropy across all predictions: {metrics['mean_entropy']:.4f}",
        f"- Average maximum probability: {metrics['mean_max_probability']:.4f}",
        "",
        "### Class-Specific Performance",
        "Brier scores indicate prediction quality for each class:",
        "- Lower scores indicate better probability estimates",
        "- Scores range from 0 (perfect) to 1 (worst possible)",
        "",
        "## Recommendations",
        "",
        "### Model Deployment",
    ])
    
    if ece < 0.1:
        report_lines.append("✅ **Model is well-calibrated** and suitable for deployment with confidence estimates")
    else:
        report_lines.append("⚠️ **Model calibration could be improved** before deployment")
    
    report_lines.extend([
        "",
        "### Uncertainty-Aware Decisions",
        "1. **High-confidence predictions** (low entropy): Accept automatically",
        "2. **Medium-confidence predictions**: Flag for review",
        "3. **High-uncertainty predictions**: Require manual inspection",
        "",
        "### Future Improvements",
        "1. Collect more training data for high-uncertainty regions",
        "2. Consider ensemble methods to reduce prediction uncertainty",
        "3. Implement temperature scaling for better calibration",
        "4. Use uncertainty estimates for active learning strategies",
        "",
        "## Technical Details",
        "",
        f"- **Model Type**: Sparse Gaussian Process with Variational Inference",
        f"- **Uncertainty Estimation**: Monte Carlo sampling (100 samples)",
        f"- **Calibration Method**: Equal-width binning (10 bins)",
        f"- **Test Set Size**: {len(analysis_df)} samples",
    ])
    
    # Save report
    with open(output_path / 'uncertainty_analysis_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info("Uncertainty analysis report saved")

def main():
    parser = argparse.ArgumentParser(description='Analyze GP model uncertainty and calibration')
    parser.add_argument('--data', required=True, help='Path to processed data CSV')
    parser.add_argument('--splits', required=True, help='Path to test split JSON')
    parser.add_argument('--model', required=True, help='Path to trained GP model')
    parser.add_argument('--output_dir', required=True, help='Directory to save uncertainty analysis')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of MC samples for uncertainty')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    logger.info(f"Starting GP uncertainty analysis")
    logger.info(f"Data: {args.data}")
    logger.info(f"Test splits: {args.splits}")
    logger.info(f"Model: {args.model}")
    logger.info(f"MC samples: {args.n_samples}")
    
    # Setup output directory
    output_path = setup_output_directory(args.output_dir)
    
    # Load data and model
    X_test, y_test, groups_test, model, likelihood, feature_cols = load_data_and_model(
        args.data, args.splits, args.model
    )
    
    if model is None:
        logger.error("Failed to load GP model")
        return
    
    # Get predictions with uncertainty
    y_pred, y_prob, prediction_uncertainty, confidence_mean, confidence_std = get_gp_predictions_with_uncertainty(
        model, likelihood, X_test, args.n_samples
    )
    
    # Compute calibration metrics
    bin_boundaries, bin_accuracy, bin_confidence, bin_count, ece = compute_calibration(
        y_test, y_prob
    )
    
    # Create visualizations
    plot_reliability_diagram(bin_boundaries, bin_accuracy, bin_confidence, bin_count, ece, output_path)
    plot_uncertainty_distributions(prediction_uncertainty, confidence_mean, confidence_std, 
                                  y_test, y_pred, output_path)
    
    # Analyze high uncertainty samples
    analysis_df = analyze_high_uncertainty_samples(
        X_test, y_test, y_pred, prediction_uncertainty, confidence_mean, groups_test, output_path
    )
    
    # Calculate additional metrics
    metrics = calculate_additional_metrics(y_test, y_prob, y_pred)
    
    # Generate report
    generate_uncertainty_report(metrics, ece, analysis_df, output_path)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path / 'uncertainty_metrics.csv', index=False)
    
    logger.info("GP uncertainty analysis completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 
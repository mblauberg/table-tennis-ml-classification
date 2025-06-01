#!/usr/bin/env python3
"""
Generate plots for trained Sparse Gaussian Process model

This script loads a saved GP model and generates:
- Training loss curve
- Uncertainty distribution plots
- Reliability diagram for calibration assessment
- Confusion matrix
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_and_model(data_path, model_path, scaler_path, pca_path, train_indices_path, val_indices_path):
    """
    Load data, trained model, scaler, and PCA transformer
    
    Args:
        data_path: Path to processed CSV file
        model_path: Path to saved model
        scaler_path: Path to saved scaler
        pca_path: Path to saved PCA transformer
        train_indices_path: Path to training indices JSON
        val_indices_path: Path to validation indices JSON
        
    Returns:
        tuple: (model, likelihood, scaler, pca, X_train, y_train, X_val, y_val, feature_names)
    """
    logger.info("Loading data, model, scaler, and PCA...")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load split indices
    with open(train_indices_path, 'r') as f:
        train_indices = json.load(f)
    with open(val_indices_path, 'r') as f:
        val_indices = json.load(f)
    
    # Identify feature columns
    exclude_cols = ['id', 'date', 'testmode', 'teststage', 'fileindex', 'age', 'playYears', 'height', 'weight']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    
    # Split data
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    
    # Load transformers
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    
    # Load model and likelihood
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Reconstruct model architecture (assuming standard configuration)
    n_components = pca.n_components_
    n_inducing = checkpoint['inducing_points'].shape[0] if 'inducing_points' in checkpoint else 1000
    
    # Create dummy inducing points for model initialization
    dummy_inducing = torch.randn(n_inducing, n_components)
    
    # Define the GP model class (must match training)
    class GPClassificationModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, num_classes=3):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0), batch_shape=torch.Size([num_classes])
            )
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, 
                    learn_inducing_locations=True
                ), num_tasks=num_classes
            )
            super().__init__(variational_strategy)
            
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes])),
                batch_shape=torch.Size([num_classes])
            )
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # Initialize model and likelihood
    model = GPClassificationModel(dummy_inducing, num_classes=3)
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_outputs, num_classes=3)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    likelihood.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Scaler loaded from {scaler_path}")
    logger.info(f"PCA loaded from {pca_path}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"PCA components: {n_components}")
    
    return model, likelihood, scaler, pca, X_train, y_train, X_val, y_val, feature_cols

def plot_training_loss_from_history(loss_history_path, output_dir):
    """
    Plot training loss curve from saved loss history
    
    Args:
        loss_history_path: Path to saved loss history
        output_dir: Directory to save plots
    """
    try:
        # Try to load loss history
        losses = joblib.load(loss_history_path)
        logger.info(f"Loaded training loss history from {loss_history_path}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.title('GP Training Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_dir) / 'gp_training_loss.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training loss plot saved to {plot_path}")
        
    except FileNotFoundError:
        logger.warning(f"Loss history file not found at {loss_history_path}. Skipping training loss plot.")
    except Exception as e:
        logger.warning(f"Could not load loss history: {e}. Skipping training loss plot.")

def plot_uncertainty_distribution(model, likelihood, scaler, pca, X_val, y_val, output_dir):
    """
    Plot uncertainty distribution by correctness
    
    Args:
        model: Trained GP model
        likelihood: GP likelihood
        scaler: Fitted scaler
        pca: Fitted PCA
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save plots
    """
    logger.info("Plotting uncertainty distribution...")
    
    # Prepare data
    X_val_scaled = scaler.transform(X_val)
    X_val_pca = pca.transform(X_val_scaled)
    X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float32)
    
    # Get predictions and uncertainties
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(X_val_tensor)
        pred_dist = likelihood(output)
        probabilities = pred_dist.probs.numpy()
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate uncertainty as entropy
        uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = predictions == y_val.values
    correct_uncertainties = uncertainties[correct_mask]
    incorrect_uncertainties = uncertainties[~correct_mask]
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(correct_uncertainties, bins=50, alpha=0.7, label='Correct', density=True)
    plt.hist(incorrect_uncertainties, bins=50, alpha=0.7, label='Incorrect', density=True)
    plt.xlabel('Uncertainty (Entropy)')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    data = [correct_uncertainties, incorrect_uncertainties]
    plt.boxplot(data, labels=['Correct', 'Incorrect'])
    plt.ylabel('Uncertainty (Entropy)')
    plt.title('Uncertainty by Correctness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'gp_uncertainty_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Uncertainty distribution plot saved to {plot_path}")
    
    return uncertainties, predictions, probabilities

def compute_ece(probabilities, predictions, y_true, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        probabilities: Predicted probabilities
        predictions: Predicted classes
        y_true: True labels
        n_bins: Number of bins for calibration
        
    Returns:
        float: ECE value
    """
    # Get maximum probabilities (confidence)
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == y_true.values).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find examples in this bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = in_bin.sum() / len(confidences)
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def plot_reliability_diagram(probabilities, predictions, y_val, output_dir, n_bins=10):
    """
    Generate reliability diagram for calibration assessment
    
    Args:
        probabilities: Predicted probabilities
        predictions: Predicted classes
        y_val: True labels
        output_dir: Directory to save plots
        n_bins: Number of bins for calibration
    """
    logger.info("Generating reliability diagram...")
    
    # Get maximum probabilities (confidence)
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == y_val.values).astype(float)
    
    # Compute ECE
    ece = compute_ece(probabilities, predictions, y_val, n_bins)
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # Create reliability diagram
    plt.figure(figsize=(8, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
    
    # Compute bin statistics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()
            
            bin_centers.append(confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(prop_in_bin)
    
    # Plot calibration curve
    if bin_centers:
        plt.scatter(bin_centers, bin_accuracies, s=[c*10 for c in bin_counts], 
                   alpha=0.7, c='red', label='Model Calibration')
        
        # Connect points
        plt.plot(bin_centers, bin_accuracies, 'r-', alpha=0.7)
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Reliability Diagram (ECE = {ece:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save plot
    plot_path = Path(output_dir) / 'gp_reliability_diagram.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Reliability diagram saved to {plot_path}")
    
    return ece

def plot_confusion_matrix(predictions, y_val, output_dir):
    """
    Plot confusion matrix for validation set
    
    Args:
        predictions: Model predictions
        y_val: True labels
        output_dir: Directory to save plots
    """
    logger.info("Generating confusion matrix...")
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, predictions)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['air swing', 'full power', 'stable'],
                yticklabels=['air swing', 'full power', 'stable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Sparse Gaussian Process - Confusion Matrix')
    
    # Save plot
    plot_path = Path(output_dir) / 'gp_confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {plot_path}")

def plot_prediction_confidence_analysis(probabilities, predictions, y_val, output_dir):
    """
    Plot prediction confidence analysis
    
    Args:
        probabilities: Predicted probabilities
        predictions: Predicted classes
        y_val: True labels
        output_dir: Directory to save plots
    """
    logger.info("Generating prediction confidence analysis...")
    
    # Get maximum probabilities (confidence)
    confidences = np.max(probabilities, axis=1)
    correct_mask = predictions == y_val.values
    
    plt.figure(figsize=(15, 5))
    
    # Confidence distribution
    plt.subplot(1, 3, 1)
    plt.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.grid(True, alpha=0.3)
    
    # Confidence vs Accuracy
    plt.subplot(1, 3, 2)
    conf_bins = np.linspace(0, 1, 11)
    bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
    bin_accuracies = []
    
    for i in range(len(conf_bins) - 1):
        mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if mask.sum() > 0:
            bin_acc = correct_mask[mask].mean()
            bin_accuracies.append(bin_acc)
        else:
            bin_accuracies.append(0)
    
    plt.plot(bin_centers, bin_accuracies, 'o-', label='Empirical')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
    plt.xlabel('Confidence Bin')
    plt.ylabel('Accuracy')
    plt.title('Confidence vs Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Class-wise confidence
    plt.subplot(1, 3, 3)
    class_names = ['Air Swing', 'Full Power', 'Stable']
    for class_idx in range(3):
        class_mask = predictions == class_idx
        if class_mask.sum() > 0:
            class_conf = confidences[class_mask]
            plt.hist(class_conf, bins=20, alpha=0.6, label=class_names[class_idx])
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence by Predicted Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'gp_confidence_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confidence analysis plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots for trained Sparse Gaussian Process model')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--model', required=True, help='Path to saved model (gp.pkl)')
    parser.add_argument('--scaler', required=True, help='Path to saved scaler (scaler.pkl)')
    parser.add_argument('--pca', required=True, help='Path to saved PCA (pca.pkl)')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    parser.add_argument('--loss_history', help='Path to saved loss history (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting plot generation for Sparse Gaussian Process...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Model file: {args.model}")
    logger.info(f"Scaler file: {args.scaler}")
    logger.info(f"PCA file: {args.pca}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data and model
    model, likelihood, scaler, pca, X_train, y_train, X_val, y_val, feature_names = load_data_and_model(
        args.data, args.model, args.scaler, args.pca, args.splits[0], args.splits[1]
    )
    
    # Generate uncertainty and prediction analysis
    uncertainties, predictions, probabilities = plot_uncertainty_distribution(
        model, likelihood, scaler, pca, X_val, y_val, output_dir
    )
    
    # Generate other plots
    plot_reliability_diagram(probabilities, predictions, y_val, output_dir)
    plot_confusion_matrix(predictions, y_val, output_dir)
    plot_prediction_confidence_analysis(probabilities, predictions, y_val, output_dir)
    
    # Plot training loss if history available
    if args.loss_history:
        plot_training_loss_from_history(args.loss_history, output_dir)
    
    logger.info("Plot generation completed successfully!")

if __name__ == "__main__":
    main() 
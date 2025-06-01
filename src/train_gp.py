#!/usr/bin/env python3
"""
Sparse Gaussian Process Training Module for COMP4702 Assignment

Implements Sparse Gaussian Process classifier with uncertainty quantification,
PCA dimensionality reduction, and reliability diagrams.

Week 11 Concepts:
- Sparse Gaussian Processes
- Uncertainty quantification
- Reliability diagrams
- Expected Calibration Error (ECE)
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.calibration import calibration_curve

# Random seed for reproducibility
SEED = 123

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seeds():
    """Set random seeds for reproducibility"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

class GPClassificationModel(gpytorch.models.ApproximateGP):
    """
    Sparse Gaussian Process Classification Model using variational inference
    """
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

def load_data_splits(data_path, train_indices_path, val_indices_path):
    """
    Load data and split indices
    
    Args:
        data_path: Path to processed CSV file
        train_indices_path: Path to training indices JSON
        val_indices_path: Path to validation indices JSON
        
    Returns:
        tuple: (X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names)
    """
    logger.info("Loading data and splits...")
    
    # Load processed data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load split indices
    with open(train_indices_path, 'r') as f:
        train_indices = json.load(f)
    with open(val_indices_path, 'r') as f:
        val_indices = json.load(f)
    
    logger.info(f"Train indices: {len(train_indices)}")
    logger.info(f"Val indices: {len(val_indices)}")
    
    # Identify feature columns (exclude id, testmode, and categorical columns)
    exclude_cols = ['id', 'testmode']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    # Split data
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    groups_train = groups.iloc[train_indices]
    
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    groups_val = groups.iloc[val_indices]
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    
    return X_train, y_train, groups_train, X_val, y_val, groups_val, feature_cols

def prepare_features(X_train, X_val, scaler_path, n_components=20):
    """
    Load existing scaler, scale features, and apply PCA dimensionality reduction
    
    Args:
        X_train: Training features
        X_val: Validation features
        scaler_path: Path to existing scaler
        n_components: Number of PCA components
        
    Returns:
        tuple: (X_train_pca, X_val_pca, scaler, pca)
    """
    logger.info("Preparing features...")
    
    # Load existing scaler
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded existing scaler from {scaler_path}")
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply PCA dimensionality reduction
    logger.info(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    logger.info(f"Reduced feature dimensionality: {X_train_pca.shape[1]}")
    
    return X_train_pca, X_val_pca, scaler, pca

def select_inducing_points(X_train, n_inducing=1000):
    """
    Select inducing points using K-means clustering
    
    Args:
        X_train: Training features (PCA transformed)
        n_inducing: Number of inducing points
        
    Returns:
        torch.Tensor: Inducing points
    """
    logger.info(f"Selecting {n_inducing} inducing points using K-means...")
    
    # Use fewer points if training set is smaller
    n_inducing = min(n_inducing, len(X_train))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_inducing, random_state=SEED, n_init=10)
    kmeans.fit(X_train)
    
    # Use cluster centroids as inducing points
    inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    logger.info(f"Selected inducing points shape: {inducing_points.shape}")
    
    return inducing_points

def train_gp_model(X_train, y_train, inducing_points, num_epochs=500, lr=0.01):
    """
    Train the Sparse Gaussian Process model
    
    Args:
        X_train: Training features (PCA transformed)
        y_train: Training labels
        inducing_points: Inducing points tensor
        num_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        tuple: (model, likelihood)
    """
    logger.info("Training Sparse Gaussian Process model...")
    
    # Convert data to tensors
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train.values, dtype=torch.long)
    
    # Initialize model and likelihood
    model = GPClassificationModel(inducing_points, num_classes=3)
    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(targets=train_y, alpha=0.01, learn_additional_noise=True)
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -likelihood(output, train_y).sum()
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            logger.info(f'Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}')
    
    logger.info(f"Training completed. Final loss: {losses[-1]:.4f}")
    
    return model, likelihood, losses

def evaluate_model(model, likelihood, X_val, y_val):
    """
    Evaluate the GP model on validation set
    
    Args:
        model: Trained GP model
        likelihood: GP likelihood
        X_val: Validation features (PCA transformed)
        y_val: Validation labels
        
    Returns:
        tuple: (predictions, probabilities, uncertainties)
    """
    logger.info("Evaluating model on validation set...")
    
    # Set to evaluation mode
    model.eval()
    likelihood.eval()
    
    # Convert to tensor
    test_x = torch.tensor(X_val, dtype=torch.float32)
    
    with torch.no_grad():
        # Make predictions
        observed_pred = likelihood(model(test_x))
        
        # Get predicted probabilities
        probabilities = observed_pred.probs.numpy()
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate uncertainties (entropy)
        uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, predictions)
    f1_macro = f1_score(y_val, predictions, average='macro')
    f1_micro = f1_score(y_val, predictions, average='micro')
    f1_weighted = f1_score(y_val, predictions, average='weighted')
    
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation F1-macro: {f1_macro:.4f}")
    logger.info(f"Validation F1-micro: {f1_micro:.4f}")
    logger.info(f"Validation F1-weighted: {f1_weighted:.4f}")
    logger.info(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
    
    # Classification report
    logger.info("Classification Report:")
    print(classification_report(y_val, predictions, target_names=['air swing', 'full power', 'stable']))
    
    return predictions, probabilities, uncertainties

def plot_training_loss(losses, output_dir):
    """
    Plot training loss curve
    
    Args:
        losses: List of training losses
        output_dir: Directory to save plots
    """
    logger.info("Plotting training loss curve...")
    
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

def plot_uncertainty_distribution(uncertainties, predictions, y_true, output_dir):
    """
    Plot uncertainty distribution by correctness
    
    Args:
        uncertainties: Prediction uncertainties
        predictions: Model predictions
        y_true: True labels
        output_dir: Directory to save plots
    """
    logger.info("Plotting uncertainty distribution...")
    
    # Separate correct and incorrect predictions
    correct_mask = predictions == y_true.values
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
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def plot_reliability_diagram(probabilities, predictions, y_true, output_dir, n_bins=10):
    """
    Generate reliability diagram for calibration assessment
    
    Args:
        probabilities: Predicted probabilities
        predictions: Predicted classes
        y_true: True labels
        output_dir: Directory to save plots
        n_bins: Number of bins for calibration
    """
    logger.info("Generating reliability diagram...")
    
    # Get maximum probabilities (confidence)
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == y_true.values).astype(float)
    
    # Compute ECE
    ece = compute_ece(probabilities, predictions, y_true, n_bins)
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

def main():
    parser = argparse.ArgumentParser(description='Sparse Gaussian Process training for table tennis classification')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Directory to save model and plots')
    parser.add_argument('--n_components', type=int, default=20, help='Number of PCA components')
    parser.add_argument('--n_inducing', type=int, default=1000, help='Number of inducing points')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    set_seeds()
    
    logger.info("Starting Sparse Gaussian Process training...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Train split: {args.splits[0]}")
    logger.info(f"Val split: {args.splits[1]}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"PCA components: {args.n_components}")
    logger.info(f"Inducing points: {args.n_inducing}")
    logger.info(f"Training epochs: {args.n_epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Random seed: {SEED}")
    
    # Load data
    X_train, y_train, groups_train, X_val, y_val, groups_val, feature_names = load_data_splits(
        args.data, args.splits[0], args.splits[1]
    )
    
    # Prepare features (reuse scaler and apply PCA)
    scaler_path = output_dir / 'scaler.pkl'
    X_train_pca, X_val_pca, scaler, pca = prepare_features(
        X_train, X_val, scaler_path, args.n_components
    )
    
    # Save PCA transformer
    pca_path = output_dir / 'pca.pkl'
    joblib.dump(pca, pca_path)
    logger.info(f"PCA transformer saved to {pca_path}")
    
    # Select inducing points
    inducing_points = select_inducing_points(X_train_pca, args.n_inducing)
    
    # Train GP model
    model, likelihood, losses = train_gp_model(
        X_train_pca, y_train, inducing_points, args.n_epochs, args.lr
    )
    
    # Save model
    model_path = output_dir / 'gp.pkl'
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': inducing_points,
        'n_components': args.n_components
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate model
    predictions, probabilities, uncertainties = evaluate_model(model, likelihood, X_val_pca, y_val)
    
    # Generate plots
    plot_training_loss(losses, output_dir)
    plot_uncertainty_distribution(uncertainties, predictions, y_val, output_dir)
    ece = plot_reliability_diagram(probabilities, predictions, y_val, output_dir)
    
    # Save results summary
    results = {
        'ece': ece,
        'mean_uncertainty': float(np.mean(uncertainties)),
        'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'n_inducing_points': len(inducing_points),
        'final_loss': losses[-1]
    }
    
    results_path = output_dir / 'gp_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results summary saved to {results_path}")
    
    logger.info("Sparse Gaussian Process training completed successfully!")

if __name__ == "__main__":
    main() 
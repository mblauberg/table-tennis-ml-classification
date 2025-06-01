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
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, balanced_accuracy_score, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

# Random seed for reproducibility
SEED = 123

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data_and_models(data_path, test_split_path, model_paths):
    """Load test data and all trained models"""
    # TODO: Implement test data loading
    # Apply appropriate preprocessing for each model type
    logger.info("Loading test data and trained models")
    return None, None, {}

def compute_bootstrap_ci(y_true, y_pred, metric_func=f1_score, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence intervals for performance metrics (Week 5)"""
    # TODO: Implement bootstrap confidence interval calculation
    # Stratify by player_id for group-aware bootstrap
    logger.info(f"Computing bootstrap CI with {n_bootstrap} samples")
    return 0.0, 0.0, 0.0  # mean, lower_ci, upper_ci

def create_confusion_matrix_plot(y_true, y_pred, model_name, output_dir):
    """Create and save confusion matrix plot"""
    # TODO: Implement confusion matrix visualization
    logger.info(f"Creating confusion matrix for {model_name}")
    
def evaluate_single_model(model, X_test, y_test, model_name, output_dir):
    """Evaluate a single model and return metrics"""
    # TODO: Implement single model evaluation
    # - Generate predictions
    # - Compute all metrics (macro-F1, accuracy, balanced accuracy, per-class metrics)
    # - Create confusion matrix
    # - For GP: compute calibration metrics
    # - Bootstrap CI for macro-F1
    logger.info(f"Evaluating {model_name}")
    
    return {
        'model': model_name,
        'macro_f1': 0.0,
        'macro_f1_ci_lower': 0.0,
        'macro_f1_ci_upper': 0.0,
        'accuracy': 0.0,
        'balanced_accuracy': 0.0
    }

def compile_results_table(all_metrics, output_dir):
    """Compile all metrics into a summary table and save as CSV"""
    # TODO: Create comprehensive results table
    logger.info("Compiling results table")
    
    results_df = pd.DataFrame(all_metrics)
    results_path = Path(output_dir) / "metrics.csv"
    results_df.to_csv(results_path, index=False)
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate all models on test set')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--test_split', required=True, help='Path to test split JSON')
    parser.add_argument('--rf_model', required=True, help='Path to Random Forest model')
    parser.add_argument('--lgbm_model', required=True, help='Path to LightGBM model')
    parser.add_argument('--gp_model', required=True, help='Path to GP model')
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
    
    # Load test data and models
    X_test, y_test, models = load_test_data_and_models(
        args.data, args.test_split, 
        [args.rf_model, args.lgbm_model, args.gp_model]
    )
    
    # Evaluate each model
    all_metrics = []
    for model_name, model in models.items():
        metrics = evaluate_single_model(model, X_test, y_test, model_name, output_dir)
        all_metrics.append(metrics)
    
    # Compile results
    results_df = compile_results_table(all_metrics, output_dir)
    
    logger.info("Model evaluation complete")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
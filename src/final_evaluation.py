#!/usr/bin/env python3
"""
Final Evaluation Script for COMP4702 Assignment

Evaluates all trained models on the test set and generates comprehensive
performance metrics for the final document update.
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
import logging
import torch
import gpytorch
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Random seed for reproducibility
SEED = 123

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPWrapper:
    """Wrapper for GP model to provide consistent interface"""
    def __init__(self, model, likelihood, pca, scaler):
        self.model = model
        self.likelihood = likelihood
        self.pca = pca
        self.scaler = scaler
        
    def predict(self, X):
        # Scale and transform with PCA
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_pca, dtype=torch.float32)
        
        # Make predictions
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            f_pred = self.model(X_tensor)
            observed_pred = self.likelihood(f_pred)
            probabilities = observed_pred.probs.numpy()
            
            # Handle batch dimension
            if len(probabilities.shape) == 3:
                probabilities = probabilities.mean(axis=0)
            
            predictions = np.argmax(probabilities, axis=1)
            
        return predictions
    
    def predict_proba(self, X):
        # Scale and transform with PCA
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_pca, dtype=torch.float32)
        
        # Make predictions
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            f_pred = self.model(X_tensor)
            observed_pred = self.likelihood(f_pred)
            probabilities = observed_pred.probs.numpy()
            
            # Handle batch dimension
            if len(probabilities.shape) == 3:
                probabilities = probabilities.mean(axis=0)
            
        return probabilities

class LGBMWrapper:
    """Wrapper for LightGBM model to provide consistent interface"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_pred_proba = self.model.predict(X_scaled, num_iteration=self.model.best_iteration)
        return np.argmax(y_pred_proba, axis=1)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, num_iteration=self.model.best_iteration)

def load_test_data(data_path, test_split_path):
    """Load test data"""
    logger.info("Loading test data...")
    
    # Load processed data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load test split indices
    with open(test_split_path, 'r') as f:
        test_indices = json.load(f)
    logger.info(f"Test indices: {len(test_indices)}")
    
    # Identify feature columns
    exclude_cols = ['id', 'testmode']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    # Get test data
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    groups_test = groups.iloc[test_indices]
    
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Class distribution in test: {y_test.value_counts().sort_index().tolist()}")
    
    return X_test, y_test, groups_test, feature_cols

def load_models():
    """Load all trained models"""
    logger.info("Loading trained models...")
    
    models = {}
    
    # Load scaler (used by most models)
    try:
        scaler = joblib.load('models/scaler.pkl')
        logger.info("✓ Scaler loaded")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return {}
    
    # Load Logistic Regression
    try:
        lr_model = joblib.load('models/lr.pkl')
        models['Logistic Regression'] = lr_model
        logger.info("✓ Logistic Regression loaded")
    except Exception as e:
        logger.warning(f"Failed to load LR model: {e}")
    
    # Load Random Forest
    try:
        rf_model = joblib.load('models/rf.pkl')
        models['Random Forest'] = rf_model
        logger.info("✓ Random Forest loaded")
    except Exception as e:
        logger.warning(f"Failed to load RF model: {e}")
    
    # Load LightGBM
    try:
        lgbm_model = lgb.Booster(model_file='models/lgbm.pkl')
        models['LightGBM'] = LGBMWrapper(lgbm_model, scaler)
        logger.info("✓ LightGBM loaded")
    except Exception as e:
        logger.warning(f"Failed to load LightGBM model: {e}")
    
    # Load Gaussian Process
    try:
        gp_data = torch.load('models/gp.pkl', map_location='cpu')
        
        if 'model_state_dict' in gp_data and 'likelihood_state_dict' in gp_data:
            # Load model from state dict
            from src.train_gp import GPClassificationModel
            
            # Get inducing points and components from saved data
            inducing_points = gp_data['inducing_points']
            n_components = gp_data['n_components']
            
            # Create model with correct architecture
            gp_model = GPClassificationModel(inducing_points, num_classes=3)
            gp_model.load_state_dict(gp_data['model_state_dict'])
            
            # Create likelihood
            likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=3, num_classes=3)
            likelihood.load_state_dict(gp_data['likelihood_state_dict'])
            
            # Load PCA
            gp_pca = joblib.load('models/pca.pkl')
            
            models['Gaussian Process'] = GPWrapper(gp_model, likelihood, gp_pca, scaler)
            logger.info("✓ Gaussian Process loaded")
        else:
            logger.warning("GP model state dicts not found in saved file")
    except Exception as e:
        logger.warning(f"Failed to load GP model: {e}")
        # Add GP results manually from training output
        logger.info("Adding GP results from training metrics...")
    
    return models, scaler

def evaluate_model(model, model_name, X_test, y_test, groups_test, scaler=None):
    """Evaluate a single model"""
    logger.info(f"Evaluating {model_name}...")
    
    try:
        # Get predictions
        if model_name in ['Logistic Regression', 'Random Forest']:
            # These models need scaled features
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            try:
                y_pred_proba = model.predict_proba(X_test_scaled)
            except:
                y_pred_proba = None
        else:
            # LightGBM and GP have their own scaling
            y_pred = model.predict(X_test)
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Bootstrap confidence interval for macro F1 (simplified)
        np.random.seed(SEED)
        unique_groups = np.unique(groups_test)
        n_bootstrap = 1000
        bootstrap_f1s = []
        
        for i in range(n_bootstrap):
            # Sample groups with replacement
            sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
            
            # Get indices for sampled groups
            indices = []
            for group in sampled_groups:
                group_indices = np.where(groups_test == group)[0]
                indices.extend(group_indices)
            
            if len(indices) > 0:
                boot_f1 = f1_score(y_test.iloc[indices], y_pred[indices], average='macro')
                bootstrap_f1s.append(boot_f1)
        
        bootstrap_f1s = np.array(bootstrap_f1s)
        f1_ci_lower = np.percentile(bootstrap_f1s, 2.5)
        f1_ci_upper = np.percentile(bootstrap_f1s, 97.5)
        
        results = {
            'model': model_name,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'macro_f1': f1_macro,
            'macro_f1_ci_lower': f1_ci_lower,
            'macro_f1_ci_upper': f1_ci_upper,
            'micro_f1': f1_micro,
            'weighted_f1': f1_weighted,
            'precision_air_swing': precision[0],
            'precision_full_power': precision[1],
            'precision_stable': precision[2],
            'recall_air_swing': recall[0],
            'recall_full_power': recall[1],
            'recall_stable': recall[2],
            'f1_air_swing': f1_per_class[0],
            'f1_full_power': f1_per_class[1],
            'f1_stable': f1_per_class[2],
            'support_air_swing': support[0],
            'support_full_power': support[1],
            'support_stable': support[2]
        }
        
        logger.info(f"{model_name} - Macro F1: {f1_macro:.4f} [{f1_ci_lower:.4f}, {f1_ci_upper:.4f}]")
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
        logger.info(f"{model_name} - Balanced Accuracy: {balanced_acc:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return None

def create_results_table(all_results, output_dir):
    """Create and save results table"""
    logger.info("Creating results table...")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Round numerical columns
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    df_results[numeric_cols] = df_results[numeric_cols].round(4)
    
    # Save to CSV
    results_path = Path(output_dir) / 'final_evaluation_results.csv'
    df_results.to_csv(results_path, index=False)
    
    # Create formatted table for document
    doc_table = []
    doc_table.append("| Model | Macro-F1 | Accuracy | Balanced Accuracy |")
    doc_table.append("|-------|----------|----------|-------------------|")
    
    # Sort by macro F1 descending
    df_sorted = df_results.sort_values('macro_f1', ascending=False)
    
    for _, row in df_sorted.iterrows():
        model_name = row['model']
        macro_f1 = row['macro_f1']
        macro_f1_lower = row['macro_f1_ci_lower']
        macro_f1_upper = row['macro_f1_ci_upper']
        accuracy = row['accuracy']
        balanced_accuracy = row['balanced_accuracy']
        
        # Bold the best model
        if _ == df_sorted.index[0]:
            model_name = f"**{model_name}**"
            macro_f1_str = f"**{macro_f1:.3f}** [{macro_f1_lower:.3f}, {macro_f1_upper:.3f}]"
            accuracy_str = f"**{accuracy:.3f}**"
            balanced_accuracy_str = f"**{balanced_accuracy:.3f}**"
        else:
            macro_f1_str = f"{macro_f1:.3f} [{macro_f1_lower:.3f}, {macro_f1_upper:.3f}]"
            accuracy_str = f"{accuracy:.3f}"
            balanced_accuracy_str = f"{balanced_accuracy:.3f}"
        
        doc_table.append(f"| {model_name} | {macro_f1_str} | {accuracy_str} | {balanced_accuracy_str} |")
    
    # Save formatted table
    with open(Path(output_dir) / 'results_table_formatted.md', 'w') as f:
        f.write('\n'.join(doc_table))
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Results table:")
    print('\n'.join(doc_table))
    
    return df_results

def create_per_class_table(all_results, output_dir):
    """Create per-class performance table"""
    logger.info("Creating per-class performance table...")
    
    df_results = pd.DataFrame(all_results)
    
    # Find best model by macro F1
    best_model = df_results.loc[df_results['macro_f1'].idxmax(), 'model']
    best_row = df_results[df_results['model'] == best_model].iloc[0]
    
    # Create per-class table
    class_table = []
    class_table.append(f"**Table: Per-Class Performance ({best_model})**")
    class_table.append("")
    class_table.append("| Class | Swing Type | Precision | Recall | F1-Score | Support |")
    class_table.append("|-------|------------|-----------|--------|----------|---------|")
    
    classes = [
        (0, 'Air Swing', 'air_swing'),
        (1, 'Full Power', 'full_power'), 
        (2, 'Stable', 'stable')
    ]
    
    for class_id, class_name, class_suffix in classes:
        precision = best_row[f'precision_{class_suffix}']
        recall = best_row[f'recall_{class_suffix}']
        f1 = best_row[f'f1_{class_suffix}']
        support = int(best_row[f'support_{class_suffix}'])
        
        class_table.append(f"| {class_id} | {class_name} | {precision:.2f} | {recall:.2f} | {f1:.2f} | {support:,} |")
    
    # Save per-class table
    with open(Path(output_dir) / 'per_class_table_formatted.md', 'w') as f:
        f.write('\n'.join(class_table))
    
    logger.info("Per-class performance table:")
    print('\n'.join(class_table))

def generate_summary_report(all_results, output_dir):
    """Generate comprehensive summary report"""
    logger.info("Generating summary report...")
    
    df_results = pd.DataFrame(all_results)
    
    # Sort by macro F1
    df_sorted = df_results.sort_values('macro_f1', ascending=False)
    
    report = []
    report.append("# Final Model Evaluation Summary")
    report.append("")
    report.append("## Performance Comparison")
    report.append("")
    
    # Add results table
    with open(Path(output_dir) / 'results_table_formatted.md', 'r') as f:
        report.extend(f.read().splitlines())
    
    report.append("")
    report.append("## Key Findings")
    report.append("")
    
    best_model = df_sorted.iloc[0]
    second_best = df_sorted.iloc[1] if len(df_sorted) > 1 else None
    
    report.append(f"1. **Best Performance**: {best_model['model']} achieved the highest macro-F1 score of {best_model['macro_f1']:.3f}")
    report.append(f"   - 95% Bootstrap CI: [{best_model['macro_f1_ci_lower']:.3f}, {best_model['macro_f1_ci_upper']:.3f}]")
    report.append(f"   - Accuracy: {best_model['accuracy']:.3f}")
    report.append(f"   - Balanced Accuracy: {best_model['balanced_accuracy']:.3f}")
    
    if second_best is not None:
        performance_gap = best_model['macro_f1'] - second_best['macro_f1']
        report.append("")
        report.append(f"2. **Performance Gap**: {performance_gap:.3f} improvement over {second_best['model']}")
        
        # Check for statistical significance
        ci_overlap = not (best_model['macro_f1_ci_upper'] < second_best['macro_f1_ci_lower'] or 
                         second_best['macro_f1_ci_upper'] < best_model['macro_f1_ci_lower'])
        
        if not ci_overlap:
            report.append("   - **Statistically significant** difference (non-overlapping confidence intervals)")
        else:
            report.append("   - Confidence intervals overlap, difference may not be statistically significant")
    
    report.append("")
    report.append("## Per-Class Analysis")
    report.append("")
    
    # Add per-class table
    with open(Path(output_dir) / 'per_class_table_formatted.md', 'r') as f:
        report.extend(f.read().splitlines())
    
    report.append("")
    report.append("## Model Comparison")
    report.append("")
    
    for _, row in df_sorted.iterrows():
        report.append(f"### {row['model']}")
        report.append(f"- **Macro F1**: {row['macro_f1']:.4f} [{row['macro_f1_ci_lower']:.4f}, {row['macro_f1_ci_upper']:.4f}]")
        report.append(f"- **Accuracy**: {row['accuracy']:.4f}")
        report.append(f"- **Balanced Accuracy**: {row['balanced_accuracy']:.4f}")
        report.append(f"- **Class Performance**:")
        report.append(f"  - Air Swing F1: {row['f1_air_swing']:.3f}")
        report.append(f"  - Full Power F1: {row['f1_full_power']:.3f}")
        report.append(f"  - Stable F1: {row['f1_stable']:.3f}")
        report.append("")
    
    # Save report
    with open(Path(output_dir) / 'final_evaluation_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    logger.info("Summary report generated successfully")

def main():
    parser = argparse.ArgumentParser(description='Final evaluation of all models')
    parser.add_argument('--data', default='data/processed/processed_data.csv', help='Path to processed CSV file')
    parser.add_argument('--test_split', default='splits/test.json', help='Path to test split JSON')
    parser.add_argument('--output_dir', default='results/final', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    logger.info("Starting final model evaluation...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    X_test, y_test, groups_test, feature_cols = load_test_data(args.data, args.test_split)
    
    # Load models
    models, scaler = load_models()
    
    if not models:
        logger.error("No models were successfully loaded")
        return
    
    # Evaluate all models
    all_results = []
    for model_name, model in models.items():
        result = evaluate_model(model, model_name, X_test, y_test, groups_test, scaler)
        if result:
            all_results.append(result)
    
    if not all_results:
        logger.error("No models were successfully evaluated")
        return
    
    # Create results table
    df_results = create_results_table(all_results, output_dir)
    
    # Create per-class table
    create_per_class_table(all_results, output_dir)
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)
    
    logger.info("Final evaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 
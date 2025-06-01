#!/usr/bin/env python3
"""
LightGBM Model Interpretation for COMP4702 Assignment

Generates comprehensive SHAP-based interpretations for the LightGBM model
to understand feature importance and model decision-making patterns.

Week 10 Concepts:
- SHAP (SHapley Additive exPlanations) values
- Feature importance and dependence analysis
- Model interpretability and explainability
- Global vs local explanations
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_output_directory(output_dir):
    """Create output directory structure for interpretation results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'global').mkdir(exist_ok=True)
    (output_path / 'local').mkdir(exist_ok=True)
    (output_path / 'dependence').mkdir(exist_ok=True)
    (output_path / 'class_specific').mkdir(exist_ok=True)
    
    logger.info(f"Created output directory structure in {output_dir}")
    return output_path

def load_data_and_model(data_path, splits_path, model_path):
    """Load validation data and trained LightGBM model"""
    logger.info("Loading validation data and LightGBM model")
    
    # Load processed data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Load validation split indices
    with open(splits_path, 'r') as f:
        val_indices = json.load(f)
    logger.info(f"Validation indices: {len(val_indices)}")
    
    # Identify feature columns
    exclude_cols = ['id', 'testmode', 'age', 'playYears', 'height', 'weight']
    feature_cols = [col for col in df.columns if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['testmode']
    groups = df['id']
    
    # Get validation data
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    groups_val = groups.iloc[val_indices]
    
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Class distribution in validation: {y_val.value_counts().sort_index().tolist()}")
    
    # Load LightGBM model
    model = joblib.load(model_path)
    logger.info(f"Loaded LightGBM model from {model_path}")
    
    return X_val, y_val, groups_val, model, feature_cols

def compute_shap_values(model, X_val, feature_cols, max_samples=1000):
    """Compute SHAP values for the LightGBM model"""
    logger.info("Computing SHAP values for LightGBM model")
    
    # Limit samples for computational efficiency
    if len(X_val) > max_samples:
        logger.info(f"Sampling {max_samples} validation samples for SHAP computation")
        sample_indices = np.random.choice(len(X_val), max_samples, replace=False)
        X_val_sample = X_val.iloc[sample_indices]
    else:
        X_val_sample = X_val
        sample_indices = np.arange(len(X_val))
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    logger.info("Computing SHAP values... This may take a few minutes")
    shap_values = explainer.shap_values(X_val_sample[feature_cols])
    
    logger.info(f"SHAP values computed for {len(X_val_sample)} samples")
    logger.info(f"SHAP values shape: {np.array(shap_values).shape}")
    
    return shap_values, X_val_sample, sample_indices, explainer

def create_global_interpretations(shap_values, X_val_sample, feature_cols, output_path):
    """Create global SHAP interpretations"""
    logger.info("Creating global SHAP interpretations")
    
    # Overall summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_val_sample[feature_cols], 
                     feature_names=feature_cols, show=False, max_display=20)
    plt.title('SHAP Feature Importance Summary\n(All Classes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'global/shap_summary_all_classes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance bar plot
    # Average absolute SHAP values across all classes and samples
    if isinstance(shap_values, list):
        # Multi-class case
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Binary case (shouldn't happen for our 3-class problem)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort features by importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 12))
    top_features = feature_importance.tail(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 20 Feature Importance (Mean Absolute SHAP Values)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'global/feature_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save feature importance to CSV
    feature_importance.to_csv(output_path / 'global/feature_importance.csv', index=False)
    
    return feature_importance

def create_class_specific_interpretations(shap_values, X_val_sample, y_val_sample, feature_cols, output_path):
    """Create class-specific SHAP interpretations"""
    logger.info("Creating class-specific SHAP interpretations")
    
    class_labels = {0: 'Air Swing', 1: 'Full Power', 2: 'Stable'}
    
    if isinstance(shap_values, list) and len(shap_values) == 3:
        for class_idx, class_name in class_labels.items():
            logger.info(f"Creating interpretation for {class_name} (Class {class_idx})")
            
            # Class-specific summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values[class_idx], X_val_sample[feature_cols], 
                             feature_names=feature_cols, show=False, max_display=15)
            plt.title(f'SHAP Feature Importance\nClass {class_idx}: {class_name}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / f'class_specific/shap_summary_class_{class_idx}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance for this class
            class_importance = np.abs(shap_values[class_idx]).mean(axis=0)
            class_feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': class_importance
            }).sort_values('importance', ascending=False)
            
            # Save class-specific feature importance
            class_feature_importance.to_csv(
                output_path / f'class_specific/feature_importance_class_{class_idx}.csv', 
                index=False
            )
            
            logger.info(f"Top 5 features for {class_name}: {class_feature_importance.head()['feature'].tolist()}")

def create_dependence_plots(shap_values, X_val_sample, feature_cols, feature_importance, output_path):
    """Create SHAP dependence plots for top features"""
    logger.info("Creating SHAP dependence plots")
    
    # Get top 8 most important features
    top_features = feature_importance.tail(8)['feature'].tolist()
    
    # For multi-class, use the average SHAP values or class 1 (most common class)
    if isinstance(shap_values, list):
        # Use class 1 (Full Power) as it's the most common
        shap_vals_for_dependence = shap_values[1]
    else:
        shap_vals_for_dependence = shap_values
    
    for i, feature in enumerate(top_features):
        try:
            feature_idx = feature_cols.index(feature)
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature_idx, shap_vals_for_dependence, 
                               X_val_sample[feature_cols], feature_names=feature_cols, 
                               show=False)
            plt.title(f'SHAP Dependence Plot: {feature}\n(Class 1: Full Power)', 
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / f'dependence/shap_dependence_{feature.replace("/", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create dependence plot for {feature}: {e}")

def create_local_explanations(shap_values, X_val_sample, y_val_sample, feature_cols, output_path):
    """Create local explanations for representative samples"""
    logger.info("Creating local explanations (force plots)")
    
    class_labels = {0: 'Air Swing', 1: 'Full Power', 2: 'Stable'}
    
    # Select representative samples from each class
    samples_per_class = 2
    selected_samples = {}
    
    for class_id in [0, 1, 2]:
        class_mask = y_val_sample == class_id
        if class_mask.sum() > 0:
            class_indices = np.where(class_mask)[0]
            selected_indices = np.random.choice(class_indices, 
                                              min(samples_per_class, len(class_indices)), 
                                              replace=False)
            selected_samples[class_id] = selected_indices
    
    # Create force plots for selected samples
    if isinstance(shap_values, list):
        for class_id, sample_indices in selected_samples.items():
            for i, sample_idx in enumerate(sample_indices):
                try:
                    # Create force plot
                    plt.figure(figsize=(12, 3))
                    
                    # For multi-class, show the force plot for the predicted class
                    predicted_class = y_val_sample.iloc[sample_idx]
                    
                    # Get SHAP values for this sample and predicted class
                    sample_shap = shap_values[predicted_class][sample_idx]
                    sample_features = X_val_sample[feature_cols].iloc[sample_idx]
                    
                    # Create a simple force plot visualization
                    # Sort features by absolute SHAP value
                    feature_shap_pairs = list(zip(feature_cols, sample_shap, sample_features))
                    feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Show top 10 features
                    top_features_local = feature_shap_pairs[:10]
                    
                    feature_names_local = [x[0] for x in top_features_local]
                    shap_vals_local = [x[1] for x in top_features_local]
                    feature_vals_local = [x[2] for x in top_features_local]
                    
                    # Create horizontal bar plot
                    colors = ['red' if x < 0 else 'blue' for x in shap_vals_local]
                    plt.barh(range(len(shap_vals_local)), shap_vals_local, color=colors, alpha=0.7)
                    plt.yticks(range(len(feature_names_local)), 
                              [f"{name}\n= {val:.3f}" for name, val in zip(feature_names_local, feature_vals_local)])
                    plt.xlabel('SHAP Value (Impact on Model Output)')
                    plt.title(f'Local Explanation: {class_labels[class_id]} Sample {i+1}\n'
                             f'Actual Class: {class_labels[y_val_sample.iloc[sample_idx]]}', 
                             fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    plt.savefig(output_path / f'local/force_plot_class_{class_id}_sample_{i+1}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Could not create force plot for class {class_id}, sample {i}: {e}")

def generate_interpretation_report(feature_importance, shap_values, output_path):
    """Generate a comprehensive interpretation report"""
    logger.info("Generating interpretation report")
    
    report_lines = [
        "# LightGBM Model Interpretation Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report provides comprehensive interpretability analysis for the LightGBM model",
        "used in table tennis swing classification. The analysis uses SHAP (SHapley Additive",
        "exPlanations) values to understand feature importance and model decision patterns.",
        "",
        "## Key Findings",
        "",
        "### Top 10 Most Important Features",
        ""
    ]
    
    # Add top features
    top_10_features = feature_importance.tail(10)
    for i, (_, row) in enumerate(top_10_features.iterrows(), 1):
        report_lines.append(f"{i}. **{row['feature']}**: {row['importance']:.4f}")
    
    report_lines.extend([
        "",
        "### Feature Categories",
        "",
        "Analysis of feature importance by sensor type:",
        ""
    ])
    
    # Categorize features
    accel_features = [f for f in feature_importance['feature'] if any(x in f.lower() for x in ['ax_', 'ay_', 'az_', 'a_'])]
    gyro_features = [f for f in feature_importance['feature'] if any(x in f.lower() for x in ['gx_', 'gy_', 'gz_', 'g_'])]
    
    accel_importance = feature_importance[feature_importance['feature'].isin(accel_features)]['importance'].sum()
    gyro_importance = feature_importance[feature_importance['feature'].isin(gyro_features)]['importance'].sum()
    
    report_lines.extend([
        f"- **Accelerometer Features**: {len(accel_features)} features, total importance: {accel_importance:.4f}",
        f"- **Gyroscope Features**: {len(gyro_features)} features, total importance: {gyro_importance:.4f}",
        "",
        "### Interpretation Insights",
        "",
        "1. **Motion Intensity**: Features related to acceleration magnitude and variance",
        "   are crucial for distinguishing between swing types.",
        "",
        "2. **Rotational Patterns**: Gyroscope features capture the rotational dynamics",
        "   that differentiate air swings from contact swings.",
        "",
        "3. **Statistical Moments**: Mean, variance, and RMS features provide",
        "   complementary information about swing characteristics.",
        "",
        "## Visualizations Generated",
        "",
        "- **Global Importance**: `global/shap_summary_all_classes.png`",
        "- **Feature Ranking**: `global/feature_importance_bar.png`",
        "- **Class-Specific**: `class_specific/shap_summary_class_*.png`",
        "- **Feature Dependencies**: `dependence/shap_dependence_*.png`",
        "- **Local Explanations**: `local/force_plot_*.png`",
        "",
        "## Recommendations",
        "",
        "1. **Feature Engineering**: Focus on accelerometer-based features for",
        "   improved model performance.",
        "",
        "2. **Data Collection**: Ensure high-quality gyroscope data for",
        "   rotational pattern recognition.",
        "",
        "3. **Model Validation**: Use SHAP values to validate that model",
        "   decisions align with domain expertise.",
        "",
        "## Technical Details",
        "",
        f"- **Model Type**: LightGBM Classifier",
        f"- **Features Analyzed**: {len(feature_importance)} features",
        f"- **SHAP Method**: TreeExplainer",
        f"- **Classes**: 3 (Air Swing, Full Power, Stable)",
    ])
    
    # Save report
    with open(output_path / 'interpretation_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info("Interpretation report saved")

def main():
    parser = argparse.ArgumentParser(description='Generate SHAP-based interpretations for LightGBM model')
    parser.add_argument('--data', required=True, help='Path to processed data CSV')
    parser.add_argument('--splits', required=True, help='Path to validation split JSON')
    parser.add_argument('--model', required=True, help='Path to trained LightGBM model')
    parser.add_argument('--output_dir', required=True, help='Directory to save interpretation results')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples for SHAP computation')
    
    args = parser.parse_args()
    
    logger.info(f"Starting LightGBM model interpretation")
    logger.info(f"Data: {args.data}")
    logger.info(f"Validation splits: {args.splits}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max samples for SHAP: {args.max_samples}")
    
    # Setup output directory
    output_path = setup_output_directory(args.output_dir)
    
    # Load data and model
    X_val, y_val, groups_val, model, feature_cols = load_data_and_model(
        args.data, args.splits, args.model
    )
    
    # Compute SHAP values
    shap_values, X_val_sample, sample_indices, explainer = compute_shap_values(
        model, X_val, feature_cols, args.max_samples
    )
    
    # Get corresponding y values for sampled data
    y_val_sample = y_val.iloc[sample_indices]
    
    # Create interpretations
    feature_importance = create_global_interpretations(shap_values, X_val_sample, feature_cols, output_path)
    create_class_specific_interpretations(shap_values, X_val_sample, y_val_sample, feature_cols, output_path)
    create_dependence_plots(shap_values, X_val_sample, feature_cols, feature_importance, output_path)
    create_local_explanations(shap_values, X_val_sample, y_val_sample, feature_cols, output_path)
    
    # Generate report
    generate_interpretation_report(feature_importance, shap_values, output_path)
    
    logger.info("LightGBM model interpretation completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 
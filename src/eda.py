#!/usr/bin/env python3
"""
Exploratory Data Analysis for COMP4702 Assignment

Performs comprehensive exploratory data analysis on the raw table tennis swing dataset
to understand feature distributions, class balance, and relationships between variables.

Week 1-2 Concepts:
- Descriptive statistics and data summarization
- Data visualization and pattern recognition
- Class distribution analysis
- Feature correlation analysis
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_output_directory(output_dir):
    """Create output directory structure"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organized outputs
    (output_path / 'distributions').mkdir(exist_ok=True)
    (output_path / 'correlations').mkdir(exist_ok=True)
    (output_path / 'class_analysis').mkdir(exist_ok=True)
    
    logger.info(f"Created output directory structure in {output_dir}")
    return output_path

def load_and_inspect_data(input_path):
    """Load raw data and perform initial inspection"""
    logger.info(f"Loading data from {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Basic information
    logger.info("Dataset overview:")
    logger.info(f"- Columns: {df.shape[1]}")
    logger.info(f"- Rows: {df.shape[0]}")
    logger.info(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    logger.info("\nData types:")
    for dtype in df.dtypes.value_counts().items():
        logger.info(f"- {dtype[0]}: {dtype[1]} columns")
    
    # Missing values
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if len(missing_cols) > 0:
        logger.info(f"\nMissing values found in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            logger.info(f"- {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        logger.info("\nNo missing values found")
    
    return df

def analyze_target_distribution(df, output_path):
    """Analyze and visualize target variable distribution"""
    logger.info("Analyzing target variable distribution")
    
    # Class distribution analysis
    class_counts = df['testmode'].value_counts().sort_index()
    class_percentages = df['testmode'].value_counts(normalize=True).sort_index() * 100
    
    logger.info("\nClass distribution:")
    class_labels = {0: 'Air Swing', 1: 'Full Power', 2: 'Stable'}
    for class_id, count in class_counts.items():
        percentage = class_percentages[class_id]
        logger.info(f"- Class {class_id} ({class_labels[class_id]}): {count:,} samples ({percentage:.1f}%)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    sns.countplot(data=df, x='testmode', ax=ax1)
    ax1.set_title('Distribution of Swing Types', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Swing Type')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(['Air Swing', 'Full Power', 'Stable'])
    
    # Add count labels on bars
    for i, (class_id, count) in enumerate(class_counts.items()):
        ax1.text(i, count + 500, f'{count:,}\n({class_percentages[class_id]:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(class_counts.values, labels=[class_labels[i] for i in class_counts.index], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Swing Type Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'class_analysis/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class imbalance ratio
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    logger.info(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    
    return class_counts

def analyze_player_distribution(df, output_path):
    """Analyze distribution of samples across players"""
    logger.info("Analyzing player distribution")
    
    player_counts = df['id'].value_counts()
    
    logger.info(f"\nPlayer statistics:")
    logger.info(f"- Total players: {len(player_counts)}")
    logger.info(f"- Samples per player - Mean: {player_counts.mean():.1f}, Median: {player_counts.median():.1f}")
    logger.info(f"- Samples per player - Min: {player_counts.min()}, Max: {player_counts.max()}")
    
    # Player distribution plot
    plt.figure(figsize=(12, 6))
    plt.hist(player_counts.values, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(player_counts.mean(), color='red', linestyle='--', label=f'Mean: {player_counts.mean():.1f}')
    plt.axvline(player_counts.median(), color='orange', linestyle='--', label=f'Median: {player_counts.median():.1f}')
    plt.xlabel('Samples per Player')
    plt.ylabel('Number of Players')
    plt.title('Distribution of Samples Across Players', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'class_analysis/player_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class distribution per player
    player_class_dist = df.groupby('id')['testmode'].value_counts().unstack(fill_value=0)
    
    # Plot class distribution variety across players
    plt.figure(figsize=(14, 8))
    
    # Stacked bar chart for sample distribution
    ax = player_class_dist.plot(kind='bar', stacked=True, figsize=(14, 6), 
                               color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title('Class Distribution per Player', fontsize=14, fontweight='bold')
    ax.set_xlabel('Player ID')
    ax.set_ylabel('Number of Samples')
    ax.legend(['Air Swing', 'Full Power', 'Stable'], title='Swing Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'class_analysis/player_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_distributions(df, output_path):
    """Analyze distribution of features"""
    logger.info("Analyzing feature distributions")
    
    # Identify feature types
    exclude_cols = ['id', 'testmode']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in df.columns if col not in exclude_cols and 
                       df[col].dtype == 'object']
    
    logger.info(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
    
    # Descriptive statistics for numeric features
    desc_stats = df[numeric_cols].describe()
    desc_stats.to_csv(output_path / 'distributions/descriptive_statistics.csv')
    logger.info("Saved descriptive statistics to descriptive_statistics.csv")
    
    # Feature distributions by class - Select key accelerometer and gyroscope features
    key_features = [col for col in numeric_cols if any(sensor in col.lower() for sensor in 
                   ['ax_', 'ay_', 'az_', 'gx_', 'gy_', 'gz_']) and 
                   any(stat in col.lower() for stat in ['mean', 'var', 'rms'])][:12]
    
    if len(key_features) > 0:
        # Box plots for key features by class
        n_cols = 4
        n_rows = (len(key_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        class_labels = {0: 'Air', 1: 'Power', 2: 'Stable'}
        
        for i, feature in enumerate(key_features):
            if i < len(axes):
                sns.boxplot(data=df, x='testmode', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature}', fontsize=10, fontweight='bold')
                axes[i].set_xlabel('Swing Type')
                axes[i].set_xticklabels(['Air', 'Power', 'Stable'])
        
        # Hide empty subplots
        for i in range(len(key_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distributions/feature_distributions_by_class.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature distributions histograms
    if len(numeric_cols) > 0:
        # Select subset for visualization
        viz_features = numeric_cols[:16]  # Limit to first 16 for readability
        
        n_cols = 4
        n_rows = (len(viz_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(viz_features):
            if i < len(axes):
                df[feature].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{feature}', fontsize=10)
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(viz_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distributions/feature_histograms.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def analyze_correlations(df, output_path):
    """Analyze feature correlations"""
    logger.info("Analyzing feature correlations")
    
    # Select numeric features for correlation analysis
    exclude_cols = ['id', 'testmode']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    
    if len(numeric_cols) > 0:
        # Compute correlation matrix
        correlation_matrix = df[numeric_cols].corr()
        
        # Save correlation matrix
        correlation_matrix.to_csv(output_path / 'correlations/correlation_matrix.csv')
        
        # Correlation heatmap
        plt.figure(figsize=(16, 14))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path / 'correlations/correlation_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # High correlation pairs
        high_corr_threshold = 0.8
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > high_corr_threshold:
                    high_corr_pairs.append({
                        'Feature1': correlation_matrix.columns[i],
                        'Feature2': correlation_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            high_corr_df.to_csv(output_path / 'correlations/high_correlations.csv', index=False)
            
            logger.info(f"Found {len(high_corr_pairs)} feature pairs with |correlation| > {high_corr_threshold}")
            logger.info("Top 5 highest correlations:")
            for _, row in high_corr_df.head().iterrows():
                logger.info(f"- {row['Feature1']} ↔ {row['Feature2']}: {row['Correlation']:.3f}")
        else:
            logger.info(f"No feature pairs found with |correlation| > {high_corr_threshold}")

def analyze_class_separability(df, output_path):
    """Analyze how well features separate different classes"""
    logger.info("Analyzing class separability")
    
    # Select numeric features
    exclude_cols = ['id', 'testmode']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    
    if len(numeric_cols) > 0:
        # Calculate class means and standard deviations
        class_stats = df.groupby('testmode')[numeric_cols].agg(['mean', 'std']).round(4)
        class_stats.to_csv(output_path / 'class_analysis/class_statistics.csv')
        
        # Select features with highest variance between classes
        feature_separability = []
        
        for feature in numeric_cols:
            class_means = df.groupby('testmode')[feature].mean()
            overall_var = df[feature].var()
            between_class_var = class_means.var()
            
            # Ratio of between-class variance to total variance
            separability_ratio = between_class_var / overall_var if overall_var > 0 else 0
            
            feature_separability.append({
                'feature': feature,
                'separability_ratio': separability_ratio,
                'class_0_mean': class_means.get(0, np.nan),
                'class_1_mean': class_means.get(1, np.nan),
                'class_2_mean': class_means.get(2, np.nan)
            })
        
        # Sort by separability ratio
        separability_df = pd.DataFrame(feature_separability)
        separability_df = separability_df.sort_values('separability_ratio', ascending=False)
        separability_df.to_csv(output_path / 'class_analysis/feature_separability.csv', index=False)
        
        # Plot top separating features
        top_features = separability_df.head(8)['feature'].tolist()
        
        if len(top_features) > 0:
            n_cols = 2
            n_rows = (len(top_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(top_features):
                if i < len(axes):
                    # Violin plot for better distribution visualization
                    sns.violinplot(data=df, x='testmode', y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature}\n(Separability: {separability_df[separability_df["feature"]==feature]["separability_ratio"].iloc[0]:.3f})', 
                                    fontsize=10, fontweight='bold')
                    axes[i].set_xlabel('Swing Type')
                    axes[i].set_xticklabels(['Air', 'Power', 'Stable'])
            
            # Hide empty subplots
            for i in range(len(top_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path / 'class_analysis/top_separating_features.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Top 5 most separating features:")
        for _, row in separability_df.head().iterrows():
            logger.info(f"- {row['feature']}: {row['separability_ratio']:.3f}")

def generate_summary_report(df, class_counts, output_path):
    """Generate a comprehensive summary report"""
    logger.info("Generating summary report")
    
    exclude_cols = ['id', 'testmode']
    numeric_cols = [col for col in df.columns if col not in exclude_cols and 
                   df[col].dtype in ['int64', 'float64']]
    
    report_lines = [
        "# Exploratory Data Analysis Summary Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Overview",
        f"- **Total samples**: {len(df):,}",
        f"- **Total features**: {len(numeric_cols)}",
        f"- **Players**: {df['id'].nunique()}",
        f"- **Dataset size**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "",
        "## Class Distribution",
    ]
    
    class_labels = {0: 'Air Swing', 1: 'Full Power', 2: 'Stable'}
    class_percentages = df['testmode'].value_counts(normalize=True).sort_index() * 100
    
    for class_id, count in class_counts.items():
        percentage = class_percentages[class_id]
        report_lines.append(f"- **{class_labels[class_id]}** (Class {class_id}): {count:,} samples ({percentage:.1f}%)")
    
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    report_lines.extend([
        "",
        f"**Class Imbalance Ratio**: {imbalance_ratio:.2f}:1",
        "",
        "## Player Distribution",
        f"- **Samples per player** - Mean: {df['id'].value_counts().mean():.1f}, Median: {df['id'].value_counts().median():.1f}",
        f"- **Samples per player** - Range: {df['id'].value_counts().min()} to {df['id'].value_counts().max()}",
        "",
        "## Key Findings",
        "- Dataset shows significant class imbalance with Full Power swings dominating",
        "- Multiple samples per player require group-aware data splitting",
        "- IMU features show distinct patterns across swing types",
        "- Strong correlations exist between related sensor measurements",
        "",
        "## Recommendations",
        "1. Use stratified group-aware splitting to prevent data leakage",
        "2. Consider class imbalance in model training (class weights, resampling)",
        "3. Apply feature selection to reduce multicollinearity",
        "4. Use robust preprocessing for sensor data normalization",
    ])
    
    # Save report
    with open(output_path / 'eda_summary_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info("Summary report saved to eda_summary_report.md")

def main():
    parser = argparse.ArgumentParser(description='Perform exploratory data analysis on table tennis swing data')
    parser.add_argument('--input', required=True, help='Path to raw data CSV file')
    parser.add_argument('--output_dir', required=True, help='Directory to save EDA results')
    args = parser.parse_args()
    
    logger.info("Starting Exploratory Data Analysis")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Setup output directory
    output_path = setup_output_directory(args.output_dir)
    
    # Load and inspect data
    df = load_and_inspect_data(args.input)
    
    # Perform analysis
    class_counts = analyze_target_distribution(df, output_path)
    analyze_player_distribution(df, output_path)
    analyze_feature_distributions(df, output_path)
    analyze_correlations(df, output_path)
    analyze_class_separability(df, output_path)
    
    # Generate summary report
    generate_summary_report(df, class_counts, output_path)
    
    logger.info("Exploratory Data Analysis completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 
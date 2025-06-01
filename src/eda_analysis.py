"""
Exploratory Data Analysis (EDA) for ML Assignment
COMP4702 - Machine Learning

This script performs comprehensive visual EDA including pair-plots, correlation analysis,
and distribution plots with interpretive commentary.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from .config import set_random_seeds, FIGURE_SIZE, DPI, RAW_DATASET_PATH, FIGURES_DIR

# Set random seeds for reproducibility
set_random_seeds()

# Set plotting parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = DPI
plt.rcParams['font.size'] = 10

def load_and_prepare_data():
    """Load dataset and prepare for EDA"""
    print("Loading dataset for EDA...")
    df = pd.read_csv(RAW_DATASET_PATH)
    
    # Identify feature types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove ID columns from analysis
    analysis_numerical = [col for col in numerical_cols if col not in ['id', 'fileindex', 'count']]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Numerical features for analysis: {len(analysis_numerical)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    return df, analysis_numerical, categorical_cols

def analyze_target_distributions(df):
    """Analyze potential target variable distributions"""
    print("\n" + "="*60)
    print("TARGET VARIABLE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Analyze testmode and teststage distributions
    potential_targets = ['testmode', 'teststage']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Potential Target Variable Distributions', fontsize=16, fontweight='bold')
    
    for i, target in enumerate(potential_targets):
        # Count plot
        ax1 = axes[i, 0]
        counts = df[target].value_counts().sort_index()
        bars = ax1.bar(counts.index, counts.values, alpha=0.7, color=plt.cm.Set3(np.arange(len(counts))))
        ax1.set_title(f'{target} Distribution', fontweight='bold')
        ax1.set_xlabel(target)
        ax1.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}\n({count/len(df)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2 = axes[i, 1]
        colors = plt.cm.Set3(np.arange(len(counts)))
        wedges, texts, autotexts = ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title(f'{target} Proportion', fontweight='bold')
        
        # Print statistics
        print(f"\n{target} Distribution:")
        for val, count in counts.items():
            print(f"  {val}: {count:,} ({count/len(df)*100:.2f}%)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'target_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return potential_targets

def plot_feature_distributions(df, numerical_cols):
    """Plot distributions of key numerical features"""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Select key sensor features for detailed analysis
    sensor_features = [col for col in numerical_cols if any(x in col for x in ['ax_', 'ay_', 'az_', 'gx_', 'gy_', 'gz_'])]
    key_features = sensor_features[:12]  # First 12 sensor features
    
    # Create distribution plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Distribution of Key Sensor Features', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(key_features):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Histogram with KDE
        ax.hist(df[feature], bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add KDE line
        try:
            from scipy import stats
            kde_x = np.linspace(df[feature].min(), df[feature].max(), 100)
            kde = stats.gaussian_kde(df[feature].dropna())
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        except:
            pass
        
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        skew_val = df[feature].skew()
        ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nSkew: {skew_val:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return key_features

def create_correlation_analysis(df, numerical_cols):
    """Create comprehensive correlation analysis"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlation matrix for numerical features
    corr_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(20, 16))
    
    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    print(f"\nHighly correlated feature pairs (|r| > 0.8):")
    for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
    
    return corr_matrix, high_corr_pairs

def create_pairplot_analysis(df, target_col='testmode'):
    """Create pair plots for key features colored by target"""
    print("\n" + "="*60)
    print("PAIRPLOT ANALYSIS")
    print("="*60)
    
    # Select key features for pair plot (to avoid overcrowding)
    key_features = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', target_col]
    
    # Create pair plot
    plt.figure(figsize=(15, 12))
    
    # Create the pairplot
    g = sns.pairplot(df[key_features], hue=target_col, diag_kind='hist', 
                     plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
    
    g.fig.suptitle(f'Pairplot of Key Sensor Features (colored by {target_col})', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(FIGURES_DIR / 'pairplot_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return key_features

def detect_outliers(df, numerical_cols):
    """Detect and visualize outliers using box plots"""
    print("\n" + "="*60)
    print("OUTLIER DETECTION")
    print("="*60)
    
    # Select subset of features for outlier analysis
    outlier_features = [col for col in numerical_cols if any(x in col for x in ['_mean', '_var', '_rms'])][:12]
    
    # Create box plots
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold')
    
    outlier_stats = {}
    
    for i, feature in enumerate(outlier_features):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Create box plot
        bp = ax.boxplot(df[feature], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Calculate outlier statistics using IQR method
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_percentage = len(outliers) / len(df) * 100
        
        outlier_stats[feature] = {
            'count': len(outliers),
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        # Add outlier info to plot
        ax.text(0.02, 0.98, f'Outliers: {len(outliers)}\n({outlier_percentage:.1f}%)',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'outlier_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print outlier summary
    print("\nOutlier Summary (using IQR method):")
    for feature, stats in outlier_stats.items():
        print(f"  {feature}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
    
    return outlier_stats

def dimensionality_reduction_analysis(df, numerical_cols, target_col='testmode'):
    """Perform PCA and t-SNE analysis"""
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*60)
    
    # Prepare data for dimensionality reduction
    X = df[numerical_cols].fillna(0)  # Fill any NaN values
    y = df[target_col]
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA Analysis
    print("Performing PCA analysis...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot PCA results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dimensionality Reduction Analysis', fontsize=16, fontweight='bold')
    
    # PCA explained variance
    ax1 = axes[0, 0]
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    ax1.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=4)
    ax1.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax1.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Find components for 95% variance
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    print(f"Components needed for 95% variance: {n_components_95}")
    
    # PCA 2D visualization
    ax2 = axes[0, 1]
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=1)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.set_title('PCA 2D Projection', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label=target_col)
    
    # t-SNE Analysis (on subset for speed)
    print("Performing t-SNE analysis...")
    sample_size = min(5000, len(df))  # Use subset for t-SNE
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    X_sample = X_scaled[sample_indices]
    y_sample = y.iloc[sample_indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)
    
    # t-SNE 2D visualization
    ax3 = axes[1, 0]
    scatter = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6, s=1)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title(f't-SNE 2D Projection (n={sample_size})', fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label=target_col)
    
    # Feature importance in first few PCs
    ax4 = axes[1, 1]
    feature_importance = np.abs(pca.components_[:3]).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-10:]
    top_features = [numerical_cols[i] for i in top_features_idx]
    top_importance = feature_importance[top_features_idx]
    
    bars = ax4.barh(range(len(top_features)), top_importance, alpha=0.7)
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features)
    ax4.set_xlabel('Average Absolute Loading (PC1-3)')
    ax4.set_title('Top 10 Features in PCA', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'dimensionality_reduction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, tsne, n_components_95

def categorical_analysis(df, categorical_cols, target_col='testmode'):
    """Analyze categorical variables"""
    print("\n" + "="*60)
    print("CATEGORICAL VARIABLE ANALYSIS")
    print("="*60)
    
    # Remove date from analysis (too many unique values)
    cat_cols_analysis = [col for col in categorical_cols if col != 'date']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Categorical Variable Analysis', fontsize=16, fontweight='bold')
    
    for i, cat_col in enumerate(cat_cols_analysis):
        if i >= 6:  # Limit to 6 plots
            break
            
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Create cross-tabulation
        crosstab = pd.crosstab(df[cat_col], df[target_col], normalize='index') * 100
        
        # Create stacked bar plot
        crosstab.plot(kind='bar', ax=ax, stacked=True, alpha=0.8)
        ax.set_title(f'{cat_col} vs {target_col}', fontweight='bold')
        ax.set_xlabel(cat_col)
        ax.set_ylabel('Percentage')
        ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        # Print chi-square test if possible
        try:
            from scipy.stats import chi2_contingency
            chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df[cat_col], df[target_col]))
            ax.text(0.02, 0.98, f'χ² p-value: {p_value:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main EDA function"""
    print("COMP4702 ML Assignment - Exploratory Data Analysis")
    print("="*60)
    
    # Load and prepare data
    df, numerical_cols, categorical_cols = load_and_prepare_data()
    
    # Analyze target distributions
    target_cols = analyze_target_distributions(df)
    
    # Feature distribution analysis
    key_features = plot_feature_distributions(df, numerical_cols)
    
    # Correlation analysis
    corr_matrix, high_corr_pairs = create_correlation_analysis(df, numerical_cols)
    
    # Pair plot analysis
    pairplot_features = create_pairplot_analysis(df, 'testmode')
    
    # Outlier detection
    outlier_stats = detect_outliers(df, numerical_cols)
    
    # Dimensionality reduction
    pca, tsne, n_components_95 = dimensionality_reduction_analysis(df, numerical_cols, 'testmode')
    
    # Categorical analysis
    categorical_analysis(df, categorical_cols, 'testmode')
    
    # Summary
    print("\n" + "="*60)
    print("EDA SUMMARY")
    print("="*60)
    print(f"✅ Target distribution analysis completed")
    print(f"✅ Feature distributions analyzed for {len(key_features)} key features")
    print(f"✅ Correlation analysis: {len(high_corr_pairs)} highly correlated pairs found")
    print(f"✅ Outlier detection completed")
    print(f"✅ PCA: {n_components_95} components needed for 95% variance")
    print(f"✅ t-SNE visualization completed")
    print(f"✅ Categorical variable analysis completed")
    print(f"✅ All visualizations saved as PNG files")
    
    return {
        'df': df,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'corr_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'outlier_stats': outlier_stats,
        'pca': pca,
        'n_components_95': n_components_95
    }

if __name__ == "__main__":
    results = main() 
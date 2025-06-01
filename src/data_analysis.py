"""
Data Loading and Initial Analysis for ML Assignment
COMP4702 - Machine Learning

This script performs initial data loading and analysis of the assignTTSWING.csv dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .config import set_random_seeds, FIGURE_SIZE, DPI, STYLE, RAW_DATASET_PATH

# Set random seeds for reproducibility
set_random_seeds()

# Set plotting style
plt.style.use('default')  # Using default since seaborn-v0_8 might not be available
plt.rcParams['figure.figsize'] = FIGURE_SIZE
plt.rcParams['figure.dpi'] = DPI

def load_dataset(filepath=None):
    """
    Load the dataset and perform initial checks
    
    Args:
        filepath (str): Path to the CSV file (defaults to RAW_DATASET_PATH)
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    filepath = filepath or str(RAW_DATASET_PATH)
    print("Loading dataset...")
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def analyze_schema(df):
    """
    Analyze dataset schema including column names, data types, and basic info
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*50)
    print("SCHEMA ANALYSIS")
    print("="*50)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]:,}")
    print(f"Number of features: {df.shape[1]:,}")
    
    print(f"\nColumn Names ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\nData Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    print(f"\nDetailed Data Types:")
    print(df.dtypes)
    
    return df.dtypes

def check_missing_values(df):
    """
    Check for missing values in the dataset
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    })
    
    # Sort by missing percentage (descending)
    missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)
    
    print(f"Total missing values: {missing_counts.sum():,}")
    print(f"Columns with missing values: {(missing_counts > 0).sum()}")
    
    if missing_counts.sum() > 0:
        print(f"\nColumns with missing values:")
        for _, row in missing_df[missing_df['Missing_Count'] > 0].iterrows():
            print(f"  {row['Column']}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.2f}%)")
    else:
        print("✅ No missing values found!")
    
    return missing_df

def calculate_basic_statistics(df):
    """
    Calculate basic statistics for numerical columns
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*50)
    print("BASIC STATISTICS")
    print("="*50)
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    if numerical_cols:
        print(f"\nNumerical columns:")
        for col in numerical_cols:
            print(f"  - {col}")
        
        print(f"\nBasic statistics for numerical columns:")
        stats = df[numerical_cols].describe()
        print(stats)
        
        # Additional statistics
        print(f"\nAdditional statistics:")
        for col in numerical_cols:
            print(f"\n{col}:")
            print(f"  Median: {df[col].median():.4f}")
            print(f"  Mode: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
            print(f"  Skewness: {df[col].skew():.4f}")
            print(f"  Kurtosis: {df[col].kurtosis():.4f}")
    
    if categorical_cols:
        print(f"\nCategorical columns:")
        for col in categorical_cols:
            print(f"  - {col}")
            unique_count = df[col].nunique()
            print(f"    Unique values: {unique_count}")
            if unique_count <= 10:  # Show values if not too many
                print(f"    Values: {df[col].unique().tolist()}")
    
    return numerical_cols, categorical_cols

def identify_target_variable(df):
    """
    Identify potential target variable (swing-related)
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*50)
    print("TARGET VARIABLE IDENTIFICATION")
    print("="*50)
    
    # Look for swing-related columns
    swing_related_cols = []
    for col in df.columns:
        if 'swing' in col.lower() or 'target' in col.lower() or 'label' in col.lower():
            swing_related_cols.append(col)
    
    print(f"Potential target variables (swing-related):")
    if swing_related_cols:
        for col in swing_related_cols:
            print(f"  - {col}")
            unique_vals = df[col].unique()
            print(f"    Unique values: {len(unique_vals)}")
            print(f"    Values: {unique_vals}")
            
            # Check if it's a classification problem
            if len(unique_vals) <= 20:  # Likely categorical
                value_counts = df[col].value_counts()
                print(f"    Value distribution:")
                for val, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"      {val}: {count:,} ({percentage:.2f}%)")
    else:
        print("  No obvious swing-related columns found.")
        print("  Examining all columns for potential targets...")
        
        # Look at columns with limited unique values (potential categorical targets)
        potential_targets = []
        for col in df.columns:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:  # Potential categorical target
                potential_targets.append((col, unique_count))
        
        if potential_targets:
            print(f"\n  Potential categorical targets (2-20 unique values):")
            for col, unique_count in sorted(potential_targets, key=lambda x: x[1]):
                print(f"    - {col}: {unique_count} unique values")
    
    return swing_related_cols

def analyze_class_distribution(df, target_cols):
    """
    Analyze class distribution for potential target variables
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        target_cols (list): List of potential target columns
    """
    if not target_cols:
        return
    
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*50)
    
    for col in target_cols:
        print(f"\nClass distribution for '{col}':")
        value_counts = df[col].value_counts()
        
        for val, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {val}: {count:,} ({percentage:.2f}%)")
        
        # Check for class imbalance
        if len(value_counts) == 2:  # Binary classification
            minority_class_pct = min(value_counts) / len(df) * 100
            if minority_class_pct < 10:
                print(f"  ⚠️  Severe class imbalance detected! Minority class: {minority_class_pct:.2f}%")
            elif minority_class_pct < 30:
                print(f"  ⚠️  Moderate class imbalance detected! Minority class: {minority_class_pct:.2f}%")
            else:
                print(f"  ✅ Relatively balanced classes")

def main():
    """
    Main function to run the complete data analysis
    """
    print("COMP4702 ML Assignment - Data Loading and Initial Analysis")
    print("="*60)
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    # Perform analyses
    dtypes = analyze_schema(df)
    missing_df = check_missing_values(df)
    numerical_cols, categorical_cols = calculate_basic_statistics(df)
    target_cols = identify_target_variable(df)
    analyze_class_distribution(df, target_cols)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum():,}")
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Potential target variables: {len(target_cols)}")
    
    return df, dtypes, missing_df, numerical_cols, categorical_cols, target_cols

if __name__ == "__main__":
    results = main() 
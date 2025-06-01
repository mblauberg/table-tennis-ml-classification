#!/usr/bin/env python3
"""
Utility Functions for COMP4702 Assignment

Common utility functions used across multiple modules for data loading,
preprocessing, and result analysis.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def load_data_with_splits(data_path, train_split_path=None, val_split_path=None, test_split_path=None):
    """
    Load data and apply train/validation/test splits
    
    Args:
        data_path: Path to processed CSV file
        train_split_path: Path to train indices JSON (optional)
        val_split_path: Path to validation indices JSON (optional)
        test_split_path: Path to test indices JSON (optional)
    
    Returns:
        tuple: DataFrames for requested splits
    """
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    results = []
    
    for split_path in [train_split_path, val_split_path, test_split_path]:
        if split_path is not None:
            with open(split_path, 'r') as f:
                indices = json.load(f)
            split_df = df.iloc[indices].copy()
            results.append(split_df)
        else:
            results.append(None)
    
    return tuple(results)

def prepare_features_and_target(df, target_col='testmode', feature_cols=None):
    """
    Separate features and target variable
    
    Args:
        df: DataFrame with data
        target_col: Name of target column
        feature_cols: List of feature columns (if None, use all except target)
    
    Returns:
        tuple: (X, y) features and target
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col and col != 'player_id']
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Prepared {X.shape[1]} features and {len(y)} samples")
    return X, y

def get_groups_from_data(df, group_col='player_id'):
    """Extract group information for GroupKFold"""
    return df[group_col].values

def save_preprocessor(preprocessor, filepath):
    """Save sklearn preprocessor to file"""
    import joblib
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, filepath)
    logger.info(f"Saved preprocessor to {filepath}")

def load_preprocessor(filepath):
    """Load sklearn preprocessor from file"""
    import joblib
    preprocessor = joblib.load(filepath)
    logger.info(f"Loaded preprocessor from {filepath}")
    return preprocessor

def setup_logging(level=logging.INFO):
    """Setup consistent logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

class ClassificationReportFormatter:
    """Helper class to format classification reports consistently"""
    
    @staticmethod
    def format_report_dict(report_dict):
        """Convert sklearn classification report dict to formatted string"""
        # TODO: Implement pretty formatting for classification reports
        return str(report_dict)
    
    @staticmethod
    def save_report(report_dict, filepath):
        """Save classification report to file"""
        formatted_report = ClassificationReportFormatter.format_report_dict(report_dict)
        
        with open(filepath, 'w') as f:
            f.write(formatted_report)
        
        logger.info(f"Saved classification report to {filepath}")

def validate_file_paths(*file_paths):
    """Validate that all required file paths exist"""
    missing_files = []
    
    for file_path in file_paths:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    logger.info("All required files validated successfully")

def create_output_directories(*dir_paths):
    """Create output directories if they don't exist"""
    for dir_path in dir_paths:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}") 
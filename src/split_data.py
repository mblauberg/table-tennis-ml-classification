#!/usr/bin/env python3
"""
Data Splitting Module for COMP4702 Assignment

Implements group-aware data partitioning using GroupKFold to ensure
no player_id leakage across train/validation/test sets.

Week 4 Concepts:
- Cross-validation strategies
- Data leakage prevention
- Stratified sampling considerations
"""

import argparse
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import GroupKFold

# Random seed for reproducibility
SEED = 123

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_group_splits(df, group_col='player_id', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create group-aware train/validation/test splits using GroupKFold (Week 4)
    
    Args:
        df: DataFrame with processed data
        group_col: Column to group by (default: 'player_id')
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' indices
    """
    # TODO: Implement GroupKFold splitting logic
    logger.info(f"Creating group-aware splits by {group_col}")
    logger.info(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Placeholder return
    n_samples = len(df)
    indices = np.arange(n_samples)
    
    return {
        'train': indices[:int(n_samples * train_ratio)].tolist(),
        'val': indices[int(n_samples * train_ratio):int(n_samples * (train_ratio + val_ratio))].tolist(),
        'test': indices[int(n_samples * (train_ratio + val_ratio)):].tolist()
    }

def save_splits(splits, output_dir):
    """Save split indices to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, indices in splits.items():
        file_path = output_path / f"{split_name}.json"
        with open(file_path, 'w') as f:
            json.dump(indices, f, indent=2)
        logger.info(f"Saved {len(indices)} {split_name} indices to {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Group-aware data splitting for table tennis classification')
    parser.add_argument('--input', required=True, help='Input processed CSV file path')
    parser.add_argument('--output_dir', required=True, help='Output directory for split JSON files')
    parser.add_argument('--group_col', default='player_id', help='Column to group by')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(SEED)
    
    logger.info(f"Starting data splitting...")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {SEED}")
    
    # Load processed data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from processed data")
    
    # Create splits
    splits = create_group_splits(df, args.group_col)
    
    # Save splits
    save_splits(splits, args.output_dir)
    
    logger.info("Data splitting complete")

if __name__ == "__main__":
    main() 
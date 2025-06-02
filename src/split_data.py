#!/usr/bin/env python3
"""
Group-Aware Data Splitting
COMP4702 Assignment - Table Tennis Swing Classification

Simple group-aware splitting to prevent data leakage.
Uses player IDs to ensure no player appears in both training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import json
import os

def create_group_aware_splits(data_path='data/processed/assignTTSWING_processed.csv', 
                             test_size=0.20, random_state=42):
    """Create group-aware train/test splits and save indices."""
    
    # Load processed data
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape}")
    print(f"Unique players: {df['id'].nunique()}")
    print(f"Class distribution: {df['testmode'].value_counts().sort_index().to_dict()}")
    
    # Prepare for splitting
    X = df.drop(['testmode'], axis=1)
    y = df['testmode']
    groups = df['id']  # Player IDs for grouping
    
    # Create train/test split ensuring no player overlap
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    # Quick validation
    train_players = set(groups.iloc[train_idx])
    test_players = set(groups.iloc[test_idx])
    assert len(train_players.intersection(test_players)) == 0, "Player overlap detected!"
    
    print(f"Training: {len(train_idx)} samples, {len(train_players)} players")
    print(f"Test: {len(test_idx)} samples, {len(test_players)} players")
    
    # Save splits
    os.makedirs('splits', exist_ok=True)
    
    with open('splits/train_indices.json', 'w') as f:
        json.dump(train_idx.tolist(), f)
    
    with open('splits/test_indices.json', 'w') as f:
        json.dump(test_idx.tolist(), f)
    
    # For validation split used in some train files
    val_size = int(len(train_idx) * 0.2)  # 20% of training data
    np.random.seed(random_state)
    val_indices = np.random.choice(train_idx, val_size, replace=False)
    train_only_indices = np.setdiff1d(train_idx, val_indices)
    
    with open('splits/val_indices.json', 'w') as f:
        json.dump(val_indices.tolist(), f)
    
    with open('splits/train_only_indices.json', 'w') as f:
        json.dump(train_only_indices.tolist(), f)
    
    print("✓ Split indices saved to splits/ directory")
    return train_idx, test_idx, val_indices

def load_splits():
    """Load saved split indices."""
    with open('splits/train_indices.json', 'r') as f:
        train_idx = np.array(json.load(f))
    with open('splits/test_indices.json', 'r') as f:
        test_idx = np.array(json.load(f))
    return train_idx, test_idx

def main():
    """Create group-aware data splits."""
    print("Creating group-aware train/test splits...")
    train_idx, test_idx, val_idx = create_group_aware_splits()
    print("✓ Data splitting completed")

if __name__ == "__main__":
    main()

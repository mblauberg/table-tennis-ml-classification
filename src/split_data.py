#!/usr/bin/env python3
"""
Group-Aware Data Splitting for Table Tennis Swing Classification
COMP4702 Assignment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import json
import os

def create_group_aware_splits(data_path='data/processed/assignTTSWING_processed.csv', 
                             test_size=0.20, random_state=123):
    """Create group-aware train/test splits preventing data leakage."""
    
    # Load preprocessed dataset and examine structure
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape}")
    print(f"Unique players: {df['id'].nunique()}")
    print(f"Class distribution: {df['testmode'].value_counts().sort_index().to_dict()}")
    
    # Prepare data for group-aware splitting
    X = df.drop(['testmode'], axis=1)  # Features (all columns except target)
    y = df['testmode']                 # Target variable (swing classifications)
    groups = df['id']                  # Player IDs for group-aware splitting
    
    # Create group-aware train/test split ensuring no player overlap
    # GroupShuffleSplit ensures no group (player) appears in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    # Validate group separation - critical for preventing data leakage
    train_players = set(groups.iloc[train_idx])
    test_players = set(groups.iloc[test_idx])
    assert len(train_players.intersection(test_players)) == 0, "Player overlap detected!"
    
    print(f"Training: {len(train_idx)} samples, {len(train_players)} players")
    print(f"Test: {len(test_idx)} samples, {len(test_players)} players")
    
    # Save split indices for consistent use across all experiments
    os.makedirs('splits', exist_ok=True)
    
    # Save training indices (used for model training and cross-validation)
    with open('splits/train_indices.json', 'w') as f:
        json.dump(train_idx.tolist(), f)
    
    # Save test indices (reserved for final evaluation only)
    with open('splits/test_indices.json', 'w') as f:
        json.dump(test_idx.tolist(), f)
    
    print("✓ Split indices saved to splits/ directory")
    print("✓ train_indices.json = training set")
    print("✓ test_indices.json = test set")
    return train_idx, test_idx

def load_splits():
    """Load previously saved train/test split indices."""
    # Load training set indices
    with open('splits/train_indices.json', 'r') as f:
        train_idx = np.array(json.load(f))
    
    # Load test set indices
    with open('splits/test_indices.json', 'r') as f:
        test_idx = np.array(json.load(f))
    return train_idx, test_idx

def main():
    """Execute group-aware data splitting for the experimental pipeline."""
    print("Creating group-aware train/test splits...")
    train_idx, test_idx = create_group_aware_splits()
    print("✓ Data splitting completed")

if __name__ == "__main__":
    main()

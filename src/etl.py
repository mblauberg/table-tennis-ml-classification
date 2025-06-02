#!/usr/bin/env python3
"""
Enhanced Data Preprocessing Pipeline
COMP4702 Assignment - Table Tennis Swing Classification

Simple preprocessing pipeline with 5 key steps:
1. Removing missing and irrelevant data
2. Converting to physical units (g, °/s, rad/s) 
3. Encoding categorical variables
4. Feature scaling with StandardScaler
5. Data validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Unit conversion factors
ACC_CONVERSION = 2.0 / 32768.0  # ±2g range to m/s²
GYRO_CONVERSION = 250.0 / 32768.0  # ±250°/s range to °/s
GYRO_TO_RAD = np.pi / 180.0  # Convert °/s to rad/s
G_TO_MS2 = 9.81  # Standard gravity

def remove_missing_and_irrelevant_data(df):
    """Step 1: Remove missing data and irrelevant columns."""
    print("1. Removing missing and irrelevant data:")
    
    # Drop rows containing "???" in any categorical column
    categorical_cols = ['age', 'playYears', 'height', 'weight']
    initial_rows = len(df)
    
    for col in categorical_cols:
        if col in df.columns:
            df = df[df[col] != '???']
    
    rows_dropped = initial_rows - len(df)
    print(f"  Dropped {rows_dropped} rows containing '???' values")
    
    # Remove irrelevant columns
    columns_to_drop = ['date', 'teststage', 'count', 'newvar1', 'newvar2', 'newvar3', 'newvar4']
    dropped_cols = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=dropped_cols)
    print(f"  Removed columns: {dropped_cols}")
    
    # Remove perfect duplicates
    if 'holdRacketHanded' in df.columns and 'handedness' in df.columns:
        if df['holdRacketHanded'].equals(df['handedness']):
            df = df.drop(columns=['holdRacketHanded'])
            print(f"  Removed duplicate column: holdRacketHanded")
    
    print(f"  Shape after cleaning: {df.shape}")
    return df

def convert_to_physical_units(df):
    """Step 2: Convert sensor values from LSB to physical units."""
    print("2. Converting to physical units:")
    
    # Identify sensor columns
    acc_columns = [col for col in df.columns if col.startswith(('ax_', 'ay_', 'az_', 'a_'))]
    gyro_columns = [col for col in df.columns if col.startswith(('gx_', 'gy_', 'gz_', 'g_'))]
    
    # Convert accelerometer data (LSB → g → m/s²)
    for col in acc_columns:
        if col in df.columns:
            df[col] = df[col] * ACC_CONVERSION * G_TO_MS2
    
    # Convert gyroscope data (LSB → °/s → rad/s)
    for col in gyro_columns:
        if col in df.columns:
            df[col] = df[col] * GYRO_CONVERSION * GYRO_TO_RAD
    
    print(f"  ✓ Accelerometer: ±2.0g → ±{2.0 * G_TO_MS2:.1f} m/s²")
    print(f"  ✓ Gyroscope: ±250.0°/s → ±{250.0 * GYRO_TO_RAD:.2f} rad/s")
    
    return df

def encode_categorical_variables(df):
    """Step 3: Encode categorical variables."""
    print("3. Encoding categorical variables:")
    
    # One-hot encode demographic buckets
    categorical_cols = ['age', 'playYears', 'height', 'weight']
    encoded_columns = []
    
    for col in categorical_cols:
        if col in df.columns:
            # Create dummy variables (drop_first=True to avoid multicollinearity)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            encoded_columns.extend(dummies.columns.tolist())
            df = df.drop(columns=[col])
            print(f"  {col}: One-hot encoded → {len(dummies.columns)} features")
    
    # Keep binary variables as 0/1
    binary_cols = ['gender', 'handedness']
    for col in binary_cols:
        if col in df.columns:
            unique_vals = sorted(df[col].unique())
            print(f"  {col}: Kept as binary (values: {unique_vals})")
    
    print(f"  ✓ Total new dummy variables: {len(encoded_columns)}")
    return df, encoded_columns

def apply_feature_scaling(df, encoded_columns):
    """Step 4: Apply StandardScaler to numeric features."""
    print("4. Applying feature scaling:")
    
    # Identify numeric sensor features (exclude categorical, binary, and ID columns)
    exclude_cols = ['id', 'testmode'] + encoded_columns + ['gender', 'handedness']
    if 'holdRacketHanded' in df.columns:
        exclude_cols.append('holdRacketHanded')
    
    numeric_features = [col for col in df.columns if col not in exclude_cols]
    
    # Apply StandardScaler to numeric features
    scaler = StandardScaler()
    if numeric_features:
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        print(f"  ✓ Standardized {len(numeric_features)} numeric features")
        print(f"  ✓ One-hot and binary features kept in {{0,1}}")
    
    return df, scaler, numeric_features

def validate_preprocessing_quality(df_original, df_processed):
    """Step 5: Validate the preprocessing quality."""
    print("\n" + "=" * 50)
    print("PREPROCESSING QUALITY VALIDATION")
    print("=" * 50)
    
    # Check shape changes
    print(f"Shape change: {df_original.shape} → {df_processed.shape}")
    
    # Check target variable distribution
    original_classes = df_original['testmode'].value_counts().sort_index()
    processed_classes = df_processed['testmode'].value_counts().sort_index()
    
    print(f"\nTarget variable distribution:")
    for class_val in [0, 1, 2]:
        original_count = original_classes.get(class_val, 0)
        processed_count = processed_classes.get(class_val, 0)
        print(f"  Class {class_val}: {original_count} → {processed_count}")
    
    # Check for missing values
    missing_count = df_processed.isnull().sum().sum()
    print(f"\nMissing values in processed data: {missing_count}")
    
    # Check ID preservation for group-aware splitting
    original_players = df_original['id'].nunique()
    processed_players = df_processed['id'].nunique()
    print(f"Unique players preserved: {original_players} → {processed_players}")
    
    print(f"Total features: {len(df_processed.columns) - 2}")  # Exclude id and testmode
    return True

def preprocess_data(input_path='data/raw/assignTTSWING.csv', 
                   output_path='data/processed/assignTTSWING_processed.csv'):
    """Apply the complete preprocessing pipeline."""
    print("=" * 60)
    print("ENHANCED DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load raw data
    df_original = pd.read_csv(input_path)
    print(f"Loaded dataset: {df_original.shape}")
    df = df_original.copy()
    
    # Step 1: Remove missing and irrelevant data
    df = remove_missing_and_irrelevant_data(df)
    
    # Step 2: Convert to physical units
    df = convert_to_physical_units(df)
    
    # Step 3: Encode categorical variables
    df, encoded_cols = encode_categorical_variables(df)
    
    # Step 4: Apply feature scaling
    df, scaler, numeric_features = apply_feature_scaling(df, encoded_cols)
    
    # Step 5: Validate preprocessing quality
    validate_preprocessing_quality(df_original, df)
    
    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"\n✓ Processed dataset saved to: {output_path}")
    
    # Save preprocessing info
    preprocessing_info = {
        'original_shape': df_original.shape,
        'processed_shape': df.shape,
        'acc_conversion_factor': ACC_CONVERSION,
        'gyro_conversion_factor': GYRO_CONVERSION,
        'gyro_to_rad_factor': GYRO_TO_RAD,
        'numeric_features': numeric_features,
        'scaler_mean': scaler.mean_.tolist() if scaler.mean_ is not None else None,
        'scaler_scale': scaler.scale_.tolist() if scaler.scale_ is not None else None
    }
    
    info_path = output_path.replace('.csv', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"✓ Preprocessing info saved to: {info_path}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Next steps:")
    print("- Run split_data.py for group-aware train/test splitting")
    print("- Use individual model training files for correlation-based feature pruning")
    
    return df

def main():
    """Main preprocessing function."""
    preprocess_data()

if __name__ == "__main__":
    main()

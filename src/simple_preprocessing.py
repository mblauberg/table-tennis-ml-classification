"""
Simple Data Preprocessing Test
"""

import pandas as pd
import numpy as np
from .config import RAW_DATASET_PATH

def main():
    print("Starting data preprocessing...")
    
    # Load dataset
    try:
        df = pd.read_csv(RAW_DATASET_PATH)
        print(f"Dataset loaded successfully: {df.shape}")
        
        # Basic info
        print(f"Columns: {len(df.columns)}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Identify potential target variables
        if 'testmode' in df.columns:
            print(f"testmode distribution: {df['testmode'].value_counts().to_dict()}")
        
        if 'teststage' in df.columns:
            print(f"teststage distribution: {df['teststage'].value_counts().to_dict()}")
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical columns: {len(numerical_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        print("Basic preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
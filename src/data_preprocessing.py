"""
Data Cleaning and Preprocessing for ML Assignment
COMP4702 - Machine Learning

This script performs comprehensive data preprocessing including:
- Irrelevant column removal
- Missing value handling
- Feature scaling and normalization
- Categorical variable encoding
- Data splitting for ML pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from .config import (
    set_random_seeds, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE,
    RAW_DATASET_PATH, PROCESSED_DATA_DIR
)

# Set random seeds for reproducibility
set_random_seeds()

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for the ML assignment
    """
    
    def __init__(self, filepath=None):
        """
        Initialize the preprocessor
        
        Args:
            filepath (str): Path to the dataset (defaults to RAW_DATASET_PATH)
        """
        self.filepath = filepath or str(RAW_DATASET_PATH)
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.removed_columns = []
        self.preprocessing_log = []
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print(f"Dataset loaded: {self.df.shape}")
        self.log_step(f"Dataset loaded with shape {self.df.shape}")
        return self
    
    def log_step(self, message):
        """Log preprocessing steps"""
        self.preprocessing_log.append(message)
        print(f"✓ {message}")
    
    def identify_irrelevant_columns(self):
        """
        Identify and remove irrelevant columns based on EDA findings
        """
        print("\n" + "="*50)
        print("IDENTIFYING IRRELEVANT COLUMNS")
        print("="*50)
        
        # Columns to remove based on analysis
        irrelevant_cols = []
        
        # ID and index columns (not useful for ML)
        id_cols = ['id', 'fileindex', 'count']
        for col in id_cols:
            if col in self.df.columns:
                irrelevant_cols.append(col)
        
        # Date column (temporal but not useful for this classification)
        if 'date' in self.df.columns:
            irrelevant_cols.append('date')
        
        # Check for columns with single unique value
        single_value_cols = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                single_value_cols.append(col)
                irrelevant_cols.append(col)
        
        # Check for columns with too many missing values (>50%)
        high_missing_cols = []
        for col in self.df.columns:
            missing_pct = self.df[col].isnull().sum() / len(self.df) * 100
            if missing_pct > 50:
                high_missing_cols.append(col)
                irrelevant_cols.append(col)
        
        # Remove duplicates
        irrelevant_cols = list(set(irrelevant_cols))
        
        print(f"Identified irrelevant columns:")
        print(f"  ID/Index columns: {id_cols}")
        print(f"  Single value columns: {single_value_cols}")
        print(f"  High missing (>50%) columns: {high_missing_cols}")
        print(f"  Total to remove: {len(irrelevant_cols)}")
        
        # Store for documentation
        self.removed_columns = irrelevant_cols
        
        # Remove columns
        if irrelevant_cols:
            self.df = self.df.drop(columns=irrelevant_cols)
            self.log_step(f"Removed {len(irrelevant_cols)} irrelevant columns")
        else:
            self.log_step("No irrelevant columns found to remove")
        
        print(f"Dataset shape after removal: {self.df.shape}")
        return self
    
    def handle_missing_values(self):
        """
        Handle missing values using appropriate techniques
        """
        print("\n" + "="*50)
        print("HANDLING MISSING VALUES")
        print("="*50)
        
        # Check for missing values
        missing_summary = self.df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) == 0:
            self.log_step("No missing values found in the dataset")
            return self
        
        print(f"Missing values found in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            pct = count / len(self.df) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
        
        # Separate numerical and categorical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numerical missing values
        numerical_missing = [col for col in missing_cols.index if col in numerical_cols]
        if numerical_missing:
            # Use median imputation for numerical features (robust to outliers)
            imputer = SimpleImputer(strategy='median')
            self.df[numerical_missing] = imputer.fit_transform(self.df[numerical_missing])
            self.log_step(f"Imputed {len(numerical_missing)} numerical columns using median")
        
        # Handle categorical missing values
        categorical_missing = [col for col in missing_cols.index if col in categorical_cols]
        if categorical_missing:
            # Use mode imputation for categorical features
            imputer = SimpleImputer(strategy='most_frequent')
            self.df[categorical_missing] = imputer.fit_transform(self.df[categorical_missing])
            self.log_step(f"Imputed {len(categorical_missing)} categorical columns using mode")
        
        # Verify no missing values remain
        remaining_missing = self.df.isnull().sum().sum()
        if remaining_missing == 0:
            self.log_step("All missing values successfully handled")
        else:
            print(f"Warning: {remaining_missing} missing values still remain")
        
        return self
    
    def encode_categorical_variables(self):
        """
        Encode categorical variables appropriately
        """
        print("\n" + "="*50)
        print("ENCODING CATEGORICAL VARIABLES")
        print("="*50)
        
        # Identify categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            self.log_step("No categorical variables found to encode")
            return self
        
        print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
        
        # Analyze each categorical column
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            print(f"  {col}: {unique_values} unique values")
            
            if unique_values <= 10:  # Use label encoding for low cardinality
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                self.log_step(f"Label encoded '{col}' ({unique_values} categories)")
            else:  # For high cardinality, consider other approaches
                print(f"    Warning: {col} has high cardinality ({unique_values})")
                # For now, still use label encoding but note this
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                self.log_step(f"Label encoded '{col}' (high cardinality: {unique_values})")
        
        return self
    
    def detect_and_handle_outliers(self, method='iqr', threshold=1.5):
        """
        Detect and optionally handle outliers
        
        Args:
            method (str): Method for outlier detection ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
        """
        print("\n" + "="*50)
        print("OUTLIER DETECTION AND HANDLING")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_summary = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = self.df[z_scores > threshold]
            
            outlier_count = len(outliers)
            outlier_pct = outlier_count / len(self.df) * 100
            
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_pct
            }
        
        # Print outlier summary
        print(f"Outlier detection using {method} method (threshold={threshold}):")
        total_outliers = 0
        for col, stats in outlier_summary.items():
            if stats['count'] > 0:
                print(f"  {col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
                total_outliers += stats['count']
        
        if total_outliers == 0:
            self.log_step("No significant outliers detected")
        else:
            # For this assignment, we'll keep outliers but note them
            # In practice, you might want to remove or cap them
            self.log_step(f"Detected outliers in multiple columns (kept for analysis)")
        
        return self
    
    def apply_feature_scaling(self, method='standard'):
        """
        Apply feature scaling to numerical variables
        
        Args:
            method (str): Scaling method ('standard', 'minmax')
        """
        print("\n" + "="*50)
        print("FEATURE SCALING")
        print("="*50)
        
        # Identify numerical columns (excluding target variables)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove potential target variables from scaling
        target_candidates = ['testmode', 'teststage']
        feature_cols = [col for col in numerical_cols if col not in target_candidates]
        
        if not feature_cols:
            self.log_step("No numerical features found for scaling")
            return self
        
        print(f"Scaling {len(feature_cols)} numerical features using {method} scaling")
        
        # Choose scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply scaling
        scaled_features = self.scaler.fit_transform(self.df[feature_cols])
        
        # Replace original features with scaled versions
        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=self.df.index)
        
        # Update dataframe
        for col in feature_cols:
            self.df[col] = scaled_df[col]
        
        self.log_step(f"Applied {method} scaling to {len(feature_cols)} features")
        
        return self
    
    def prepare_target_variables(self):
        """
        Prepare and analyze target variables
        """
        print("\n" + "="*50)
        print("TARGET VARIABLE PREPARATION")
        print("="*50)
        
        target_candidates = ['testmode', 'teststage']
        
        for target in target_candidates:
            if target in self.df.columns:
                print(f"\n{target} distribution:")
                value_counts = self.df[target].value_counts().sort_index()
                for val, count in value_counts.items():
                    pct = count / len(self.df) * 100
                    print(f"  Class {val}: {count:,} ({pct:.1f}%)")
                
                # Check for class imbalance
                min_class_pct = (value_counts.min() / len(self.df)) * 100
                if min_class_pct < 10:
                    print(f"  ⚠️  Class imbalance detected (smallest class: {min_class_pct:.1f}%)")
        
        self.log_step("Target variable analysis completed")
        return self
    
    def split_data(self, target_col='testmode', stratify=True):
        """
        Split data into train, validation, and test sets
        
        Args:
            target_col (str): Target variable column name
            stratify (bool): Whether to stratify the split
        """
        print("\n" + "="*50)
        print("DATA SPLITTING")
        print("="*50)
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Prepare features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target column: {target_col}")
        
        # First split: train+val vs test
        stratify_param = y if stratify else None
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, 
            stratify=stratify_param
        )
        
        # Second split: train vs val
        val_size = VALIDATION_SIZE / (1 - TEST_SIZE)  # Adjust for remaining data
        stratify_param = y_temp if stratify else None
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=RANDOM_SEED,
            stratify=stratify_param
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Print class distributions
        print(f"\nClass distributions:")
        for split_name, y_split in [('Train', self.y_train), ('Val', self.y_val), ('Test', self.y_test)]:
            dist = y_split.value_counts(normalize=True).sort_index() * 100
            dist_str = ', '.join([f"Class {k}: {v:.1f}%" for k, v in dist.items()])
            print(f"  {split_name}: {dist_str}")
        
        self.log_step(f"Data split into train/val/test with stratification={stratify}")
        return self
    
    def save_processed_data(self, prefix='processed_'):
        """
        Save processed datasets
        
        Args:
            prefix (str): Prefix for saved files
        """
        print("\n" + "="*50)
        print("SAVING PROCESSED DATA")
        print("="*50)
        
        # Save full processed dataset
        dataset_path = PROCESSED_DATA_DIR / f'{prefix}dataset.csv'
        self.df.to_csv(dataset_path, index=False)
        self.log_step(f"Saved full processed dataset: {dataset_path}")
        
        # Save train/val/test splits if available
        if self.X_train is not None:
            # Combine features and target for each split
            train_data = pd.concat([self.X_train, self.y_train], axis=1)
            val_data = pd.concat([self.X_val, self.y_val], axis=1)
            test_data = pd.concat([self.X_test, self.y_test], axis=1)
            
            train_path = PROCESSED_DATA_DIR / f'{prefix}train.csv'
            val_path = PROCESSED_DATA_DIR / f'{prefix}val.csv'
            test_path = PROCESSED_DATA_DIR / f'{prefix}test.csv'
            
            train_data.to_csv(train_path, index=False)
            val_data.to_csv(val_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            self.log_step("Saved train/val/test splits")
        
        return self
    
    def generate_preprocessing_report(self):
        """
        Generate a comprehensive preprocessing report
        """
        print("\n" + "="*60)
        print("PREPROCESSING REPORT")
        print("="*60)
        
        report = []
        report.append("# Data Preprocessing Report")
        report.append("## COMP4702 Machine Learning Assignment\n")
        
        report.append("### Dataset Overview")
        report.append(f"- **Original shape:** {pd.read_csv(self.filepath).shape}")
        report.append(f"- **Final shape:** {self.df.shape}")
        report.append(f"- **Features removed:** {len(self.removed_columns)}")
        report.append(f"- **Preprocessing steps:** {len(self.preprocessing_log)}\n")
        
        report.append("### Removed Columns")
        if self.removed_columns:
            for col in self.removed_columns:
                report.append(f"- `{col}`")
        else:
            report.append("- None")
        report.append("")
        
        report.append("### Categorical Encoding")
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                classes = list(encoder.classes_)
                report.append(f"- `{col}`: {len(classes)} categories")
        else:
            report.append("- No categorical variables encoded")
        report.append("")
        
        report.append("### Feature Scaling")
        if self.scaler:
            scaler_type = type(self.scaler).__name__
            report.append(f"- **Method:** {scaler_type}")
            report.append(f"- **Features scaled:** {len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 'Multiple'}")
        else:
            report.append("- No scaling applied")
        report.append("")
        
        report.append("### Data Splits")
        if self.X_train is not None:
            report.append(f"- **Training:** {len(self.X_train):,} samples")
            report.append(f"- **Validation:** {len(self.X_val):,} samples")
            report.append(f"- **Test:** {len(self.X_test):,} samples")
        else:
            report.append("- No data splits created")
        report.append("")
        
        report.append("### Preprocessing Steps")
        for i, step in enumerate(self.preprocessing_log, 1):
            report.append(f"{i}. {step}")
        
        # Save report
        with open('preprocessing_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("✓ Preprocessing report saved: preprocessing_report.md")
        
        # Print summary
        for line in report:
            print(line)
        
        return self

def main():
    """
    Main preprocessing pipeline
    """
    print("COMP4702 ML Assignment - Data Preprocessing Pipeline")
    print("="*60)
    
    # Initialize preprocessor (uses RAW_DATASET_PATH by default)
    preprocessor = DataPreprocessor()
    
    # Execute preprocessing pipeline
    preprocessor.load_data() \
                .identify_irrelevant_columns() \
                .handle_missing_values() \
                .encode_categorical_variables() \
                .detect_and_handle_outliers() \
                .apply_feature_scaling(method='standard') \
                .prepare_target_variables() \
                .split_data(target_col='testmode', stratify=True) \
                .save_processed_data() \
                .generate_preprocessing_report()
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return preprocessor

if __name__ == "__main__":
    preprocessor = main() 
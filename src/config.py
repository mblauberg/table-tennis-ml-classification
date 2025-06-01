"""
Configuration file for COMP4702 Machine Learning Assignment
Focus: Demonstrating course understanding through data-driven model selection
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, 
                  DOCS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
DATASET_PATH = RAW_DATA_DIR / "assignTTSWING.csv"

# Processed data paths
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_dataset.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"

# Figure directories (organized by analysis type)
EDA_FIGURES_DIR = FIGURES_DIR / "eda_plots"
MODEL_FIGURES_DIR = FIGURES_DIR / "model_analysis"
COMPARISON_FIGURES_DIR = FIGURES_DIR / "comparisons"

# Create figure subdirectories
for fig_dir in [EDA_FIGURES_DIR, MODEL_FIGURES_DIR, COMPARISON_FIGURES_DIR]:
    fig_dir.mkdir(parents=True, exist_ok=True)

# Model directories (organized by type)
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
ENSEMBLE_MODELS_DIR = MODELS_DIR / "ensemble"
SVM_MODELS_DIR = MODELS_DIR / "svm"

# Create model subdirectories
for model_dir in [BASELINE_MODELS_DIR, ENSEMBLE_MODELS_DIR, SVM_MODELS_DIR]:
    model_dir.mkdir(parents=True, exist_ok=True)

# Results directories (organized by analysis type)
PERFORMANCE_RESULTS_DIR = RESULTS_DIR / "performance_metrics"
FEATURE_RESULTS_DIR = RESULTS_DIR / "feature_analysis"
HYPERPARAMETER_RESULTS_DIR = RESULTS_DIR / "hyperparameter_tuning"

# Create results subdirectories
for results_dir in [PERFORMANCE_RESULTS_DIR, FEATURE_RESULTS_DIR, HYPERPARAMETER_RESULTS_DIR]:
    results_dir.mkdir(parents=True, exist_ok=True)

# Dataset configuration
TARGET_COLUMN = "testmode"
FEATURE_COLUMNS = None  # Will be determined during data loading
CATEGORICAL_FEATURES = ["gender", "handedness", "teststage"]
NUMERICAL_FEATURES = None  # Will be determined during data loading

# Data preprocessing configuration
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # From remaining training data
RANDOM_STATE = 42
STRATIFY = True

# Cross-validation configuration
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Model configurations for course demonstration

# Logistic Regression (Baseline) - Course concepts: Linear classification, MLE, regularization
LOGISTIC_REGRESSION_CONFIG = {
    'regularization_types': ['l1', 'l2', 'elasticnet'],
    'C_values': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratios': [0.1, 0.3, 0.5, 0.7, 0.9],  # For elasticnet
    'max_iter': 1000,
    'class_weight': ['balanced', None],
    'solver': 'liblinear'  # Supports L1 regularization
}

# Random Forest (Advanced) - Course concepts: Ensemble learning, bagging, feature importance
RANDOM_FOREST_CONFIG = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'bootstrap': [True, False],
    'random_state': RANDOM_STATE
}

# Support Vector Machine (Advanced) - Course concepts: Margin maximization, kernel methods
SVM_CONFIG = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
    'degree': [2, 3, 4],  # For polynomial kernel
    'class_weight': ['balanced', None],
    'random_state': RANDOM_STATE
}

# Feature selection configuration
FEATURE_SELECTION_CONFIG = {
    'univariate_k': [10, 15, 20, 25, 30],
    'rfe_features': [10, 15, 20, 25, 30],
    'l1_alpha': [0.001, 0.01, 0.1, 1.0]
}

# Class imbalance handling strategies
CLASS_IMBALANCE_STRATEGIES = {
    'class_weights': ['balanced'],
    'sampling_methods': ['smote', 'random_oversample', 'random_undersample'],
    'threshold_tuning': True
}

# Evaluation metrics for course demonstration
EVALUATION_METRICS = [
    'accuracy',
    'precision_macro',
    'recall_macro',
    'f1_macro',
    'precision_weighted',
    'recall_weighted',
    'f1_weighted',
    'roc_auc_ovr'  # One-vs-Rest for multi-class
]

# Visualization configuration
FIGURE_SIZE = (10, 8)
DPI = 300
STYLE = 'seaborn-v0_8'

# Random seeds for reproducibility (course emphasis on reproducible results)
RANDOM_SEEDS = [42, 123, 456, 789, 101112]

# Course understanding demonstration parameters
COURSE_CONCEPTS = {
    'logistic_regression': [
        'Linear decision boundaries',
        'Maximum likelihood estimation',
        'Sigmoid function',
        'L1/L2 regularization',
        'Bias-variance tradeoff',
        'Feature scaling importance'
    ],
    'random_forest': [
        'Bootstrap aggregating (bagging)',
        'Random subspaces',
        'Ensemble learning',
        'Feature importance (Gini)',
        'Out-of-bag error',
        'Overfitting prevention',
        'Bias-variance decomposition'
    ],
    'svm': [
        'Margin maximization',
        'Support vectors',
        'Kernel trick',
        'Feature mapping',
        'Lagrange multipliers',
        'Regularization (C parameter)',
        'Kernel selection'
    ]
}

# Data characteristics that justify model selection
DATA_CHARACTERISTICS = {
    'type': 'Tabular sensor data',
    'features': 46,  # Numerical features
    'samples': 97355,
    'target_classes': 3,
    'class_distribution': [0.077, 0.759, 0.164],  # Approximate
    'missing_values': False,
    'feature_scales': 'Mixed (requires normalization)',
    'linearity': 'Mixed (some linear, some non-linear relationships)',
    'outliers': 'Present but valid sensor readings',
    'correlations': 'High between related sensors'
}

# Model justification based on data characteristics
MODEL_JUSTIFICATIONS = {
    'logistic_regression': {
        'data_reasons': [
            'Linear relationships in many sensor features',
            'High dimensionality benefits from regularization',
            'Probabilistic output suitable for imbalanced classes',
            'Fast training for hyperparameter exploration'
        ],
        'course_concepts': [
            'Linear classification foundation',
            'Regularization for high-dimensional data',
            'Maximum likelihood estimation',
            'Feature scaling necessity'
        ]
    },
    'random_forest': {
        'data_reasons': [
            'Sensor interactions likely non-linear',
            'Robust to outliers in sensor readings',
            'Built-in feature importance for 46 features',
            'Handles class imbalance through sampling'
        ],
        'course_concepts': [
            'Ensemble learning for complex patterns',
            'Bootstrap aggregating reduces variance',
            'Feature importance interpretation',
            'Overfitting prevention through averaging'
        ]
    },
    'svm': {
        'data_reasons': [
            'Effective with high-dimensional data (46 features)',
            'Kernel methods capture non-linear sensor patterns',
            'Margin maximization for clear class separation',
            'Robust to outliers through support vector focus'
        ],
        'course_concepts': [
            'Margin maximization principle',
            'Kernel trick for non-linear relationships',
            'Support vector concept',
            'Regularization through C parameter'
        ]
    }
}

# Analysis phases for course demonstration
ANALYSIS_PHASES = {
    'phase_1': 'Data Understanding & EDA with Course Concepts',
    'phase_2': 'Baseline Implementation (Logistic Regression)',
    'phase_3': 'Advanced Model 1 (Random Forest)',
    'phase_4': 'Advanced Model 2 (SVM)',
    'phase_5': 'Comparative Analysis & Course Integration'
}

# Documentation requirements
DOCUMENTATION_REQUIREMENTS = {
    'theoretical_background': 'Course concept explanation for each model',
    'data_justification': 'Why each model fits our specific data',
    'implementation_details': 'Show understanding through proper coding',
    'results_interpretation': 'Analyze using course concepts',
    'course_connections': 'Explicit links to lecture material'
}

print(f"✅ Configuration loaded successfully")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📊 Dataset path: {DATASET_PATH}")
print(f"🎯 Target column: {TARGET_COLUMN}")
print(f"🔬 Focus: Course understanding demonstration through data-driven model selection") 
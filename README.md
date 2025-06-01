# COMP4702 Machine Learning Assignment: Table Tennis Swing Classification

This repository contains a complete machine learning pipeline for classifying table tennis swings using wearable IMU sensor data. The project demonstrates core COMP4702 concepts through systematic implementation of preprocessing, model training, and evaluation workflows.

## Quick Start

### Prerequisites
- Conda package manager
- Python 3.10+

### Setup Environment
```bash
# Clone repository and navigate to project
cd Assignment/

# Create and activate conda environment
conda env create -f environment.yml
conda activate ml_assignment

# Verify data exists
ls data/raw/assignTTSWING.csv
```

### Run Complete Pipeline
```bash
# 1. Data preprocessing (ETL)
python src/etl.py \
  --input data/raw/assignTTSWING.csv \
  --output data/processed/processed_data.csv

# 2. Create train/validation/test splits
python src/split_data.py \
  --input data/processed/processed_data.csv \
  --output_dir splits/

# 3. Exploratory Data Analysis
python src/eda.py \
  --input data/raw/assignTTSWING.csv \
  --output_dir results/eda

# 4. Train all models
python src/train_lr.py \
  --data data/processed/processed_data.csv \
  --splits splits/train.json splits/val.json \
  --output_dir models

python src/train_rf.py \
  --data data/processed/processed_data.csv \
  --splits splits/train.json splits/val.json \
  --output_dir models

python src/train_lgbm.py \
  --data data/processed/processed_data.csv \
  --splits splits/train.json splits/val.json \
  --output_dir models

python src/train_gp.py \
  --data data/processed/processed_data.csv \
  --splits splits/train.json splits/val.json \
  --output_dir models

# 5. Evaluate all models
python src/evaluate.py \
  --data data/processed/processed_data.csv \
  --test_split splits/test.json \
  --output_dir results

# 6. Bootstrap confidence intervals
python src/bootstrap.py \
  --data data/processed/processed_data.csv \
  --splits splits/test.json \
  --output_dir results/bootstrap
```

## Project Overview

### Dataset
- **Source**: Table tennis swing IMU data (Dryad DOI 10.5061/dryad.0zpc8677f)
- **Size**: 46MB CSV file with ~97,350 samples after preprocessing
- **Features**: 44 numeric features from 6-axis IMU sensor statistics
- **Target**: testmode - 3-class classification (0: air swing, 1: full power, 2: stable)
- **Challenge**: Class imbalance (7.7% / 75.9% / 16.4%) and player-specific variations

### Models Implemented

1. **Logistic Regression (Baseline)**
   - Multinomial classification with L2 regularization
   - Linear decision boundary baseline
   - Interpretable coefficients

2. **Random Forest (Ensemble)**
   - Bootstrap aggregating for robust predictions
   - Optuna hyperparameter optimization
   - Feature importance analysis

3. **LightGBM (Advanced Ensemble)**
   - Gradient boosting with early stopping
   - SHAP interpretability analysis
   - Optimized leaf-wise tree growth

4. **Sparse Gaussian Process (Bayesian)**
   - Uncertainty quantification with prediction confidence
   - PCA dimensionality reduction (20 components)
   - Inducing point approximation for scalability

### COMP4702 Concept Mapping

| Week | Topic | Implementation |
|------|-------|----------------|
| 1-2  | Exploratory Data Analysis | Feature distributions, correlation analysis, class separability |
| 3-5  | Data Engineering & Validation | Unit conversion, outlier filtering, GroupKFold splitting |
| 6    | Preprocessing & Dimensionality | StandardScaler, PCA for computational efficiency |
| 9    | Ensemble Methods | Random Forest (bagging), LightGBM (boosting) |
| 10   | Interpretability | SHAP values, feature importance visualization |
| 11   | Bayesian Methods | GP uncertainty quantification, calibration analysis |

## Directory Structure

```
Assignment/
├── data/
│   ├── raw/assignTTSWING.csv          # Original dataset
│   └── processed/processed_data.csv   # Cleaned data
├── splits/                            # Train/val/test indices
│   ├── train.json
│   ├── val.json
│   └── test.json
├── models/                           # Trained model artifacts
│   ├── lr.pkl
│   ├── rf.pkl
│   ├── lgbm.pkl
│   ├── gp.pkl
│   └── scaler.pkl
├── results/                          # Evaluation outputs and plots
│   ├── eda/                         # Exploratory data analysis
│   ├── bootstrap/                   # Bootstrap confidence intervals
│   └── confusion_matrix_*.png
├── src/                             # Source code modules
│   ├── etl.py
│   ├── split_data.py
│   ├── eda.py
│   ├── train_lr.py
│   ├── train_rf.py
│   ├── train_lgbm.py
│   ├── train_gp.py
│   ├── evaluate.py
│   ├── bootstrap.py
│   └── utils.py
├── tasks/                           # Task management (Taskmaster)
├── logs/                           # Training and execution logs
├── environment.yml                  # Conda environment
├── DOCUMENT.md                      # Main assignment report
└── README.md                        # This file
```

## Module Descriptions

### `src/eda.py`
Exploratory Data Analysis:
- Comprehensive statistical analysis of features
- Class distribution and imbalance analysis
- Feature correlation and separability analysis
- Player distribution patterns

### `src/etl.py`
Data extraction, transformation, and loading:
- Converts raw LSB values to physical units (g-force, degrees/second)
- Applies signal processing (median despike, Butterworth filtering)
- Physics-based outlier removal (||acceleration|| > 16g)
- Implements Week 3-5 preprocessing concepts

### `src/split_data.py`
Group-aware data partitioning:
- Uses StratifiedGroupKFold to prevent player leakage
- Creates 61/18.7/20.3% train/validation/test splits
- Saves indices as JSON for reproducibility

### `src/train_lr.py`
Logistic Regression training:
- Multinomial classification baseline
- Optuna hyperparameter optimization
- StandardScaler fitting and persistence

### `src/train_rf.py`
Random Forest training pipeline:
- Optuna hyperparameter optimization with GroupKFold validation
- Feature importance extraction
- Out-of-bag error estimation

### `src/train_lgbm.py`
LightGBM training with interpretability:
- Advanced gradient boosting with early stopping
- SHAP value computation for model explanation
- Class weight handling for imbalanced data

### `src/train_gp.py`
Sparse Gaussian Process implementation:
- GPytorch-based variational inference
- PCA preprocessing for computational efficiency
- Uncertainty quantification and prediction confidence

### `src/evaluate.py`
Comprehensive model evaluation:
- Performance metrics for all trained models
- Confusion matrix generation
- Bootstrap confidence intervals
- Calibration analysis for probabilistic models

### `src/bootstrap.py`
Bootstrap Confidence Intervals:
- Group-aware stratified bootstrap sampling
- Robust uncertainty quantification
- Statistical significance testing
- Visualization of bootstrap distributions

### `src/utils.py`
Shared utility functions:
- Data loading and preprocessing helpers
- Evaluation metrics and visualization functions
- Bootstrap confidence interval calculations
- Logging configuration

## Key Features

### Reproducibility
- Fixed random seeds (SEED = 123)
- Environment specification via `environment.yml`
- Comprehensive logging throughout pipeline
- Modular design for easy debugging

### Statistical Rigor
- Group-aware cross-validation prevents data leakage
- Bootstrap confidence intervals quantify uncertainty
- Multiple evaluation metrics address class imbalance
- Stratified sampling maintains class distributions

### Course Integration
- Each module explicitly references relevant COMP4702 weeks
- Implementation choices justified by course theory
- Progressive complexity from linear baselines to Bayesian approaches
- Comprehensive evaluation with uncertainty quantification

## Expected Runtime

- ETL: ~2 minutes
- Data splitting: ~30 seconds
- EDA: ~3 minutes
- Logistic Regression training: ~5 minutes
- Random Forest training: ~10 minutes (100 Optuna trials)
- LightGBM training: ~15 minutes (100 Optuna trials + SHAP)
- GP training: ~30 minutes (PCA + variational optimization)
- Evaluation: ~5 minutes
- Bootstrap analysis: ~5 minutes

**Total Pipeline**: ~1.5 hours on modern hardware

## Results Location

After running the complete pipeline:

### Performance Metrics
- **Overall results**: `results/metrics.csv`
- **Bootstrap CIs**: `results/bootstrap/bootstrap_ci_summary.csv`
- **EDA summary**: `results/eda/eda_summary_report.md`

### Visualizations
- **Confusion matrices**: `results/confusion_matrix_*.png`
- **EDA plots**: `results/eda/`
- **Bootstrap distributions**: `results/bootstrap/distributions/`
- **Bootstrap comparisons**: `results/bootstrap/comparisons/`

### Analysis Reports
- **Main academic report**: `DOCUMENT.md`
- **EDA findings**: `results/eda/eda_summary_report.md`
- **Bootstrap analysis**: `results/bootstrap/bootstrap_analysis_report.md`

## Task Management

This project uses Task Master for project management. Key completed components:

✅ Project structure setup  
✅ ETL implementation with signal processing  
✅ Group-aware data splitting  
✅ Four model implementations (LR, RF, LightGBM, GP)  
✅ Comprehensive evaluation framework  
✅ Bootstrap confidence intervals  
✅ Exploratory data analysis  
✅ Academic documentation (DOCUMENT.md)  

## Performance Summary

Based on bootstrap confidence intervals (95% CI):

1. **Random Forest**: Highest macro-F1 performance with robust uncertainty estimates
2. **LightGBM**: Strong ensemble performance with excellent interpretability
3. **Gaussian Process**: Bayesian uncertainty quantification for confidence-aware predictions
4. **Logistic Regression**: Solid linear baseline with interpretable coefficients

All models significantly outperform random chance (33.3%) demonstrating successful learning from IMU sensor statistics.

## Documentation

The primary deliverable is `DOCUMENT.md`, containing:
- Comprehensive methodology description
- Statistical analysis and results
- Model comparison and interpretation
- Academic-style writing with proper citations
- Week-by-week concept mapping to COMP4702 curriculum

For technical details and implementation specifics, refer to individual module docstrings and the comprehensive logging output.

## Troubleshooting

### Environment Issues
```bash
# If environment creation fails
conda clean --all
conda env create -f environment.yml

# If PyTorch/GPytorch issues occur
conda install pytorch torchvision torchaudio -c pytorch
pip install gpytorch
```

### Memory Issues
- GP training may require 8GB+ RAM
- Reduce `n_inducing` points if needed
- Consider reducing PCA components for very large datasets

### Performance Issues
- Use `n_jobs=-1` for parallel processing
- Reduce Optuna `n_trials` for faster iteration
- Monitor GPU usage for PyTorch operations

## License & Attribution

This implementation is for educational purposes as part of COMP4702 coursework at the University of Queensland. The original dataset is available via Dryad (DOI 10.5061/dryad.0zpc8677f) under appropriate licensing terms.

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
  --output data/processed/processed.csv

# 2. Create train/validation/test splits
python src/split_data.py \
  --input data/processed/processed.csv \
  --output_dir splits/

# 3. Train baseline Random Forest
python src/train_rf.py \
  --data data/processed/processed.csv \
  --train_split splits/train.json \
  --val_split splits/val.json \
  --output models/rf.pkl

# 4. Train advanced LightGBM
python src/train_lgbm.py \
  --data data/processed/processed.csv \
  --train_split splits/train.json \
  --val_split splits/val.json \
  --output models/lgbm.pkl

# 5. Train Sparse Gaussian Process
python src/train_gp.py \
  --data data/processed/processed.csv \
  --train_split splits/train.json \
  --val_split splits/val.json \
  --output models/gp.pkl

# 6. Evaluate all models
python src/evaluate.py \
  --data data/processed/processed.csv \
  --test_split splits/test.json \
  --rf_model models/rf.pkl \
  --lgbm_model models/lgbm.pkl \
  --gp_model models/gp.pkl \
  --output_dir results/
```

## Project Overview

### Dataset
- **Source**: Table tennis swing IMU data (Dryad DOI 10.5061/dryad.0zpc8677f)
- **Size**: 46MB CSV file with multivariate time-series features
- **Target**: testmode: 3-class classification (air swing, full power, stable)
- **Challenge**: Player-specific variations require group-aware validation

### Models Implemented

1. **Random Forest (Baseline)**
   - Bootstrap aggregating for robust predictions
   - Optuna hyperparameter optimization
   - Feature importance analysis

2. **LightGBM (Advanced Ensemble)**
   - Gradient boosting with early stopping
   - SHAP interpretability analysis
   - Optimized leaf-wise tree growth

3. **Sparse Gaussian Process (Bayesian)**
   - Uncertainty quantification with prediction confidence
   - PCA dimensionality reduction (20 components)
   - Inducing point approximation (1,000 points)

### COMP4702 Concept Mapping

| Week | Topic | Implementation |
|------|-------|----------------|
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
│   └── processed/processed.csv        # Cleaned data
├── splits/                            # Train/val/test indices
├── models/                           # Trained model artifacts
├── results/                          # Evaluation outputs and plots
├── src/                             # Source code modules
├── docs/sections/                   # Documentation templates
├── environment.yml                  # Conda environment
├── DOCUMENT.md                      # Main assignment report
└── README.md                        # This file
```

## Module Descriptions

### `src/etl.py`
Data extraction, transformation, and loading:
- Converts raw LSB values to physical units (g-force, degrees/second)
- Applies data quality filters (NaN removal, outlier detection)
- Implements Week 3-5 preprocessing concepts

### `src/split_data.py`
Group-aware data partitioning:
- Uses GroupKFold to prevent player leakage
- Creates 70/15/15% train/validation/test splits
- Saves indices as JSON for reproducibility

### `src/train_rf.py`
Random Forest training pipeline:
- Optuna hyperparameter optimization with GroupKFold validation
- StandardScaler fitting and persistence
- Feature importance extraction

### `src/train_lgbm.py`
LightGBM training with interpretability:
- Advanced gradient boosting with early stopping
- SHAP value computation for model explanation
- Reuses preprocessing from Random Forest

### `src/train_gp.py`
Sparse Gaussian Process implementation:
- GPytorch-based variational inference
- PCA preprocessing for computational efficiency
- Uncertainty quantification and calibration analysis

### `src/evaluate.py`
Comprehensive model evaluation:
- Bootstrap confidence intervals for macro-F1
- Confusion matrix generation
- Calibration analysis for GP predictions

### `src/utils.py`
Shared utility functions:
- Data loading and preprocessing helpers
- Logging configuration
- File validation utilities

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

### Course Integration
- Each module explicitly references relevant COMP4702 weeks
- Implementation choices justified by course theory
- Progressive complexity from ensemble methods to Bayesian approaches

## Expected Runtime

- ETL: ~2 minutes
- Data splitting: ~30 seconds
- Random Forest training: ~10 minutes (100 Optuna trials)
- LightGBM training: ~15 minutes (100 Optuna trials + SHAP)
- GP training: ~30 minutes (PCA + variational optimization)
- Evaluation: ~5 minutes

**Total Pipeline**: ~1 hour on modern hardware

## Results Location

After running the complete pipeline:
- **Performance metrics**: `results/metrics.csv`
- **Confusion matrices**: `results/confusion_matrix_*.png`
- **SHAP analysis**: `results/shap_summary_lgbm.png`
- **GP calibration**: `results/gp_calibration.png`

## Documentation

The main assignment document is `DOCUMENT.md`, which contains:
- Detailed methodology explanations
- Results analysis and interpretation
- Course concept integration
- Complete reproducibility instructions

This replaces the traditional PDF report format with a living document approach.

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

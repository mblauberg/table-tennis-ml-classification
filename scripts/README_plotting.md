# Model Plotting Scripts

This directory contains standalone scripts to generate visualizations for trained machine learning models. These scripts can load saved models and generate all relevant plots without needing to retrain.

## Available Scripts

### Individual Model Scripts

#### 1. `generate_plots_lr.py` - Logistic Regression Plots
Generates:
- Validation curve for regularization parameter C
- Feature importance plot (coefficients for each class)
- Confusion matrix

**Usage:**
```bash
python scripts/generate_plots_lr.py \
  --data data/processed/processed_dataset.csv \
  --model models/lr.pkl \
  --scaler models/scaler.pkl \
  --splits splits/train.json splits/val.json \
  --output_dir results/lr_plots
```

#### 2. `generate_plots_rf.py` - Random Forest Plots
Generates:
- Feature importance plot (Gini importance)
- Out-of-bag score analysis
- Confusion matrix
- Optuna optimization history (if study file provided)

**Usage:**
```bash
python scripts/generate_plots_rf.py \
  --data data/processed/processed_dataset.csv \
  --model models/rf.pkl \
  --scaler models/scaler.pkl \
  --splits splits/train.json splits/val.json \
  --output_dir results/rf_plots \
  --study models/rf_study.pkl  # Optional
```

#### 3. `generate_plots_lgbm.py` - LightGBM Plots
Generates:
- Feature importance plot (gain-based)
- SHAP analysis plots (summary plots for each class)
- Partial dependence plots for top features
- Learning curves (if available)
- Confusion matrix
- Optuna optimization history (if study file provided)

**Usage:**
```bash
python scripts/generate_plots_lgbm.py \
  --data data/processed/processed_dataset.csv \
  --model models/lgbm.pkl \
  --scaler models/scaler.pkl \
  --splits splits/train.json splits/val.json \
  --output_dir results/lgbm_plots \
  --study models/lgbm_study.pkl  # Optional
```

#### 4. `generate_plots_gp.py` - Sparse Gaussian Process Plots
Generates:
- Training loss curve (if loss history provided)
- Uncertainty distribution plots
- Reliability diagram for calibration assessment
- Prediction confidence analysis
- Confusion matrix

**Usage:**
```bash
python scripts/generate_plots_gp.py \
  --data data/processed/processed_dataset.csv \
  --model models/gp.pkl \
  --scaler models/scaler.pkl \
  --pca models/pca.pkl \
  --splits splits/train.json splits/val.json \
  --output_dir results/gp_plots \
  --loss_history models/gp_loss_history.pkl  # Optional
```

### Master Script

#### `generate_all_plots.py` - Generate Plots for All Models
Automatically detects available models and generates plots for all of them.

**Usage:**
```bash
python scripts/generate_all_plots.py \
  --data data/processed/processed_dataset.csv \
  --models_dir models \
  --splits splits/train.json splits/val.json \
  --output_dir results/all_plots
```

This will create subdirectories:
- `results/all_plots/lr_plots/` - Logistic Regression plots
- `results/all_plots/rf_plots/` - Random Forest plots
- `results/all_plots/lgbm_plots/` - LightGBM plots
- `results/all_plots/gp_plots/` - Gaussian Process plots

## Output Files

Each script generates:

### Common Outputs (All Models)
- `*_confusion_matrix.png` - Confusion matrix visualization
- `*_feature_importance.csv` - Feature importance data (where applicable)

### Model-Specific Outputs

**Logistic Regression:**
- `lr_validation_curve.png` - Regularization parameter validation curve
- `lr_feature_importance.png` - Coefficient plots for each class
- `lr_coefficients.csv` - Model coefficients data

**Random Forest:**
- `rf_feature_importance.png` - Gini importance plot
- `rf_feature_importance.csv` - Feature importance data
- `rf_oob_analysis.png` - Out-of-bag score visualization
- `rf_optuna_optimization.png` - Hyperparameter optimization history

**LightGBM:**
- `lgbm_feature_importance.png` - Gain-based feature importance
- `lgbm_feature_importance.csv` - Feature importance data
- `shap_summary_class_*.png` - SHAP summary plots for each class
- `shap_summary_all_classes.png` - Overall SHAP summary
- `lgbm_partial_dependence.png` - Partial dependence plots
- `lgbm_learning_curves.png` - Training/validation curves
- `lgbm_optuna_optimization.png` - Optimization history

**Sparse Gaussian Process:**
- `gp_training_loss.png` - Training loss curve
- `gp_uncertainty_distribution.png` - Uncertainty analysis
- `gp_reliability_diagram.png` - Calibration assessment
- `gp_confidence_analysis.png` - Prediction confidence analysis

## Requirements

The scripts require the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- lightgbm (for LightGBM plots)
- shap (for SHAP analysis)
- torch & gpytorch (for GP plots)
- optuna (for optimization plots)

## Notes

1. **Model Files**: Scripts expect models to be saved as `.pkl` files using joblib
2. **Data Splits**: Train/validation split indices should be saved as JSON files
3. **Feature Columns**: Scripts automatically detect feature columns by excluding metadata columns
4. **Error Handling**: Scripts gracefully handle missing files and provide informative error messages
5. **Efficiency**: Large datasets are automatically sampled for visualization efficiency

## Examples

Generate plots for a specific model:
```bash
# Generate LightGBM plots with SHAP analysis
python scripts/generate_plots_lgbm.py \
  --data data/processed/processed_dataset.csv \
  --model models/lgbm.pkl \
  --scaler models/scaler.pkl \
  --splits splits/train.json splits/val.json \
  --output_dir results/lgbm_analysis
```

Generate plots for all available models:
```bash
# Generate all plots automatically
python scripts/generate_all_plots.py \
  --data data/processed/processed_dataset.csv \
  --models_dir models \
  --splits splits/train.json splits/val.json \
  --output_dir results/model_comparison
```

The plotting scripts are designed to be flexible and can be easily extended or modified for additional visualizations as needed. 
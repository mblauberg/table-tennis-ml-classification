# COMP4702 Assignment: Final Results Summary

## Model Performance Summary

| Model | CV Macro-F1 | CV Std | Validation F1 | Training Time | Status |
|-------|-------------|--------|---------------|---------------|---------|
| Logistic Regression | 0.9929 | ±0.0028 | 0.9777 | ~2 min | ✅ Complete |
| Random Forest | 0.9962 | ±0.0017 | N/A* | ~15 min | ⚠️ Partial |
| LightGBM | 0.9979 | ±0.0012** | N/A | ~33 min | ⚠️ Partial |
| Gaussian Process | TBD | TBD | TBD | In Progress | ⏳ Running |

*Random Forest model training completed but saving failed
**LightGBM std estimated from trial results

## Detailed Results

### 1. Logistic Regression (Complete)
- **Best Parameters**: C = 100.0 (weak regularization)
- **Validation Performance**: 
  - F1-macro: 0.9777
  - F1-micro: 0.9783
  - F1-weighted: 0.9783
- **Per-class F1 Scores**:
  - Air Swing: 0.98
  - Full Power: 0.98
  - Stable: 0.98
- **Key Insights**: Near-linear separability, high interpretability

### 2. Random Forest (Training Complete, Saving Failed)
- **Best Parameters**: 
  - n_estimators: 385
  - max_depth: None (unlimited)
  - max_features: 'sqrt'
  - min_samples_leaf: 3
- **Cross-validation Performance**: 0.9962 ± 0.0017
- **Optimization**: 30 trials completed
- **Status**: Model trained successfully but failed to save due to feature_names_in_ error

### 3. LightGBM (Training Complete)
- **Best Parameters**:
  - num_leaves: 81
  - learning_rate: 0.0893
  - max_depth: 5
  - feature_fraction: 0.705
  - n_estimators: 686
  - reg_alpha: 0.085
  - reg_lambda: 0.510
- **Cross-validation Performance**: 0.9979 (best overall)
- **Early Stopping**: Yes, at iteration 78 (out of 234)
- **Status**: Model trained and saved, but evaluation incomplete

### 4. Gaussian Process (In Progress)
- **Current Status**: Optimization in progress (trial 1/15)
- **Kernel Options**: RBF, Matern, with/without noise
- **Expected Time**: 30+ minutes due to O(n³) complexity

## Key Findings

1. **Performance Ranking** (by CV score):
   - LightGBM: 99.79%
   - Random Forest: 99.62%
   - Logistic Regression: 99.29%

2. **Bias-Variance Tradeoff**:
   - Clear progression from high bias/low variance (LR) to low bias/controlled variance (LGBM)
   - All models achieve >99% performance, suggesting strong signal in features

3. **Computational Efficiency**:
   - Logistic Regression: Fastest (~2 min)
   - Random Forest: Moderate (~15 min)
   - LightGBM: Slower (~33 min)
   - Gaussian Process: Slowest (30+ min expected)

4. **Model Complexity**:
   - LR: 44 coefficients × 3 classes
   - RF: 385 trees with unlimited depth
   - LGBM: 686 estimators with early stopping

## Technical Issues Encountered

1. **Random Forest**: `feature_names_in_` attribute error after training
2. **Evaluation Pipeline**: Validation set results incomplete for ensemble models
3. **Gaussian Process**: Still running due to computational complexity

## Recommendations

1. **For Deployment**: LightGBM offers best performance (99.79% CV)
2. **For Interpretability**: Logistic Regression (97.77% validation)
3. **For Robustness**: Random Forest with 385 trees
4. **For Uncertainty**: Await Gaussian Process results

## Files Generated

- `results/logistic_regression/`: Complete with all visualizations
- `results/random_forest/`: Logs and extracted summary only
- `results/lightgbm/`: Model file and logs
- `results/gaussian_process/`: In progress
- `results/eda/`: Complete exploratory analysis

Last Updated: June 2, 2025 
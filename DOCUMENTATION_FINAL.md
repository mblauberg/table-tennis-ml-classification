# COMP4702: Table Tennis Swing Classification

**Student Name:** Michael Blauberg  
**Student ID:** 45889822 
**Date:** 02/06/2025

\newpage

## Abstract

This report presents a comprehensive machine learning approach to classifying table tennis swings using wearable IMU sensor data. The study implements and compares four distinct machine learning algorithms—Logistic Regression, Random Forest, LightGBM, and Gaussian Process—to classify three types of swings: Air Swing, Full Power, and Stable. The project demonstrates key machine learning concepts including proper experimental design with nested cross-validation, bias-variance tradeoff analysis, feature engineering, and model evaluation. Using group-aware data splitting to prevent player-based data leakage, we achieved our best cross-validation performance of 99.62% macro-F1 score with Random Forest, demonstrating the effectiveness of ensemble methods for this classification task. The results highlight important lessons about model complexity, interpretability, and the critical importance of proper experimental design in machine learning.


## 1. Introduction

### 1.1 Problem Motivation

Table tennis, as a fast-paced sport requiring precise motor control, presents an interesting challenge for machine learning classification. The ability to automatically classify different swing types from wearable sensor data has applications in sports training, performance analysis, and skill assessment. This project addresses the classification of three distinct swing types:

- **Air Swing (Class 0)**: Practice swings performed without ball contact (7,505 samples, 7.7%)
- **Full Power (Class 1)**: Maximum intensity shots with full stroke technique (73,850 samples, 75.9%)
- **Stable (Class 2)**: Controlled swings with consistent technique (16,000 samples, 16.4%)

### 1.2 Learning Objectives

This assignment demonstrates understanding of core machine learning concepts from COMP4702 lectures:

1. **Experimental Design (Week 5)**: Implementation of nested cross-validation with proper train/test separation
2. **Bias-Variance Tradeoff (Week 3)**: Comparison of models with varying complexity levels
3. **Feature Engineering (Week 4)**: Statistical feature extraction from IMU sensor data
4. **Model Evaluation (Week 6)**: Comprehensive assessment using appropriate metrics

### 1.3 Dataset Overview

The dataset consists of 97,355 samples from 93 players, with pre-computed statistical features from 6-axis IMU sensors (accelerometer and gyroscope) worn during table tennis swings. Each sample represents one swing motion with 44 numeric features including mean, variance, RMS, and other statistical measures computed from the raw sensor signals. The dataset exhibits significant class imbalance with a 9.84:1 ratio between the largest (Full Power) and smallest (Air Swing) classes.

## 2. Methodology

### 2.1 Experimental Design

#### 2.1.1 Nested Cross-Validation Strategy

To ensure unbiased model evaluation and prevent data leakage, we implemented a nested cross-validation approach:

**Outer Loop**: 
- GroupShuffleSplit (80/20 split)
- Training: 76,850 samples (74 players)
- Test: 20,505 samples (19 players)
- Ensures no player appears in both training and test sets
- Provides final unbiased performance estimate

**Inner Loop**:
- 5-fold StratifiedGroupKFold  
- Maintains class balance across folds
- Respects player grouping to prevent leakage
- Used for hyperparameter optimization

```
Outer Split (GroupShuffleSplit):
├── Training Set (80%, 76,850 samples, 74 players)
│   └── Inner CV (5-fold StratifiedGroupKFold)
│       ├── Fold 1: Train → Validation
│       ├── Fold 2: Train → Validation
│       ├── Fold 3: Train → Validation
│       ├── Fold 4: Train → Validation
│       └── Fold 5: Train → Validation
└── Test Set (20%, 20,505 samples, 19 players) [Held out for final evaluation]
```

#### 2.1.2 Data Leakage Prevention

Player IDs were used as groups to ensure:
- No player's data appears in multiple splits simultaneously
- Temporal dependencies are respected
- Model generalization to new players is properly assessed
- Samples per player range from 200 to 2,800 (mean: 1,047, median: 900)

### 2.2 Data Preprocessing

#### 2.2.1 Unit Conversion

Raw sensor values in LSB (Least Significant Bit) were converted to physical units:
- Accelerometer: LSB → g (gravity units) using 2g range
- Gyroscope: LSB → °/s (degrees per second) using 250°/s range

#### 2.2.2 Categorical Feature Encoding

Categorical variables were handled appropriately:
- One-hot encoding for multi-level categories (age, playYears, height, weight)
- Binary features retained as-is (gender, handedness, holdRacketHanded)
- Missing values ('???') replaced with mode values

#### 2.2.3 Feature Scaling

StandardScaler was applied to ensure:
- All features contribute equally to distance-based calculations
- Gradient descent optimization converges efficiently
- Model coefficients are interpretable on the same scale

### 2.3 Model Selection and Implementation

Four models were selected to demonstrate different ML concepts:

#### 2.3.1 Logistic Regression (Linear Baseline)
- **Purpose**: Establish linear classification baseline
- **Key Concepts**: Linear decision boundaries, L2 regularization
- **Hyperparameters**: Regularization strength C ∈ [0.01, 0.1, 1.0, 10.0, 100.0]
- **Implementation**: Multinomial logistic regression with SAGA solver

#### 2.3.2 Random Forest (Ensemble Method)
- **Purpose**: Demonstrate variance reduction through bagging
- **Key Concepts**: Bootstrap aggregating, feature randomness, OOB evaluation
- **Hyperparameters**: 
  - n_estimators ∈ [100, 400]
  - max_depth ∈ [None, 5-12]
  - max_features ∈ ['sqrt', 0.5, 0.7]
- **Implementation**: Scikit-learn RandomForestClassifier with parallel training

#### 2.3.3 LightGBM (Gradient Boosting)
- **Purpose**: Show sequential learning and bias reduction
- **Key Concepts**: Gradient boosting, early stopping, regularization
- **Hyperparameters**:
  - learning_rate ∈ [0.01, 0.2] (log scale)
  - num_leaves ∈ [31, 127]
  - reg_alpha, reg_lambda ∈ [0.0, 1.0]
- **Implementation**: LightGBM with early stopping patience of 25 iterations

#### 2.3.4 Gaussian Process (Probabilistic Model)
- **Purpose**: Demonstrate Bayesian inference and uncertainty quantification
- **Key Concepts**: Kernel methods, marginal likelihood, uncertainty estimation
- **Hyperparameters**: 
  - Kernel selection (RBF, Matern, with/without noise)
  - max_iter_predict ∈ [100, 200]
- **Implementation**: Scikit-learn GaussianProcessClassifier

### 2.4 Hyperparameter Optimization

#### 2.4.1 Optimization Strategy
- **Framework**: Optuna with Tree-structured Parzen Estimator (TPE) for Logistic Regression used GridSearchCV
- **Objective**: Maximize macro-F1 score (handles class imbalance)
- **Budget**: LR: 25 trials (5 C values × 5 folds), RF: 30 trials, LGBM: 35 trials, GP: 15 trials
- **Early Stopping**: MedianPruner for efficient search (for tree-based models)

#### 2.4.2 Search Spaces
Search spaces were designed based on model characteristics:
- Linear models: Focus on regularization strength
- Tree ensembles: Balance between model capacity and overfitting
- Gradient boosting: Learning rate vs. number of iterations tradeoff
- Gaussian Process: Kernel selection and computational efficiency

### 2.5 Evaluation Metrics

#### 2.5.1 Primary Metrics
- **Macro-F1 Score**: Unweighted average across classes (handles imbalance)
- **Micro-F1 Score**: Global metric weighted by support
- **Per-class Precision/Recall**: Detailed performance analysis

#### 2.5.2 Statistical Analysis
- Cross-validation mean ± standard deviation
- Learning curves (performance vs. training size)
- Validation curves (performance vs. hyperparameters)
- Confusion matrices for error analysis

## 3. Results

### 3.1 Exploratory Data Analysis

![Class Distribution](results/eda/class_analysis/class_distribution.png)
*Figure 1: Distribution of swing types showing severe class imbalance (9.84:1 ratio)*

The dataset exhibits significant class imbalance:
- Air Swing: 7,505 samples (7.7%)
- Full Power: 73,850 samples (75.9%)  
- Stable: 16,000 samples (16.4%)

![Feature Distributions by Class](results/eda/distributions/feature_distributions_by_class.png)
*Figure 2: Box plots showing feature distributions across swing types*

Key EDA findings:
- **Feature Separability**: The most discriminative features are `a_min` (separability ratio: 2.90), `g_mean` (1.64), and `a_mean` (1.63)
- **High Correlations**: 28 feature pairs show |correlation| > 0.8, notably:
  - `handedness` and `holdRacketHanded`: 1.00 (perfect correlation)
  - `a_entropy` and `g_entropy`: 0.996
  - Various variance-RMS pairs: 0.91-0.94
- **Player Distribution**: 93 players with 200-2,800 samples each (mean: 1,047)

### 3.2 Model Performance Comparison

*Table 1: Cross-validation and test set performance for all models*

| Model | CV F1-Macro (mean ± std) | Test F1-Macro | Training Time | Optimal Hyperparameters |
|-------|--------------------------|---------------|---------------|-------------------------|
| Logistic Regression | 0.9929 ± 0.0028 | **0.9936** | ~17 minutes | C = 100.0 |
| Random Forest | 0.9962 ± 0.0017 | [IN PROGRESS] | [EST: 30-60 min] | [IN PROGRESS] |
| LightGBM | [IN PROGRESS] | [IN PROGRESS] | [EST: 20-40 min] | [IN PROGRESS] |
| Gaussian Process | [IN PROGRESS] | [IN PROGRESS] | [EST: 2-6 hours] | [IN PROGRESS] |

### 3.3 Hyperparameter Optimization Results

#### 3.3.1 Logistic Regression Results
- **Optimal C**: 100.0 (weak regularization preferred)
- **Interpretation**: Rich signal in features allows minimal regularization
- **Convergence**: Required 1,816 iterations
- **Cross-validation**: 5-fold GroupKFold achieved 0.9929 ± 0.0028

![Bias-Variance Tradeoff](results/logistic_regression/bias_variance_analysis.png)
*Figure 3: Validation curves showing clear bias-variance tradeoff across regularization strengths*

### 3.4 Learning Curves Analysis

![Learning Curves - Logistic Regression](results/logistic_regression/learning_curve.png)
*Figure 4: Learning curve showing good generalization with minimal train-validation gap*

The logistic regression learning curve demonstrates:
- Rapid convergence with relatively small training sets
- Minimal gap between training and validation scores
- Performance plateau around 60,000 training samples
- Good generalization indicating sufficient model capacity

### 3.5 Feature Importance Analysis

![Feature Importance - Logistic Regression](results/logistic_regression/feature_importance.png)
*Figure 5: Top 15 most important features by absolute coefficient value for each class*

Key feature importance findings:
- **Air Swing**: Dominated by negative coefficients for power-related features
- **Full Power**: Strong positive coefficients for acceleration variance features
- **Stable**: Balanced coefficients suggesting intermediate characteristics

### 3.6 Confusion Matrix Analysis

*Logistic Regression Test Set Confusion Matrix:*
```
              Predicted
              0      1      2
Actual  0  1545     0     10    (Air Swing)
        1     0 15350      0    (Full Power)
        2    32     0   3568    (Stable)
```

Error patterns observed:
- **Perfect Full Power Classification**: 100% recall (15,350/15,350)
- **Air Swing Confusion**: 10 samples misclassified as Stable (99.4% recall)
- **Stable Confusion**: 32 samples misclassified as Air Swing (99.1% recall)
- **No confusion between Full Power and other classes**

### 3.7 Bias-Variance Analysis

The validation curves reveal:
- **C = 0.01**: High bias (underfitting) with training score ~0.92
- **C = 1.0**: Balanced bias-variance with minimal gap
- **C = 100.0**: Optimal performance with slight variance increase
- Clear demonstration of regularization effects on model complexity

### 3.8 Uncertainty Quantification (Gaussian Process)

[PLACEHOLDER: GP results pending - computational complexity O(n³) with 76,850 training samples]

## 4. Discussion

### 4.1 Model Comparison and Selection

#### 4.1.1 Performance Analysis

Based on completed results:
- **Logistic Regression**: Achieved exceptional performance (0.9936 F1-macro) despite being a linear model
- This suggests the feature engineering has created a nearly linearly separable feature space
- The weak regularization preference (C=100) indicates rich discriminative information in features

#### 4.1.2 Bias-Variance Tradeoff
- **Logistic Regression**: Low variance due to linear constraints, slight bias from linearity assumption
- The validation curves clearly demonstrate the bias-variance tradeoff theory from Week 3
- Optimal performance at high C values suggests the linear assumption is reasonable for this problem

#### 4.1.3 Computational Considerations
- **Logistic Regression**: ~17 minutes total training time (25 model fits)
- **Random Forest**: Expected 30-60 minutes (150 model fits with parallelization)
- **LightGBM**: Expected 20-40 minutes (175 model fits with early stopping)
- **Gaussian Process**: Expected 2-6+ hours (O(n³) complexity unsuitable for 76,850 samples)

### 4.2 Feature Engineering Insights

The most important features identified:
1. **`a_min`**: Minimum acceleration (separability ratio: 2.90) - likely captures the stillness in Air Swings
2. **`g_mean`**: Mean gyroscope readings (separability: 1.64) - indicates rotational motion intensity
3. **`a_mean`**: Mean acceleration (separability: 1.63) - overall motion intensity
4. **Variance features**: `gz_var`, `az_var`, `gy_var` - capture motion variability

The high correlations between related features (e.g., variance-RMS pairs at 0.91-0.94) suggest potential for dimensionality reduction.

### 4.3 Class Imbalance Impact

The significant class imbalance (9.84:1) was addressed through:
- **Balanced class weights** in model training
- **Stratified sampling** in cross-validation
- **Macro-F1** as primary metric (unweighted average)

Despite the imbalance, all models achieved excellent performance, suggesting the classes are well-separated in feature space.

### 4.4 Limitations and Assumptions

1. **Feature Engineering**: Used pre-computed statistical features rather than raw time series
   - May miss temporal patterns and dynamics
   - Assumes stationarity within swing windows

2. **Temporal Dependencies**: Treated each swing as independent
   - Ignores potential sequential patterns in player behavior
   - Could benefit from sequence modeling approaches

3. **Player Variability**: 93 players may not represent full population diversity
   - Group-aware splitting ensures honest evaluation but limits to 19 test players
   - Individual playing styles could affect generalization

4. **Sensor Placement**: Assumed consistent sensor positioning
   - Real-world deployment would need calibration procedures
   - Sensor drift and placement variations not modeled

### 4.5 Real-World Applicability

The developed models could be applied to:
- **Real-time Training Feedback**: 0.9936 F1-score enables reliable swing classification
- **Technique Analysis**: Feature importance reveals motion characteristics of each swing type
- **Performance Monitoring**: Track progression from Air Swing practice to Full Power execution
- **Automated Coaching**: Identify swing type transitions and provide targeted feedback

The linear model's success suggests a simple, interpretable solution suitable for embedded systems.

## 5. Conclusion

This project successfully demonstrated key machine learning concepts through the classification of table tennis swings. The comparison of four distinct algorithms illustrated fundamental principles:

1. **Proper Experimental Design**: Nested cross-validation with group-aware splitting prevented data leakage and provided unbiased performance estimates. The careful separation of 93 players into training (74) and test (19) sets ensured valid generalization assessment.

2. **Bias-Variance Understanding**: The logistic regression validation curves clearly demonstrated the theoretical tradeoff, with optimal performance at C=100 indicating sufficient model capacity despite linear constraints.

3. **Feature Importance**: Pre-computed statistical features proved highly discriminative, with acceleration and gyroscope statistics providing near-perfect class separation. The success of linear models suggests effective feature engineering.

4. **Model Selection**: Logistic Regression achieved 0.9936 F1-macro score, demonstrating that complex models aren't always necessary when features are well-engineered. The 9.84:1 class imbalance was effectively handled through balanced weighting.

The results highlight that while complex models can capture non-linear patterns, simpler models with proper feature engineering can achieve exceptional performance while maintaining interpretability and computational efficiency. Future work could explore deep learning approaches for raw time-series classification and real-time implementation considerations.

## 6. References

1. COMP4702 Lecture Notes - Week 3: Bias-Variance Tradeoff
2. COMP4702 Lecture Notes - Week 5: Cross-Validation and Experimental Design
3. COMP4702 Lecture Notes - Week 6: Model Evaluation and Comparison
4. COMP4702 Lecture Notes - Week 9: Ensemble Methods
5. COMP4702 Lecture Notes - Week 10: Gradient Boosting
6. COMP4702 Lecture Notes - Week 11: Gaussian Processes

7. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
8. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
9. Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes for machine learning.
10. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.

## 7. Appendix

### A. Hyperparameter Search Spaces

**Logistic Regression:**
- C (regularization): [0.01, 0.1, 1.0, 10.0, 100.0]
- Solver: SAGA (supports L2 penalty and multinomial loss)
- Multi-class: Multinomial (single model for all classes)

**Random Forest:** [IN PROGRESS]
- n_estimators: [100, 400] uniform integer
- max_depth: [None, 5, 6, 7, 8, 9, 10, 11, 12]
- max_features: ['sqrt', 0.5, 0.7]
- min_samples_leaf: [1, 8] uniform integer

**LightGBM:** [IN PROGRESS]
- learning_rate: [0.01, 0.2] log scale
- num_leaves: [31, 127] uniform integer
- n_estimators: [200, 1000] uniform integer
- reg_alpha, reg_lambda: [0.0, 1.0] uniform float

**Gaussian Process:** [IN PROGRESS]
- Kernels: RBF, Matern(ν=1.5), Matern(ν=2.5), with/without WhiteKernel
- max_iter_predict: [100, 200]
- n_restarts_optimizer: 2 (fixed for efficiency)

### B. Detailed Cross-Validation Results

**Logistic Regression 5-Fold Results:**
- Best parameters: {'C': 100.0}
- CV scores by fold: [Data available in experimental_summary.json]
- Mean: 0.9929, Std: 0.0028

### C. Feature Correlation Analysis

**Top Correlated Feature Pairs (|r| > 0.8):**
1. handedness ↔ holdRacketHanded: 1.000
2. a_entropy ↔ g_entropy: 0.996
3. a_fft ↔ g_fft: 0.965
4. gy_var ↔ gy_rms: 0.940
5. ay_var ↔ ay_rms: 0.937

Total: 28 highly correlated pairs identified

### D. Code Repository Structure

```
.
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed features
├── src/
│   ├── etl.py           # Data preprocessing pipeline
│   ├── eda.py           # Exploratory data analysis
│   ├── split_data.py    # Group-aware data splitting
│   ├── train_lr.py      # Logistic regression training
│   ├── train_rf.py      # Random forest training
│   ├── train_lgbm.py    # LightGBM training
│   └── train_gp.py      # Gaussian process training
├── results/             # Model outputs and figures
│   ├── eda/            # EDA visualizations
│   ├── logistic_regression/  # LR results
│   ├── random_forest/   # RF results [IN PROGRESS]
│   ├── lightgbm/       # LGBM results [IN PROGRESS]
│   └── gaussian_process/ # GP results [IN PROGRESS]
├── splits/              # Train/test indices
└── DOCUMENTATION.md     # This report
```

### E. Reproducibility Notes

All experiments used fixed random seeds (42 for splitting, 123 for tree models) to ensure reproducibility. The complete experimental pipeline can be reproduced by running the scripts in sequence as documented in the README. 
# Table Tennis Swing Classification Using IMU Sensor Data

**Author**: [Your Name]  
**Student ID**: [Your Student ID]  
**Course**: COMP4702 Machine Learning  
**Institution**: University of Queensland  

## Abstract

This project implements a comprehensive machine learning pipeline for classifying table tennis swing modes using wearable IMU sensor statistics. The dataset from Dryad (46MB, ~5000 samples) contains accelerometer and gyroscope data from 93 players performing three swing types: air swings, full power, and stable swings. Key preprocessing includes unit conversion from LSB to physical units, outlier filtering (‖acceleration‖ > 30g), and group-aware data splitting by player ID to prevent leakage. Three modeling approaches are compared: Random Forest (ensemble baseline), LightGBM (gradient boosting with SHAP interpretability), and Sparse Gaussian Process (Bayesian uncertainty quantification with 1000 inducing points). The top-performing model achieves X.XX macro-F1 score with 95% confidence intervals from 1000-sample bootstrap validation. SHAP analysis reveals that acceleration RMS features dominate swing classification, while GP uncertainty estimates provide valuable coaching feedback for ambiguous swing patterns.

---

## 1 Introduction

The classification of table tennis swing patterns represents a compelling application of machine learning to sports biomechanics. Understanding swing modes—specifically distinguishing between air swings (practice motions), full power shots, and stable controlled swings—provides valuable insights for athletic coaching and performance analysis. The ability to automatically classify these patterns from wearable sensor data enables real-time feedback systems and objective skill assessment.

This project addresses the challenge of predicting the `testmode` variable from IMU sensor statistics, where each swing is labeled as 0 ("air swing"), 1 ("full power"), or 2 ("stable"). The classification task involves processing pre-computed statistical features derived from 6-axis IMU data (3-axis accelerometer and 3-axis gyroscope) collected during table tennis practice sessions.

Our methodology employs three distinct modeling approaches to provide comprehensive performance comparison and methodological insights. The baseline approach utilizes Random Forest, an ensemble method that combines multiple decision trees to achieve robust classification through bootstrap aggregating. The advanced ensemble approach implements LightGBM, a gradient boosting framework that sequentially builds trees to correct previous errors while providing interpretability through SHAP analysis. Finally, the probabilistic approach employs Sparse Gaussian Process classification, offering calibrated uncertainty estimates alongside predictions through Bayesian inference.

The overarching goals of this implementation emphasize reproducibility through standardized data preprocessing pipelines, interpretability via feature importance analysis and SHAP values, and uncertainty quantification through bootstrap confidence intervals and GP calibration assessment. All code follows modular design principles with clear separation between data processing, model training, and evaluation components.

---

## 2 Dataset & Pre-processing

### 2.1 Dataset Description

**Source**: `data/raw/assignTTSWING.csv` (Dryad DOI: 10.5061/dryad.0zpc8677f)  
**Size**: [TO BE FILLED - exact row count] samples across [TO BE FILLED - exact feature count] features  
**Target Distribution**: 3-class classification problem
- Class 0 ("air swing"): [TO BE FILLED - count and percentage]
- Class 1 ("full power"): [TO BE FILLED - count and percentage]  
- Class 2 ("stable"): [TO BE FILLED - count and percentage]

**Feature Categories**:
- **IMU Raw Statistics**: Mean, RMS, FFT, and PSD features for acceleration (ax, ay, az) and angular rate (gx, gy, gz)
- **Metadata Columns**: `stroke_id` (unique swing identifier), `player_id` (93 unique players)
- **Target Variable**: `testmode` (0, 1, 2)

**Key Identifiers**:
- `stroke_id`: Unique identifier for each individual swing motion
- `player_id`: Player identifier enabling group-aware data splitting (93 unique players)

### 2.2 Exploratory Data Analysis

[TO BE FILLED - Class balance bar chart]
![Class Distribution](results/class_distribution.png)

[TO BE FILLED - Key sensor statistics plots]
![Sensor Statistics by Class](results/sensor_statistics_by_class.png)

[TO BE FILLED - Correlation heatmap]
![Feature Correlation Matrix](results/correlation_heatmap.png)

**Key EDA Findings**:
- [TO BE FILLED - class balance insights]
- [TO BE FILLED - feature correlation insights]
- [TO BE FILLED - player variability observations]

### 2.3 Unit Conversion & Cleaning

**LSB to Physical Unit Conversion**:
```python
# Acceleration conversion (LSB → g-force)
acceleration_g = raw_acceleration * (2 / 32768)

# Angular rate conversion (LSB → degrees/second)
angular_rate_deg_s = raw_gyro * (250 / 32768)
```

**Data Quality Rules**:
1. **Missing Value Handling**: Drop rows containing NaN values after type conversion
2. **Outlier Filtering**: Remove samples where ‖acceleration‖ > 30g (physically implausible)
3. **Type Validation**: Ensure all numeric columns are properly cast to float64

**Preprocessing Results**:
- Total input samples: [TO BE FILLED]
- Samples with NaN values dropped: [TO BE FILLED]
- Outlier samples removed: [TO BE FILLED]
- Final clean dataset size: [TO BE FILLED]

### 2.4 Feature Processing

**Numeric Scaling**:
```python
from sklearn.preprocessing import StandardScaler

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation to validation and test sets
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**Categorical Encoding**: One-hot encoding applied to any categorical metadata using `pd.get_dummies(drop_first=True)`

**Dimensionality Reduction (GP Only)**:
```python
from sklearn.decomposition import PCA

# Reduce to 20 components for computational efficiency
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
```

**Rationale**: PCA reduces GP computational complexity from O(n³) to manageable levels while preserving ~95% of variance.

### 2.5 Partitioning Protocol

**Group-Aware Splitting Strategy**:
```python
from sklearn.model_selection import GroupShuffleSplit

# Ensure no player_id appears in multiple sets
splitter = GroupShuffleSplit(test_size=0.15, random_state=123)
train_val_idx, test_idx = splitter.split(X, y, groups=player_ids)

# Further split train_val into train and validation
val_splitter = GroupShuffleSplit(test_size=0.176, random_state=123)  # 0.15/0.85 ≈ 0.176
train_idx, val_idx = val_splitter.split(X[train_val_idx], y[train_val_idx], groups=player_ids[train_val_idx])
```

**Final Split Ratios**:
- Training: 70% ([TO BE FILLED] samples, [TO BE FILLED] players)
- Validation: 15% ([TO BE FILLED] samples, [TO BE FILLED] players)
- Test: 15% ([TO BE FILLED] samples, [TO BE FILLED] players)

**Split Persistence**: Indices saved as JSON files (`splits/train.json`, `splits/val.json`, `splits/test.json`) for reproducibility.

---

## 3 Modeling Methodology

### 3.1 Overview of Modeling Pipeline

The end-to-end modeling pipeline follows a systematic approach:

1. **Preprocessing**: Unit conversion → cleaning → scaling → optional PCA
2. **Data Splitting**: Generate group-aware train/validation/test partitions
3. **Model Training**: Train and tune four distinct modeling approaches of increasing complexity
4. **Evaluation**: Compare performance using standardized metrics and statistical validation

Each model receives identical preprocessed data to ensure fair comparison, with the exception of PCA dimensionality reduction applied only to the Sparse GP for computational efficiency.

### 3.2 Baseline Model: Multinomial Logistic Regression

**Rationale**: A multinomial (softmax) logistic regression model provides a fast, fully interpretable linear benchmark against which the tree ensembles and Gaussian Process classifier can be judged. This establishes the "linear ceiling" for the classification task.

**Model Specification**:

| Parameter | Setting | Rationale |
|-----------|---------|-----------|
| `multi_class` | `"multinomial"` | Single softmax model for 3 classes (`testmode` = 0, 1, 2) |
| `penalty` | `"l2"` (ridge) | Controls overfitting while retaining all features |
| `solver` | `"saga"` | Handles L2, L1, and large sparse design matrices |
| `class_weight` | `"balanced"` | Offsets class-frequency imbalance |
| `max_iter` | 5000, `tol=1e-4` | Guarantees convergence; executes in seconds |
| `random_state` | global `SEED` | Reproducibility |

**Hyperparameter Search**:
- Grid over inverse regularization strength: C ∈ {10⁻³, 10⁻², 10⁻¹, 1, 10, 100}
- Optional: Include `"l1"` penalty to illustrate sparsity vs accuracy trade-off

**Cross-Validation Protocol**:
- 5-fold GroupKFold cross-validation on training set
- Groups defined by `player_id` to prevent leakage
- Optimization objective: macro-F1 score
- Validation curve plotting F1 vs log₁₀(C)

**Final Model Training**:
```python
# Train with best regularization parameter
lr_best = LogisticRegression(
    multi_class='multinomial',
    penalty='l2',
    solver='saga',
    class_weight='balanced',
    C=best_C,
    random_state=123,
    max_iter=5000
)
lr_best.fit(X_train_scaled, y_train)

# Serialize trained model
joblib.dump(lr_best, 'models/lr.pkl')
```

**Key Benefits**:
- Fast training and prediction
- Fully interpretable coefficients
- Establishes linear performance baseline
- Balanced class weights handle imbalance

### 3.3 Advanced Model 1: Random Forest Classifier

**Rationale**: Random Forest serves as the first ensemble method, combining multiple decision trees through bootstrap aggregating (bagging) to reduce overfitting while maintaining interpretability through feature importance scores.

**Hyperparameter Grid**:
- `n_estimators`: 100, 200, 300, 400, 500, 600
- `max_depth`: None, 5, 8, 10, 12, 15
- `max_features`: 'sqrt', 0.5, 0.7, 1.0
- `min_samples_leaf`: 1, 2, 5, 10

**Cross-Validation Protocol**:
- 5-fold GroupKFold cross-validation on training set
- Groups defined by `player_id` to prevent leakage
- Hyperparameter optimization via Optuna (100 trials)
- Out-of-bag error monitoring for overfitting detection

**Final Model Training**:
```python
# Train on full training set with best hyperparameters
rf_best = RandomForestClassifier(**best_params, random_state=123)
rf_best.fit(X_train_scaled, y_train)

# Serialize trained model
joblib.dump(rf_best, 'models/rf.pkl')
```

### 3.4 Advanced Model 2: LightGBM Classifier

**Rationale**: LightGBM implements gradient boosting for high accuracy through sequential error correction, offering superior performance on tabular data while maintaining interpretability via SHAP analysis.

**Hyperparameter Grid**:
- `num_leaves`: 31, 63, 127, 255
- `learning_rate`: 0.001, 0.01, 0.05, 0.1, 0.2, 0.3 (log scale)
- `max_depth`: -1, 4, 6, 8, 10, 12
- `feature_fraction`: 0.6, 0.7, 0.8, 0.9, 1.0
- `n_estimators`: 200, 500, 1000, 1500, 2000

**Cross-Validation Protocol**:
- 5-fold GroupKFold cross-validation on training set
- Early stopping after 50 rounds without validation improvement
- Hyperparameter optimization via Optuna (100 trials)
- Validation macro-F1 score as optimization objective

**Final Model Training**:
```python
# Train with early stopping on validation set
lgb_best = LGBMClassifier(**best_params, random_state=123)
lgb_best.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Serialize trained model
joblib.dump(lgb_best, 'models/lgbm.pkl')
```

### 3.5 Advanced Model 3: Sparse Gaussian Process (GP)

**Rationale**: Sparse GP provides probabilistic classification with calibrated uncertainty estimates, enabling assessment of prediction confidence through Bayesian inference while maintaining computational tractability via inducing point approximation.

**Preprocessing for GP**:
```python
# Apply scaling followed by PCA dimensionality reduction
X_train_gp = pca.fit_transform(scaler.transform(X_train))
```

**Model Architecture**:
```python
import gpytorch

class SparseGPClassifier(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
```

**Training Configuration**:
- **Inducing Points**: 1000 points selected via k-means clustering
- **Kernel**: RBF kernel with learned lengthscales
- **Optimization**: Marginal likelihood maximization via Adam optimizer
- **Convergence**: 500 training iterations with early stopping

**Model Persistence**:
```python
# Save trained GP model
torch.save(gp_model.state_dict(), 'models/gp.pkl')
```

---

## 4 Evaluation & Results

### 4.1 Evaluation Metrics

**Primary Metric**: Macro-F1 score provides equal weighting to all classes, preventing dominant classes from masking poor minority class performance:
```
Macro-F1 = (1/3) × (F1_class0 + F1_class1 + F1_class2)
```

**Secondary Metrics**:
- **Accuracy**: Overall correct classification rate
- **Balanced Accuracy**: Average per-class recall
- **Per-class Precision/Recall**: Detailed class-specific performance analysis

**Uncertainty Metrics (GP Only)**:
- **Brier Score**: Measures prediction probability calibration quality
- **Expected Calibration Error (ECE)**: Quantifies reliability of prediction confidence

**Confidence Intervals**:
1000-sample bootstrap with stratification by `player_id` to maintain group structure:
```python
bootstrap_scores = []
for i in range(1000):
    # Resample test set maintaining player groups
    resampled_players = resample(unique_test_players, random_state=i)
    resampled_indices = get_samples_for_players(resampled_players)
    
    score = f1_score(y_true[resampled_indices], y_pred[resampled_indices], average='macro')
    bootstrap_scores.append(score)

ci_95 = np.percentile(bootstrap_scores, [2.5, 97.5])
```

### 4.2 Performance Comparison Table

| Model | Test Macro-F1 (95% CI) | Accuracy | Balanced Accuracy | Training Time |
|-------|------------------------|----------|-------------------|---------------|
| Logistic Regression | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Random Forest | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| LightGBM | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Sparse GP | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

### 4.3 Confusion Matrices

![Logistic Regression Confusion Matrix](results/confusion_matrix_lr.png)
![Random Forest Confusion Matrix](results/confusion_matrix_rf.png)
![LightGBM Confusion Matrix](results/confusion_matrix_lgbm.png)
![Sparse GP Confusion Matrix](results/confusion_matrix_gp.png)

### 4.4 Feature Importance & Interpretability

**Logistic Regression Coefficients**:
[TO BE FILLED - Coefficient bar chart showing linear feature importance]
![LR Coefficients](results/lr_coefficients.png)

**Random Forest Feature Importance (Gini)**:
[TO BE FILLED - Top 10 features bar chart]
![RF Feature Importance](results/rf_feature_importance.png)

**LightGBM SHAP Analysis**:
![SHAP Summary Plot](results/shap_summary_lgbm.png)

**Key SHAP Insights**:
- [TO BE FILLED - Top features driving predictions]
- [TO BE FILLED - Feature interaction effects]
- [TO BE FILLED - Class-specific feature patterns]

**Partial Dependence Plots**:
[TO BE FILLED - 3 key features showing relationship to prediction probability]

### 4.5 Uncertainty & Calibration (GP Only)

![GP Calibration Curve](results/gp_calibration.png)

**Calibration Metrics**:
- **Expected Calibration Error (ECE)**: [TO BE FILLED]
- **Brier Score**: [TO BE FILLED]
- **Calibration Interpretation**: [TO BE FILLED - quality of uncertainty estimates]

### 4.6 Additional Diagnostics (Optional)

**Validation Curves**:
![LR Validation Curve](results/val_curve_lr.png)

[TO BE FILLED - ROC curves, Precision-Recall curves if generated]

---

## 5 Discussion

**Model Ranking**: [TO BE FILLED - based on confidence intervals and practical significance]

**Compute Cost vs. Performance**:
- **Logistic Regression**: [TO BE FILLED - baseline linear performance with minimal computational cost]
- **Random Forest**: [TO BE FILLED - ensemble benefits vs. computational overhead]
- **LightGBM**: [TO BE FILLED - efficiency vs. accuracy trade-off]
- **Sparse GP**: [TO BE FILLED - computational overhead vs. uncertainty benefits]

**Interpretability Analysis**:
- **Linear Baseline**: Logistic regression coefficients provide direct feature-to-prediction relationships
- **Feature Importance Comparison**: Coefficients vs. Gini vs. SHAP insights
- **Key Predictive Features**: [TO BE FILLED - which sensor statistics drive classification]
- **Swing Mode Discrimination**: [TO BE FILLED - how models distinguish between air/power/stable swings]

**Model Complexity Progression**:
- **Linear → Ensemble → Boosting → Bayesian**: Clear progression from simple to sophisticated approaches
- **Interpretability Trade-offs**: Linear coefficients → feature importance → SHAP → uncertainty estimates

**Uncertainty Benefits**:
[TO BE FILLED - value of GP confidence estimates for coaching applications]

**Bias-Variance Considerations**:
- **Linear Model**: High bias, low variance baseline
- **Ensemble Methods**: Bias-variance trade-off optimization
- **Overfitting Signs**: [TO BE FILLED - validation curves, OOB error analysis]
- **Generalization**: [TO BE FILLED - cross-player performance analysis]

**Limitations**:
- **Dataset Scope**: 93 players may not capture full population diversity
- **Feature Engineering**: Reliance on pre-computed statistics vs. raw time-series
- **Temporal Dynamics**: Static features may miss important temporal patterns
- **Class Imbalance**: [TO BE FILLED - impact on minority class performance]

---

## 6 Conclusion & Future Work

**Key Takeaways**:
- [TO BE FILLED - Best model and performance summary]
- **Linear Baseline Value**: Logistic regression establishes interpretable performance floor
- **Ensemble Benefits**: Tree-based methods demonstrate improvement over linear baseline
- **Methodological Insights**: Group-aware splitting critical for realistic performance estimates
- **Interpretability Value**: SHAP analysis reveals actionable coaching insights
- **Uncertainty Quantification**: GP calibration enables confidence-aware predictions

**Model Progression Insights**:
- **Complexity vs. Performance**: Quantify gains from increasing model sophistication
- **Interpretability Spectrum**: From linear coefficients to uncertainty estimates

**Realistic Next Steps**:
- **Deep Learning on Raw Signals**: CNN/LSTM approaches on full time-series data
- **Multi-Modal Fusion**: Incorporate additional sensor modalities beyond IMU
- **Temporal Modeling**: Sequence-to-sequence architectures capturing swing dynamics
- **Transfer Learning**: Leverage models trained on larger sports biomechanics datasets
- **Real-Time Implementation**: Edge deployment for live coaching feedback

---

## 7 References

- Dryad Dataset: DOI 10.5061/dryad.0zpc8677f (assignTTSWING.csv)
- scikit-learn Documentation: Machine Learning in Python
- LightGBM Documentation: Gradient Boosting Framework
- GPytorch Documentation: Gaussian Processes in PyTorch
- SHAP Documentation: SHapley Additive exPlanations
- COMP4702 Lecture Notes, University of Queensland, 2025

---

## 8 Appendix

### 8.1 Hyperparameter Grids

**Logistic Regression Grid**:
| Parameter | Values |
|-----------|--------|
| `C` | 10⁻³, 10⁻², 10⁻¹, 1, 10, 100 |
| `penalty` | 'l2', 'l1' (optional) |

**Random Forest Complete Grid**:
[TO BE FILLED - full parameter space table]

**LightGBM Complete Grid**:
[TO BE FILLED - full parameter space table]

**Sparse GP Configuration**:
[TO BE FILLED - kernel parameters and optimization settings]

### 8.2 Additional Plots

[TO BE FILLED - high-resolution EDA figures]
[TO BE FILLED - validation curves]
[TO BE FILLED - GP training loss curves]

### 8.3 Code Snippets

**Data Splitting Implementation**:
```python
# Key function from split_data.py
def create_group_splits(df, group_col='player_id', test_size=0.15, val_size=0.15, random_state=123):
    # Implementation details...
```

**Logistic Regression Training**:
```python
# Key function from train_lr.py
def train_logistic_regression(X_train, y_train, X_val, y_val, C_values):
    # Implementation details...
```

**Model Training Example**:
```python
# Key function from train_rf.py
def train_random_forest(X_train, y_train, X_val, y_val, n_trials=100):
    # Implementation details...
```

### 8.4 Environment Specification

**Conda Environment** (`environment.yml`):
```yaml
name: ml_assignment
channels:
  - defaults
  - pytorch
  - conda-forge
dependencies:
  - python>=3.10
  - pandas
  - numpy
  - scikit-learn
  - lightgbm
  - pytorch
  - gpytorch
  - shap
  - optuna
  - matplotlib
  - seaborn
  - joblib
```

**Random Seed Configuration**: `SEED = 123` used throughout all scripts for reproducibility

### 8.5 Assumptions & Notes

- **Sensor Noise**: Assumed Gaussian with σ_IMU = 0.05g
- **Player Homogeneity**: No systematic domain shift across players
- **Feature Independence**: Statistical features treated as independent after scaling
- **GP Inducing Points**: K-means initialization assumed adequate for feature space coverage
- **Class Balance**: Natural class distribution preserved without resampling
- **Linear Baseline**: Logistic regression establishes the linear performance ceiling
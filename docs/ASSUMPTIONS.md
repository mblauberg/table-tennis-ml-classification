# ASSUMPTIONS.md - Data Preprocessing Assumptions
## COMP4702 Machine Learning Assignment

### Data Quality Assumptions

**1. Missing Values**
- **Assumption**: The dataset is complete with no missing values based on initial analysis
- **Justification**: EDA showed 0 missing values across all 97,355 samples
- **Impact**: No imputation strategies needed, preserving original data integrity

**2. Outlier Handling**
- **Assumption**: Outliers represent valid sensor readings and should be retained
- **Justification**: Sensor data naturally contains extreme values during different movement phases
- **Impact**: Outliers kept to preserve natural data distribution and avoid information loss

### Feature Engineering Assumptions

**3. Irrelevant Column Identification**
- **Assumption**: ID columns (id, fileindex, count) and date are not predictive features
- **Justification**: These are metadata/administrative fields not related to movement patterns
- **Impact**: Reduces dimensionality and focuses on meaningful sensor data

**4. Feature Scaling Method**
- **Assumption**: StandardScaler is appropriate for sensor data normalization
- **Justification**: EDA showed approximately normal distributions for most features
- **Impact**: Ensures all features contribute equally to distance-based algorithms

**5. Target Variable Selection**
- **Assumption**: 'testmode' is the primary target variable for classification
- **Justification**: 
  - More interpretable classes (0, 1, 2) 
  - Reasonable class distribution (7.7%, 75.9%, 16.4%)
  - 'teststage' available as alternative target with 4 classes
- **Impact**: Focuses modeling efforts on testmode classification initially

### Data Splitting Assumptions

**6. Stratified Sampling**
- **Assumption**: Stratified splitting preserves class distribution across train/val/test sets
- **Justification**: Handles class imbalance by maintaining proportions in each split
- **Impact**: Ensures representative samples in all data splits

**7. Split Ratios**
- **Assumption**: 60% train, 20% validation, 20% test split is appropriate
- **Justification**: Standard practice providing sufficient training data while reserving adequate test data
- **Impact**: Balances model training capacity with robust evaluation

### Categorical Encoding Assumptions

**8. Label Encoding Strategy**
- **Assumption**: Label encoding is sufficient for categorical variables
- **Justification**: Most categorical variables have low cardinality (≤10 categories)
- **Impact**: Preserves ordinal relationships where they exist, simpler than one-hot encoding

**9. Gender and Handedness Encoding**
- **Assumption**: Binary/categorical demographic variables can be label encoded
- **Justification**: These are nominal variables with clear distinct categories
- **Impact**: Converts categorical data to numerical format for ML algorithms

### Temporal Assumptions

**10. Time Independence**
- **Assumption**: Individual samples are independent despite temporal collection
- **Justification**: Each row represents a distinct movement measurement
- **Impact**: Allows standard ML approaches without time series considerations

**11. Date Column Exclusion**
- **Assumption**: Temporal information (date) is not relevant for movement classification
- **Justification**: Focus is on movement patterns, not when they were recorded
- **Impact**: Simplifies model by removing temporal complexity

### Class Imbalance Assumptions

**12. Imbalance Handling Strategy**
- **Assumption**: Class imbalance will be addressed during model training, not preprocessing
- **Justification**: Preserves original data distribution for initial analysis
- **Impact**: Defers sampling strategies (SMOTE, etc.) to modeling phase

**13. Minority Class Significance**
- **Assumption**: All classes (including 7.7% minority class) contain meaningful patterns
- **Justification**: Each class represents distinct movement modes worth predicting
- **Impact**: Justifies keeping all classes rather than combining or removing small classes

### Feature Correlation Assumptions

**14. Multicollinearity Tolerance**
- **Assumption**: High correlations between sensor features are expected and acceptable
- **Justification**: Related sensor measurements naturally correlate (e.g., ax_mean vs ax_var)
- **Impact**: Feature selection/reduction deferred to modeling phase if needed

**15. Feature Completeness**
- **Assumption**: All provided sensor features contribute meaningful information
- **Justification**: Features represent comprehensive sensor data collection
- **Impact**: Retains full feature set for initial modeling attempts

### Validation Assumptions

**16. Cross-Validation Strategy**
- **Assumption**: 5-fold stratified cross-validation will provide robust model evaluation
- **Justification**: Standard practice balancing computational cost with statistical reliability
- **Impact**: Guides model selection and hyperparameter tuning approach

**17. Evaluation Metrics**
- **Assumption**: Accuracy, precision, recall, and F1-score are appropriate for this classification task
- **Justification**: Standard multiclass classification metrics suitable for imbalanced data
- **Impact**: Provides comprehensive model performance assessment

---

### Assumption Validation Plan

**During Model Development:**
1. Monitor model performance to validate scaling assumptions
2. Assess feature importance to confirm relevance assumptions
3. Evaluate class prediction quality to validate target selection
4. Test alternative preprocessing strategies if assumptions prove limiting

**Risk Mitigation:**
- Document alternative approaches for each major assumption
- Prepare fallback strategies (e.g., different scaling, target variables)
- Plan iterative refinement based on model performance feedback 
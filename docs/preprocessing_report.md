# Data Preprocessing Report
## COMP4702 Machine Learning Assignment

### Dataset Overview
- **Original shape:** (97355, 51)
- **Final shape:** (97355, 47)
- **Features removed:** 4
- **Preprocessing steps:** 13

### Removed Columns
- `fileindex`
- `count`
- `id`
- `date`

### Categorical Encoding
- `age`: 3 categories
- `playYears`: 3 categories
- `height`: 3 categories
- `weight`: 3 categories

### Feature Scaling
- **Method:** StandardScaler
- **Features scaled:** 45

### Data Splits
- **Training:** 58,413 samples
- **Validation:** 19,471 samples
- **Test:** 19,471 samples

### Preprocessing Steps
1. Dataset loaded with shape (97355, 51)
2. Removed 4 irrelevant columns
3. No missing values found in the dataset
4. Label encoded 'age' (4 categories)
5. Label encoded 'playYears' (4 categories)
6. Label encoded 'height' (4 categories)
7. Label encoded 'weight' (4 categories)
8. Detected outliers in multiple columns (kept for analysis)
9. Applied standard scaling to 45 features
10. Target variable analysis completed
11. Data split into train/val/test with stratification=True
12. Saved full processed dataset: processed_dataset.csv
13. Saved train/val/test splits
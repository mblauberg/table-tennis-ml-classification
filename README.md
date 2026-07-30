# Table-tennis stroke classification from IMU data

Classifying table-tennis swing types from 6-axis IMU sensor data, built for
COMP4702 (Machine Learning) at the University of Queensland. The project
compares four classifiers — logistic regression, random forest, LightGBM and a
Gaussian process — on an imbalanced three-class problem, with the experimental
design centred on preventing player-identity leakage.

## Results

Evaluated on a held-out test set of players never seen in training:

| Model | F1-macro | ROC-AUC | Notes |
|-------|----------|---------|-------|
| LightGBM | 0.960 | 0.997 | Best overall; see the early-stopping caveat below |
| Random forest | 0.957 | 0.997 | Optuna-tuned, feature importances |
| Gaussian process | 0.928 | 0.991 | Calibrated probabilities, uncertainty estimates |
| Logistic regression | 0.917 | 0.980 | Interpretable baseline, balanced class weights |

The classes are heavily imbalanced (roughly 8% / 76% / 16%), so macro-F1 is
the headline metric.

## Method

- **Signal processing** (`src/etl.py`): raw 16-bit LSB readings converted to
  m/s² and rad/s, physics-based outlier filtering (‖acceleration‖ > 16 g),
  then 44 statistical features (RMS, variance, entropy, FFT-based) per swing
  from the 6-axis stream.
- **Split** (`src/split_data.py`): an 80/20 player-disjoint holdout using
  `GroupShuffleSplit`, with an assertion that no player appears on both sides.
  Swings from the same player are correlated, so a row-wise random split would
  overstate every model's performance.
- **Tuning**: Optuna (TPE) with group-aware cross-validation inside the
  training set only.
- **Validation**: bootstrap confidence intervals on test metrics; calibration
  curves for the probabilistic models.

## Limitations

- In the final LightGBM run, early stopping monitors the held-out test set
  (`src/train_lgbm.py`), so its reported test F1 is mildly optimistic. The
  other models are unaffected. Fixing this properly means carving a validation
  fold from the training players.
- Single holdout split, not repeated or nested cross-validation — the
  group-aware CV is used for tuning, not for the headline estimate.
- The dataset is course-provided and not redistributable, which is why there
  is no `data/` directory here. The pipeline expects
  `data/raw/assignTTSWING.csv`; without it the code documents the approach but
  is not runnable end-to-end. The trained artifacts and course report are also
  not included.

## Repository contents

```
src/
├── etl.py                     # Signal processing and feature engineering
├── split_data.py              # Player-disjoint train/test split
├── eda.py                     # Exploratory data analysis
├── train_lr.py                # Logistic regression
├── train_rf.py                # Random forest (Optuna)
├── train_lgbm.py              # LightGBM (Optuna)
├── train_gp.py                # Gaussian process
├── train_all_models.py        # Orchestration
└── comprehensive_analysis.py  # Cross-model comparison and plots
```

## Running it

```bash
conda env create -f environment.yml
conda activate ml_assignment

# with data/raw/assignTTSWING.csv in place:
python src/train_all_models.py            # everything (~90 min)
python src/train_all_models.py lgbm rf    # selected models
```

## Context

Solo project for COMP4702 at UQ. The accompanying report covered methodology,
ablations and error analysis; it is not published here.

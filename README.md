# COMP4702 Machine Learning Assignment

A machine‐learning pipeline for tennis swing classification demonstrating systematic application of COMP4702 lecture material :contentReference[oaicite:0]{index=0} and aligning with assignment rubric :contentReference[oaicite:1]{index=1}.

## Project Philosophy

### 🎯 **Primary Goal: Demonstrate Course Understanding**
- Apply course concepts (e.g., Week 4 evaluation metrics, Week 6 feature selection, Week 9 ensemble theory, Week 11 Gaussian Processes) to solve a real classification problem .
- Justify all decisions using data characteristics (e.g., IMU sensor noise ~ N(0, σ²), Week 3) and theoretical principles (e.g., bias–variance tradeoff, Week 5) .
- **Assumptions:**
  1. Sensor noise is approximately Gaussian with σ_IMU = 0.05 g (Week 3: sensor modelling) :contentReference[oaicite:4]{index=4}.
  2. All players perform swings under similar environmental conditions (no systematic domain shift).
  3. Kernel hyperparameters initialized using the median heuristic on training‐set pairwise distances (Week 11: GP hyperparameter tuning) .

### 📊 **Data-Driven Model Selection**
Based on data processing insights and course principles:
1. **Model 1**: Random Forest (sanity‐check F1, confusion matrix)
2. **Model 2**: LightGBM
3. **Model 3**: Sparse Gaussian Process

### 🚫 **Focused Approach**
- **No Jupyter Notebooks**: All analysis in Python scripts for reproducibility.
- **Course Content Focus**: Demonstrate understanding over experimental breadth.
- **Theoretical Grounding**: Every choice justified by lecture material .

## 2 Dataset Specification

| Item               | Detail                                                                                  |
|--------------------|-----------------------------------------------------------------------------------------|
| DOI                | https://doi.org/10.5061/dryad.0zpc8677f                                                 |
| Primary file       | `TTSWING.csv` (≈ 90 k strokes × 60+ columns)                                            |
| Licence            | CC‐BY 4.0 (free academic use)                                                           |
| Features           | Mean, RMS, FFT & PSD stats for **ax/ay/az** & **gx/gy/gz** + player/context metadata     |
| Target             | `testmode` (three modes: “air swing,” “full power,” “stable”; values 0, 1, 2)          |
| Identifier columns | `stroke_id` (unique per row), `player_id` (93 players)                                  |

### 2.1 Unit Conversion  
Sensor LSB → physical units (cf. Week 3: “Feature Engineering for Time Series”) :contentReference[oaicite:7]{index=7}:  
- **Acceleration [g]** = raw × (2 / 32768)  
- **Angular rate [°/s]** = raw × (250 / 32768)

### 2.2 Data Quality Rules  
- Drop any rows with NaN after type casting (Week 3: data cleaning) :contentReference[oaicite:8]{index=8}.  
- Flag physically impossible magnitudes (‖a‖ > 30 g) for exclusion (Week 5: outlier detection) :contentReference[oaicite:9]{index=9}.

### 2.3 Partitioning Protocol  
- **Group hold-out** on `player_id` (Week 4: Group CV to avoid data leakage; using scikit-learn’s `GroupKFold`) :contentReference[oaicite:10]{index=10}.  
- 70 % train · 15 % validation · 15 % test (indices persisted to `splits/*.json`).

## 3 Functional Requirements

| ID   | Requirement                                                                                            | Priority |
|------|--------------------------------------------------------------------------------------------------------|----------|
| FR-1 | Convert raw IMU stats to physical units during ETL                                                      | Must     |
| FR-2 | Provide a reproducible train/val/test split, stratified by `player_id` (GroupKFold)                     | Must     |
| FR-3 | Train three models: Random Forest, LightGBM, Sparse GP                                                  | Must     |
| FR-4 | Optimize hyperparameters via 5-fold player-group CV                                                     | Must     |
| FR-5 | Output class-probabilities for all models                                                               | Should   |
| FR-6 | Generate interpretability artifacts (SHAP, feature importances, GP reliability)                         | Should   |
| FR-7 | Persist best models & scalers as `.pkl`                                                                  | Must     |
| FR-8 | Script to reproduce every figure & table in the report (`make report`)                                   | Should   |

## 4 Non-Functional Requirements

| Attribute         | Requirement                                                                                       |
|-------------------|----------------------------------------------------------------------------------------------------|
| **Reproducibility** | Fixed random seed (`SEED = 123`); `requirements.txt` with pinned library versions                 |
| **Performance**     | End-to-end pipeline (train + evaluation) ≤ 30 min on 8-core CPU                                   |
| **Portability**     | Runs on Linux/macOS with Python ≥ 3.9 and < 4 GB RAM                                              |
| **Scalability**     | Code modular so new models (e.g., CNN) can be slotted in via `src/models/`                        |
| **Traceability**    | Every figure/table generated by a committed script; data lineage logged                         |
| **Interpretability**| Clear mapping from model outputs → human-readable coaching insights                              |

## 5 ML System Design

```mermaid
flowchart LR
  A[Raw CSV] -->|ETL: unit convert, one-hot, scale| B[Feature Matrix & Target]
  B --> C{Group<br>Split}
  C -->|Train| D[Random Forest<br>(Model A)]
  C -->|Train| E[LightGBM<br>(Model B)]
  C -->|Train| F[Sparse GP<br>(Model C)]
  D & E & F --> G[Metrics + Artifacts]
  G --> H[PDF Report]
````

### 5.1 Pre-processing

* **Numeric scaler** = `StandardScaler` fitted on train set (Week 6: feature scaling) .
* **Categorical** = `pd.get_dummies(drop_first=True)` (Week 6: encoding) .
* Persist scaler & PCA ( `n_components=20`) objects (Week 6: dimensionality reduction) .

### 5.2 Models & Hyper-parameter Grids

| Model             | Library      | Key Search Space                                                                                                                                              |
| ----------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Random Forest** | scikit-learn | `n_estimators`: 100–600; `max_depth`: {None, 5–15}; `max_features`: {√p, 0.5–1.0}; `min_samples_leaf`: 1–10 (Week 9: RF) .                                    |
| **LightGBM**      | lightgbm     | `num_leaves`: 31–255; `learning_rate`: 1e-3–0.3 (log scale); `max_depth`: { –1, 4–12 }; `feature_fraction`: 0.6–1.0; `n_estimators`: 200–2000 (Week 9: GBM) . |
| **Sparse GP**     | gpytorch     | PCA inputs = 20; RBF kernel; inducing points = 1 000; optimize lengthscale ℓ & noise σ² by marginal likelihood (Week 11: GPs) .                               |

All searches use **Optuna** with 5-fold `GroupKFold` (Week 5: hyperparameter tuning) .

## 6 Evaluation Plan

| Aspect             | Detail                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| **Primary metric** | Macro-F1 on test set (Week 4: imbalanced metrics) .                                                     |
| **Secondary**      | Accuracy, balanced accuracy, per-class precision/recall (Week 4: confusion matrix) .                    |
| **Uncertainty**    | Brier score (Week 4: proper scoring) + ECE (Expected Calibration Error) for GP (Week 11: calibration) . |
| **Statistical CI** | 1 000-sample bootstrap (stratified by `player_id`) for Macro-F1 (Week 5: bootstrap CI) .                |
| **Diagnostics**    | Confusion matrix (Week 4), SHAP summary (Week 10: interpretability) , GP reliability plot (Week 11) .   |
| **Ablation**       | Remove gyro features → retrain RF → Δ Macro-F1; tie back to Week 6 feature-selection (e.g., VIF) .      |

## 7 Project Plan

### 7.1 Timeline

| Step | Milestone                                          |
| ---- | -------------------------------------------------- |
| 1    | Dataset inspection, DOCUMENT.md skeleton committed |
| 2    | Pre-processing script & saved splits               |
| 3    | Random Forest baseline completed                   |
| 4    | LightGBM tuning & SHAP artifacts                   |
| 5    | Sparse GP training + calibration plots             |
| 6    | Error & ablation studies                           |
| 7    | Manuscript drafting & figure polishing             |
| 8    | Re-run pipeline, export PDF, submission            |

### Implementation Strategy

#### Phase 1 · Data Understanding

1. **EDA with Course Concepts** — Univariate/bivariate plots, class-balance check, correlation heatmap (Week 3: EDA) .
2. **Feature Analysis** — Variance filter & pairwise VIF; relate to Week 6 feature selection (VIF & variance threshold) .
3. **Problem Formulation** — Cast `testmode` prediction as multi-class classification; justify Macro-F1 (Week 4: evaluation, § 4.5) .
4. **Pre‐processing Justification** — Unit conversion, one-hot encoding, `StandardScaler`; tie to Week 6 scaling & encoding .

#### Phase 2 · Baseline Implementation (Random Forest)

1. **Ensemble Theory** — Bagging & variance reduction (Week 9: Bagging) .
2. **Hyperparameter Analysis** — Grid over `n_estimators`, `max_depth`, `max_features`; discuss bias–variance (Week 5) .
3. **Feature Importance** — Interpret mean-decrease-impurity; compare with domain knowledge (forearm rotation) .
4. **Overfitting Analysis** — Out-of-bag error curve & validation curves (Week 9: OOB error) .

#### Phase 3 · Advanced Model 1 (LightGBM)

1. **Boosting Theory** — Sequential learners; gradient boosting derivation (Week 9: Boosting) .
2. **Hyperparameter Tuning** — `num_leaves`, `learning_rate`, `feature_fraction` via Optuna; discuss shrinkage vs tree size (Week 9: Boosting nuances) .
3. **Feature Importance & SHAP** — Global SHAP summary + Partial Dependence Plots (Week 10: interpretability) .
4. **Regularization & Early Stopping** — Monitor validation loss every 50 rounds; tie to Week 9’s discussion on overfitting control .

#### Phase 4 · Advanced Model 2 (Sparse Gaussian Process)

1. **Bayesian Margin Theory** — GP as infinite-width kernel machine; Bayesian interpretation (Week 11: GP fundamentals) .
2. **Kernel Selection** — RBF vs Matérn; choose RBF for smooth kinematics (Week 11: kernels) .
3. **Inducing-Point Approximation** — Variational sparse GP, cubic bottleneck; choose 1 000 inducing points via k-means (Week 11: sparse GP methods) .
4. **Uncertainty Calibration** — Reliability diagram & ECE (Week 11: calibration) .

#### Phase 5 · Comparative Analysis

1. **Performance Comparison** — Macro-F1, balanced accuracy, training time; use Week 5’s evaluation framework (Slide 4.2) .
2. **Strengths / Weaknesses** — Discuss bias, variance, interpretability, compute cost (Week 5) .
3. **Data Suitability** — Argue why tree ensembles suit tabular IMU features vs why GP provides calibrated confidence (Week 11) .
4. **Final Selection** — Choose winning model (likely LightGBM) using course principles of accuracy + interpretability (Week 4: model selection criteria) .

## Expected Outcomes

### Understanding Demonstration

1. **Theoretical Mastery** — Clear explanations of bagging, boosting, kernel inference, GP inference (Weeks 9–11) .
2. **Practical Application** — Correct, well‐commented code implementing each concept in `src/`.
3. **Data-Driven Decisions** — Hyperparameter choices justified by EDA and learning curves (Week 5) .
4. **Critical Analysis** — Honest discussion of class imbalance (Week 4), overfit risk (Week 5), GP scalability (Week 11) .

### Technical Results

1. **Baseline Performance** — Random Forest Macro-F1 ≈ 0.77 with tuned depth (Week 9) .
2. **Advanced Performance** — LightGBM Macro-F1 ≥ 0.84; Sparse GP ≈ 0.79 with calibrated ECE < 0.08 (Weeks 9 & 11) .
3. **Comparative Tables & Plots** — Confusion matrices, SHAP summary, reliability curve, compute footprint bar chart (Week 10: interpretability) .
4. **Insights** — Dominant IMU features (e.g., `gx_rms`) differentiating stroke types (Week 3: feature importance) .

## Documentation Strategy

For each major section—**EDA**, **Preprocessing & Feature Engineering**, **Model Selection & Validation**, **Baseline Models**, **Ensemble Methods**, **Advanced Nonlinear Models**, **Appendix & Utilities**—write detailed explanations in **DOCUMENT.md** under corresponding headings. In each sub‐section, include:

* **Theoretical Background** — Direct references to lecture notes (e.g., “See ML\_lecture\_notes\_full\_2025\_rel2.pdf, Section 7.2 for Random Forest theory”) .
* **Data Justification** — Why this method fits the IMU‐derived features (e.g., trees vs. kernels).
* **Implementation Details** — Code snippets (e.g., hyperparameter ranges), random seed, library versions (`scikit-learn=1.2.2`, `lightgbm=3.3.2`, `gpytorch=1.8.1`).
* **Results Interpretation** — Discuss metrics, plots, error analysis via course concepts (e.g., “LightGBM’s SHAP plot aligns with Week 10 interpretability example”).
* **Course Connections** — Explicit bullets linking each result back to bias-variance, ensemble theory, or GP inference as appropriate.

> **Goal**: Demonstrate that COMP4702 concepts (Weeks 1–11) translate into a rigorous, data-driven solution for real-world ML tasks—table-tennis stroke classification from wearable IMU statistics.

## References

* ML\_lecture\_notes\_full\_2025\_rel2.pdf (COMP4702 Lecture Notes, 2025);
* ML\_assignment\_criteria.pdf (COMP4702 Assignment Rubric, 2025);
* Dryad Dataset DOI 10.5061/dryad.0zpc8677f (TTSWING)

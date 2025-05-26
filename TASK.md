<!-- TASK.md -->

# Machine Learning Assignment – Task Board

> **Legend**  
> ◻︎ To Do ▪︎ In Progress ✓ Done

## Phase 0 – Repository & Environment
- ◻︎ Initialise Git repository and push to remote.  
- ◻︎ Add `assignTTSWING.csv`, `ML_assignment.pdf`, `ML_assignment_criteria.pdf` to repo.  
- ◻︎ Create Python 3.11 environment (`requirements.txt`).  
- ◻︎ Configure `.gitignore`, optional DVC.

## Phase A – Data Understanding & Pre-processing (`SCOPE.md`)
- ◻︎ Load data and print schema / basic stats.  
- ◻︎ Visual EDA: pair-plots, correlation heat-map, class distribution.  
- ◻︎ Drop irrelevant or ID-like columns.  
- ◻︎ Handle missing values (imputation / row drop).  
- ◻︎ Scale/normalise numerical features.  
- ◻︎ Assess need for PCA; implement if ≥ 90 % variance captured in ≤ 20 components.  
- ◻︎ Decide on problem framing (classification vs regression).  
- ◻︎ Commit updates & finalise `SCOPE.md`.

## Phase B – Modelling (one sub-section per model)
### B1 Baseline Linear / Logistic
- ◻︎ Build pipeline with scaling.  
- ◻︎ Hyper-parameter grid search (C, penalty).  
- ◻︎ Save metrics & coefficients to `model_linear.md`.

### B2 Tree-based Ensemble (RF / XGBoost / LightGBM)
- ◻︎ Implement model with cross-validated grid search.  
- ◻︎ Plot feature importances.  
- ◻︎ Document in `model_tree.md`.

### B3 Neural Network (MLP or 1-D CNN)
- ◻︎ Define architecture & callbacks.  
- ◻︎ Track learning curves.  
- ◻︎ Log results in `model_nn.md`.

## Phase C – Evaluation & Comparison (`COMPARE.md`)
- ◻︎ Consolidate metrics into a single table.  
- ◻︎ Draw ROC/PR or residual plots as appropriate.  
- ◻︎ Perform statistical significance test or cross-fold variance check.  
- ◻︎ Write interpretive commentary (½–1 page).

## Phase D – Reporting
- ◻︎ Draft abstract and introduction.  
- ◻︎ Insert key figures & tables.  
- ◻︎ Ensure every rubric line item is addressed.  
- ◻︎ Peer/self review for clarity & formatting.

## Phase E – Finalisation
- ◻︎ Convert markdown → PDF via pandoc or LaTeX.  
- ◻︎ Tag release (`v1.0`) and push.  
- ◻︎ Submit PDF & repo link.

---

**Progress at a glance**

| Phase | % complete |
|-------|------------|
| Repo & Env | 0 % |
| Pre-processing | 0 % |
| Modelling | 0 % |
| Evaluation | 0 % |
| Report | 0 % |
| Finalisation | 0 % |

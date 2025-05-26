<!-- PLAN.md -->

# Machine Learning Assignment – Project Plan

## 1 · Project Overview
We will build a fully-reproducible ML pipeline that predicts the **swing-related target** contained in `assignTTSWING.csv` (classification or regression to be finalised after EDA).  
The end product is a short report (≈ 8–12 pages, PDF) plus version-controlled code and markdown documentation that meet every criterion in *ML_assignment.pdf* and *ML_assignment_criteria.pdf*.

## 2 · Objectives
1. **Understand & clean the data** – identify unusable columns, handle missing values, scale features, and optionally apply PCA.  
2. **Benchmark three model families** – a linear/baseline model, a tree-based ensemble, and a neural network (or other advanced model best suited to the task).  
3. **Evaluate & compare** models on a held-out test set with rubric-aligned metrics.  
4. **Explain results** – feature importance, error analysis, and links to theory.  
5. **Deliver polished documentation** – every major step logged in markdown files inside the repo.

## 3 · Scope & Deliverables
| Artefact | File(s) | Description |
|----------|---------|-------------|
| **Data SCOPE log** | `SCOPE.md` | Decisions made during cleaning & EDA (with key plots). |
| **Model notebooks / scripts** | `src/` or `notebooks/` | One file per model family, mirroring `model_<name>.md` docs. |
| **Model docs** | `model_<name>.md` | Hyper-parameters, training curves, metrics, discussion. |
| **Comparison & Conclusion** | `COMPARE.md` | Side-by-side metric table, statistical tests, final choice. |
| **Final report** | `ML_assignment.pdf` | Typeset submission compiled from markdown sources. |

## 4 · Technology & Tools
- **Cursor IDE** with built-in LLM agent for code + prose co-editing.  
- **Python 3.11** environment (Conda or venv).  
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`/`seaborn`, `imbalanced-learn`, `shap`, `torch` or `tensorflow` (if NN chosen).  
- **Version control**: Git (commit early & often).  
- **Data versioning** (optional but recommended): DVC or Git-LFS.

## 5 · Workflow
1. **Setup** – clone repo, create environment, add dataset & rubric PDFs.  
2. **EDA & Cleaning** (Phase A) – update `SCOPE.md`.  
3. **Modelling** (Phase B) – build three pipelines in parallel.  
4. **Evaluation** (Phase C) – generate metrics & plots, populate `model_*.md`.  
5. **Comparison** (Phase D) – fill `COMPARE.md`, decide best model.  
6. **Reporting** (Phase E) – convert markdown → LaTeX/PDF, final proof-read.  

## 6 · Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| *Dataset imbalance* | Use stratified splits, class weights, or SMOTE. |
| *Model over-fitting* | Cross-validation, learning-curve checks, regularisation. |
| *Markdown bloat* | Split docs by phase; keep large outputs in `/figures`. |
| *Time crunch* | Milestone dates in TASK.md; weekly progress reviews. |

## 7 · Timeline (indicative)
| Phase | Deadline |
|-------|----------|
| Setup & EDA complete | **Day 3** |
| All three models trained | **Day 7** |
| Comparison & draft report | **Day 10** |
| Final PDF & repo tidy-up | **Day 12** |

---

**Author:** *<your-name>*  **Last updated:** <!-- auto-update -->

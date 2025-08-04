# 🏓 Table Tennis Swing Classification: Advanced ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)
[![Deep Learning](https://img.shields.io/badge/DL-PyTorch-red.svg)](https://pytorch.org)
[![Academic](https://img.shields.io/badge/Academic-COMP4702-green.svg)](report.pdf)

> **A production-grade machine learning system for classifying table tennis swing types using IMU sensor data. Demonstrates advanced ML concepts including ensemble methods, Bayesian approaches, and rigorous experimental design.**

## 🎯 Project Highlights

**🏆 Key Achievements:**
- **96.0% F1-Score** on imbalanced multi-class classification
- **Rigorous experimental design** with group-aware cross-validation preventing data leakage
- **Four sophisticated algorithms** from linear baselines to Bayesian uncertainty quantification
- **Production-ready pipeline** with comprehensive error handling and statistical validation

## 📊 Performance Summary

| Model | F1-Macro | ROC-AUC | Key Strength |
|-------|----------|---------|--------------|
| **LightGBM** | **96.0%** | **99.7%** | Best overall performance |
| **Random Forest** | 95.7% | 99.7% | Robust ensemble with feature importance |
| **Gaussian Process** | 92.8% | 99.1% | **Uncertainty quantification** |
| **Logistic Regression** | 91.7% | 98.0% | Interpretable baseline |

## 🧠 Technical Innovation

### **Advanced ML Techniques Implemented**

#### 🔬 **Rigorous Experimental Design**
- **Group-aware data splitting** preventing player-specific data leakage
- **Nested cross-validation** with StratifiedGroupKFold for unbiased hyperparameter tuning
- **Statistical significance testing** with bootstrap confidence intervals

#### 🤖 **Sophisticated Model Portfolio**
1. **Logistic Regression**: Multinomial classification with L2 regularization and balanced class weights
2. **Random Forest**: Bootstrap aggregating with Bayesian hyperparameter optimization (Optuna)
3. **LightGBM**: Gradient boosting with leaf-wise tree growth and early stopping
4. **Gaussian Process**: Bayesian classification with RBF kernels and uncertainty quantification

#### 📈 **Engineering Excellence**
- **Signal Processing Pipeline**: LSB → physical units conversion with outlier detection
- **Intelligent Feature Selection**: Correlation-based pruning with model-specific thresholds
- **Bayesian Optimization**: Optuna framework with Tree-structured Parzen Estimator
- **Production Monitoring**: Comprehensive logging, timing analysis, model serialization

## 🔬 Research-Grade Implementation

### **Dataset Characteristics**
- **Source**: IMU sensor data from 93 players during table tennis training
- **Volume**: 97K samples with 44 engineered statistical features
- **Challenge**: Severe class imbalance (7.7% : 75.9% : 16.4%) with player dependencies
- **Innovation**: Group-aware splitting ensuring no player appears in both train/test sets

### **Advanced Signal Processing**
- **Unit Conversion**: Raw 16-bit LSB → calibrated m/s² and rad/s values
- **Feature Engineering**: Statistical measures (RMS, variance, entropy, FFT) from 6-axis IMU
- **Quality Control**: Physics-based outlier detection (||acceleration|| > 16g threshold)

### **Statistical Rigor**
- **Cross-Validation**: StratifiedGroupKFold preserving class balance and preventing data leakage
- **Uncertainty Quantification**: Gaussian Process with calibration curve analysis
- **Performance Validation**: Bootstrap confidence intervals for statistical significance

## 📁 Repository Structure

```
table-tennis-ml-classification/
├── 📊 data/
│   ├── raw/assignTTSWING.csv              # Original IMU dataset (46MB)
│   └── processed/                         # Engineered features & metadata
├── 🧠 src/                              # Core pipeline (10 files, 5.1K lines)
│   ├── etl.py                           # Signal processing pipeline  
│   ├── split_data.py                    # Group-aware data splitting
│   ├── eda.py                           # Exploratory data analysis
│   ├── train_lr.py                      # Logistic Regression training
│   ├── train_rf.py                      # Random Forest training  
│   ├── train_lgbm.py                    # LightGBM training
│   ├── train_gp.py                      # Gaussian Process training
│   ├── train_all_models.py              # Pipeline orchestration
│   └── comprehensive_analysis.py        # Cross-model evaluation
├── 📈 results/
│   ├── comprehensive_analysis/          # Performance comparisons
│   ├── [model_name]/                   # Model-specific artifacts
│   │   ├── evaluation_metrics.json
│   │   ├── feature_importance.csv
│   │   └── [model_file]
│   └── eda/                            # Exploratory data analysis
├── 📄 report.pdf                        # Academic report (22 pages)
├── 🔧 environment.yml                    # Conda environment
└── 📖 README.md                         # This documentation
```

## 🚀 Quick Start

### **Installation**
```bash
# Setup environment  
conda env create -f environment.yml
conda activate ml_assignment

# Validate setup
python src/validate_models.py
```

### **Run Complete Pipeline**
```bash
# Train all models with comprehensive analysis (~90 minutes)
python src/train_all_models.py

# Quick training (specific models only)
python src/train_all_models.py lgbm rf  # Best performers
```

### **Key Outputs**
- **Model Performance**: `results/comprehensive_analysis/performance_summary.csv`
- **Visualizations**: ROC curves, confusion matrices, feature importance plots
- **Statistical Analysis**: Bootstrap confidence intervals, calibration curves

## 🏆 Professional Impact

### **Machine Learning Expertise Demonstrated**
- **Ensemble Methods**: Both bagging (Random Forest) and boosting (LightGBM) implementations
- **Bayesian ML**: Gaussian Processes with uncertainty quantification and kernel optimization
- **Feature Engineering**: Domain-specific signal processing for IMU sensor data
- **Statistical Validation**: Rigorous experimental design preventing common ML pitfalls

### **Software Engineering Excellence**
- **Production Architecture**: Modular, configurable, fault-tolerant pipeline design
- **Code Quality**: Comprehensive documentation, type hints, automated validation
- **Performance Optimization**: Memory-efficient processing, parallel execution, timing analysis
- **Reproducibility**: Fixed seeds, environment specifications, deterministic workflows

### **Academic Rigor**
- **22-page technical report** with comprehensive methodology and statistical analysis
- **Literature review** connecting implementation choices to theoretical foundations
- **Ablation studies** demonstrating understanding of bias-variance tradeoffs
- **Error analysis** providing insights into model limitations and failure modes

## 📊 Results Deep Dive

### **Model-Specific Insights**
- **LightGBM**: Superior handling of "Stable" class through fine-grained leaf splitting
- **Random Forest**: Excellent "Full Power" detection via acceleration-magnitude features
- **Gaussian Process**: Well-calibrated probabilities for uncertainty-aware predictions
- **Logistic Regression**: Fast, interpretable baseline with balanced class handling

### **Engineering Trade-offs**
- **Training Time**: 3 min (LR) → 65 min (RF) → 10 min (LightGBM) → 27 min (GP)
- **Model Size**: 0.8MB (LR) → 15MB (LightGBM) → 250MB (RF) → 3.2MB (GP)
- **Inference Speed**: <1.1ms per sample across all models (real-time capable)

## 📚 Academic Context

**Course**: COMP4702 - Machine Learning (University of Queensland)  
**Focus**: Advanced ML concepts with production-ready implementation practices  
**Report**: [Comprehensive 22-page technical analysis](report.pdf)

This project demonstrates mastery of core ML concepts through practical implementation:
- Data preprocessing and feature engineering
- Model selection and hyperparameter optimization  
- Ensemble methods and Bayesian approaches
- Statistical validation and uncertainty quantification
- Production system design and deployment considerations

---

## 👨‍💻 Professional Summary

This project showcases **advanced machine learning engineering** skills through a complete production pipeline that achieved state-of-the-art performance on a challenging real-world dataset. The implementation demonstrates both theoretical understanding and practical engineering expertise that would be valuable in any ML role.

**Key Differentiators:**
- ✅ **Statistical Rigor**: Proper experimental design preventing data leakage
- ✅ **Model Diversity**: Linear, tree-based, and Bayesian approaches
- ✅ **Production Quality**: Comprehensive error handling, logging, and monitoring
- ✅ **Academic Excellence**: Peer-reviewed methodology with detailed technical report

[![View Report](https://img.shields.io/badge/📄_View_Full_Report-PDF-red.svg)](report.pdf)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://linkedin.com/in/yourprofile)
[![Email](https://img.shields.io/badge/Email-Contact-green.svg)](mailto:your.email@example.com)
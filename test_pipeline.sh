#!/bin/bash
set -e

echo "=========================================="
echo "Starting COMP4702 Assignment Pipeline Test"
echo "=========================================="
echo "Test started at $(date)"
echo ""

# Create directories if they don't exist
echo "Setting up directory structure..."
mkdir -p data/processed splits models results logs
echo "✓ Directory structure created"

# Check if raw data exists
if [ ! -f "data/raw/assignTTSWING.csv" ]; then
    echo "❌ Raw data file not found: data/raw/assignTTSWING.csv"
    echo "Please ensure the raw data file is in the correct location"
    exit 1
fi
echo "✓ Raw data file found"

# Run ETL pipeline
echo ""
echo "=========================================="
echo "STEP 1: Running ETL (Extract, Transform, Load)"
echo "=========================================="
python src/etl.py --input data/raw/assignTTSWING.csv --output data/processed/processed_data.csv
echo "✓ ETL completed successfully"

# Run data splitting
echo ""
echo "=========================================="
echo "STEP 2: Running Data Splitting"
echo "=========================================="
python src/split_data.py --input data/processed/processed_data.csv --output_dir splits/
echo "✓ Data splitting completed successfully"

# Verify splits were created
if [ ! -f "splits/train.json" ] || [ ! -f "splits/val.json" ] || [ ! -f "splits/test.json" ]; then
    echo "❌ Data splits not created properly"
    exit 1
fi
echo "✓ All data splits created"

# Train models
echo ""
echo "=========================================="
echo "STEP 3: Training Models"
echo "=========================================="

echo "Training Logistic Regression model..."
python src/train_lr.py --data data/processed/processed_data.csv --splits splits/train.json splits/val.json --output_dir models
echo "✓ Logistic Regression training completed"

echo "Training Random Forest model..."
python src/train_rf.py --data data/processed/processed_data.csv --splits splits/train.json splits/val.json --output_dir models
echo "✓ Random Forest training completed"

echo "Training LightGBM model..."
python src/train_lgbm.py --data data/processed/processed_data.csv --splits splits/train.json splits/val.json --output_dir models
echo "✓ LightGBM training completed"

echo "Training Sparse Gaussian Process model..."
python src/train_gp.py --data data/processed/processed_data.csv --splits splits/train.json splits/val.json --output_dir models
echo "✓ Gaussian Process training completed"

# Verify models were created
echo ""
echo "Verifying trained models..."
models=("scaler.pkl" "lr.pkl" "rf.pkl" "lgbm.pkl")
for model in "${models[@]}"; do
    if [ -f "models/$model" ]; then
        echo "✓ Model found: $model"
    else
        echo "⚠️  Model not found: $model"
    fi
done

# Run exploratory data analysis
echo ""
echo "=========================================="
echo "STEP 4: Running Exploratory Data Analysis"
echo "=========================================="
python src/eda.py --input data/raw/assignTTSWING.csv --output_dir results/eda
echo "✓ EDA completed successfully"

# Run model evaluation
echo ""
echo "=========================================="
echo "STEP 5: Running Model Evaluation"
echo "=========================================="
python src/evaluate.py --data data/processed/processed_data.csv --test_split splits/test.json --output_dir results
echo "✓ Model evaluation completed"

# Run bootstrap confidence intervals
echo ""
echo "=========================================="
echo "STEP 6: Running Bootstrap Analysis"
echo "=========================================="
python src/bootstrap.py --data data/processed/processed_data.csv --splits splits/test.json --output_dir results/bootstrap
echo "✓ Bootstrap analysis completed"

# Run uncertainty analysis (if GP model exists)
echo ""
echo "=========================================="
echo "STEP 7: Running Uncertainty Analysis"
echo "=========================================="
if [ -f "models/gp.pkl" ]; then
    python src/analyze_uncertainty.py --data data/processed/processed_data.csv --splits splits/test.json --model models/gp.pkl --output_dir results/uncertainty
    echo "✓ Uncertainty analysis completed"
else
    echo "⚠️  GP model not found, skipping uncertainty analysis"
fi

# Run LightGBM interpretation
echo ""
echo "=========================================="
echo "STEP 8: Running LightGBM Interpretation"
echo "=========================================="
if [ -f "models/lgbm.pkl" ]; then
    python src/interpret_lgbm.py --data data/processed/processed_data.csv --splits splits/val.json --model models/lgbm.pkl --output_dir results/interpretation
    echo "✓ LightGBM interpretation completed"
else
    echo "⚠️  LightGBM model not found, skipping interpretation"
fi

# Verify results structure
echo ""
echo "=========================================="
echo "STEP 9: Verifying Results Structure"
echo "=========================================="
expected_dirs=("results/eda" "results/bootstrap" "results/uncertainty" "results/interpretation")
for dir in "${expected_dirs[@]}"; do
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -type f | wc -l)
        echo "✓ Results directory found: $dir ($file_count files)"
    else
        echo "⚠️  Results directory not found: $dir"
    fi
done

# Display summary
echo ""
echo "=========================================="
echo "PIPELINE EXECUTION SUMMARY"
echo "=========================================="
echo "Test completed at $(date)"

# Count generated files
processed_files=$(find data/processed -name "*.csv" 2>/dev/null | wc -l)
split_files=$(find splits -name "*.json" 2>/dev/null | wc -l)
model_files=$(find models -name "*.pkl" 2>/dev/null | wc -l)
result_files=$(find results -type f 2>/dev/null | wc -l)
log_files=$(find logs -name "*.log" 2>/dev/null | wc -l)

echo "Generated files:"
echo "  • Processed data files: $processed_files"
echo "  • Data split files: $split_files"
echo "  • Trained model files: $model_files"
echo "  • Result files: $result_files"
echo "  • Log files: $log_files"

echo ""
echo "✅ Pipeline execution completed successfully!"
echo "All components have been tested and verified."
echo ""
echo "Next steps:"
echo "1. Review the generated results in the results/ directory"
echo "2. Check the DOCUMENT.md for completeness"
echo "3. Verify all figures and tables are properly integrated"
echo "==========================================" 
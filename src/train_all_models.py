#!/usr/bin/env python3
"""
COMP4702 Assignment: Complete Model Training Pipeline

Trains all models sequentially with robust error handling and performs comprehensive analysis.

Usage:
    python src/train_all_models.py                    # Train all models
    python src/train_all_models.py lr rf             # Train only LR and RF
    python src/train_all_models.py --help            # Show help
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import json

# Configuration
ALL_MODELS = [
    {
        'name': 'Logistic Regression',
        'key': 'lr',
        'script': 'src/train_lr.py',
        'expected_time': 5  # minutes
    },
    {
        'name': 'Random Forest', 
        'key': 'rf',
        'script': 'src/train_rf.py',
        'expected_time': 15  # minutes
    },
    {
        'name': 'LightGBM',
        'key': 'lgbm',
        'script': 'src/train_lgbm.py', 
        'expected_time': 20  # minutes
    },
    {
        'name': 'Gaussian Process',
        'key': 'gp',
        'script': 'src/train_gp.py',
        'expected_time': 30  # minutes
    }
]

ANALYSIS_SCRIPT = 'src/comprehensive_analysis.py'
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def parse_arguments():
    """Parse command line arguments for flexible pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Train machine learning models for COMP4702 Assignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  lr, logistic        Logistic Regression (~5 min)
  rf, forest          Random Forest (~15 min)  
  lgbm, lightgbm      LightGBM (~20 min)
  gp, gaussian        Gaussian Process (~30 min)

Examples:
  python src/train_all_models.py                 # Train all models
  python src/train_all_models.py lr rf           # Train LR and RF only
  python src/train_all_models.py lgbm gp         # Train LightGBM and GP only
  python src/train_all_models.py --no-analysis  # Train all but skip analysis
        """
    )
    
    parser.add_argument(
        'models', 
        nargs='*', 
        help='Models to train (lr, rf, lgbm, gp). If none specified, trains all models.'
    )
    
    parser.add_argument(
        '--no-analysis', 
        action='store_true',
        help='Skip comprehensive analysis after training'
    )
    
    parser.add_argument(
        '--list-models', 
        action='store_true',
        help='List available models and exit'
    )
    
    return parser.parse_args()

def get_models_to_train(requested_models):
    """
    Resolve user input to specific model configurations.
    
    Args:
        requested_models: User-specified model identifiers
    
    Returns:
        list: Model configuration dictionaries
    """
    if not requested_models:
        # No models specified, train all
        return ALL_MODELS
    
    # Create mapping for model lookup
    model_map = {}
    for model in ALL_MODELS:
        model_map[model['key']] = model
        model_map[model['name'].lower().replace(' ', '')] = model
        # Add some common aliases
        if model['key'] == 'rf':
            model_map['forest'] = model
        elif model['key'] == 'lr':
            model_map['logistic'] = model
        elif model['key'] == 'lgbm':
            model_map['lightgbm'] = model
        elif model['key'] == 'gp':
            model_map['gaussian'] = model
    
    selected_models = []
    invalid_models = []
    
    for requested in requested_models:
        requested_lower = requested.lower()
        if requested_lower in model_map:
            model = model_map[requested_lower]
            if model not in selected_models:  # Avoid duplicates
                selected_models.append(model)
        else:
            invalid_models.append(requested)
    
    if invalid_models:
        print(f"Error: Unknown models: {', '.join(invalid_models)}")
        print(f"Available models: {', '.join([m['key'] for m in ALL_MODELS])}")
        sys.exit(1)
    
    return selected_models

def list_available_models():
    """Print available models and exit"""
    print("Available Models:")
    print("================")
    for model in ALL_MODELS:
        print(f"  {model['key']:8} - {model['name']} (~{model['expected_time']} min)")
    print(f"\nExample: python src/train_all_models.py lr rf")
    sys.exit(0)

def log_message(message, level="INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    
    # Also save to log file
    with open(LOG_DIR / "training_pipeline.log", "a") as f:
        f.write(log_entry + "\n")

def check_prerequisites():
    """Validate pipeline prerequisites before execution."""
    log_message("Checking prerequisites...")
    
    # Check if data files exist
    required_files = [
        "data/processed/assignTTSWING_processed.csv",
        "splits/train_indices.json", 
        "splits/test_indices.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        log_message(f"Missing required files: {missing_files}", "ERROR")
        return False
    
    # Check if model scripts exist
    for model in ALL_MODELS:
        if not Path(model['script']).exists():
            log_message(f"Missing model script: {model['script']}", "ERROR")
            return False
    
    if not Path(ANALYSIS_SCRIPT).exists():
        log_message(f"Missing analysis script: {ANALYSIS_SCRIPT}", "ERROR")
        return False
    
    log_message("✓ All prerequisites satisfied")
    return True

def run_model_training(model):
    """Execute individual model training with comprehensive error handling."""
    log_message(f"Starting {model['name']} training...")
    log_message(f"Expected runtime: ~{model['expected_time']} minutes")
    
    start_time = time.time()
    
    try:
        # Run the model training script (no timeout - let it run as long as needed)
        result = subprocess.run(
            [sys.executable, model['script']],
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # minutes
        
        if result.returncode == 0:
            log_message(f"✓ {model['name']} completed successfully in {elapsed_time:.2f} minutes")
            return {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            log_message(f"✗ {model['name']} failed with return code {result.returncode}", "ERROR")
            log_message(f"STDOUT: {result.stdout}", "DEBUG")
            log_message(f"STDERR: {result.stderr}", "ERROR")
            return {
                'status': 'failed',
                'elapsed_time': elapsed_time,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except Exception as e:
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        log_message(f"✗ {model['name']} failed with exception: {str(e)}", "ERROR")
        return {
            'status': 'exception',
            'elapsed_time': elapsed_time,
            'error': str(e)
        }

def run_comprehensive_analysis():
    """Execute comprehensive model comparison and visualization."""
    log_message("Starting comprehensive analysis...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, ANALYSIS_SCRIPT],
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        
        if result.returncode == 0:
            log_message(f"✓ Comprehensive analysis completed successfully in {elapsed_time:.2f} minutes")
            return {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            log_message(f"✗ Comprehensive analysis failed with return code {result.returncode}", "ERROR")
            log_message(f"STDERR: {result.stderr}", "ERROR")
            return {
                'status': 'failed',
                'elapsed_time': elapsed_time,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except Exception as e:
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        log_message(f"✗ Comprehensive analysis failed with exception: {str(e)}", "ERROR")
        return {
            'status': 'exception',
            'elapsed_time': elapsed_time,
            'error': str(e)
        }

def save_pipeline_results(model_results, analysis_result, total_time, models_to_train):
    """Save comprehensive pipeline execution summary."""
    pipeline_results = {
        'pipeline_start': datetime.now().isoformat(),
        'total_pipeline_time_minutes': total_time,
        'models': {},
        'analysis': analysis_result,
        'summary': {
            'total_models': len(models_to_train),
            'successful_models': sum(1 for r in model_results.values() if r['status'] == 'success'),
            'failed_models': sum(1 for r in model_results.values() if r['status'] != 'success'),
            'analysis_successful': analysis_result['status'] == 'success'
        }
    }
    
    # Add model results
    for i, model in enumerate(models_to_train):
        model_name = model['name'].replace(' ', '_').lower()
        pipeline_results['models'][model_name] = model_results[i]
    
    # Save results
    with open(LOG_DIR / 'pipeline_results.json', 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    
    log_message(f"Pipeline results saved to {LOG_DIR / 'pipeline_results.json'}")

def print_summary(model_results, analysis_result, total_time, models_to_train):
    """Display comprehensive pipeline execution summary."""
    log_message("=" * 80)
    log_message("PIPELINE EXECUTION SUMMARY")
    log_message("=" * 80)
    
    successful_models = []
    failed_models = []
    
    for i, model in enumerate(models_to_train):
        result = model_results[i]
        if result['status'] == 'success':
            successful_models.append(f"{model['name']} ({result['elapsed_time']:.1f}m)")
        else:
            failed_models.append(f"{model['name']} ({result['status']})")
    
    log_message(f"Total pipeline time: {total_time:.2f} minutes")
    log_message(f"Successful models ({len(successful_models)}/{len(models_to_train)}): {', '.join(successful_models)}")
    
    if failed_models:
        log_message(f"Failed models ({len(failed_models)}): {', '.join(failed_models)}", "WARNING")
    
    if analysis_result['status'] == 'success':
        log_message(f"✓ Comprehensive analysis completed ({analysis_result['elapsed_time']:.1f}m)")
    else:
        log_message(f"✗ Comprehensive analysis failed ({analysis_result['status']})", "ERROR")
    
    log_message("=" * 80)
    
    # Provide next steps
    if len(successful_models) > 0:
        log_message("NEXT STEPS:")
        log_message("1. Check results/ directory for model outputs")
        if analysis_result['status'] == 'success':
            log_message("2. View comprehensive analysis in results/comprehensive_analysis/")
        log_message("3. Review logs/ for detailed execution logs")
    else:
        log_message("No models completed successfully. Check logs for debugging.", "ERROR")

def main():
    """Execute complete model training and analysis pipeline."""
    # Parse command line arguments for flexible pipeline execution
    args = parse_arguments()
    
    # Handle special flags
    if args.list_models:
        list_available_models()
    
    # Resolve user model selection to internal configurations
    models_to_train = get_models_to_train(args.models)
    
    pipeline_start = time.time()
    
    log_message("="*80)
    log_message("COMP4702 ASSIGNMENT - MODEL TRAINING PIPELINE")
    log_message("="*80)
    
    if args.models:
        model_names = [m['name'] for m in models_to_train]
        log_message(f"Training selected models: {', '.join(model_names)}")
    else:
        log_message("Training all models")
    
    if args.no_analysis:
        log_message("Analysis will be skipped (--no-analysis flag)")
    
    # Validate all required files and dependencies before starting
    if not check_prerequisites():
        log_message("Pipeline aborted due to missing prerequisites", "ERROR")
        return 1
    
    # Execute model training sequentially with comprehensive error handling
    model_results = {}
    for i, model in enumerate(models_to_train):
        model_results[i] = run_model_training(model)
        
        # Brief pause between models to ensure clean resource separation
        if i < len(models_to_train) - 1:
            log_message("Pausing 10 seconds before next model...")
            time.sleep(10)
    
    # Execute comprehensive analysis pipeline (creates visualizations and comparisons)
    if args.no_analysis:
        log_message("Skipping comprehensive analysis as requested")
        analysis_result = {
            'status': 'skipped',
            'elapsed_time': 0
        }
    else:
        log_message("All model training attempts completed. Starting analysis...")
        analysis_result = run_comprehensive_analysis()
    
    # Calculate total time
    pipeline_end = time.time()
    total_time = (pipeline_end - pipeline_start) / 60
    
    # Document complete pipeline execution with detailed results and timing
    save_pipeline_results(model_results, analysis_result, total_time, models_to_train)
    print_summary(model_results, analysis_result, total_time, models_to_train)
    
    # Return appropriate exit code
    successful_models = sum(1 for r in model_results.values() if r['status'] == 'success')
    if successful_models == 0:
        return 1  # Complete failure
    elif successful_models < len(models_to_train):
        return 2  # Partial success
    else:
        return 0  # Complete success

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
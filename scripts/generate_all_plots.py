#!/usr/bin/env python3
"""
Generate plots for all trained models

This script loads saved models and generates visualizations for:
- Logistic Regression: validation curve, feature importance, confusion matrix
- Random Forest: feature importance, confusion matrix, OOB analysis, optimization history
- LightGBM: feature importance, SHAP analysis, partial dependence, confusion matrix, optimization history
- Sparse GP: training loss, uncertainty distribution, reliability diagram, confusion matrix, confidence analysis
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path, args):
    """
    Run a plotting script with given arguments
    
    Args:
        script_path: Path to the script to run
        args: List of arguments to pass to the script
    """
    try:
        cmd = [sys.executable, str(script_path)] + args
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully completed {script_path.name}")
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path.name}: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error running {script_path.name}: {e}")
        raise

def check_model_exists(model_path):
    """Check if a model file exists"""
    if Path(model_path).exists():
        logger.info(f"Found model: {model_path}")
        return True
    else:
        logger.warning(f"Model not found: {model_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate plots for all trained models')
    parser.add_argument('--data', required=True, help='Path to processed CSV file')
    parser.add_argument('--models_dir', required=True, help='Directory containing saved models')
    parser.add_argument('--splits', nargs=2, required=True, help='Paths to train and val JSON files')
    parser.add_argument('--output_dir', required=True, help='Base directory to save plots')
    parser.add_argument('--scripts_dir', default='scripts', help='Directory containing plotting scripts')
    
    args = parser.parse_args()
    
    # Set up paths
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    scripts_dir = Path(args.scripts_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting plot generation for all models...")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Define model configurations
    model_configs = [
        {
            'name': 'Logistic Regression',
            'script': scripts_dir / 'generate_plots_lr.py',
            'model_file': models_dir / 'lr.pkl',
            'scaler_file': models_dir / 'scaler.pkl',
            'output_subdir': output_dir / 'lr_plots',
            'extra_args': []
        },
        {
            'name': 'Random Forest',
            'script': scripts_dir / 'generate_plots_rf.py',
            'model_file': models_dir / 'rf.pkl',
            'scaler_file': models_dir / 'scaler.pkl',
            'output_subdir': output_dir / 'rf_plots',
            'extra_args': ['--study', str(models_dir / 'rf_study.pkl')] if (models_dir / 'rf_study.pkl').exists() else []
        },
        {
            'name': 'LightGBM',
            'script': scripts_dir / 'generate_plots_lgbm.py',
            'model_file': models_dir / 'lgbm.pkl',
            'scaler_file': models_dir / 'scaler.pkl',
            'output_subdir': output_dir / 'lgbm_plots',
            'extra_args': ['--study', str(models_dir / 'lgbm_study.pkl')] if (models_dir / 'lgbm_study.pkl').exists() else []
        },
        {
            'name': 'Sparse Gaussian Process',
            'script': scripts_dir / 'generate_plots_gp.py',
            'model_file': models_dir / 'gp.pkl',
            'scaler_file': models_dir / 'scaler.pkl',
            'output_subdir': output_dir / 'gp_plots',
            'extra_args': ['--pca', str(models_dir / 'pca.pkl')]
        }
    ]
    
    # Process each model
    success_count = 0
    total_count = len(model_configs)
    
    for config in model_configs:
        logger.info(f"\n--- Processing {config['name']} ---")
        
        # Check if required files exist
        if not config['script'].exists():
            logger.error(f"Script not found: {config['script']}")
            continue
            
        if not check_model_exists(config['model_file']):
            logger.warning(f"Skipping {config['name']} - model not found")
            continue
            
        if not check_model_exists(config['scaler_file']):
            logger.warning(f"Skipping {config['name']} - scaler not found")
            continue
        
        # For GP model, check PCA file
        if config['name'] == 'Sparse Gaussian Process':
            pca_file = models_dir / 'pca.pkl'
            if not check_model_exists(pca_file):
                logger.warning(f"Skipping {config['name']} - PCA transformer not found")
                continue
        
        # Create output subdirectory
        config['output_subdir'].mkdir(parents=True, exist_ok=True)
        
        # Prepare arguments
        script_args = [
            '--data', args.data,
            '--model', str(config['model_file']),
            '--scaler', str(config['scaler_file']),
            '--splits', args.splits[0], args.splits[1],
            '--output_dir', str(config['output_subdir'])
        ]
        
        # Add extra arguments
        script_args.extend(config['extra_args'])
        
        try:
            # Run the plotting script
            run_script(config['script'], script_args)
            success_count += 1
            logger.info(f"✓ Successfully generated plots for {config['name']}")
            
        except Exception as e:
            logger.error(f"✗ Failed to generate plots for {config['name']}: {e}")
            continue
    
    logger.info(f"\n--- Summary ---")
    logger.info(f"Successfully processed: {success_count}/{total_count} models")
    
    if success_count > 0:
        logger.info(f"Plots saved to: {output_dir}")
        logger.info("\nGenerated plot directories:")
        for config in model_configs:
            if config['output_subdir'].exists() and any(config['output_subdir'].iterdir()):
                logger.info(f"  - {config['name']}: {config['output_subdir']}")
    
    if success_count == total_count:
        logger.info("All models processed successfully!")
    elif success_count == 0:
        logger.error("No models were processed successfully!")
        sys.exit(1)
    else:
        logger.warning(f"Some models failed to process ({total_count - success_count} failures)")

if __name__ == "__main__":
    main() 
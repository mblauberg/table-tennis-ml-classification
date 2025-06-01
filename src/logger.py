#!/usr/bin/env python3
"""
Logging System for COMP4702 Assignment

Provides comprehensive logging functionality for tracking execution of all
pipeline components with proper timestamps, system information, and execution metrics.

Features:
- File and console logging with configurable levels
- Execution time tracking decorators
- System information logging
- Structured log formatting
- Automatic directory creation
"""

import logging
import os
import sys
import time
import platform
import functools
from datetime import datetime
from pathlib import Path


def setup_logger(name, log_file, level=logging.INFO, overwrite=False):
    """
    Set up logger with file and console handlers
    
    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level (int): Logging level
        overwrite (bool): Whether to overwrite existing log file
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing handlers if logger already exists
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Create file handler
    file_mode = 'w' if overwrite else 'a'
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_system_info(logger):
    """
    Log comprehensive system information
    
    Args:
        logger (logging.Logger): Logger instance
    """
    logger.info("="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*60)
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)


def log_execution_time(func):
    """
    Decorator to log function execution time and parameters
    
    Args:
        func: Function to decorate
    
    Returns:
        function: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get logger for the module containing the function
        logger = logging.getLogger(func.__module__)
        
        # Log function start
        logger.info(f"Starting {func.__name__}")
        if kwargs:
            logger.debug(f"Function arguments: {kwargs}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log successful completion
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Completed {func.__name__} successfully in {execution_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            # Log error
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


def log_data_info(logger, data, data_name="data"):
    """
    Log information about a dataset
    
    Args:
        logger (logging.Logger): Logger instance
        data: Dataset (pandas DataFrame or similar)
        data_name (str): Name of the dataset for logging
    """
    try:
        logger.info(f"{data_name.upper()} INFORMATION")
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if hasattr(data, 'dtypes'):
            logger.info(f"Column types: {dict(data.dtypes)}")
        
        if hasattr(data, 'isnull'):
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"Null values found: {dict(null_counts[null_counts > 0])}")
            else:
                logger.info("No null values found")
                
    except Exception as e:
        logger.warning(f"Could not log data info for {data_name}: {str(e)}")


def log_model_info(logger, model, model_name="model"):
    """
    Log information about a trained model
    
    Args:
        logger (logging.Logger): Logger instance
        model: Trained model object
        model_name (str): Name of the model for logging
    """
    try:
        logger.info(f"{model_name.upper()} INFORMATION")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Log scikit-learn model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            logger.info(f"Model parameters: {params}")
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            n_features = len(model.feature_importances_)
            logger.info(f"Number of features: {n_features}")
            logger.debug(f"Feature importances shape: {model.feature_importances_.shape}")
        
        # Log classes if available
        if hasattr(model, 'classes_'):
            logger.info(f"Classes: {model.classes_}")
            
    except Exception as e:
        logger.warning(f"Could not log model info for {model_name}: {str(e)}")


def log_performance_metrics(logger, metrics_dict, stage="evaluation"):
    """
    Log performance metrics in a structured format
    
    Args:
        logger (logging.Logger): Logger instance
        metrics_dict (dict): Dictionary of metric names and values
        stage (str): Stage of evaluation (train, validation, test)
    """
    logger.info(f"{stage.upper()} METRICS")
    logger.info("-" * 40)
    
    for metric_name, metric_value in metrics_dict.items():
        if isinstance(metric_value, (list, tuple)) and len(metric_value) == 2:
            # Assume it's a confidence interval
            logger.info(f"{metric_name}: {metric_value[0]:.4f} [{metric_value[1][0]:.4f}, {metric_value[1][1]:.4f}]")
        elif isinstance(metric_value, float):
            logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")
    
    logger.info("-" * 40)


class ExecutionTimer:
    """Context manager for timing code blocks"""
    
    def __init__(self, logger, description):
        self.logger = logger
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.logger.info(f"Starting: {self.description}")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.description} in {execution_time:.2f} seconds")
        else:
            self.logger.error(f"Failed: {self.description} after {execution_time:.2f} seconds")


def setup_pipeline_logging(component_name, logs_dir="logs", level=logging.INFO):
    """
    Set up logging for a pipeline component with standard configuration
    
    Args:
        component_name (str): Name of the pipeline component
        logs_dir (str): Directory for log files
        level (int): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    log_file = Path(logs_dir) / f"{component_name}.log"
    logger = setup_logger(component_name, log_file, level, overwrite=True)
    
    # Log header information
    logger.info(f"Initializing {component_name} pipeline component")
    log_system_info(logger)
    
    return logger


# Example usage and testing
if __name__ == "__main__":
    # Test the logging system
    logger = setup_pipeline_logging("test_logger", "logs")
    
    @log_execution_time
    def test_function(x, y, verbose=False):
        """Test function for demonstrating logging"""
        import time
        logger.info(f"Processing {x} and {y}")
        time.sleep(1)  # Simulate work
        return x + y
    
    # Test function execution logging
    result = test_function(5, 3, verbose=True)
    logger.info(f"Function result: {result}")
    
    # Test performance metrics logging
    metrics = {
        "accuracy": 0.8542,
        "f1_score": 0.7834,
        "precision": 0.8123,
        "recall": 0.7567
    }
    log_performance_metrics(logger, metrics, "test")
    
    # Test execution timer
    with ExecutionTimer(logger, "test operation"):
        time.sleep(0.5)
    
    logger.info("Logging system test completed successfully") 
import logging
import os
from datetime import datetime
from pathlib import Path

# Keep track of created loggers to avoid duplicates
_loggers = {}

# Global logger for backward compatibility (non-experiment files)
logger = logging.getLogger("mamba_ts_forecasting")
logger.setLevel(logging.INFO)

def configure_logger(log_path=None, script_name=None):
    """
    Configure logger with both console and file output (for non-experiment files)
    
    Args:
        log_path: Specific log file path. If None, auto-generates based on script_name
        script_name: Name of the script for auto-generating log filename
    """
    global logger
    
    # Avoid adding multiple handlers to the same logger
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Stream (console) handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_path is None and script_name:
        # For non-experiment files, use the outputs/logs directory
        logs_dir = Path("../outputs/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"{script_name}_{timestamp}.log"
    
    if log_path:
        # Ensure log directory exists
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f"Logging to file: {log_path}")

    # Prevent logs from being passed to the root logger
    logger.propagate = False

    return logger

def get_experiment_logger(script_name):
    """
    Get a configured logger for experiment scripts
    
    Args:
        script_name: Name of the script (e.g., 'experiment_runner', 'main', etc.)
    
    Returns:
        A logger instance that writes to both console and file
    """
    # Return existing logger if already created
    if script_name in _loggers:
        return _loggers[script_name]
    
    # Create new logger instance for this script
    experiment_logger = logging.getLogger(f"mamba_ts_forecasting.{script_name}")
    experiment_logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers to the same logger
    if experiment_logger.handlers:
        _loggers[script_name] = experiment_logger
        return experiment_logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Stream (console) handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    experiment_logger.addHandler(ch)

    # File handler - use absolute path to ensure it works from any directory
    current_dir = Path(__file__).parent  # src directory
    project_root = current_dir.parent    # project root
    logs_dir = project_root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{script_name}_{timestamp}.log"
    
    # Ensure log directory exists
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    experiment_logger.addHandler(fh)
    experiment_logger.info(f"Logging to file: {log_path}")

    # Prevent logs from being passed to the root logger
    experiment_logger.propagate = False
    
    # Store logger in cache
    _loggers[script_name] = experiment_logger
    return experiment_logger
# ========== Standard Library Imports ==========
import os
import json

# ========== Third-Party Imports ==========
import torch
import numpy as np
import torch.nn as nn

# ========== Local Imports ==========
from logger import logger

# Global device cache to avoid repeated detection
_device_cache = None
_device_info_logged = False


def get_device():
    """
    Returns the device to be used for training (GPU or CPU) with detailed logging.
    Uses caching to avoid repeated GPU detection and logging.
    """
    global _device_cache, _device_info_logged

    # Return cached device if already detected
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        _device_cache = torch.device("cuda")

        # Only log detailed GPU info once
        if not _device_info_logged:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.2f} GB) - CUDA {torch.version.cuda}")
            _device_info_logged = True

        return _device_cache
    else:
        if not _device_info_logged:
            logger.warning("CUDA not available. Using CPU for training")
            _device_info_logged = True
        _device_cache = torch.device("cpu")
        return _device_cache


def clear_gpu_cache():
    """
    Clear GPU cache to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")


def calculate_class_weights(y_train):
    """
    Calculate class weights for multi-label classification.
    Args:
        y_train (np.ndarray): Training labels (multi-label, shape: [n_samples, n_classes])
    Returns:
        torch.FloatTensor: Class weights tensor
    """
    n_samples, n_classes = y_train.shape
    positives = np.sum(y_train, axis=0)
    negatives = n_samples - positives
    class_weights = negatives / (positives + 1e-6)
    return torch.FloatTensor(class_weights)


def calculate_focal_loss_alpha(y_train, is_multilabel=False):
    """
    Calculate optimal alpha value for Focal Loss based on class distribution.
    
    For binary classification:
        - Alpha represents weight for positive class (class 1)
        - Calculated as proportion of negative class: alpha = n_negative / n_total
        - This balances the loss contribution from majority vs minority class
    
    For multi-label classification:
        - Returns mean alpha across all classes
        - Each class's alpha is proportion of negative samples for that class
    
    Args:
        y_train (np.ndarray): Training labels
            - Binary: shape [n_samples] with values 0 or 1
            - Multi-label: shape [n_samples, n_classes] with values 0 or 1
        is_multilabel (bool): Whether this is multi-label classification
    
    Returns:
        float: Optimal alpha value for Focal Loss
    """
    if is_multilabel:
        # Multi-label: calculate mean alpha across all classes
        n_samples, n_classes = y_train.shape
        positives = np.sum(y_train, axis=0)
        negatives = n_samples - positives
        # Alpha for each class is proportion of negative samples
        alphas = negatives / n_samples
        # Return mean alpha across classes
        mean_alpha = np.mean(alphas)
        logger.debug(f"Multi-label alpha calculation: per-class alphas = {alphas}, mean = {mean_alpha:.3f}")
        return float(mean_alpha)
    else:
        # Binary classification: alpha = proportion of negative class
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        total = len(y_train)
        
        if len(class_counts) == 2:
            # Calculate proportion of class 0 (negative/normal)
            alpha = class_counts.get(0, 0) / total
            logger.debug(f"Binary alpha calculation: Normal={class_counts.get(0, 0)}, "
                        f"Abnormal={class_counts.get(1, 0)}, alpha={alpha:.3f}")
            return float(alpha)
        else:
            # Fallback to standard value if not exactly 2 classes
            logger.warning(f"Binary classification expected 2 classes, found {len(class_counts)}. Using default alpha=0.25")
            return 0.25


def get_loss_function(config, y_train=None):
    """
    Returns the appropriate loss function (criterion) for training or evaluation.
    Args:
        config (dict): Configuration dictionary.
        y_train (np.ndarray, optional): Training labels for class weights.
    Returns:
        nn.Module: Loss function.
    """
    is_multilabel = config.get('is_multilabel', False)
    use_focal_loss = config.get('use_focal_loss', False)
    device = get_device()
    
    if is_multilabel:
        if use_focal_loss:
            # Import here to avoid circular imports
            from training_utils import FocalLoss
            # Calculate alpha based on actual class distribution
            if y_train is not None:
                alpha = calculate_focal_loss_alpha(y_train, is_multilabel=True)
            else:
                alpha = 0.25  # Default fallback
            criterion = FocalLoss(alpha=alpha, gamma=2, binary_mode=False)
            logger.info(f"Using Focal Loss for multi-label classification (alpha={alpha:.3f}, gamma=2)")
        elif y_train is not None:
            class_weights = calculate_class_weights(y_train).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            logger.info(f"Using BCEWithLogitsLoss with class weights: {class_weights.cpu().numpy()}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss (no class weights)")
    else:
        # Binary classification
        if use_focal_loss:
            from training_utils import FocalLoss
            # Calculate alpha based on PTB-XL class distribution
            if y_train is not None:
                alpha = calculate_focal_loss_alpha(y_train, is_multilabel=False)
                logger.info(f"Using Focal Loss for binary classification (alpha={alpha:.3f}, gamma=2)")
                logger.info(f"Alpha computed from training data to balance class distribution")
            else:
                alpha = 0.44 # Default fallback
                logger.info(f"Using Focal Loss for binary classification (alpha={alpha:.3f}, gamma=2)")
                logger.warning("No training data provided, using default alpha")
            criterion = FocalLoss(alpha=alpha, gamma=2, binary_mode=True)
        elif y_train is not None:
            # Calculate class weights for binary classification
            unique, counts = np.unique(y_train, return_counts=True)
            class_counts = dict(zip(unique, counts))
            # Weight for minority class (typically class 1 - abnormal)
            if len(class_counts) == 2:
                n_samples = len(y_train)
                weight_for_0 = n_samples / (2 * class_counts.get(0, 1))
                weight_for_1 = n_samples / (2 * class_counts.get(1, 1))
                class_weights = torch.FloatTensor([weight_for_0, weight_for_1]).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                logger.info(f"Using CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
                logger.info(f"Class distribution - Normal: {class_counts.get(0, 0)}, Abnormal: {class_counts.get(1, 0)}")
            else:
                criterion = nn.CrossEntropyLoss()
                logger.warning("Binary classification expected 2 classes, using unweighted loss")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using CrossEntropyLoss (no class weights)")
    return criterion


def get_config(config_base_path=None):
    """
    Load configuration for the specified dataset and model.
    Args:
        config_base_path (str, optional): Base path to config.json. Defaults to parent dir.
    Returns:
        dict: Configuration dictionary.
    """
    if not config_base_path:
        config_base_path = os.path.join(os.path.dirname(__file__), "../configs")

    config_path = os.path.abspath(os.path.join(config_base_path, "config.json"))

    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def get_experiments_dir(dataset, model):
    """
    Set up the environment and create a new experiment directory with sequential ordering.
    Args:
        dataset (str): Dataset name.
        model (str): Model name.
    Returns:
        str: Path to the new experiment directory.
    """
    base_dir = os.path.join(r"./experiments", dataset, model)
    os.makedirs(base_dir, exist_ok=True)
    # Get the highest run number
    existing_runs = [
        int(d.split("_")[-1]) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]
    next_run = max(existing_runs, default=0) + 1
    # Create the new experiment directory
    experiment_dir = os.path.join(base_dir, f"run_{next_run}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

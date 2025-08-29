
# ========== Standard Library Imports ==========
import os
import json

# ========== Third-Party Imports ==========
import torch
import numpy as np
import torch.nn as nn

# ========== Local Imports ==========
from logger import logger


def get_device():
    """
    Returns the device to be used for training (GPU or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_class_weights(y_train):
    """
    Calculate class weights for multi-label classification.
    Args:
        y_train (np.ndarray): Training labels (multi-label, shape: [n_samples, n_classes])
    Returns:
        torch.FloatTensor: Class weights tensor
    """
    n_samples, n_classes = y_train.shape
    class_counts = np.sum(y_train, axis=0)
    class_weights = n_samples / (n_classes * (class_counts + 1e-6))  # Avoid divide by zero
    return torch.FloatTensor(class_weights)


def get_loss_function(config, y_train=None):
    """
    Returns the appropriate loss function (criterion) for training or evaluation.
    Args:
        config (dict): Configuration dictionary.
        y_train (np.ndarray, optional): Training labels for class weights (multi-label).
    Returns:
        nn.Module: Loss function.
    """
    is_multilabel = config.get('is_multilabel', False)
    device = get_device()
    if is_multilabel:
        if y_train is not None:
            class_weights = calculate_class_weights(y_train).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            if logger:
                logger.info(f"Using BCEWithLogitsLoss with class weights: {class_weights}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            if logger:
                logger.info("Using BCEWithLogitsLoss (no class weights)")
    else:
        criterion = nn.CrossEntropyLoss()
        if logger:
            logger.info("Using CrossEntropyLoss for binary classification.")
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
        config_base_path = os.path.join(os.path.dirname(__file__), "..")

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
    base_dir = os.path.join(r"./../experiments", dataset, model)
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

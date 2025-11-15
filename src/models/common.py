import os
import json
import torch

from models.lstm.model import LSTMClassifier
from models.mamba.model import MambaClassifier
from logger import logger

from models.hybrid_serial.model import HybridSerialClassifier
from models.hybrid_serial_rev.model import HybridSerialReversedClassifier
from models.hybrid_parallel.model import HybridParallelClassifier
from models.hybrid_crossattn.model import HybridCrossAttentionClassifier
from utils import get_device


def get_model_class(model_type):
    """
    Get the model class based on the model type.

    Args:
        model_type (str): Type of the model ("lstm", "mamba").

    Returns:
        class: Model class corresponding to the model type.
    """
    if model_type.lower() == "lstm":
        return LSTMClassifier
    elif model_type.lower() == "mamba":
        return MambaClassifier
    elif model_type.lower() == "hybrid_serial":
        return HybridSerialClassifier
    elif model_type.lower() == "hybrid_serial_rev":
        return HybridSerialReversedClassifier
    elif model_type.lower() == "hybrid_parallel":
        return HybridParallelClassifier
    elif model_type.lower() == "hybrid_crossattn":
        return HybridCrossAttentionClassifier
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def setup_model(model_type, input_channels, num_classes, config=None):
    """
    Setup model with hyperparameters from config.
    Returns model
    """
    device = get_device()  # This will use cached device without logging
    model_class = get_model_class(model_type)
    
    # Get model-specific hyperparameters from config
    model_params = {}
    if config and 'model_hyperparameters' in config:
        model_params = config['model_hyperparameters'].get(model_type.lower(), {})
        logger.info(f"Using hyperparameters for {model_type.upper()}: {model_params}")
    
    # Create model with hyperparameters
    model = model_class(input_channels, num_classes, **model_params)
    
    # Move model to device and log the information
    model = model.to(device)
    
    # Count and log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model_type.upper()} - Parameters: {total_params:,} (trainable: {trainable_params:,}) - Device: {device}")
    
    return model


def load_model(model_type, input_channels, num_classes, model_dir, config=None):
    """
    Load model weights for evaluation/inference with configurable hyperparameters.
    Returns model
    """
    device = get_device()
    model_class = get_model_class(model_type)
    
    # Try to load config from experiment directory first, then fall back to provided config
    experiment_config_path = os.path.join(model_dir, "config.json")
    experiment_config = None
    
    if os.path.exists(experiment_config_path):
        try:
            with open(experiment_config_path, 'r') as f:
                experiment_config = json.load(f)
            logger.info(f"Loaded experiment config from {experiment_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load experiment config: {e}")
    
    # Use experiment config if available, otherwise fall back to provided config
    config_to_use = experiment_config if experiment_config else config
    
    # Get model-specific hyperparameters from config
    model_params = {}
    if config_to_use and 'model_hyperparameters' in config_to_use:
        model_params = config_to_use['model_hyperparameters'].get(model_type.lower(), {})
        logger.info(f"Using hyperparameters for {model_type.upper()}: {model_params}")
    
    # Create model with hyperparameters and move to device
    model = model_class(input_channels, num_classes, **model_params).to(device)
    
    # Load saved weights
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

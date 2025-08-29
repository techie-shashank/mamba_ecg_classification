import os
import torch

from models.lstm.model import LSTMClassifier
from models.mamba.model import MambaClassifier
from src.utils import get_device


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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def setup_model(model_type, input_channels, num_classes):
    """
    Setup model and criterion for training.
    Returns model, criterion
    """
    device = get_device()
    model_class = get_model_class(model_type)
    model = model_class(input_channels, num_classes).to(device)
    return model


def load_model(model_type, input_channels, num_classes, model_dir):
    """
    Load model weights for evaluation/inference.
    Returns model
    """
    device = get_device()
    model_class = get_model_class(model_type)
    model = model_class(input_channels, num_classes).to(device)
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

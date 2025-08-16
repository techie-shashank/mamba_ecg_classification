import json
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.classifiers.fcn import FCNClassifier
from models.classifiers.lstm import LSTMClassifier
from data.data_loader import load_and_prepare
from utils import load_and_prepare_data, train_model, get_config_for_training, get_model_class
from logger import logger, configure_logger


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--model", type=str, required=True, choices=["fcn", "lstm", "mamba"], help="Model type")
    return parser.parse_args()

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


def calculate_class_weights(y_train):
    n_samples, n_classes = y_train.shape
    class_counts = np.sum(y_train, axis=0)
    class_weights = n_samples / (n_classes * (class_counts + 1e-6))  # Avoid divide by zero
    return torch.FloatTensor(class_weights)


def setup_model(model_name, X_train, y_train, device, learning_rate):
    """
    Initialize the model, criterion, and optimizer.

    Args:
        args (argparse.Namespace): Parsed arguments.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        device (torch.device): Device to load the model on.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tuple: Model, criterion, optimizer.
    """
    sample_X = X_train[0]
    time_steps = sample_X.shape[0]
    input_channels = sample_X.shape[1]
    logger.info(f"Input shape: {sample_X.shape}, Time steps: {time_steps}, Input channels: {input_channels}")
    num_classes = len(y_train[0]) if y_train.ndim > 1 else 2

    model_class = get_model_class(model_name)
    model = model_class(input_channels, time_steps, num_classes).to(device)

    if num_classes > 2:
        class_weights = calculate_class_weights(y_train).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        logger.info(f"Using BCEWithLogitsLoss with class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss for binary classification.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train(dataset_name, model_name, save_dir):
    config = get_config_for_training(dataset=dataset_name, model=model_name)

    # Save the config dictionary as a JSON file
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract parameters from config
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    # Prepare data
    loaders, datasets, data_arrays, metadata = load_and_prepare(dataset_name, config)
    train_loader = loaders['train']
    val_loader = loaders['val']
    X_train, y_train = data_arrays['train']

    # Model setup
    model, criterion, optimizer = setup_model(model_name, X_train, y_train, device, learning_rate)

    # Train the model
    logger.info(f"Starting training for model: {model_name.upper()}")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=epochs, is_multilabel=config['is_multilabel'])

    # Save the model
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    args = parse_arguments()
    experiments_dir = get_experiments_dir(args.dataset, args.model)
    log_path = os.path.join(experiments_dir, "train.log")
    configure_logger(log_path)
    train(args.dataset, args.model, experiments_dir)
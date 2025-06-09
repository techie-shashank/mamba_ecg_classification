import json
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, multilabel_confusion_matrix, ConfusionMatrixDisplay
)
import numpy as np
import logging

from torch.utils.data import DataLoader

from src.data.dataset_handler import DatasetHandler
from src.models.classifiers.fcn import FCNClassifier
from src.models.classifiers.lstm import LSTMClassifier
from src.models.classifiers.mamba_model import MambaClassifier
from src.logger import logger


def get_model_class(model_type):
    """
    Get the model class based on the model type.

    Args:
        model_type (str): Type of the model ("fcn", "lstm").

    Returns:
        class: Model class corresponding to the model type.
    """
    if model_type.lower() == "fcn":
        return FCNClassifier
    elif model_type.lower() == "lstm":
        return LSTMClassifier
    elif model_type.lower() == "mamba":
        return MambaClassifier
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_config_for_training(dataset, model):
    """
    Load configuration for the specified dataset and model.
    Args:
        dataset (str): Dataset name.
        model (str): Model name.
    Returns:
        dict: Configuration dictionary.
    """
    base_config_path = f"./../configs/{dataset}/base.json"
    model_config_path = f"./../configs/{dataset}/{model}.json"

    # Load base configuration
    with open(base_config_path, "r") as base_file:
        base_config = json.load(base_file)

    # Load model-specific configuration
    with open(model_config_path, "r") as model_file:
        model_config = json.load(model_file)

    # Merge configurations
    config = {**base_config, **model_config}
    return config

def get_config_for_testing(config_base_path):
    """
    Load configuration for the specified dataset and model.
    Args:
        dataset (str): Dataset name.
        model (str): Model name.
    Returns:
        dict: Configuration dictionary.
    """
    config_path = os.path.join(config_base_path, "config.json")

    # Load model-specific configuration
    with open(config_path, "r") as model_file:
        config = json.load(model_file)

    return config


def calculate_store_metrics(y_true, y_prob, threshold=0.5, save_dir="metrics_results", class_names=None):
    """
    Evaluate model predictions and save metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Threshold for binary classification.
        save_dir (str): Directory to save the metrics results.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert probabilities to binary predictions
    y_pred = (y_prob > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    try:
        roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        roc_auc = float('nan')  # In case AUC cannot be computed due to label imbalance

    cm = multilabel_confusion_matrix(y_true, y_pred)

    # Save metrics to a JSON file
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Plot confusion matrices
    n_classes = cm.shape[0]
    cols = 4
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n_classes):
        disp = ConfusionMatrixDisplay(cm[i],
                                      display_labels=[f'Not {class_names[i]}' if class_names else 'Negative',
                                                      class_names[i] if class_names else f'Class {i}']
                                      )
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(class_names[i] if class_names else f'Class {i}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    for j in range(n_classes, len(axes)):
        fig.delaxes(axes[j])  # remove extra axes

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.close()

    logger.info("Metrics and confusion matrices saved successfully.")


def load_and_prepare_data(dataset_name, config, split="train", batch_size=None):
    """
    Utility function to load, preprocess, and create datasets and dataloaders.

    Args:
        dataset_name (str): Name of the dataset.
        config (dict): Configuration dictionary.
        split (str): Data split to use ("train", "val", "test").
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: DataLoader for the specified split.
        DatasetHandler: DatasetHandler object for further use.
        np.ndarray: Input data (X) for the specified split.
        np.ndarray: Labels (Y) for the specified split.
    """
    dataset_handler = DatasetHandler(dataset_name=dataset_name, config=config)

    logger.info(f"Loading data for dataset: {dataset_name}")
    X, Y = dataset_handler.load_data()
    logger.info("Loaded raw data: %s", X.shape)
    logger.info("Labels shape: %s", Y.shape)

    logger.info("Splitting data into training, validation, and test sets.")
    X_train, y_train, X_val, y_val, X_test, y_test = dataset_handler.split_data(X, Y)

    if split == "train":
        X_split, y_split = X_train, y_train
    elif split == "val":
        X_split, y_split = X_val, y_val
    elif split == "test":
        X_split, y_split = X_test, y_test
    else:
        raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', or 'test'.")

    logger.info(f"Preprocessing {split} data.")
    X_split, y_split = dataset_handler.preprocess_data(X_split, y_split)
    logger.info(f"{split.capitalize()} data preprocessing completed.")

    logger.info(f"Creating {split} DataLoader.")
    dataset = dataset_handler.get_dataset(X_split, y_split)
    dataloader = DataLoader(dataset, batch_size=batch_size or config["batch_size"], shuffle=(split == "train"))
    logger.info(f"{split.capitalize()} DataLoader created successfully.")

    return dataloader, dataset_handler, X_split, y_split

def setup_model(model_type, dataset_name, num_classes, input_channels, time_steps, device, model_dir):
    """
    Initialize and load the trained model.

    Args:
        model_type (str): Type of the model ("fcn", "lstm", etc.).
        dataset_name (str): Name of the dataset.
        num_classes (int): Number of output classes.
        input_channels (int): Number of input channels.
        time_steps (int): Number of time steps in the input data.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model ready for evaluation.
    """
    model_class = get_model_class(model_type)
    model = model_class(input_channels, time_steps, num_classes).to(device)
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, test_loader, criterion, device, is_multilabel, logger):
    """
    Evaluate the model on the test set.

    Args:
        model (torch.nn.Module): Loaded model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        is_multilabel (bool): Whether the task is multilabel classification.
        logger (logging.Logger): Logger for logging evaluation metrics.

    Returns:
        tuple: (average test loss, test accuracy, all probabilities, all labels)
    """
    test_loss = 0
    correct, total = 0, 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_multilabel:
                y_batch = y_batch.float()
            else:
                y_batch = y_batch.long()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            if is_multilabel:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                correct += (preds == y_batch.int()).all(dim=1).sum().item()
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()

            total += y_batch.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    logger.info(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return avg_test_loss, test_acc, all_probs, all_labels


def evaluate_and_save_metrics(dataset_name, model_name, model, test_loader, criterion, device, is_multilabel, logger, dataset_handler, save_dir):
    """
    Evaluate the model and save metrics.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        model (torch.nn.Module): Loaded model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        is_multilabel (bool): Whether the task is multilabel classification.
        logger (logging.Logger): Logger for logging evaluation metrics.
        dataset_handler (DatasetHandler): Dataset handler object.

    Returns:
        None
    """
    avg_test_loss, test_acc, all_probs, all_labels = evaluate_model(
        model, test_loader, criterion, device, is_multilabel, logger
    )

    # Concatenate all batches

    # If multiclass single-label (CrossEntropy), convert to one-hot for evaluation
    if not is_multilabel:
        num_classes = all_probs.shape[1]
        y_true_onehot = np.zeros_like(all_probs)
        y_true_onehot[np.arange(all_labels.shape[0]), all_labels] = 1
        all_labels = y_true_onehot

    # Evaluate and save metrics
    calculate_store_metrics(
        all_labels, all_probs, threshold=0.5, save_dir=f"{save_dir}/metrics_results",
        class_names=list(dataset_handler.handler.mlb.classes_) if is_multilabel else dataset_handler.handler.binary_classes
    )


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, is_multilabel=False):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch = y_batch.float() if is_multilabel else y_batch.long()

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_batch = y_batch.float() if is_multilabel else y_batch.long()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                if is_multilabel:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()
                    correct += (preds == y_batch.int()).all(dim=1).sum().item()
                    total += y_batch.size(0)
                else:
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        logger.info(f"[Epoch {epoch+1}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}%")

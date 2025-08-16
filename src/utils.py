import json
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             roc_auc_score, hamming_loss, multilabel_confusion_matrix,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
import numpy as np
import logging

from torch.utils.data import DataLoader

from data.data_loader import load_and_prepare
from models.classifiers.fcn import FCNClassifier
from models.classifiers.lstm import LSTMClassifier
from models.classifiers.mamba_model import MambaClassifier
from logger import logger


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


def plot_multilabel_confusion_matrices(multilabel_cm, class_names, save_path):
    n_classes = len(class_names)
    cols = 4
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n_classes):
        disp = ConfusionMatrixDisplay(multilabel_cm[i],
                                      display_labels=[f'Not {class_names[i]}', class_names[i]])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(class_names[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    for j in range(n_classes, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_binary_classification_metrics(y_true_binary, y_pred_binary, y_prob_binary, class_names):
    """Binary classification metrics for Normal vs Abnormal"""

    try:
        auc = roc_auc_score(y_true_binary, y_prob_binary)
    except ValueError:
        auc = float('nan')

    metrics = {
        "binary_accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "binary_f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "binary_precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "binary_recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "binary_auc": auc,
        "confusion_matrix": confusion_matrix(y_true_binary, y_pred_binary).tolist(),
        "classification_report": classification_report(y_true_binary, y_pred_binary, target_names=class_names, zero_division=0, output_dict=True)
    }
    return metrics

def calculate_multilabel_metrics(y_true_multi, y_pred_multi, y_prob_multi, class_names):
    """Multi-label metrics for diagnostic superclasses"""

    try:
        macro_roc_auc = roc_auc_score(y_true_multi, y_prob_multi, average='macro')
    except ValueError:
        macro_roc_auc = float('nan')

    metrics = {
        "macro_f1": f1_score(y_true_multi, y_pred_multi, average='macro', zero_division=0),
        "macro_precision": precision_score(y_true_multi, y_pred_multi, average='macro', zero_division=0),
        "macro_recall": recall_score(y_true_multi, y_pred_multi, average='macro', zero_division=0),
        "macro_roc_auc": macro_roc_auc,
        "hamming_loss": hamming_loss(y_true_multi, y_pred_multi),
        "subset_accuracy": accuracy_score(y_true_multi, y_pred_multi),
        "classification_report": classification_report(y_true_multi, y_pred_multi, target_names=class_names, output_dict=True, zero_division=0),
        "multilabel_confusion_matrix": multilabel_confusion_matrix(y_true_multi, y_pred_multi).tolist()
    }
    return metrics


def calculate_store_metrics(y_true, y_prob, y_pred, save_dir="metrics_results", class_names=None, is_multilabel=False):
    """
    Evaluate model predictions and save metrics, including classification report.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Threshold for binary classification.
        save_dir (str): Directory to save the metrics results.
        class_names (list, optional): List of class names.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if is_multilabel:
        print(f"[Evaluation] Running Multi-Label Classification Evaluation...")

        metrics = calculate_multilabel_metrics(
            y_true_multi=y_true,
            y_pred_multi=y_pred,
            y_prob_multi=y_prob,
            class_names=class_names
        )

        print("\n[Multi-Label Classification Results]")
        print(f"Macro Precision : {metrics['macro_precision']:.4f}")
        print(f"Macro Recall    : {metrics['macro_recall']:.4f}")
        print(f"Macro F1 Score  : {metrics['macro_f1']:.4f}")
        print(f"Macro ROC AUC   : {metrics['macro_roc_auc']:.4f}")

        # Optional: Save confusion matrix plot for multi-label
        plot_multilabel_confusion_matrices(
            np.array(metrics['multilabel_confusion_matrix']),
            class_names,
            os.path.join(save_dir, 'multilabel_confusion_matrices.png')
        )

    else:
        print(f"[Evaluation] Running Binary Classification Evaluation...")

        metrics = calculate_binary_classification_metrics(
            y_true_binary=y_true,
            y_pred_binary=y_pred,
            y_prob_binary=y_prob,
            class_names=class_names
        )

        # ✅ Print Important Binary Metrics
        print("\n[Binary Classification Results]")
        print(f"Accuracy      : {metrics['binary_accuracy']:.4f}")
        print(f"Precision     : {metrics['binary_precision']:.4f}")
        print(f"Recall        : {metrics['binary_recall']:.4f}")
        print(f"F1 Score      : {metrics['binary_f1']:.4f}")
        print(f"ROC AUC       : {metrics['binary_auc']:.4f}")

    # Save metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"[Evaluation] Metrics saved at {save_dir}")


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
    data_loader, X, Y = load_and_prepare(dataset_name, config)
    return data_loader, dataset_name, X, Y

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
    all_preds = []

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
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    logger.info(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    return avg_test_loss, test_acc, all_probs, all_labels, all_preds


def evaluate_and_save_metrics(model, test_loader, criterion, device, is_multilabel, classes, save_dir):
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
    avg_test_loss, test_acc, all_probs, all_labels, all_preds = evaluate_model(
        model, test_loader, criterion, device, is_multilabel, logger
    )

    # Evaluate and save metrics
    calculate_store_metrics(
        all_labels, all_probs, all_preds, save_dir=f"{save_dir}/metrics_results",
        class_names=classes, is_multilabel=is_multilabel
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

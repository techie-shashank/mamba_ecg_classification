
# ========== Standard Library Imports ==========
import os
import json

# ========== Third-Party Imports ==========
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ========== Local Imports ==========
from evaluation.metrics import calculate_binary_classification_metrics, calculate_multilabel_metrics
from evaluation.plots import plot_multilabel_confusion_matrices
from utils import get_device
from logger import logger


def evaluate_and_save_metrics(
    model: torch.nn.Module,
    criterion,
    test_loader,
    config,
    classes,
    save_dir: str,
    logger
) -> None:
    """
    Evaluate the model and save metrics to disk.
    Args:
        model (torch.nn.Module): Trained model.
        criterion: Loss function.
        test_loader: DataLoader for test set.
        config (dict): Configuration dict.
        classes (list): Class names.
        save_dir (str): Directory to save metrics.
        logger: Logger instance.
    """
    is_multilabel = config['is_multilabel']
    _, _, all_probs, all_labels, all_preds = evaluate_model(
        model, test_loader, criterion, is_multilabel
    )
    calculate_store_metrics(
        all_labels, all_probs, all_preds, save_dir=f"{save_dir}/metrics_results",
        class_names=classes, is_multilabel=is_multilabel
    )


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    criterion,
    is_multilabel
) -> tuple:
    """
    Evaluate the model on the test set and return (avg_test_loss, test_acc, all_probs, all_labels, all_preds).
    Args:
        model (torch.nn.Module): Trained model.
        test_loader: DataLoader for test set.
        criterion: Loss function.
        is_multilabel (bool): Whether the task is multi-label.
    Returns:
        tuple: (avg_test_loss, test_acc, all_probs, all_labels, all_preds)
    """
    device = get_device()
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            # Ensure labels are the correct type for the loss function
            if is_multilabel:
                labels_for_loss = labels.float()
            else:
                labels_for_loss = labels.long()
            outputs = model(inputs)
            if is_multilabel:
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                correct += (preds == labels.cpu().numpy().astype(int)).all(axis=1).sum()
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                correct += (preds == labels.cpu().numpy()).sum()
                labels = labels.cpu().numpy()
            loss = criterion(outputs, labels_for_loss)
            test_loss += loss.item()
            all_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
            total += len(labels)
    if total == 0:
        logger.warning("Test set is empty. No evaluation performed.")
        return 0.0, 0.0, np.array([]), np.array([]), np.array([])
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    logger.info(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    return avg_test_loss, test_acc, all_probs, all_labels, all_preds


def calculate_store_metrics(
    y_true,
    y_prob,
    y_pred,
    save_dir: str = "metrics_results",
    class_names=None,
    is_multilabel: bool = False,
) -> None:
    """
    Evaluate model predictions and save metrics, including classification report.
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        y_pred: Predicted labels.
        save_dir (str): Directory to save metrics.
        class_names (list, optional): Class names.
        is_multilabel (bool): Whether the task is multi-label.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if is_multilabel:
        logger.info("[Evaluation] Running Multi-Label Classification Evaluation...")
        metrics = calculate_multilabel_metrics(
            y_true_multi=y_true,
            y_pred_multi=y_pred,
            y_prob_multi=y_prob,
            class_names=class_names
        )
        logger.info("\n[Multi-Label Classification Results]")
        logger.info(f"Macro Precision : {metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall    : {metrics['macro_recall']:.4f}")
        logger.info(f"Macro F1 Score  : {metrics['macro_f1']:.4f}")
        logger.info(f"Macro ROC AUC   : {metrics['macro_roc_auc']:.4f}")
        plot_multilabel_confusion_matrices(
            np.array(metrics['multilabel_confusion_matrix']),
            class_names,
            os.path.join(save_dir, 'multilabel_confusion_matrices.png')
        )
    else:
        logger.info("[Evaluation] Running Binary Classification Evaluation...")
        metrics = calculate_binary_classification_metrics(
            y_true_binary=y_true,
            y_pred_binary=y_pred,
            y_prob_binary=y_prob,
            class_names=class_names
        )
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved at {cm_path}")

    logger.info("\n[Binary Classification Results]")
    logger.info(f"Accuracy      : {metrics['binary_accuracy']:.4f}")
    logger.info(f"Precision     : {metrics['binary_precision']:.4f}")
    logger.info(f"Recall        : {metrics['binary_recall']:.4f}")
    logger.info(f"F1 Score      : {metrics['binary_f1']:.4f}")
    logger.info(f"ROC AUC       : {metrics['binary_auc']:.4f}")
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"[Evaluation] Metrics saved at {save_dir}")

# Metric calculation functions for evaluation

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, multilabel_confusion_matrix,
    classification_report
)

def calculate_binary_classification_metrics(y_true_binary, y_pred_binary, y_prob_binary, class_names):
    """
    Compute binary classification metrics for Normal vs Abnormal.
    Args:
        y_true_binary (array-like): Ground truth binary labels.
        y_pred_binary (array-like): Predicted binary labels.
        y_prob_binary (array-like): Predicted probabilities.
        class_names (list): Class names.
    Returns:
        dict: Dictionary of binary classification metrics.
    """
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
        "classification_report": classification_report(
            y_true_binary, 
            y_pred_binary, 
            target_names=class_names, 
            labels=list(range(len(class_names))),  # Specify expected labels
            zero_division=0, 
            output_dict=True
        )
    }
    return metrics

def calculate_multilabel_metrics(y_true_multi, y_pred_multi, y_prob_multi, class_names):
    """
    Compute multi-label metrics for diagnostic superclasses.
    Args:
        y_true_multi (array-like): Ground truth multi-labels.
        y_pred_multi (array-like): Predicted multi-labels.
        y_prob_multi (array-like): Predicted probabilities.
        class_names (list): Class names.
    Returns:
        dict: Dictionary of multi-label classification metrics.
    """
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
        "classification_report": classification_report(
            y_true_multi, 
            y_pred_multi, 
            target_names=class_names, 
            labels=list(range(len(class_names))),  # Specify expected labels
            output_dict=True, 
            zero_division=0
        ),
        "multilabel_confusion_matrix": multilabel_confusion_matrix(y_true_multi, y_pred_multi).tolist()
    }
    return metrics

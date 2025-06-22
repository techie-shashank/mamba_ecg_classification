import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def normalize_ecg_signals(X, eps=1e-8):
    """
    Normalize ECG signals using Z-score normalization.
    Args:
        X (np.ndarray): Raw ECG signals.
        eps (float): Small value to avoid division by zero.
    Returns:
        np.ndarray: Normalized ECG signals.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std < eps, eps, std)  # Prevent division by zero
    return (X - mean) / std


def filter_non_empty_labels(X, Y, label_col='diagnostic'):
    """
    Filter out rows where the label column is empty or not a list/tuple.

    Args:
        X (np.ndarray): Raw ECG signals.
        Y (pd.DataFrame): Annotation DataFrame.
        label_col (str): Name of the label column containing lists of labels.

    Returns:
        tuple: Filtered X and Y DataFrame.
    """
    non_empty_mask = Y[label_col].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)

    # Find ecg_ids removed
    indexes_removed = Y.index[~non_empty_mask].tolist()
    logging.info(f"Removed Indexes (empty labels): {indexes_removed}")

    Y_filtered = Y[non_empty_mask]
    X_filtered = X[non_empty_mask.values]
    return X_filtered, Y_filtered

def preprocess_data(X, Y, mlb=None):
    """
    Preprocess ECG data and annotations.
    Args:
        X (np.ndarray): Raw ECG signals.
        Y (pd.DataFrame): Annotation DataFrame.
        mlb (MultiLabelBinarizer, optional): Pre-fitted MultiLabelBinarizer.
    Returns:
        tuple: Preprocessed ECG signals, binarized labels, and MultiLabelBinarizer.
    """
    X, Y = filter_non_empty_labels(X, Y, 'diagnostic_superclass')
    logging.info("Empty labels removed - X shape: %s, Y shape: %s", X.shape, Y.shape)
    Y = Y.diagnostic_superclass

    # Binarize labels
    if mlb is None:
        mlb = MultiLabelBinarizer()
        Y_bin = mlb.fit_transform(Y)
    else:
        Y_bin = mlb.transform(Y)

    # Normalize ECG signals
    X_norm = normalize_ecg_signals(X)
    return X_norm, Y_bin, mlb


def preprocess_data_binary(X, Y):
    """
    Preprocess ECG data and annotations for binary classification: Normal (0) vs Abnormal (1).

    Args:
        X (np.ndarray): Raw ECG signals, shape (n_samples, n_channels, n_timesteps)
        Y (list of list or pd.Series): Multi-label list of SCP codes per sample, e.g., [['NORM'], ['MI', 'STTC'], ...]

    Returns:
        tuple: Normalized ECG signals, binary labels (0=normal, 1=abnormal)
    """
    import ast

    X, Y = filter_non_empty_labels(X, Y, 'diagnostic_superclass')
    Y = Y.diagnostic_superclass
    binary_labels = []
    for labels in Y:
        # If Y is string representations of list, convert to list
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        # Define normal = only 'NORM' present, otherwise abnormal
        if len(labels) == 1 and 'NORM' in labels:
            binary_labels.append(0)
        else:
            binary_labels.append(1)

    X_norm = normalize_ecg_signals(X)
    y_binary = np.array(binary_labels)
    return X_norm, y_binary, ["Normal", "Abnormal"]
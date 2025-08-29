

# ========== PTB-XL Data Utilities ==========
"""
Unified PTB-XL Data Utilities
----------------------------
This module combines dataset, preprocessing, loading, splitting, and visualization utilities for PTB-XL ECG data.
"""


# ========== Standard Library Imports ==========
import os
import ast
import logging
from collections import Counter

# ========== Third-Party Imports ==========
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ===================== Dataset Class =====================

class PTBXL(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# ===================== Preprocessing Functions =====================
def normalize_ecg_signals(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize ECG signals using Z-score normalization.
    Args:
        X: Raw ECG signals.
        eps: Small value to avoid division by zero.
    Returns:
        Normalized ECG signals.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std < eps, eps, std)
    return (X - mean) / std

def filter_non_empty_labels(X: np.ndarray, Y: pd.DataFrame, label_col: str = 'diagnostic') -> tuple[np.ndarray, pd.DataFrame]:
    """
    Filter out rows where the label column is empty or not a list/tuple.
    Returns filtered X and Y.
    """
    non_empty_mask = Y[label_col].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)
    indexes_removed = Y.index[~non_empty_mask].tolist()
    logging.info(f"Removed Indexes (empty labels): {indexes_removed}")
    Y_filtered = Y[non_empty_mask]
    X_filtered = X[non_empty_mask.values]
    return X_filtered, Y_filtered

def preprocess_label_multilabel(Y: pd.DataFrame) -> tuple[np.ndarray, MultiLabelBinarizer]:
    """
    Binarize multi-labels for multi-class classification.
    Returns binarized labels and the MultiLabelBinarizer.
    """
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(Y['diagnostic_superclass'])
    return y, mlb

def preprocess_labels_binary(Y: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Convert labels to binary: Normal (0) vs Abnormal (1).
    Returns binary labels and class names.
    """
    binary_labels = []
    for labels in Y['diagnostic_superclass']:
        if isinstance(labels, str):
            try:
                labels = ast.literal_eval(labels)
            except Exception as e:
                logging.warning(f"Could not parse label: {labels} ({e})")
                labels = []
        if len(labels) == 1 and 'NORM' in labels:
            binary_labels.append(0)
        else:
            binary_labels.append(1)
    return np.array(binary_labels), ["Normal", "Abnormal"]

def preprocess_labels(X: np.ndarray, Y: pd.DataFrame, config: dict) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    """
    Preprocess ECG data and annotations according to config.
    Returns filtered X, Y, processed y, and class names.
    """
    X, Y = filter_non_empty_labels(X, Y, 'diagnostic_superclass')
    logging.info("Empty labels removed - X shape: %s, Y shape: %s", X.shape, Y.shape)
    is_multilabel = config.get("is_multilabel", False)
    if is_multilabel:
        y, mlb = preprocess_label_multilabel(Y)
        classes = list(mlb.classes_)
    else:
        y, binary_classes = preprocess_labels_binary(Y)
        classes = binary_classes
    return X, Y, y, classes

def apply_standardizer(X: np.ndarray, ss: StandardScaler) -> np.ndarray:
    """
    Apply a fitted StandardScaler to a batch of ECG signals, preserving original shape.
    Args:
        X: Array of ECG signals (samples, ...).
        ss: Fitted StandardScaler instance.
    Returns:
        Standardized ECG signals with the same shape as input.
    """
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp

def preprocess_signals(X_train: np.ndarray, X_validation: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize train, validation, and test ECG signals using a scaler fit on the training set.
    Args:
        X_train: Training ECG signals.
        X_validation: Validation ECG signals.
        X_test: Test ECG signals.
    Returns:
        Tuple of standardized (X_train, X_validation, X_test).
    """
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

# ===================== Data Loading Functions =====================

def load_annotations(data_dir: str, limit: int = None) -> pd.DataFrame:
    """
    Load PTB-XL annotation dataframe and parse scp_codes column.
    Args:
        data_dir: Directory containing ptbxl_database.csv.
        limit: Optional row limit for debugging or subsampling.
    Returns:
        DataFrame with parsed scp_codes.
    """
    try:
        df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    except Exception as e:
        logging.error(f"Failed to load ptbxl_database.csv: {e}")
        raise
    def safe_eval(x):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            logging.warning(f"Could not parse scp_codes: {x} ({e})")
            return []
    df.scp_codes = df.scp_codes.apply(safe_eval)
    if limit:
        df = df[:limit]
    return df


def load_signals(data_dir: str, sampling_rate: int, annotation_df: pd.DataFrame) -> np.ndarray:
    """
    Load raw ECG signals for PTB-XL records listed in annotation_df.
    Args:
        data_dir: Directory containing signal files.
        sampling_rate: 100 or 500 (determines filename column).
        annotation_df: DataFrame with file paths.
    Returns:
        Array of ECG signals.
    """
    file_column = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    signals = []
    for f in annotation_df[file_column]:
        try:
            signal, _ = wfdb.rdsamp(os.path.join(data_dir, f))
            signals.append(signal)
        except Exception as e:
            logging.warning(f"Could not load file {f}: {e}")
    return np.array(signals)


def load_aggregation_map(data_dir: str) -> pd.DataFrame:
    """
    Load diagnostic aggregation mapping for PTB-XL.
    Args:
        data_dir: Directory containing scp_statements.csv.
    Returns:
        DataFrame with only diagnostic rows.
    """
    try:
        agg_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
    except Exception as e:
        logging.error(f"Failed to load scp_statements.csv: {e}")
        raise
    return agg_df[agg_df.diagnostic == 1.0]


def aggregate_superclasses(annotation_df: pd.DataFrame, agg_map: pd.DataFrame) -> pd.Series:
    """
    Aggregate diagnostic labels into superclasses for each record.
    Args:
        annotation_df: DataFrame with scp_codes column.
        agg_map: DataFrame with diagnostic_class mapping.
    Returns:
        Series of lists of diagnostic superclasses.
    """
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_map.index:
                c = agg_map.loc[key].diagnostic_class
                if str(c) != 'nan':
                    tmp.append(c)
        return list(set(tmp))
    return annotation_df.scp_codes.apply(aggregate_diagnostic)


def load_data(data_dir: str, sampling_rate: int, limit: int = None) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load both annotation data and raw ECG signals for PTB-XL.
    Args:
        data_dir: Directory containing PTB-XL files.
        sampling_rate: 100 or 500.
        limit: Optional row limit.
    Returns:
        Tuple of (signals array, annotation DataFrame with diagnostic_superclass column).
    """
    annotation_df = load_annotations(data_dir, limit=limit)
    signals = load_signals(data_dir, sampling_rate, annotation_df)
    agg_map = load_aggregation_map(data_dir)
    annotation_df['diagnostic_superclass'] = aggregate_superclasses(annotation_df, agg_map)
    return signals, annotation_df

# ===================== Data Splitting =====================
def split_train_test(
    X: np.ndarray, Y: pd.DataFrame, y: np.ndarray, test_fold: int = 10, val_ratio: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = y[(Y.strat_fold != test_fold)]
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = y[Y.strat_fold == test_fold]
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=1 - val_ratio, random_state=42
    )
    # logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# ===================== Visualization =====================
def visualize_raw_data_distribution(data_tuple: tuple[np.ndarray, pd.DataFrame]) -> None:
    """
    Visualize the raw diagnostic superclass length distribution for the PTB-XL dataset.
    Uses matplotlib and seaborn for plotting.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    X, Y = data_tuple
    all_labels = [label for sublist in Y['diagnostic_superclass'] for label in sublist]
    label_counts = Counter(all_labels)
    logging.info(f"Label distribution: {label_counts}")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), palette='viridis')
    plt.title('PTB-XL Diagnostic Superclass Distribution')
    plt.xlabel('Diagnostic Superclass')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ===================== Main Data Preparation Interface =====================

def load_and_prepare_ptbxl(config: dict) -> tuple[dict, dict, dict, dict]:
    """
    Load, preprocess, split, and prepare the PTBXL dataset.
    Returns data loaders, datasets, data arrays, and metadata.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        tuple: (data_loaders, datasets, data_arrays, metadata)
    """
    signals, annotation_df = load_data(
        data_dir=config["data_dir"],
        sampling_rate=config["sampling_rate"],
        limit=config.get("limit")
    )
    # visualize_raw_data_distribution((signals, annotation_df))  # Uncomment to visualize
    X, Y, y, classes = preprocess_labels(signals, annotation_df, config)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_test(X, Y, y)
    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)
    train_dataset = PTBXL(X_train, y_train)
    val_dataset = PTBXL(X_val, y_val)
    test_dataset = PTBXL(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    data_arrays = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    metadata = {"classes": classes}
    return data_loaders, datasets, data_arrays, metadata

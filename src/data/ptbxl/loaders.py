import pandas as pd
import os
import ast
import wfdb
import numpy as np


def load_annotations(data_dir, limit=None):
    """
    Load annotation data from the PTB-XL dataset.
    Args:
        data_dir (str): Directory containing the PTB-XL dataset.
        limit (int, optional): Limit the number of records to load.
    Returns:
        pd.DataFrame: Annotation DataFrame.
    """
    Y = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    Y = Y.reset_index()
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    if limit:
        Y = Y[:limit]
    return Y

def load_raw_data(data_dir, sampling_rate, df):
    """
    Load raw ECG signal data.
    Args:
        data_dir (str): Directory containing the PTB-XL dataset.
        df (pd.DataFrame): DataFrame containing file paths.
    Returns:
        np.ndarray: Array of raw ECG signals.
    """
    file_column = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    data = [
        wfdb.rdsamp(
            os.path.join(data_dir, f)
        ) for f in df[file_column]
    ]
    return np.array([signal for signal, meta in data])

def load_aggregation_mapping(data_dir):
    """
    Load diagnostic aggregation mapping from scp_statements.csv.
    Returns:
        data_dir (str): Directory containing the PTB-XL dataset.
        pd.DataFrame: Aggregation DataFrame.
    """
    agg_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
    return agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic_labels(Y, agg_df):
    """
    Aggregate diagnostic labels into superclasses.
    Args:
        Y (pd.DataFrame): Annotation DataFrame.
        agg_df (pd.DataFrame): Aggregation DataFrame.
    Returns:
        pd.Series: Aggregated diagnostic superclasses.
    """
    def aggregate(y_dic):
        return list(set(agg_df.loc[key].diagnostic_class for key in y_dic.keys() if key in agg_df.index))
    return Y.scp_codes.apply(aggregate)

def load_data(data_dir, sampling_rate, limit=None):
    """
    Load both annotation data and raw ECG signals.
    Args:
        data_dir (str): Directory containing the PTB-XL dataset.
        limit (int, optional): Limit the number of records to load.
    Returns:
        tuple: Raw ECG signals (X) and annotation DataFrame (Y).
    """
    # Load annotations
    Y = load_annotations(data_dir, limit=limit)

    # Load raw ECG data
    X = load_raw_data(data_dir, sampling_rate, Y)

    # Load aggregation mapping
    agg_df = load_aggregation_mapping(data_dir)

    # Aggregate diagnostic labels
    Y['diagnostic_superclass'] = aggregate_diagnostic_labels(Y, agg_df)

    return X, Y



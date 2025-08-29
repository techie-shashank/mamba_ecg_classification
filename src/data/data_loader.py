
# ========== Local Imports ==========
from data.ptbxl import load_and_prepare_ptbxl

def load_and_prepare(dataset_name, config):
    """
    Load, preprocess, split, and prepare the dataset.
    Args:
        dataset_name (str): Name of the dataset.
        config (dict): Configuration dictionary.
    Returns:
        tuple: (data_loaders, datasets, data_arrays, metadata)
    """
    if dataset_name == "ptbxl":
        return load_and_prepare_ptbxl(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

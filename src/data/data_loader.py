from data.ptbxl import load_and_prepare_ptbxl


def load_and_prepare(dataset_name, config):
    """
    Load, preprocess, split, and prepare the dataset.

    Args:
        dataset_name (str): Name of the dataset.
        config (dict): Configuration dictionary.
        split (str): Data split ('train', 'val', 'test').
        batch_size (int): Batch size for data loading.

    Returns:
        tuple: DataLoader, input data (X), and labels (Y).
    """
    if dataset_name == "ptbxl":
        return load_and_prepare_ptbxl(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

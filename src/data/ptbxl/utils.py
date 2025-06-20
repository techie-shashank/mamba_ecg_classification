import numpy as np
from sklearn.model_selection import train_test_split

from src.logger import logger


def split_train_test(X, Y, test_fold=10, val_ratio=0.5):
    """
    Split data into train, validation, and test sets based on strat_fold.
    Args:
        X (np.ndarray): Raw ECG signals.
        Y (pd.DataFrame): Annotation DataFrame.
        test_fold (int): Fold to use as the test set.
        val_ratio (float): Proportion of test data to use as validation.
    Returns:
        tuple: Train, validation, and test splits (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)]

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold]

    # Split test data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=1 - val_ratio, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

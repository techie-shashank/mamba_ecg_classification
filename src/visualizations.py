# ========== Standard Library Imports ==========
import os

# ========== Third-Party Imports ==========
import numpy as np
import matplotlib.pyplot as plt

# ========== Local Imports ===============
from logger import logger

def plot_data_samples(X, y, num_samples=5, channel=0, title_prefix="Sample", save_dir=None):
    """
    Plot a few input samples and their labels.
    Args:
        X: np.ndarray or torch.Tensor, shape (N, T, C) or (N, C, T)
        y: np.ndarray or torch.Tensor, shape (N,)
        num_samples: int, number of samples to plot
        channel: int, which channel to plot if multichannel
        title_prefix: str, prefix for plot titles
    """
    if hasattr(X, 'cpu'):
        X = X.cpu().numpy()
    if hasattr(y, 'cpu'):
        y = y.cpu().numpy()
    for i in range(min(num_samples, len(X))):
        # Determine shape: (T, C) or (C, T)
        sample = X[i]
        if sample.shape[0] < sample.shape[-1]:
            # (T, C)
            n_channels = sample.shape[1]
            time = np.arange(sample.shape[0])
            signals = [sample[:, ch] for ch in range(n_channels)]
        else:
            # (C, T)
            n_channels = sample.shape[0]
            time = np.arange(sample.shape[1])
            signals = [sample[ch, :] for ch in range(n_channels)]

        fig, axes = plt.subplots(n_channels, 1, figsize=(8, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]
        for ch in range(n_channels):
            axes[ch].plot(time, signals[ch])
            axes[ch].set_ylabel(f"Ch {ch+1}")
        axes[0].set_title(f"{title_prefix} {i} | Label: {y[i]}")
        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_{i}.png")
            plt.savefig(save_path)
            logger.info(f"Saved data sample plot to {save_path}")
        else:
            plt.show()
            logger.info(f"Displayed data sample plot for {title_prefix} {i} | Label: {y[i]}")
        plt.close(fig)

# Prediction vs ground truth plotting
def plot_predictions_vs_ground_truth(X, y_true, y_pred, num_samples=1, channel=0, save_dir=None):
    """
    Plot model predictions vs. ground truth for a few samples.
    """
    if hasattr(X, 'cpu'):
        X = X.cpu().numpy()
    for i in range(min(num_samples, len(X))):
        sample = X[i]
        if sample.shape[0] < sample.shape[-1]:
            n_channels = sample.shape[1]
            time = np.arange(sample.shape[0])
            signals = [sample[:, ch] for ch in range(n_channels)]
        else:
            n_channels = sample.shape[0]
            time = np.arange(sample.shape[1])
            signals = [sample[ch, :] for ch in range(n_channels)]

        fig, axes = plt.subplots(n_channels, 1, figsize=(8, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]
        for ch in range(n_channels):
            axes[ch].plot(time, signals[ch])
            axes[ch].set_ylabel(f"Ch {ch+1}")
        axes[0].set_title(f"Sample {i} | True: {y_true[i]}, Pred: {y_pred[i]}")
        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"prediction_vs_gt_{i}.png")
            plt.savefig(save_path)
            logger.info(f"Saved prediction vs ground truth plot to {save_path}")
        else:
            plt.show()
            logger.info(f"Displayed prediction vs ground truth plot for sample {i}")
        plt.close(fig)

# Training curves plotting
def plot_loss_and_accuracy_curves(train_losses, val_losses, val_accuracies, save_dir=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        loss_path = os.path.join(save_dir, 'loss_curves.png')
        acc_path = os.path.join(save_dir, 'val_accuracy_curve.png')
    else:
        loss_path = None
        acc_path = None

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    if loss_path:
        plt.savefig(loss_path)
        logger.info(f"Saved loss curves to {loss_path}")
    else:
        plt.show()
        logger.info("Displayed loss curves plot interactively.")
    plt.close()

    plt.figure()
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy Curve')
    if acc_path:
        plt.savefig(acc_path)
        logger.info(f"Saved validation accuracy curve to {acc_path}")
    else:
        plt.show()
        logger.info("Displayed validation accuracy curve plot interactively.")
    plt.close()

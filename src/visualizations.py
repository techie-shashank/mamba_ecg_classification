# ========== Standard Library Imports ==========
import os
import torch

# ========== Third-Party Imports ==========
import numpy as np
import matplotlib.pyplot as plt

# ========== Local Imports ===============
from logger import logger


# Prediction vs ground truth plotting
def plot_predictions_vs_ground_truth(
        model, X_test, y_test, is_multilabel, class_names, test_annotation_df, num_samples=1,
        save_dir=None, sampling_rate=1.0
):
    """
    Plot model predictions vs. ground truth for a few samples.
    """
    # Visualize predictions vs ground truth for a few samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    rng = np.random.default_rng()
    indices = rng.choice(len(X_test), size=num_samples, replace=False)
    ecg_ids = test_annotation_df.iloc[indices].index.to_numpy()
    X = X_test[indices]
    y_true = y_test[indices]
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        if is_multilabel:
            y_pred = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        else:
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    if hasattr(X, 'cpu'):
        X = X.cpu().numpy()
    for i in range(min(num_samples, len(X))):
        sample = X[i]
        n_channels = sample.shape[1]
        time = np.arange(sample.shape[0]) / sampling_rate
        signals = [sample[:, ch] for ch in range(n_channels)]

        fig, axes = plt.subplots(n_channels, 1, figsize=(8, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]
        for ch in range(n_channels):
            axes[ch].plot(time, signals[ch])
            axes[ch].set_ylabel(f"Ch {ch+1}")
            
        # Convert label indices to strings if class_names is provided, support multi-label
        def label_to_str(label):
            if isinstance(label, (list, np.ndarray)):
                return ','.join([class_names[idx] if idx < len(class_names) else str(idx) for idx, v in enumerate(label) if v])
            elif label < len(class_names):
                return class_names[label]
            else:
                return str(label)
                
        true_label = label_to_str(y_true[i])
        pred_label = label_to_str(y_pred[i])
        axes[0].set_title(f"ECG ID {ecg_ids[i]} | True: {true_label}, Pred: {pred_label}")
        axes[-1].set_xlabel("Time (s)")
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

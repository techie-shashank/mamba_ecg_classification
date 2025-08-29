"""
Plotting utilities for evaluation
"""
# ========== Third-Party Imports ==========
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_multilabel_confusion_matrices(multilabel_cm, class_names, save_path):
    """
    Plot confusion matrices for each class in a multi-label classification task.
    Args:
        multilabel_cm (np.ndarray): Array of confusion matrices (n_classes, 2, 2).
        class_names (list): List of class names.
        save_path (str): Path to save the resulting plot.
    """
    n_classes = len(class_names)
    cols = 4
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for i in range(n_classes):
        disp = ConfusionMatrixDisplay(multilabel_cm[i],
                                      display_labels=[f'Not {class_names[i]}', class_names[i]])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(class_names[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    for j in range(n_classes, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

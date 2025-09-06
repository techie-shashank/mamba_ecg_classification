# ========== Standard Library Imports ==========
import os
import torch

# ========== Third-Party Imports ==========
import numpy as np
import matplotlib.pyplot as plt

# ========== Local Imports ===============
from logger import logger


# Training curves plotting
def plot_loss_and_accuracy_curves(train_losses, val_losses, val_accuracies, save_dir=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        loss_path = os.path.join(save_dir, 'main_loss_curves.png')
        acc_path = os.path.join(save_dir, 'main_val_accuracy_curve.png')
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


# ========== Standard Library Imports ==========
import os
import json
import argparse

# ========== Third-Party Imports ==========
import torch
import torch.optim as optim

# ========== Local Imports ==========
from data.data_loader import load_and_prepare
from utils import get_config, get_experiments_dir, get_device, get_loss_function
from models.common import setup_model
from logger import logger, configure_logger
from visualizations import plot_loss_and_accuracy_curves


def parse_arguments():
    """
    Parse command-line arguments for training.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "mamba"], help="Model type")
    return parser.parse_args()


def train_model(model, train_loader, val_loader, criterion, optimizer, device, config, save_dir=None):
    """
    Train the model and evaluate on validation set after each epoch.
    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
        config: Configuration dict
    """
    num_epochs = config.get("epochs", 10)
    is_multilabel = config.get("is_multilabel", False)

    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch = y_batch.float() if is_multilabel else y_batch.long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_batch = y_batch.float() if is_multilabel else y_batch.long()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                if is_multilabel:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()
                    correct += (preds == y_batch.int()).all(dim=1).sum().item()
                    total += y_batch.size(0)
                else:
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total if total > 0 else 0.0
        logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

    # Plot loss and accuracy curves using centralized function
    # Save plots to experiment directory if available in config
    plot_loss_and_accuracy_curves(train_losses, val_losses, val_accuracies, save_dir=save_dir)


def train(dataset_name, model_name, save_dir):
    """
    Main training pipeline: loads data, sets up model, trains, and saves model.
    Args:
        dataset_name: str
        model_name: str
        save_dir: str
    """
    config = get_config()
    device = get_device()
    # Save the config dictionary as a JSON file
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    # Prepare data
    loaders, _, data_arrays, _ = load_and_prepare(dataset_name, config)
    train_loader = loaders['train']
    val_loader = loaders['val']
    X_train, y_train = data_arrays['train']
    # Model setup
    sample_X = X_train[0]
    input_channels = sample_X.shape[1]
    num_classes = len(y_train[0]) if y_train.ndim > 1 else 2
    model = setup_model(model_name, input_channels, num_classes)
    criterion = get_loss_function(config, y_train)
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    # Train the model
    logger.info(f"Starting training for model: {model_name.upper()}")
    plot_dir = os.path.join(save_dir, "plots")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, config, save_dir=plot_dir)
    # Save the model
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    args = parse_arguments()
    experiments_dir = get_experiments_dir(args.dataset, args.model)
    log_path = os.path.join(experiments_dir, "train.log")
    configure_logger(log_path)
    train(args.dataset, args.model, experiments_dir)
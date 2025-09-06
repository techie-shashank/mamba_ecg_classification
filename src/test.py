# ========== Standard Library Imports ==========
import os
import argparse
import torch

# ========== Local Imports ==========
import utils
from models.common import load_model
from evaluation.evaluate import evaluate_and_save_metrics
from logger import logger, configure_logger
from data.data_loader import load_and_prepare


def parse_arguments():
    """
    Parse command-line arguments for testing.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "mamba", "hybrid_serial"], help="Model type")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--run_number", type=int, default=1, help="Run number of the experiment (default: 1)")
    return parser.parse_args()


def test_model(dataset_name, model_name, base_dir):
    """
    Test a trained model and save evaluation metrics.
    Args:
        dataset_name: str
        model_name: str
        base_dir: str (directory containing model and config)
    """
    config = utils.get_config(base_dir)
    plot_dir = os.path.join(base_dir, "plots")

    # Prepare data
    loaders, _, data_arrays, metadata, _ = load_and_prepare(dataset_name, config)
    test_loader = loaders['test']
    train_loader = loaders['train']  # Get train loader for linear probing
    X_test, y_test = data_arrays['test']

    # Model setup
    sample_X = X_test[0]
    input_channels = sample_X.shape[1]
    num_classes = len(y_test[0]) if y_test.ndim > 1 else 2
    model = load_model(model_name, input_channels, num_classes, base_dir, config)
    criterion = utils.get_loss_function(config)

    # Evaluation
    classes = metadata['classes']

    evaluate_and_save_metrics(
        model, criterion, test_loader, config, classes, base_dir, logger, 
        model_type=model_name, train_loader=train_loader, generate_tsne=True, enable_linear_probe=True
    )

if __name__ == "__main__":
    args = parse_arguments()
    run_number = args.run_number 
    base_dir = os.path.join(
        os.path.join(r"./experiments", args.dataset, args.model),
        f"run_{run_number}"
    )
    log_path = os.path.join(base_dir, "test.log")
    configure_logger(log_path)
    test_model(args.dataset, args.model, base_dir)

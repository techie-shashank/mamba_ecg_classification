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
from visualizations import plot_data_samples, plot_predictions_vs_ground_truth


def parse_arguments():
    """
    Parse command-line arguments for testing.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "mamba"], help="Model type")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
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
    loaders, _, data_arrays, metadata = load_and_prepare(dataset_name, config)
    test_loader = loaders['test']
    X_test, y_test = data_arrays['test']

    # Visualize a few data samples
    plot_data_samples(X_test, y_test, num_samples=5, save_dir=plot_dir)
    # Model setup
    sample_X = X_test[0]
    input_channels = sample_X.shape[1]
    num_classes = len(y_test[0]) if y_test.ndim > 1 else 2
    model = load_model(model_name, input_channels, num_classes, base_dir)
    criterion = utils.get_loss_function(config)
    # Evaluation
    classes = metadata['classes']
    evaluate_and_save_metrics(model, criterion, test_loader, config, classes, base_dir, logger)

    # Visualize predictions vs ground truth for a few samples
    # Run model on a few test samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    num_samples = 1
    X_vis = X_test[:num_samples]
    y_vis = y_test[:num_samples]
    with torch.no_grad():
        X_tensor = torch.tensor(X_vis, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        if num_classes > 2:
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
    plot_predictions_vs_ground_truth(X_vis, y_vis, preds, num_samples=num_samples, save_dir=plot_dir)


if __name__ == "__main__":
    args = parse_arguments()
    # Optionally, allow for run_number override (future extension)
    run_number = None
    if run_number:
        base_dir = os.path.join(
            os.path.join(r"./../experiments", args.dataset, args.model),
            f"run_{run_number}"
        )
    else:
        base_dir = f"./../saved_model/{args.dataset}/{args.model}"
    log_path = os.path.join(base_dir, "test.log")
    configure_logger(log_path)
    test_model(args.dataset, args.model, base_dir)

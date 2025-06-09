import argparse
import os

import torch
from src import utils
from src.logger import logger, configure_logger


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--model", type=str, required=True, choices=["fcn", "lstm", "mamba"], help="Model type")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    return parser.parse_args()


def test_model(dataset_name, model_name, base_dir):
    config = utils.get_config_for_testing(base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract parameters from config
    batch_size = config["batch_size"]

    # Prepare data
    test_loader, dataset_handler, X_test, y_test = utils.load_and_prepare_data(
        dataset_name=dataset_name, config=config, split="test", batch_size=batch_size
    )
    # Model setup
    sample_X = X_test[0]
    time_steps = sample_X.shape[0]
    input_channels = sample_X.shape[1]
    num_classes = len(y_test[0]) if y_test.ndim > 1 else 2

    model = utils.setup_model(model_name, dataset_name, num_classes, input_channels, time_steps, device, base_dir)

    # Evaluation
    criterion = torch.nn.BCEWithLogitsLoss() if num_classes > 2 else torch.nn.CrossEntropyLoss()
    is_multilabel = num_classes > 2

    utils.evaluate_and_save_metrics(dataset_name, model_name, model, test_loader, criterion, device, is_multilabel, logger, dataset_handler, base_dir)


if __name__ == "__main__":
    args = parse_arguments()

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

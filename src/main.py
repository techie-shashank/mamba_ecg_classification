
# ========== Standard Library Imports ==========
import os
import sys
import argparse
from pathlib import Path

# Add src directory to path if not already there
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# ========== Local Imports ==========
from logger import configure_logger, logger
from train import train
from test import test_model
from utils import get_experiments_dir
from dotenv import load_dotenv


def parse_arguments():
    """
    Parse command-line arguments for main pipeline.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "mamba", "hybrid_serial"], help="Model type")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    dataset_name = args.dataset
    model_name = args.model
    experiments_dir = get_experiments_dir(dataset_name, model_name)

    log_path = os.path.join(experiments_dir, "main.log")
    configure_logger(log_path)

    # Train the model
    logger.info(f"Starting training for dataset: {dataset_name}, model: {model_name}")
    train(dataset_name, model_name, experiments_dir)

    # Test the model
    logger.info(f"Starting evaluation for dataset: {dataset_name}, model: {model_name}")
    test_model(dataset_name, model_name, experiments_dir)

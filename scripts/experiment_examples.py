#!/usr/bin/env python3
from scripts.experiment_runner import ExperimentRunner
from src.logger import get_experiment_logger
from dotenv import load_dotenv

# Setup logging
logger = get_experiment_logger('experiment_examples')

COMPREHENSIVE_SEARCH = {
    "datasets": ["ptbxl"],
    "models": ["lstm", "mamba", "hybrid_serial", "hybrid_serial_rev"],
    "hyperparameter_grids": {
        "lstm": {
            "hidden_size": [128, 256],
            "num_layers": [2, 3],
            "dropout": [0.3, 0.5]
        },
        "mamba": {
            "d_model": [128, 256],
            "d_state": [16, 32, 64],
            "d_conv": [2, 4],
            "expand": [2]
        },
        "hybrid_serial": {
            "d_model": [64, 128],
            "d_state": [16, 32],
            "d_conv": [2, 4],
            "lstm_hidden": [64, 128],
            "dropout": [0.3]
        },
        "hybrid_serial_rev": {
            "d_model": [64, 128],
            "d_state": [16, 32],
            "d_conv": [2, 4],
            "lstm_hidden": [64, 128],
            "dropout": [0.3]
        }
    },
    "global_param_grid": {
        "batch_size": [64],
        "learning_rate": [0.0001],
        "epochs": [100],
        "is_multilabel": [True, False],
        "sampling_rate": [100, 500],
        "use_focal_loss": [True]
    }
}

def run_comprehensive_search():
    """Run comprehensive hyperparameter search"""

    runner = ExperimentRunner()
    runner.generate_experiment_configs(**COMPREHENSIVE_SEARCH)
    # Save experiment plan
    runner.save_experiment_plan("outputs/comprehensive_search.json")
    logger.info(f"Generated {len(runner.experiment_configs)} experiments for comprehensive search")
    return runner

if __name__ == "__main__":
    load_dotenv()
    runner = run_comprehensive_search()
    runner.run_all_experiments()

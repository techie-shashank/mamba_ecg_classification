#!/usr/bin/env python3
from scripts.experiment_runner import ExperimentRunner
from src.logger import get_experiment_logger
from dotenv import load_dotenv

# Setup logging
logger = get_experiment_logger('experiment_examples')

# Defined configurations with comparable architecture sizes
# Based on best-performing experiment results
# Parameter counts matched to successful runs
DEFINED_CONFIGS = {
    "datasets": ["ptbxl"],
    "models": ["resnet50"],
    "hyperparameter_grids": {
        "lstm": {
            "hidden_size": [192],
            "num_layers": [2],
            "dropout": [0.3]
        },
        "mamba": {
            "d_model": [256],
            "d_state": [16],
            "d_conv": [4],
            "expand": [2]
        },
        "patchtst": {
            "seq_len": [1000],
            "patch_len": [16],
            "stride": [8],
            "d_model": [160],
            "n_heads": [2],
            "num_layers": [1],
            "dropout": [0.1]
        },
        "autoformer": {
            "seq_len": [1000],
            "prediction_length": [1],
            "d_model": [128],
            "encoder_layers": [2],
            "decoder_layers": [1],
            "encoder_attention_heads": [8],
            "decoder_attention_heads": [8],
            "dropout": [0.1]
        },
        "resnet50": {
            "fc_hidden_size": [128],
            "dropout": [0.3]
        },
        "hybrid_serial": {
            "d_model": [160],
            "d_state": [16],
            "d_conv": [4],
            "lstm_hidden": [128],
            "lstm_layers": [2],
            "dropout": [0.3]
        }
    },
    "global_param_grid": {
        "batch_size": [64],
        "learning_rate": [0.0001],
        "epochs": [100],
        "is_multilabel": [False, True],     # Binary classification
        "sampling_rate": [100],       # Standard 100Hz
        "use_focal_loss": [True]
    }
}

# Alternative: Grid search for experimentation (commented out)
COMPREHENSIVE_SEARCH = {
    "datasets": ["ptbxl"],
    "models": ["lstm", "mamba", "patchtst", "resnet50", "autoformer", "hybrid_serial", "hybrid_serial_rev", "hybrid_parallel", "hybrid_crossattn"],
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
        "patchtst": {
            "seq_len": [1000],
            "patch_len": [16, 32],
            "stride": [8, 16],
            "d_model": [128, 256],
            "n_heads": [4, 8],
            "num_layers": [2, 3],
            "dropout": [0.1, 0.2]
        },
        "autoformer": {
            "seq_len": [1000],
            "prediction_length": [1],
            "d_model": [128, 256],
            "encoder_layers": [2, 3],
            "decoder_layers": [1, 2],
            "encoder_attention_heads": [4, 8],
            "decoder_attention_heads": [4, 8],
            "dropout": [0.1, 0.2]
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
        },
        "hybrid_parallel": {
            "d_model": [64, 128],
            "d_state": [16, 32],
            "d_conv": [2, 4],
            "lstm_hidden": [64, 128],
            "dropout": [0.3],
            "fusion_method": ["concat"]
        },
        "hybrid_crossattn": {
            "d_model": [128],
            "d_state": [16, 32],
            "d_conv": [2, 4],
            "lstm_hidden": [128],
            "dropout": [0.3],
            "num_attn_heads": [4, 8],
            "fusion_method": ["concat"]
        },
        "resnet50": {
            "fc_hidden_size": [128],
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

def run_defined_experiments():
    """Run experiments with defined configurations (comparable architecture sizes)"""
    runner = ExperimentRunner()
    runner.generate_experiment_configs(**DEFINED_CONFIGS)
    # Save experiment plan
    runner.save_experiment_plan("outputs/defined_experiments.json")
    logger.info(f"Generated {len(runner.experiment_configs)} experiments with defined configurations")
    logger.info("All models use d_model/hidden_size ∈ {256, 128}: LSTM(256/2), MAMBA(256), PatchTST(128), Hybrids(256/64)")
    return runner

def run_comprehensive_search():
    """Run comprehensive hyperparameter search (grid search)"""
    runner = ExperimentRunner()
    runner.generate_experiment_configs(**COMPREHENSIVE_SEARCH)
    # Save experiment plan
    runner.save_experiment_plan("outputs/comprehensive_search.json")
    logger.info(f"Generated {len(runner.experiment_configs)} experiments for comprehensive search")
    return runner

if __name__ == "__main__":
    load_dotenv()
    
    # Run defined experiments (default - comparable architectures)
    runner = run_defined_experiments()
    
    # Uncomment below to run comprehensive grid search instead
    # runner = run_comprehensive_search()
    
    runner.run_all_experiments()

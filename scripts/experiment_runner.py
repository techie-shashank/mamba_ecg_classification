#!/usr/bin/env python3
"""
Experiment Runner for Multiple Configuration Testing
Allows running multiple experiments with different hyperparameters and models
"""

import os
import json
import argparse
import itertools
import subprocess
import time
import hashlib
import copy
from pathlib import Path
from typing import Dict, List, Any
from logger import get_experiment_logger

# Setup logging with file output
logger = get_experiment_logger('experiment_runner')


class ExperimentRunner:
    """Class to manage and run multiple experiments with different configurations"""

    def __init__(self, base_config_path: str = "./configs/config.json"):
        """
        Initialize the experiment runner

        Args:
            base_config_path: Path to the base configuration file
        """
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()
        self.experiment_configs = []

    def _load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file"""
        try:
            with open(self.base_config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded base configuration from {self.base_config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading base config: {e}")
            raise

    def generate_hyperparameter_combinations(self, hyperparameter_grids: Dict[str, Dict[str, List]]) -> List[Dict]:
        """
        Generate all combinations of hyperparameters for experiments

        Args:
            hyperparameter_grids: Dictionary with model names and their hyperparameter grids

        Returns:
            List of experiment configurations
        """
        all_configs = []

        for model_name, param_grid in hyperparameter_grids.items():
            # Get all parameter names and their possible values
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            # Generate all combinations
            for combination in itertools.product(*param_values):
                config = self.base_config.copy()
                config['model'] = model_name

                # Update model-specific hyperparameters
                model_params = dict(zip(param_names, combination))
                if 'model_hyperparameters' not in config:
                    config['model_hyperparameters'] = {}
                if model_name not in config['model_hyperparameters']:
                    config['model_hyperparameters'][model_name] = {}

                config['model_hyperparameters'][model_name].update(model_params)
                all_configs.append(config)

        return all_configs

    def generate_experiment_configs(self,
                                  datasets: List[str],
                                  models: List[str],
                                  hyperparameter_grids: Dict[str, Dict[str, List]] = None,
                                  global_param_grid: Dict[str, List] = None) -> None:
        """
        Generate experiment configurations

        Args:
            datasets: List of dataset names to test
            models: List of model names to test
            hyperparameter_grids: Model-specific hyperparameter grids
            global_param_grid: Global parameters to vary (batch_size, learning_rate, etc.)
        """
        self.experiment_configs = []

        # Default hyperparameter grids if not provided
        if hyperparameter_grids is None:
            hyperparameter_grids = {
                'lstm': {
                    'hidden_size': [64, 128],
                    'num_layers': [1, 2],
                    'dropout': [0.2, 0.3]
                },
                'mamba': {
                    'd_model': [128, 256],
                    'd_state': [16, 32],
                    'expand': [2, 4]
                },
                'hybrid_serial': {
                    'd_model': [128, 256],
                    'lstm_hidden': [64, 128],
                    'dropout': [0.2, 0.3]
                }
            }

        # Default global parameter grid
        if global_param_grid is None:
            global_param_grid = {
                'batch_size': [32, 64],
                'learning_rate': [0.001, 0.01],
                'epochs': [10, 20]
            }

        # Generate global parameter combinations
        global_param_names = list(global_param_grid.keys())
        global_param_values = list(global_param_grid.values())

        for dataset in datasets:
            for model in models:
                # Get model-specific hyperparameters
                model_hyperparams = hyperparameter_grids.get(model, {})
                
                # Generate hyperparameter combinations for this specific model
                if model_hyperparams:
                    param_names = list(model_hyperparams.keys())
                    param_values = list(model_hyperparams.values())
                    
                    # Generate all combinations for this model
                    for param_combination in itertools.product(*param_values):
                        model_params = dict(zip(param_names, param_combination))
                        
                        # Combine with global parameters
                        for global_combination in itertools.product(*global_param_values):
                            global_params = dict(zip(global_param_names, global_combination))
                            
                            # Create config with deep copy to avoid reference issues
                            config = copy.deepcopy(self.base_config)
                            config.update(global_params)
                            config['dataset'] = dataset
                            config['model'] = model
                            
                            # Update model-specific hyperparameters
                            config['model_hyperparameters'][model].update(model_params)
                            self.experiment_configs.append(config)
                else:
                    # No hyperparameters for this model, just use global params
                    for global_combination in itertools.product(*global_param_values):
                        global_params = dict(zip(global_param_names, global_combination))
                        
                        config = copy.deepcopy(self.base_config)
                        config.update(global_params)
                        config['dataset'] = dataset
                        config['model'] = model
                        
                        self.experiment_configs.append(config)

        logger.info(f"Generated {len(self.experiment_configs)} experiment configurations")

    def _create_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Create a hash of the configuration to identify unique experiments
        
        Args:
            config: Experiment configuration
            
        Returns:
            Hash string representing the configuration
        """
        # Create a copy and remove non-essential fields for hashing
        config_for_hash = config.copy()
        
        # Remove fields that don't affect the experiment uniqueness
        fields_to_ignore = ['experiment_name', 'timestamp', 'output_dir']
        for field in fields_to_ignore:
            config_for_hash.pop(field, None)
        
        # Only include hyperparameters for the specific model being run
        if 'model_hyperparameters' in config_for_hash and 'model' in config:
            current_model = config['model']
            original_model_hyperparams = config_for_hash['model_hyperparameters']
            
            # Create new model_hyperparameters dict with only the current model
            if current_model in original_model_hyperparams:
                config_for_hash['model_hyperparameters'] = {
                    current_model: original_model_hyperparams[current_model]
                }
            else:
                # If no hyperparameters for this model, remove the field entirely
                config_for_hash.pop('model_hyperparameters', None)
        
        # Sort the config to ensure consistent hashing
        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _check_experiment_exists(self, config: Dict[str, Any]) -> bool:
        """
        Check if an experiment with the same configuration already exists
        
        Args:
            config: Experiment configuration to check
            
        Returns:
            True if experiment exists, False otherwise
        """
        dataset = config['dataset']
        model = config['model']
        
        # Check in experiments directory
        experiments_dir = Path("./experiments") / dataset / model
        
        if not experiments_dir.exists():
            return False
        
        config_hash = self._create_config_hash(config)
        
        # Check all run directories
        for run_dir in experiments_dir.glob("run_*"):
            if run_dir.is_dir():
                config_file = run_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            existing_config = json.load(f)
                        
                        existing_hash = self._create_config_hash(existing_config)
                        
                        if config_hash == existing_hash:
                            logger.info(f"⏭️  Skipping duplicate experiment in {run_dir}")
                            return True
                            
                    except Exception as e:
                        logger.warning(f"Error reading config from {config_file}: {e}")
                        continue
        
        return False

    def save_experiment_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Save experiment configuration to file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def run_single_experiment(self, config: Dict[str, Any], experiment_id: int, skip_existing: bool = True) -> bool:
        """
        Run a single experiment with given configuration

        Args:
            config: Experiment configuration
            experiment_id: Unique experiment identifier
            skip_existing: Whether to skip if experiment with same config exists

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if experiment already exists
            if skip_existing and self._check_experiment_exists(config):
                logger.info(f"⏭️  Experiment {experiment_id} skipped - already exists with same configuration")
                return True  # Return True since we're considering this a "success"
            
            dataset = config['dataset']
            model = config['model']

            # Create temporary config file for this experiment
            temp_config_path = f"temp/temp_config_{experiment_id}.json"
            self.save_experiment_config(config, temp_config_path)

            # Prepare command
            cmd = [
                'python', 'src/main.py',
                '--dataset', dataset,
                '--model', model
            ]

            logger.info(f"Running experiment {experiment_id}: {dataset} + {model}")
            logger.info(f"Config: batch_size={config.get('batch_size')}, lr={config.get('learning_rate')}, epochs={config.get('epochs')}")

            # Run the experiment
            start_time = time.time()

            # Copy config to working directory
            with open('./configs/config.json', 'w') as f:
                json.dump(config, f, indent=2)

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), r".."))

            end_time = time.time()
            duration = end_time - start_time

            if result.returncode == 0:
                logger.info(f"✅ Experiment {experiment_id} completed successfully in {duration:.2f}s")
                return True
            else:
                logger.error(f"❌ Experiment {experiment_id} failed:")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Exception in experiment {experiment_id}: {e}")
            return False
        finally:
            # Clean up temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    def run_all_experiments(self, max_parallel: int = 1, skip_existing: bool = True) -> None:
        """
        Run all generated experiments

        Args:
            max_parallel: Maximum number of parallel experiments (currently supports 1)
            skip_existing: Whether to skip experiments with existing configurations
        """
        if not self.experiment_configs:
            logger.error("No experiment configurations found. Run generate_experiment_configs() first.")
            return

        logger.info(f"Starting {len(self.experiment_configs)} experiments...")
        if skip_existing:
            logger.info("⏭️  Will skip experiments with existing configurations")

        successful = 0
        failed = 0
        skipped = 0

        for i, config in enumerate(self.experiment_configs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Experiment {i+1}/{len(self.experiment_configs)}")
            logger.info(f"{'='*50}")

            # Check if experiment exists before running
            if skip_existing and self._check_experiment_exists(config):
                logger.info(f"⏭️  Skipping experiment {i+1} - already exists")
                skipped += 1
                continue

            if self.run_single_experiment(config, i+1, skip_existing=False):  # Don't double-check in run_single
                successful += 1
            else:
                failed += 1

            # Small delay between experiments
            time.sleep(2)

        logger.info(f"\n{'='*50}")
        logger.info(f"EXPERIMENT SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total experiments: {len(self.experiment_configs)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped (existing): {skipped}")
        logger.info(f"Actually run: {successful + failed}")
        if len(self.experiment_configs) > 0:
            logger.info(f"Success rate: {successful/(successful + failed)*100:.1f}%" if (successful + failed) > 0 else "No experiments run")

    def save_experiment_plan(self, filepath: str) -> None:
        """Save the experiment plan to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.experiment_configs, f, indent=2)
        logger.info(f"Experiment plan saved to {filepath}")

    def load_experiment_plan(self, filepath: str) -> None:
        """Load experiment plan from a file"""
        with open(filepath, 'r') as f:
            self.experiment_configs = json.load(f)
        logger.info(f"Loaded {len(self.experiment_configs)} experiments from {filepath}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Run multiple ML experiments")
    parser.add_argument("--datasets", nargs='+', default=['ptbxl'], help="Datasets to test")
    parser.add_argument("--models", nargs='+', default=['lstm', 'mamba'], help="Models to test")
    parser.add_argument("--config", default="config.json", help="Base configuration file")
    parser.add_argument("--save-plan", help="Save experiment plan to file")
    parser.add_argument("--load-plan", help="Load experiment plan from file")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't run experiments")
    parser.add_argument("--force", action="store_true", help="Run all experiments even if they already exist")

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(args.config)

    if args.load_plan:
        # Load existing plan
        runner.load_experiment_plan(args.load_plan)
    else:
        # Generate new experiment configurations
        runner.generate_experiment_configs(
            datasets=args.datasets,
            models=args.models
        )

    # Save plan if requested
    if args.save_plan:
        runner.save_experiment_plan(args.save_plan)

    if args.dry_run:
        logger.info(f"DRY RUN: Would run {len(runner.experiment_configs)} experiments")
        for i, config in enumerate(runner.experiment_configs[:5]):  # Show first 5
            logger.info(f"Experiment {i+1}: {config['dataset']} + {config['model']} "
                       f"(batch_size={config.get('batch_size')}, lr={config.get('learning_rate')})")
        if len(runner.experiment_configs) > 5:
            logger.info(f"... and {len(runner.experiment_configs) - 5} more")
    else:
        # Run all experiments
        skip_existing = not args.force  # Skip existing unless --force is used
        runner.run_all_experiments(skip_existing=skip_existing)


if __name__ == "__main__":
    main()

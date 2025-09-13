#!/usr/bin/env python3
"""
Experiment Data Parser and Storage
Parses all experiments from the experiments folder and stores data in a structured format
"""

import os
import sys
import json
import pandas as pd
import sqlite3
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Setup paths for imports  
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

# Add both to path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from src.logger import get_experiment_logger

# Setup logging with file output
logger = get_experiment_logger('experiment_data_parser')


class ExperimentDataParser:
    """Class to parse and store experiment data from experiments folder"""

    def __init__(self, experiments_root: str = "../experiments", db_path: str = "outputs/experiments.db"):
        """
        Initialize the data parser

        Args:
            experiments_root: Root directory containing all experiments
            db_path: Path to SQLite database file
        """
        self.experiments_root = Path(experiments_root)
        self.db_path = db_path
        self.parsed_data = []

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with experiments table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Drop existing table and recreate with updated schema
        cursor.execute('DROP TABLE IF EXISTS experiments')
        logger.info("Dropped existing experiments table")

        cursor.execute('''
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                dataset TEXT,
                model TEXT,
                run_id TEXT,
                experiment_path TEXT,
                
                -- Configuration parameters
                batch_size INTEGER,
                epochs INTEGER,
                learning_rate REAL,
                is_multilabel BOOLEAN,
                sampling_rate INTEGER,
                limit_samples INTEGER,
                
                -- Model hyperparameters (JSON stored as TEXT)
                model_hyperparameters TEXT,
                
                -- Training metrics
                total_training_time REAL,
                total_inference_time REAL,
                total_parameters INTEGER,
                trainable_parameters INTEGER,
                
                -- Test metrics
                accuracy REAL,
                macro_f1 REAL,
                macro_precision REAL,
                macro_recall REAL,
                macro_auc REAL,
                
                -- Linear probe metrics
                linear_probe_accuracy REAL,
                linear_probe_f1 REAL,
                linear_probe_precision REAL,
                linear_probe_recall REAL,
                linear_probe_auc REAL,
                linear_probe_cv_accuracy REAL,
                linear_probe_cv_std REAL,
                
                -- Additional test metrics (JSON stored as TEXT)
                all_test_metrics TEXT,
                
                -- File paths for plots and visualizations (JSON stored as TEXT)
                experiment_files TEXT,
                
                -- Metadata
                created_at TEXT,
                parsed_at TEXT
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database recreated with updated schema at {self.db_path}")

    def parse_all_experiments(self) -> pd.DataFrame:
        """
        Parse all experiments from the experiments folder

        Returns:
            DataFrame containing all experiment data
        """
        self.parsed_data = []

        # Scan all experiment directories
        experiment_paths = list(self.experiments_root.glob("*/*/run_*"))
        logger.info(f"Found {len(experiment_paths)} experiment directories")

        for exp_path in experiment_paths:
            if exp_path.is_dir():
                exp_data = self._parse_single_experiment(exp_path)
                if exp_data:
                    self.parsed_data.append(exp_data)

        logger.info(f"Successfully parsed {len(self.parsed_data)} experiments")
        return pd.DataFrame(self.parsed_data)

    def _parse_single_experiment(self, exp_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse data from a single experiment directory

        Args:
            exp_path: Path to experiment directory

        Returns:
            Dictionary containing experiment data or None if parsing failed
        """
        try:
            # Extract basic info from path
            parts = exp_path.parts
            dataset = parts[-3]
            model = parts[-2]
            run_id = parts[-1]

            exp_data = {
                'dataset': dataset,
                'model': model.upper(),
                'run_id': run_id,
                'experiment_path': str(exp_path),
                'timestamp': self._get_experiment_timestamp(exp_path),
                'created_at': self._get_creation_time(exp_path),
                'parsed_at': datetime.now().isoformat()
            }

            # Load configuration data
            config_data = self._load_config(exp_path)
            exp_data.update(config_data)

            # Parse training logs
            training_metrics = self._parse_training_log(exp_path)
            exp_data.update(training_metrics)

            # Load test metrics
            test_metrics = self._load_test_metrics(exp_path)
            exp_data.update(test_metrics)

            # Load linear probe metrics
            linear_probe_metrics = self._load_linear_probe_metrics(exp_path)
            exp_data.update(linear_probe_metrics)

            return exp_data

        except Exception as e:
            logger.warning(f"Error parsing experiment {exp_path}: {e}")
            return None

    def _get_experiment_timestamp(self, exp_path: Path) -> str:
        """Get experiment timestamp from directory modification time"""
        try:
            timestamp = datetime.fromtimestamp(exp_path.stat().st_mtime)
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return "Unknown"

    def _get_creation_time(self, exp_path: Path) -> str:
        """Get experiment creation time"""
        try:
            ctime = datetime.fromtimestamp(exp_path.stat().st_ctime)
            return ctime.isoformat()
        except:
            return "Unknown"

    def _load_config(self, exp_path: Path) -> Dict[str, Any]:
        """Load configuration from config.json"""
        config_data = {}
        config_path = exp_path / "config.json"

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Extract basic training settings
                config_data.update({
                    'batch_size': config.get('batch_size'),
                    'epochs': config.get('epochs'),
                    'learning_rate': config.get('learning_rate'),
                    'is_multilabel': config.get('is_multilabel'),
                    'sampling_rate': config.get('sampling_rate'),
                    'limit_samples': config.get('limit')
                })

                # Store model hyperparameters as JSON string
                if 'model_hyperparameters' in config:
                    config_data['model_hyperparameters'] = json.dumps(config['model_hyperparameters'])

            except Exception as e:
                logger.warning(f"Error reading config from {config_path}: {e}")

        return config_data

    def _parse_training_log(self, exp_path: Path) -> Dict[str, Any]:
        """Parse training log for performance metrics"""
        metrics = {}
        
        # Try both main.log and train.log (in that order of preference)
        log_files = ["main.log", "train.log", "test.log"]
        
        for log_file in log_files:
            log_path = exp_path / log_file
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        content = f.read()

                    # Extract training time - multiple patterns to try
                    training_time = None
                    
                    # Pattern 1: Look for explicit training completion message
                    time_pattern = r'Training completed - Total time: ([\d\.]+)s'
                    time_match = re.search(time_pattern, content)
                    if time_match:
                        training_time = float(time_match.group(1))
                    else:
                        # Pattern 2: Calculate from "Starting training" to "Model saved"
                        start_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) .* Starting training for model:'
                        end_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) .* Model saved to'
                        
                        start_match = re.search(start_pattern, content)
                        end_match = re.search(end_pattern, content)
                        
                        if start_match and end_match:
                            try:
                                from datetime import datetime
                                start_time = datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                                end_time = datetime.strptime(end_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                                training_time = (end_time - start_time).total_seconds()
                            except Exception as e:
                                logger.debug(f"Error calculating training time from timestamps: {e}")
                    
                    if training_time is not None:
                        metrics['total_training_time'] = training_time

                    # Extract inference time - pattern matches your format
                    inference_pattern = r'Inference timing - Total: ([\d\.]+)s'
                    inference_match = re.search(inference_pattern, content)
                    if inference_match:
                        metrics['total_inference_time'] = float(inference_match.group(1))

                    # Extract parameter count - updated pattern to match your log format
                    param_pattern = r'Parameters: ([\d,]+) \(trainable: ([\d,]+)\)'
                    param_match = re.search(param_pattern, content)
                    if param_match:
                        total_params = int(param_match.group(1).replace(',', ''))
                        trainable_params = int(param_match.group(2).replace(',', ''))
                        metrics.update({
                            'total_parameters': total_params,
                            'trainable_parameters': trainable_params
                        })
                    
                    # If we found data in this file, break and don't check the other
                    if metrics:
                        break

                except Exception as e:
                    logger.warning(f"Error parsing training log from {log_path}: {e}")

        return metrics

    def _load_test_metrics(self, exp_path: Path) -> Dict[str, Any]:
        """Load test metrics from metrics.json"""
        test_metrics = {}
        metrics_path = exp_path / "metrics_results" / "metrics.json"

        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                # Determine if this is binary or multilabel classification
                is_binary = 'binary_accuracy' in metrics or 'binary_f1' in metrics
                is_multilabel = 'macro_f1' in metrics or 'subset_accuracy' in metrics

                # Extract accuracy first (different field names for binary vs multilabel)
                if is_binary and 'binary_accuracy' in metrics:
                    accuracy_value = metrics['binary_accuracy']
                    if isinstance(accuracy_value, (int, float)) and pd.notna(accuracy_value):
                        test_metrics['accuracy'] = accuracy_value
                elif is_multilabel and 'subset_accuracy' in metrics:
                    accuracy_value = metrics['subset_accuracy']
                    if isinstance(accuracy_value, (int, float)) and pd.notna(accuracy_value):
                        test_metrics['accuracy'] = accuracy_value

                # Extract other metrics based on classification type
                if is_binary:
                    # Binary classification metrics
                    binary_mapping = {
                        'binary_f1': 'macro_f1',
                        'binary_precision': 'macro_precision',
                        'binary_recall': 'macro_recall',
                        'binary_auc': 'macro_auc'
                    }
                    
                    for metric_key, column_name in binary_mapping.items():
                        if metric_key in metrics:
                            value = metrics[metric_key]
                            if isinstance(value, (int, float)) and pd.notna(value):
                                test_metrics[column_name] = value

                elif is_multilabel:
                    # Multilabel classification metrics  
                    multilabel_mapping = {
                        'macro_f1': 'macro_f1',
                        'macro_precision': 'macro_precision',
                        'macro_recall': 'macro_recall',
                        'macro_roc_auc': 'macro_auc'
                    }
                    
                    for metric_key, column_name in multilabel_mapping.items():
                        if metric_key in metrics:
                            value = metrics[metric_key]
                            if isinstance(value, (int, float)) and pd.notna(value):
                                test_metrics[column_name] = value

                # Store all metrics as JSON for detailed analysis
                test_metrics['all_test_metrics'] = json.dumps(metrics)

                # Store file paths for plots and visualizations
                file_paths = self._get_experiment_files(exp_path)
                test_metrics['experiment_files'] = json.dumps(file_paths)

            except Exception as e:
                logger.warning(f"Error reading test metrics from {metrics_path}: {e}")

        return test_metrics

    def _load_linear_probe_metrics(self, exp_path: Path) -> Dict[str, Any]:
        """Load linear probe metrics from linear_probe_*_metrics.json"""
        linear_metrics = {}
        metrics_dir = exp_path / "metrics_results"
        
        if metrics_dir.exists():
            # Look for linear probe metrics files (pattern: linear_probe_<model>_metrics.json)
            linear_probe_files = list(metrics_dir.glob("linear_probe_*_metrics.json"))
            
            if linear_probe_files:
                # Use the first found linear probe metrics file
                linear_probe_path = linear_probe_files[0]
                try:
                    with open(linear_probe_path, 'r') as f:
                        probe_metrics = json.load(f)
                    
                    logger.debug(f"Found linear probe metrics in {linear_probe_path}")
                    
                    # Extract key linear probe metrics
                    # Handle both binary and multilabel classification formats
                    if 'accuracy' in probe_metrics:
                        # Binary classification format
                        linear_metrics['linear_probe_accuracy'] = probe_metrics['accuracy']
                        linear_metrics['linear_probe_f1'] = probe_metrics.get('macro_f1', 0)
                        linear_metrics['linear_probe_precision'] = probe_metrics.get('macro_precision', 0)
                        linear_metrics['linear_probe_recall'] = probe_metrics.get('macro_recall', 0)
                        linear_metrics['linear_probe_auc'] = probe_metrics.get('macro_auc', 0)
                    elif 'overall_accuracy' in probe_metrics:
                        # Multilabel classification format
                        linear_metrics['linear_probe_accuracy'] = probe_metrics['overall_accuracy']
                        linear_metrics['linear_probe_f1'] = probe_metrics.get('macro_f1', 0)
                        linear_metrics['linear_probe_precision'] = probe_metrics.get('macro_precision', 0)
                        linear_metrics['linear_probe_recall'] = probe_metrics.get('macro_recall', 0)
                        linear_metrics['linear_probe_auc'] = probe_metrics.get('macro_auc', 0)
                    
                    # Cross-validation metrics (available in both formats)
                    linear_metrics['linear_probe_cv_accuracy'] = probe_metrics.get('cv_mean_accuracy', 0)
                    linear_metrics['linear_probe_cv_std'] = probe_metrics.get('cv_std_accuracy', 0)
                    
                    logger.debug(f"Extracted linear probe metrics: {linear_metrics}")
                    
                except Exception as e:
                    logger.warning(f"Error reading linear probe metrics from {linear_probe_path}: {e}")
        
        return linear_metrics

    def _get_experiment_files(self, exp_path: Path) -> Dict[str, str]:
        """Get paths to experiment files (plots, confusion matrices, etc.)"""
        file_paths = {}
        
        # Check for plots folder
        plots_dir = exp_path / "plots"
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                file_key = f"plot_{plot_file.stem}"
                # Store absolute path
                file_paths[file_key] = str(plot_file.resolve())
            
            # Check for per-class t-SNE plots (multilabel)
            tsne_per_class_dir = plots_dir / "tsne_visualization_per_class"
            if tsne_per_class_dir.exists():
                for class_plot in tsne_per_class_dir.glob("*.png"):
                    file_key = f"tsne_class_{class_plot.stem}"
                    # Store absolute path
                    file_paths[file_key] = str(class_plot.resolve())
        
        # Check for metrics results folder
        metrics_dir = exp_path / "metrics_results"
        if metrics_dir.exists():
            # Confusion matrices
            for conf_file in metrics_dir.glob("*confusion*.png"):
                file_key = f"confusion_{conf_file.stem}"
                # Store absolute path
                file_paths[file_key] = str(conf_file.resolve())
            
            # Any other visualization files
            for viz_file in metrics_dir.glob("*.png"):
                if "confusion" not in viz_file.name:
                    file_key = f"viz_{viz_file.stem}"
                    # Store absolute path
                    file_paths[file_key] = str(viz_file.resolve())
        
        return file_paths

    def save_to_database(self, df: pd.DataFrame) -> None:
        """
        Save experiment data to SQLite database

        Args:
            df: DataFrame containing experiment data
        """
        conn = sqlite3.connect(self.db_path)

        try:
            # Clear existing data
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments")

            # Insert new data
            df.to_sql('experiments', conn, if_exists='append', index=False)

            conn.commit()
            logger.info(f"Saved {len(df)} experiments to database {self.db_path}")

        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            conn.rollback()
        finally:
            conn.close()

    def save_to_csv(self, df: pd.DataFrame, filepath: str = "outputs/experiments_data.csv") -> None:
        """
        Save experiment data to CSV file

        Args:
            df: DataFrame containing experiment data
            filepath: Path to save CSV file
        """
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} experiments to CSV {filepath}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")

    def load_from_database(self) -> pd.DataFrame:
        """
        Load experiment data from SQLite database

        Returns:
            DataFrame containing experiment data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM experiments", conn)
            conn.close()
            logger.info(f"Loaded {len(df)} experiments from database")
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return pd.DataFrame()

    def get_experiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics of experiments

        Args:
            df: DataFrame containing experiment data

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_experiments': len(df),
            'datasets': list(df['dataset'].unique()) if 'dataset' in df.columns else [],
            'models': list(df['model'].unique()) if 'model' in df.columns else [],
            'date_range': {
                'earliest': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'latest': df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }

        # Performance statistics
        if 'accuracy' in df.columns:
            summary['accuracy_stats'] = {
                'mean': df['accuracy'].mean(),
                'std': df['accuracy'].std(),
                'min': df['accuracy'].min(),
                'max': df['accuracy'].max()
            }

        return summary

    def update_experiment_data(self) -> pd.DataFrame:
        """
        Complete pipeline to parse, save, and return experiment data

        Returns:
            DataFrame containing all experiment data
        """
        logger.info("Starting experiment data update...")

        # Parse all experiments
        df = self.parse_all_experiments()

        if not df.empty:
            # Save to database
            self.save_to_database(df)

            # Save to CSV
            self.save_to_csv(df)

            # Print summary
            summary = self.get_experiment_summary(df)
            logger.info(f"Summary: {summary}")
        else:
            logger.warning("No experiment data found!")

        return df


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Parse and store experiment data")
    parser.add_argument("--experiments-dir", default="../experiments", help="Experiments root directory")
    parser.add_argument("--db-path", default="outputs/experiments.db", help="SQLite database path")
    parser.add_argument("--csv-path", default="outputs/experiments_data.csv", help="CSV output path")
    parser.add_argument("--summary-only", action="store_true", help="Only show summary, don't update data")

    args = parser.parse_args()

    # Initialize parser
    parser = ExperimentDataParser(args.experiments_dir, args.db_path)

    if args.summary_only:
        # Load existing data and show summary
        df = parser.load_from_database()
        if not df.empty:
            summary = parser.get_experiment_summary(df)
            logger.info(json.dumps(summary, indent=2, default=str))
        else:
            logger.warning("No data found in database")
    else:
        # Update experiment data
        df = parser.update_experiment_data()
        logger.info(f"Updated experiment database with {len(df)} experiments")


if __name__ == "__main__":
    load_dotenv()
    main()

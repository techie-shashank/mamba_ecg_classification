#!/usr/bin/env python3
"""
Standalone Integrated Gradients Interpretability Analysis Script

This script provides interpretability analysis for trained time series models using
Integrated Gradients with Captum. It can analyze any trained LSTM, Mamba, or Hybrid model
and generate 12-lead ECG attribution visualizations.

Usage:
    python interpretability_analysis.py --model lstm --dataset ptbxl --run_number 1
    python interpretability_analysis.py --model mamba --dataset ptbxl --run_number 2 --patient_id "Custom_Patient_001"
    python interpretability_analysis.py --model hybrid_serial --dataset ptbxl --run_number 3 --sample_idx 5
"""

# ========== Standard Library Imports ==========
import os
import sys
import argparse
import torch
from dotenv import load_dotenv

# ========== Local Imports ==========
# Setup paths for imports  
from pathlib import Path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

# Add both to path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from src import utils
from src.models.common import load_model
from src.logger import logger, configure_logger
from src.data.data_loader import load_and_prepare
from src.evaluation.integrated_gradients_utils import IGAttributor


def parse_arguments():
    """
    Parse command-line arguments for interpretability analysis.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate Integrated Gradients interpretability analysis for trained time series models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze LSTM model run 1 on PTB-XL dataset for single ECG ID
    python interpretability_analysis.py --model lstm --dataset ptbxl --run_number 1 --ecg_id 12345
    
    # Analyze multiple ECG IDs at once
    python interpretability_analysis.py --model lstm --dataset ptbxl --run_number 1 --ecg_id 1 2 5 10 20
    
    # Analyze specific ECGs with custom output directory
    python interpretability_analysis.py --model mamba --dataset ptbxl --run_number 2 --ecg_id 67890 11111 --output_dir "./custom_output"
    
    # Generate analysis for different model type with multiple ECGs
    python interpretability_analysis.py --model hybrid_serial --dataset ptbxl --run_number 3 --ecg_id 100 200 300
        """
    )
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, 
                       choices=["lstm", "mamba", "hybrid_serial"], 
                       help="Model type to analyze")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset name (e.g., ptbxl)")
    parser.add_argument("--run_number", type=int, required=True,
                       help="Run number of the trained model to analyze")
    parser.add_argument("--ecg_id", type=int, nargs='+', required=True,
                       help="ECG ID(s) of the sample(s) to analyze. Can specify multiple IDs: --ecg_id 1 2 5 10")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (default: model's run directory/interpretability)")
    
    return parser.parse_args()


def setup_paths_and_config(args):
    """
    Setup file paths and load configuration.
    Args:
        args: Parsed command line arguments
    Returns:
        tuple: (base_dir, config, output_dir)
    """
    # Construct base directory path (relative to src directory)
    base_dir = os.path.join(
        "..", "experiments", args.dataset, args.model, f"run_{args.run_number}"
    )
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Model directory not found: {base_dir}")
    
    # Load configuration
    config = utils.get_config(base_dir)
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, 'interpretability')
    
    os.makedirs(output_dir, exist_ok=True)
    
    return base_dir, config, output_dir


def load_model_and_data(args, base_dir, config):
    """
    Load the trained model and prepare specific ECG data.
    Args:
        args: Parsed command line arguments
        base_dir: Base directory containing model files
        config: Model configuration
    Returns:
        tuple: (model, ecg_samples_dict, classes, metadata)
    """
    logger.info(f"Loading {args.model} model from {base_dir}")
    
    # Support for multiple ECG IDs
    ecg_ids = args.ecg_id if isinstance(args.ecg_id, list) else [args.ecg_id]
    logger.info(f"Analyzing ECG IDs: {ecg_ids}")
    
    # Load annotations and setup data processing
    from src.data.ptbxl import load_annotations, load_aggregation_map, aggregate_superclasses, preprocess_labels
    import pandas as pd
    import numpy as np
    import wfdb
    import ast
    
    # Load annotations to check if ECG IDs exist
    logger.info(f"Loading annotations to find ECG IDs: {ecg_ids}")
    
    try:
        df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], 'ptbxl_database.csv'), index_col='ecg_id')
    except Exception as e:
        raise FileNotFoundError(f"Failed to load ptbxl_database.csv: {e}")
    
    # Check if all ECG IDs exist
    missing_ids = [ecg_id for ecg_id in ecg_ids if ecg_id not in df.index]
    if missing_ids:
        available_ids = list(df.index[:10])  # Show first 10 available IDs
        raise ValueError(f"ECG IDs {missing_ids} not found in dataset. "
                        f"Available ECG IDs (first 10): {available_ids}")
    
    # Parse scp_codes
    def safe_eval(x):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            logger.warning(f"Could not parse scp_codes: {x} ({e})")
            return []
    
    df.scp_codes = df.scp_codes.apply(safe_eval)
    
    # Load aggregation map and add diagnostic superclass
    agg_map = load_aggregation_map(os.environ["DATA_DIR"])
    df['diagnostic_superclass'] = aggregate_superclasses(df, agg_map)
    
    # Get classes from a sample to determine all possible classes for the model
    logger.info("Determining all possible classes from dataset configuration...")
    sample_df = df.head(100)  # Use first 100 records to determine classes
    sample_signals = np.random.rand(100, 1000, 12)  # Dummy signals for class determination
    _, _, _, classes = preprocess_labels(sample_signals, sample_df, config)
    
    logger.info(f"Number of classes: {len(classes)}")
    logger.info(f"Classes: {classes}")
    
    # Load ECG signals for all specified IDs
    ecg_samples_dict = {}
    ecg_records_dict = {}
    
    sampling_rate = config["sampling_rate"]
    file_column = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    
    for ecg_id in ecg_ids:
        logger.info(f"Loading ECG ID {ecg_id}")
        
        # Get the specific ECG record
        ecg_record = df.loc[ecg_id]
        file_path = ecg_record[file_column]
        
        logger.info(f"Loading ECG signal from {file_path}")
        try:
            signal, _ = wfdb.rdsamp(os.path.join(os.environ["DATA_DIR"], file_path))
            # Store without batch dimension for individual processing
            ecg_samples_dict[ecg_id] = signal
            ecg_records_dict[ecg_id] = ecg_record
            logger.info(f"Loaded ECG ID {ecg_id}, signal shape: {signal.shape}")
        except Exception as e:
            logger.error(f"Could not load ECG file {file_path} for ID {ecg_id}: {e}")
            continue
    
    if not ecg_samples_dict:
        raise RuntimeError("No ECG signals could be loaded successfully")
    
    # Model setup (use first sample to determine dimensions)
    first_sample = next(iter(ecg_samples_dict.values()))
    input_channels = first_sample.shape[1]
    num_classes = len(classes)
    
    logger.info(f"Input channels: {input_channels}, Output classes: {num_classes}")
    
    # Load trained model
    model = load_model(args.model, input_channels, num_classes, base_dir, config)
    
    # Get ground truth for each ECG
    ecg_metadata = {}
    for ecg_id in ecg_samples_dict.keys():
        ecg_record = ecg_records_dict[ecg_id]
        
        # Create single-sample annotation dataframe for ground truth extraction
        single_ecg_df = pd.DataFrame([ecg_record]).copy()
        single_ecg_df.index = [ecg_id]
        signal_with_batch = ecg_samples_dict[ecg_id][np.newaxis, ...]  # Add batch dimension for preprocessing
        
        try:
            # Use the existing single ECG dataframe and process labels properly
            _, true_labels, _, _ = preprocess_labels(signal_with_batch, single_ecg_df, config)
            if len(true_labels) > 0:
                true_class_idx = np.argmax(true_labels[0])
                true_class = classes[true_class_idx]
                logger.info(f"ECG ID {ecg_id} - Ground truth class: {true_class} (index: {true_class_idx})")
            else:
                # Fallback: try to get ground truth directly from the ECG record
                diagnostic_superclass = ecg_record.get('diagnostic_superclass', [])
                if diagnostic_superclass and len(diagnostic_superclass) > 0:
                    true_class = diagnostic_superclass[0] if isinstance(diagnostic_superclass, list) else str(diagnostic_superclass)
                    logger.info(f"ECG ID {ecg_id} - Ground truth class from diagnostic_superclass: {true_class}")
                else:
                    true_class = "Unknown"
                    logger.warning(f"ECG ID {ecg_id} - No ground truth labels found in dataset")
        except Exception as e:
            logger.warning(f"ECG ID {ecg_id} - Error determining ground truth: {e}")
            # Fallback: try to get ground truth directly from the ECG record
            try:
                diagnostic_superclass = ecg_record.get('diagnostic_superclass', [])
                if diagnostic_superclass and len(diagnostic_superclass) > 0:
                    true_class = diagnostic_superclass[0] if isinstance(diagnostic_superclass, list) else str(diagnostic_superclass)
                    logger.info(f"ECG ID {ecg_id} - Ground truth class from diagnostic_superclass fallback: {true_class}")
                else:
                    true_class = "Unknown"
                    logger.warning(f"ECG ID {ecg_id} - No diagnostic_superclass found in ECG record")
            except Exception as e2:
                logger.warning(f"ECG ID {ecg_id} - Complete fallback failed: {e2}")
                true_class = "Unknown"
        
        ecg_metadata[ecg_id] = {
            "true_class": true_class,
            "ecg_record": ecg_record
        }

    metadata = {
        "classes": classes, 
        "ecg_metadata": ecg_metadata,
        "processed_ecg_ids": list(ecg_samples_dict.keys())
    }

    return model, ecg_samples_dict, classes, metadata
def analyze_samples_interpretability(model, ecg_samples_dict, classes, args, output_dir, metadata):
    """
    Perform Integrated Gradients analysis on the specified ECG samples.
    Args:
        model: Trained PyTorch model
        ecg_samples_dict: Dict[int, numpy array] - ECG samples keyed by ECG ID
        classes: List of class names
        args: Parsed command line arguments
        output_dir: Directory to save analysis results
        metadata: Dict containing metadata for each ECG ID
    """
    logger.info("Initializing Integrated Gradients analysis...")
    
    # Initialize IG attributor
    ig_explainer = IGAttributor(model)
    
    # Get prediction device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Process each ECG sample
    ecg_metadata = metadata["ecg_metadata"]
    total_ecgs = len(ecg_samples_dict)
    
    for idx, (ecg_id, ecg_sample) in enumerate(ecg_samples_dict.items(), 1):
        logger.info(f"Processing ECG {idx}/{total_ecgs}: ID {ecg_id}")
        
        # Create analysis subdirectory with ECG ID subfolder
        analysis_dir = os.path.join(output_dir, 'sample_attributions', f'ecg_{ecg_id}')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Standardize the ECG sample (using the same approach as training)
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        logger.info(f"Preprocessing ECG ID {ecg_id}")
        logger.info(f"Raw sample shape: {ecg_sample.shape}")
        
        # Create a StandardScaler and fit it on the current sample
        # Note: For a single sample, we'll use simple z-score normalization per channel
        sample_shape = ecg_sample.shape
        standardized_sample = np.zeros_like(ecg_sample)
        
        for channel in range(sample_shape[1]):
            channel_data = ecg_sample[:, channel]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                standardized_sample[:, channel] = (channel_data - mean) / std
            else:
                standardized_sample[:, channel] = channel_data - mean
        
        # Convert to tensor and add batch dimension [1, T, F]
        sample = torch.tensor(standardized_sample, dtype=torch.float32).unsqueeze(0)
        
        logger.info(f"Analyzing ECG ID {ecg_id}")
        logger.info(f"Sample shape: {sample.shape}")
        
        # Get model predictions for this sample
        sample_tensor = sample.to(device)
        
        with torch.no_grad():
            outputs = model(sample_tensor)
            # Get prediction probabilities
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = classes[predicted_class_idx]
            max_probability = probabilities[predicted_class_idx]
        
        # Get ground truth from metadata
        true_class = ecg_metadata[ecg_id]["true_class"]
        
        logger.info(f"ECG ID {ecg_id} - True class: {true_class} | Predicted class: {predicted_class} (confidence: {max_probability:.3f})")
        
        # Create attribution plots for all classes with prediction info
        attribution_data = ig_explainer.visualize_attribution_for_all_classes(
            sample, 
            class_names=classes, 
            save_dir=analysis_dir,
            ecg_id=ecg_id,
            true_class=true_class,
            predicted_class=predicted_class,
            prediction_probabilities=probabilities
        )
        
        logger.info(f"Generated attribution visualizations for ECG ID: {ecg_id}")
    
    logger.info(f"Interpretability analysis completed for all {total_ecgs} ECG samples!")
    logger.info(f"Results saved to: {output_dir}/sample_attributions/")


# Backward compatibility function
def analyze_sample_interpretability(model, ecg_sample, classes, args, output_dir, metadata):
    """
    Legacy function for single ECG analysis - maintained for backward compatibility.
    Converts single ECG to dict format and calls the new multi-ECG function.
    """
    # Convert single ECG to dict format
    ecg_id = args.ecg_id[0] if isinstance(args.ecg_id, list) else args.ecg_id
    ecg_samples_dict = {ecg_id: ecg_sample}
    
    # Convert metadata to new format
    new_metadata = {
        "classes": metadata["classes"],
        "ecg_metadata": {
            ecg_id: {
                "true_class": metadata["true_class"],
                "ecg_record": metadata["ecg_record"]
            }
        }
    }
    
    # Call the new multi-ECG function
    analyze_samples_interpretability(model, ecg_samples_dict, classes, args, output_dir, new_metadata)


def main():
    """
    Main function to run interpretability analysis.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Change to src directory to ensure data loading works correctly
    original_dir = os.getcwd()
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    os.chdir(src_dir)
    
    try:
        # Setup paths and configuration
        base_dir, config, output_dir = setup_paths_and_config(args)
        
        # Configure logging
        log_path = os.path.join(output_dir, "interpretability_analysis.log")
        configure_logger(log_path)
        
        logger.info("="*60)
        logger.info("INTEGRATED GRADIENTS INTERPRETABILITY ANALYSIS")
        logger.info("="*60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Run number: {args.run_number}")
        logger.info(f"ECG IDs: {args.ecg_id}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*60)
        
        # Load model and data
        model, ecg_samples_dict, classes, metadata = load_model_and_data(args, base_dir, config)
        
        # Perform interpretability analysis
        analyze_samples_interpretability(model, ecg_samples_dict, classes, args, output_dir, metadata)
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info(f"Check results in: {output_dir}")
        logger.info("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        # Always restore the original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    load_dotenv()
    main()

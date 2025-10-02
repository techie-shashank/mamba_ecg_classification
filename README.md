# Comparative Study of LSTM, MAMBA, and  Hybrid Architectures for ECG Classification

A comprehensive deep learning framework for time series forecasting using state-of-the-art architectures including LSTM and Mamba models. Designed for ECG classification and extensible to other time series tasks with full interpretability analysis.

## Key Features

- **Multi-Model Support**: LSTM and Mamba architectures for time series classification
- **Interpretability Tools**: Integrated Gradients analysis with multi-ECG batch processing
- **Experiment Management**: Comprehensive tracking with SQLite database and Streamlit dashboard
- **ECG Classification**: Specialized pipeline for PTB-XL dataset with 12-lead ECG analysis
- **Development Tools**: VS Code debugging configurations and organized project structure

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n mamba_new python=3.10
conda activate mamba_new

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Quick start with default configuration
cd scripts
python quick_start.py
```

### 3. Model Training
```bash
# Train LSTM model
python src/train.py --model lstm --config configs/config.json

# Train Mamba model  
python src/train.py --model mamba --config configs/config.json
```

### 4. Interpretability Analysis
```bash
# Single ECG analysis
cd scripts
python interpretability_analysis.py --ecg_id 12345 --model_path ../saved_model/ptbxl/

# Multi-ECG batch analysis
python interpretability_analysis.py --ecg_ids 12345,67890,11111 --model_path ../saved_model/ptbxl/
```

### 5. Experiment Dashboard
```bash
# Launch interactive dashboard
cd scripts
streamlit run experiment_dashboard.py
```

## Project Structure

```
├── configs/           # Configuration files (config.json)
├── data/             # Dataset storage (PTB-XL, etc.)
├── outputs/          # Generated outputs
│   ├── logs/         # All logging output
│   └── interpretability/  # Analysis results
├── scripts/          # User-facing scripts
│   ├── quick_start.py          # Complete pipeline runner
│   ├── interpretability_analysis.py  # Multi-ECG analysis tool
│   ├── experiment_dashboard.py # Streamlit dashboard
│   └── experiment_runner.py    # Hyperparameter search
├── src/              # Core source code
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # LSTM and Mamba architectures
│   ├── evaluation/   # Metrics and interpretability
│   └── train.py      # Training pipeline
└── .vscode/          # VS Code debugging configurations
```

## Development

### VS Code Integration
Pre-configured launch configurations for debugging:
- Train Model
- Test Model  
- Interpretability Analysis
- Quick Start Pipeline

### Logging
All logs automatically saved to `outputs/logs/` with timestamps and experiment tracking.

### Configuration Management
- Main config: `configs/config.json` (absolute paths)
- Legacy config: `config.json` (relative paths, for backward compatibility)

## Citation
If you use this codebase in your research, please cite appropriately.

## License
This project is licensed under the MIT License.


# Mamba Time Series Forecasting

This repository contains code and experiments for time series forecasting using various deep learning models, including LSTM, FCN, and Mamba architectures. The project is organized for reproducible research and extensibility to new datasets and models.

## Project Structure

- `config.json` — Project configuration file.
- `requirements.txt` — Python dependencies.
- `data/` — Raw and processed datasets.
- `experiments/` — Experiment results and logs for different models and runs.
- `notebooks/` — Jupyter notebooks for exploration and analysis.
- `remote_runs/` — Results from remote or distributed runs.
- `saved_model/` — Saved model checkpoints.
- `scripts/` — Utility scripts (e.g., visualizations).
- `src/` — Source code for data loading, training, evaluation, and models.

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   Place your datasets in the `data/` directory as described in the documentation or use the provided scripts to download and preprocess data.

3. **Run experiments**
   Use the scripts in `src/` or Jupyter notebooks in `notebooks/` to train and evaluate models.

## Main Components

- **Data Loading:** `src/data/`
- **Model Architectures:** `src/models/`
- **Training & Evaluation:** `src/train.py`, `src/evaluation/`
- **Visualization:** `scripts/visualizations.py`, `src/visualizations.py`

## Citation
If you use this codebase in your research, please cite appropriately.

## License
This project is licensed under the MIT License.

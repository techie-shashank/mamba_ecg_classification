#!/usr/bin/env python3
"""
Clean Integrated Gradients for ECG Time Series Visualization

Simple implementation focused on 12-lead ECG attribution visualization
using Integrated Gradients for LSTM, MAMBA, and Hybrid models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from captum.attr import IntegratedGradients
from logger import logger
from utils import get_device


class IGAttributor:
    """
    Clean Integrated Gradients implementation for ECG attribution visualization
    """
    
    def __init__(self, model, device=None):
        """
        Initialize Integrated Gradients attributor
        
        Args:
            model: trained PyTorch model (LSTM, MAMBA, Hybrid)
            device: "cuda" or "cpu" (auto-detected if None)
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        
        # For RNN/LSTM models, we need a wrapper that handles training mode
        self.model_wrapper = self._create_model_wrapper(self.model)
        self.ig = IntegratedGradients(self.model_wrapper)
        
        logger.info(f"Initialized IGAttributor on {self.device}")
    
    def _create_model_wrapper(self, model):
        """Create a wrapper that handles LSTM training mode for gradient computation"""
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                # Set to training mode for gradient computation
                original_mode = self.model.training
                self.model.train()
                
                # Disable dropout for consistent results
                for module in self.model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.eval()
                
                try:
                    output = self.model(x)
                finally:
                    # Restore original mode
                    self.model.train(original_mode)
                    
                return output
        
        return ModelWrapper(model)

    def explain(self, inputs, baseline=None, target_label=0):
        """
        Compute IG attributions for a batch of samples
        
        Args:
            inputs: tensor [B, T, F] (batch, timesteps, features)
            baseline: tensor [B, T, F] (default = zeros)
            target_label: int (which label to explain)
            
        Returns:
            attributions: tensor [B, T, F] (same shape as input)
            delta: convergence measure
        """
        inputs = inputs.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(inputs).to(self.device)

        attributions, delta = self.ig.attribute(
            inputs=inputs,
            baselines=baseline,
            target=target_label,
            return_convergence_delta=True
        )
        return attributions, delta

    def plot_12lead_attribution(self, sample, attributions, ecg_id, label_name, model_name=None, 
                               save_path=None, true_class=None, predicted_class=None, class_probability=None):
        """
        Plot all 12 ECG leads with attribution overlays
        
        Args:
            sample: np.array [T, 12] - ECG signal data
            attributions: np.array [T, 12] - attribution values
            ecg_id: int - ECG identifier
            label_name: str - class label name
            model_name: str - model name (optional)
            save_path: str - path to save plot (optional)
            true_class: str - ground truth class
            predicted_class: str - predicted class
            class_probability: float - prediction probability for this class
        """
        T, L = sample.shape
        assert L == 12, f"Expected 12 leads, got {L}"
        
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        # Normalize attribution per lead for visualization (avoid divide-by-zero)
        max_per_lead = np.max(np.abs(attributions), axis=0)
        max_per_lead[max_per_lead == 0] = 1.0
        norm_attr = np.abs(attributions) / max_per_lead  # in [0,1] per lead

        fig, axes = plt.subplots(12, 1, figsize=(14, 2.0*12), sharex=True)
        time = np.arange(T)
        
        for i in range(12):
            # Plot ECG signal
            axes[i].plot(time, sample[:, i], color='black', linewidth=0.7)
            
            # Get y-axis limits for scaling attribution
            ymin, ymax = axes[i].get_ylim()
            
            # Fill with normalized attribution scaled to signal amplitude range
            fill = norm_attr[:, i] * (ymax - ymin) * 0.8  # scale to visually fit
            axes[i].fill_between(time, ymin, ymin + fill, color='red', alpha=0.4)
            
            # Set lead name as y-label
            axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=20, va='center')
            axes[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axes[i].grid(True, alpha=0.3)

        # Set title with prediction information
        supt = f"ECG ID {ecg_id} — Attribution for {label_name}"
        if class_probability is not None:
            supt += f" (P={class_probability:.3f})"
        if true_class and predicted_class:
            prediction_info = f"True: {true_class} | Pred: {predicted_class}"
            supt += f"\n{prediction_info}"
        if model_name:
            supt += f" — Model: {model_name}"
        plt.suptitle(supt, fontsize=12)
        plt.xlabel("Time (samples)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved 12-lead attribution plot: {save_path}")
        else:
            plt.show()

    def visualize_attribution_for_all_classes(self, sample, class_names, save_dir=None, ecg_id=None, 
                                             true_class=None, predicted_class=None, prediction_probabilities=None,
                                             model_name=None):
        """
        Create 12-lead ECG attribution visualizations for all classes
        
        Args:
            sample: tensor [1, T, F] - single ECG sample
            class_names: List of class names
            save_dir: Directory to save plots (optional)
            ecg_id: ECG identifier for plot titles
            true_class: Ground truth class name
            predicted_class: Model's predicted class name
            prediction_probabilities: Array of prediction probabilities for all classes
            model_name: str - actual model name (LSTM, Mamba, Hybrid)
            
        Returns:
            dict: Attribution data for all classes
        """
        sample = sample.to(self.device)
        signal = sample.squeeze().cpu().numpy()  # [T, F]
        
        # Ensure we have 12 leads
        if signal.shape[1] != 12:
            logger.warning(f"Expected 12 ECG leads, got {signal.shape[1]}. Using available leads.")
            # Pad or truncate to 12 leads if needed
            if signal.shape[1] < 12:
                # Pad with zeros
                padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                signal = np.concatenate([signal, padding], axis=1)
            else:
                # Truncate to first 12 leads
                signal = signal[:, :12]
        
        attribution_results = {}
        
        for i, label in enumerate(class_names):
            # Get attributions for this class
            attributions, delta = self.explain(sample, target_label=i)
            attr = attributions.squeeze().cpu().detach().numpy()  # [T, F]
            
            # Ensure attributions match signal dimensions
            if attr.shape[1] != 12:
                if attr.shape[1] < 12:
                    # Pad with zeros
                    padding = np.zeros((attr.shape[0], 12 - attr.shape[1]))
                    attr = np.concatenate([attr, padding], axis=1)
                else:
                    # Truncate to first 12 leads
                    attr = attr[:, :12]
            
            # Store results
            attribution_results[label] = {
                'attributions': attr,
                'convergence_delta': delta.item()
            }
            
            # Create 12-lead visualization
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"ig_attribution_12lead_{label}.png")
                
                # Get prediction probability for this class
                class_prob = prediction_probabilities[i] if prediction_probabilities is not None else None
                
                self.plot_12lead_attribution(
                    signal, attr, ecg_id, label, 
                    model_name=model_name or "Unknown",  # Use passed model name or fallback
                    save_path=save_path,
                    true_class=true_class,
                    predicted_class=predicted_class,
                    class_probability=class_prob
                )
            else:
                class_prob = prediction_probabilities[i] if prediction_probabilities is not None else None
                self.plot_12lead_attribution(
                    signal, attr, ecg_id, label,
                    true_class=true_class,
                    predicted_class=predicted_class, 
                    class_probability=class_prob
                )
        
        return attribution_results

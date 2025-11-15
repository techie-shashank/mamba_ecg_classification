#!/usr/bin/env python3
"""
Shared Embedding Extraction Module

Provides unified embedding extraction functionality for both t-SNE visualization and linear probing.
Ensures consistency between evaluations and avoids redundant computation.
"""

import numpy as np
import torch
from logger import logger


def extract_model_embeddings(model, data_loader, model_type: str, device):
    """
    Extract embeddings from a specific model type - shared function for t-SNE and linear probing
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial', 'hybrid_serial_rev', 'hybrid_parallel', 'hybrid_crossattn')
        device: Device to run inference on
        
    Returns:
        tuple: (embeddings, labels) as numpy arrays
    """
    model.eval()
    embeddings = []
    labels = []
    
    # Store original forward method
    original_forward = None
    captured_embedding = None
    
    def capture_embedding_hook(module, input, output):
        nonlocal captured_embedding
        if model_type.lower() == 'lstm':
            # For LSTM, output is (output, (h_n, c_n))
            # We want the hidden state h_n from the last layer
            if isinstance(output, tuple) and len(output) >= 2:
                h_n, c_n = output[1]  # (h_n, c_n)
                # h_n shape: (num_layers, batch, hidden_size)
                # Take the last layer's hidden state
                captured_embedding = h_n[-1].detach()  # (batch, hidden_size)
            else:
                # Fallback: use the main output
                captured_embedding = output.detach()
        elif model_type.lower() == 'mamba':
            # For Mamba, we capture the last layer's output before classification
            captured_embedding = output.detach()
        elif model_type.lower() in ['hybrid_serial', 'hybrid_serial_rev', 'hybrid_parallel', 'hybrid_crossattn']:
            # For hybrid models, capture the input to the FC layer (embeddings after fusion/processing)
            if len(input) > 0:
                captured_embedding = input[0].detach()  # Input to FC layer
            else:
                captured_embedding = output.detach()
        else:
            # Default: use output as-is
            captured_embedding = output.detach()
    
    # Register hook based on model type
    hook_handle = None
    try:
        if model_type.lower() == 'lstm':
            # Hook to the LSTM layer
            for name, module in model.named_modules():
                if 'lstm' in name.lower() and hasattr(module, 'forward'):
                    hook_handle = module.register_forward_hook(capture_embedding_hook)
                    break
        elif model_type.lower() == 'mamba':
            # Hook to the last Mamba block or backbone
            for name, module in model.named_modules():
                if ('mamba' in name.lower() or 'backbone' in name.lower()) and hasattr(module, 'forward'):
                    hook_handle = module.register_forward_hook(capture_embedding_hook)
        elif model_type.lower() in ['hybrid_serial', 'hybrid_serial_rev', 'hybrid_parallel', 'hybrid_crossattn']:
            # For hybrid models, hook to the FC layer to capture its input (embeddings)
            hook_registered = False
            for name, module in model.named_modules():
                if 'fc' in name.lower() and hasattr(module, 'forward'):
                    hook_handle = module.register_forward_hook(capture_embedding_hook)
                    logger.debug(f"Registered hook for {model_type} on FC module: {name}")
                    hook_registered = True
                    break
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, batch_labels = batch[0].to(device), batch[1].to(device)
                
                # Reset captured embedding
                captured_embedding = None
                
                # Forward pass
                _ = model(inputs)
                
                # Check if embedding was captured
                if captured_embedding is not None:
                    # Handle different embedding shapes
                    if captured_embedding.dim() > 2:
                        # If 3D, take the last timestep or mean across time
                        captured_embedding = captured_embedding.mean(dim=1)  # Mean across time dimension
                    
                    embeddings.append(captured_embedding.cpu().numpy())
                    labels.append(batch_labels.cpu().numpy())
                else:
                    logger.warning(f"No embedding captured for batch {batch_idx} in {model_type}")
        
    finally:
        # Clean up hook
        if hook_handle is not None:
            hook_handle.remove()
    
    if len(embeddings) > 0:
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        logger.info(f"Extracted {model_type} embeddings: shape {embeddings.shape}, labels shape: {labels.shape}")
        return embeddings, labels
    else:
        logger.warning(f"No embeddings extracted for {model_type}")
        return np.array([]), np.array([])


def extract_embeddings_from_dataloaders(model, train_loader, test_loader, model_type: str, device):
    """
    Extract embeddings from both train and test dataloaders - shared function
    
    Args:
        model: The trained model
        train_loader: Training DataLoader (can be None for t-SNE only)
        test_loader: Test DataLoader
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial', 'hybrid_serial_rev', 'hybrid_parallel', 'hybrid_crossattn')
        device: Device to run inference on
        
    Returns:
        tuple: (train_embeddings, train_labels, test_embeddings, test_labels)
               train_embeddings and train_labels will be empty arrays if train_loader is None
    """
    logger.info(f"Extracting embeddings from dataloaders for {model_type}...")
    
    # Extract test embeddings (always needed)
    test_embeddings, test_labels = extract_model_embeddings(model, test_loader, model_type, device)
    
    # Extract training embeddings (only if train_loader provided)
    if train_loader is not None:
        train_embeddings, train_labels = extract_model_embeddings(model, train_loader, model_type, device)
    else:
        train_embeddings, train_labels = np.array([]), np.array([])
        logger.info("No train_loader provided, skipping training embeddings extraction")
    
    return train_embeddings, train_labels, test_embeddings, test_labels

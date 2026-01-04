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
        model_type: Type of model ('lstm', 'mamba', 'patchtst', 'autoformer', 'hybrid_serial', 'hybrid_serial_rev', 'hybrid_parallel', 'hybrid_crossattn', 'resnet50')
        device: Device to run inference on
        
    Returns:
        tuple: (embeddings, labels) as numpy arrays
    """
    model.eval()
    embeddings = []
    labels = []
    captured_embedding = None
    
    def capture_embedding_hook(module, input, output):
        nonlocal captured_embedding
        # Capture the input to the classification layer (the learned embeddings)
        if len(input) > 0:
            captured_embedding = input[0].detach()
        else:
            captured_embedding = output.detach()
    
    # Register hook to the final classification layer
    hook_handle = None
    try:
        # Try to directly access the fc attribute (most reliable)
        if hasattr(model, 'fc'):
            hook_handle = model.fc.register_forward_hook(capture_embedding_hook)
            logger.debug(f"Registered hook for {model_type} on model.fc")
        else:
            # Fallback: search for fc module by name
            for name, module in model.named_modules():
                if name == 'fc' or name.endswith('.fc'):
                    hook_handle = module.register_forward_hook(capture_embedding_hook)
                    logger.debug(f"Registered hook for {model_type} on {name}")
                    break
        
        if hook_handle is None:
            logger.warning(f"Could not find FC layer for {model_type}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, batch_labels = batch[0].to(device), batch[1].to(device)
                captured_embedding = None
                
                # Forward pass
                _ = model(inputs)
                
                if captured_embedding is not None:
                    # Handle 3D embeddings by averaging across time dimension
                    if captured_embedding.dim() > 2:
                        captured_embedding = captured_embedding.mean(dim=1)
                    
                    embeddings.append(captured_embedding.cpu().numpy())
                    labels.append(batch_labels.cpu().numpy())
                else:
                    logger.warning(f"No embedding captured for batch {batch_idx} in {model_type}")
        
    finally:
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
        model_type: Type of model ('lstm', 'mamba', 'patchtst', 'hybrid_serial', 'hybrid_serial_rev', 'hybrid_parallel', 'hybrid_crossattn', 'resnet50')
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

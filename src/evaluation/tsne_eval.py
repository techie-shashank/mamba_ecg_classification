#!/usr/bin/env python3
"""
t-SNE Evaluation Module

Generates t-SNE visualizations during model evaluation and saves them to the experiment plots folder.
Uses shared embedding extraction for consistency with linear probing.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from logger import logger
from evaluation.embedding_extractor import extract_model_embeddings


def generate_tsne_from_embeddings(embeddings, labels, model_type: str, class_names, save_path: str, is_multilabel: bool = False):
    """
    Generate t-SNE visualization from pre-extracted embeddings
    
    Args:
        embeddings: Pre-extracted embeddings array
        labels: Corresponding labels array
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial')
        class_names: List of class names
        save_path: Path to save the plot
        is_multilabel: Whether this is a multilabel classification task
    """
    try:
        logger.info(f"Generating t-SNE visualization from pre-extracted embeddings for {model_type.upper()}...")
        
        if len(embeddings) == 0:
            logger.warning(f"No embeddings provided for {model_type}, skipping t-SNE")
            return
        
        # Determine perplexity based on sample size
        n_samples = len(embeddings)
        perplexity = min(30, max(5, n_samples // 3))  # Ensure perplexity is reasonable
        logger.info(f"Using perplexity: {perplexity} for {n_samples} samples")
        
        # Apply t-SNE transformation
        logger.info("Applying t-SNE transformation...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualizations based on task type
        if is_multilabel:
            # For multilabel, create separate plots for each class
            logger.info("Creating separate t-SNE visualizations for each class...")
            plots_dir = os.path.dirname(save_path)
            per_class_dir = os.path.join(plots_dir, 'tsne_visualization_per_class')
            os.makedirs(per_class_dir, exist_ok=True)
            
            for i, class_name in enumerate(class_names):
                plt.figure(figsize=(10, 8))
                
                # Get class-specific labels
                class_labels = labels[:, i]
                
                # Plot points with different colors for presence/absence of class
                scatter_positive = plt.scatter(embeddings_2d[class_labels == 1, 0], 
                                             embeddings_2d[class_labels == 1, 1], 
                                             c='red', alpha=0.6, s=20, label=f'{class_name} Present')
                scatter_negative = plt.scatter(embeddings_2d[class_labels == 0, 0], 
                                             embeddings_2d[class_labels == 0, 1], 
                                             c='blue', alpha=0.6, s=20, label=f'{class_name} Absent')
                
                plt.title(f't-SNE Visualization - {class_name} Class ({model_type.upper()})')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save individual class plot
                class_plot_path = os.path.join(per_class_dir, f'{class_name.lower()}_tsne.png')
                plt.tight_layout()
                plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"t-SNE plot for {class_name} saved to: {class_plot_path}")
            
            # Create multilabel overview plot
            logger.info("Creating multilabel overview t-SNE plot...")
            plt.figure(figsize=(16, 10))
            
            # Define distinct colors for single classes
            base_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']  # Vibrant distinct colors
            
            # Create label combination strings for each sample
            label_combinations = []
            for sample_labels in labels:
                active_classes = [class_names[i] for i in range(len(class_names)) if sample_labels[i] == 1]
                if len(active_classes) == 0:
                    label_combinations.append('None')
                else:
                    label_combinations.append(' + '.join(sorted(active_classes)))
            
            # Find unique label combinations and their counts
            from collections import Counter
            combination_counts = Counter(label_combinations)
            
            # Determine threshold for "considerable" number of samples
            total_samples = len(label_combinations)
            min_threshold = max(3, total_samples * 0.05)  # At least 3 samples or 5% of total
            
            # Separate significant combinations from others
            significant_combinations = []
            other_combinations = []
            
            for combo, count in combination_counts.items():
                if count >= min_threshold:
                    significant_combinations.append(combo)
                else:
                    other_combinations.append(combo)
            
            # Sort significant combinations by count (descending)
            significant_combinations.sort(key=lambda x: combination_counts[x], reverse=True)
            
            logger.info(f"Found {len(significant_combinations)} significant label combinations (≥{min_threshold:.0f} samples):")
            for combo in significant_combinations:
                count = combination_counts[combo]
                logger.info(f"  {combo}: {count} samples")
            
            if other_combinations:
                others_count = sum(combination_counts[combo] for combo in other_combinations)
                logger.info(f"Grouping {len(other_combinations)} rare combinations as 'Others': {others_count} samples")
            
            # Create color mapping for significant combinations
            import matplotlib.colors as mcolors
            combination_colors = {}
            markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '+', '<', '>', '1', '2', '3', '4']
            
            for i, combo in enumerate(significant_combinations):
                if combo == 'None':
                    combination_colors[combo] = '#CCCCCC'  # Gray for no labels
                elif ' + ' not in combo:  # Single class
                    # Find which single class this is
                    for j, class_name in enumerate(class_names):
                        if combo == class_name:
                            combination_colors[combo] = base_colors[j]
                            break
                else:  # Multiple classes - create a mixed color
                    # For combinations, use distinct colors
                    combo_classes = combo.split(' + ')
                    if len(combo_classes) == 2:
                        combination_colors[combo] = '#8E44AD'  # Purple for 2-class combinations
                    elif len(combo_classes) == 3:
                        combination_colors[combo] = '#D35400'  # Orange for 3-class combinations
                    else:
                        combination_colors[combo] = '#2C3E50'  # Dark blue for 4+ class combinations
            
            # Add color for "Others"
            if other_combinations:
                combination_colors['Others'] = '#7F8C8D'  # Medium gray for others
            
            # Plot significant combinations
            for i, combo in enumerate(significant_combinations):
                combo_mask = np.array([label_combinations[j] == combo for j in range(len(label_combinations))])
                count = combination_counts[combo]
                marker = markers[i % len(markers)]
                
                # Determine marker size based on combination complexity and frequency
                if combo == 'None':
                    size = 30
                    alpha = 0.4
                elif ' + ' not in combo:  # Single class
                    size = max(40, min(80, 30 + count * 2))  # Size based on frequency
                    alpha = 0.8
                else:  # Multiple classes
                    size = max(50, min(100, 40 + count * 2 + len(combo.split(' + ')) * 5))
                    alpha = 0.9
                
                plt.scatter(embeddings_2d[combo_mask, 0], 
                          embeddings_2d[combo_mask, 1], 
                          c=combination_colors[combo], 
                          alpha=alpha, 
                          s=size,
                          marker=marker,
                          label=f'{combo} ({count})',
                          edgecolors='white' if ' + ' not in combo else 'black', 
                          linewidth=0.8 if ' + ' not in combo else 1.2)
            
            # Plot "Others" if they exist
            if other_combinations:
                others_mask = np.array([label_combinations[j] in other_combinations for j in range(len(label_combinations))])
                others_count = np.sum(others_mask)
                
                if others_count > 0:
                    plt.scatter(embeddings_2d[others_mask, 0], 
                              embeddings_2d[others_mask, 1], 
                              c=combination_colors['Others'], 
                              alpha=0.5, 
                              s=25,
                              marker='.',
                              label=f'Others ({others_count})',
                              edgecolors='none')
            
            plt.title(f'Multilabel t-SNE Overview - Major Label Combinations ({model_type.upper()})', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('t-SNE Component 1', fontsize=14)
            plt.ylabel('t-SNE Component 2', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
            plt.grid(True, alpha=0.3)
            
        else:
            # For single-label classification
            logger.info("Creating single-label t-SNE visualization...")
            plt.figure(figsize=(10, 8))
            
            # Convert labels if needed
            if labels.ndim > 1:
                plot_labels = np.argmax(labels, axis=1)
            else:
                plot_labels = labels
            
            # Create scatter plot with different colors for each class
            unique_labels = np.unique(plot_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = plot_labels == label
                class_name = class_names[int(label)] if int(label) < len(class_names) else f'Class {label}'
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=[colors[i]], alpha=0.6, s=20, label=class_name)
            
            plt.title(f't-SNE Visualization ({model_type.upper()})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
        
        # Save the main plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_type = "Multilabel overview" if is_multilabel else "Binary classification"
        logger.info(f"{plot_type} t-SNE plot saved to: {save_path}")
        
        # Save embedding data for potential future use
        embeddings_path = save_path.replace('.png', '_embeddings.npz')
        np.savez(embeddings_path, embeddings_2d=embeddings_2d, labels=labels, 
                 original_embeddings=embeddings)
        logger.info(f"t-SNE embedding data saved to: {embeddings_path}")
        
    except Exception as e:
        logger.error(f"Error generating t-SNE plot for {model_type}: {e}")
        import traceback
        traceback.print_exc()


def generate_tsne_plot(model, data_loader, model_type: str, class_names, save_path: str, device, is_multilabel: bool = False):
    """
    Generate t-SNE visualization for a single model and save to plots folder
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial')
        class_names: List of class names
        save_path: Path to save the plot
        device: Device to run inference on
        is_multilabel: Whether this is a multilabel classification task
    """
    # Extract embeddings using shared function
    embeddings, labels = extract_model_embeddings(model, data_loader, model_type, device)
    
    # Generate t-SNE from extracted embeddings
    generate_tsne_from_embeddings(embeddings, labels, model_type, class_names, save_path, is_multilabel)


def evaluate_with_tsne(model, test_loader, model_type: str, class_names, plots_dir: str, device, is_multilabel: bool = False):
    """
    Generate t-SNE visualization for model evaluation
    
    Args:
        model: The trained model
        test_loader: Test DataLoader
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial')
        class_names: List of class names
        plots_dir: Directory to save t-SNE plots
        device: Device to run inference on
        is_multilabel: Whether this is a multilabel classification task
    """
    logger.info(f"Generating t-SNE visualization for {model_type.upper()}...")
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate t-SNE plot
    tsne_plot_path = os.path.join(plots_dir, 'tsne_visualization.png')
    generate_tsne_plot(model, test_loader, model_type, class_names, tsne_plot_path, device, is_multilabel)


def evaluate_with_tsne_from_embeddings(test_embeddings, test_labels, model_type: str, class_names, plots_dir: str, is_multilabel: bool = False):
    """
    Generate t-SNE visualization from pre-extracted embeddings
    
    Args:
        test_embeddings: Pre-extracted test embeddings
        test_labels: Pre-extracted test labels
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial')
        class_names: List of class names
        plots_dir: Directory to save t-SNE plots
        is_multilabel: Whether this is a multilabel classification task
    """
    logger.info(f"Generating t-SNE visualization from pre-extracted embeddings for {model_type.upper()}...")
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate t-SNE plot from embeddings
    tsne_plot_path = os.path.join(plots_dir, 'tsne_visualization.png')
    generate_tsne_from_embeddings(test_embeddings, test_labels, model_type, class_names, tsne_plot_path, is_multilabel)

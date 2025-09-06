#!/usr/bin/env python3
"""
Linear Probing Evaluation Module

Implements linear probing on extracted embeddings to evaluate representation quality.
Provides fair comparison across model architectures with PCA normalization and consistent preprocessing.
Uses shared embedding extraction for consistency with t-SNE visualization.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

from logger import logger
from evaluation.embedding_extractor import extract_embeddings_from_dataloaders


def plot_linear_probe_training(train_embeddings, train_labels, test_embeddings, test_labels,
                              model_type: str, plots_dir: str, is_multilabel: bool = False,
                              random_seed: int = 42):
    """
    Create learning curve plot for linear probing
    
    Args:
        train_embeddings: Training embeddings
        train_labels: Training labels  
        test_embeddings: Test embeddings
        test_labels: Test labels
        model_type: Type of model
        plots_dir: Directory to save plots
        is_multilabel: Whether multilabel classification
        random_seed: Random seed
    """
    try:
        logger.info(f"Creating linear probe learning curve for {model_type.upper()}...")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Standardize embeddings
        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        
        # Configure classifier
        if is_multilabel:
            base_classifier = LogisticRegression(random_state=random_seed, max_iter=1000, solver='liblinear')
            classifier = MultiOutputClassifier(base_classifier)
            scoring = 'accuracy'
        else:
            classifier = LogisticRegression(random_state=random_seed, max_iter=1000, solver='liblinear')
            scoring = 'accuracy'
        
        # Learning curves
        logger.info("Generating learning curves...")
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            classifier, train_embeddings_scaled, 
            train_labels,
            train_sizes=train_sizes,
            cv=3,
            scoring=scoring,
            n_jobs=1,
            random_state=random_seed
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create the learning curve plot
        plt.figure(figsize=(10, 6))
        
        # Plot training and validation curves
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', linewidth=2, 
                label=f'Training Score', markersize=6)
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', linewidth=2, 
                label=f'Validation Score', markersize=6)
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title(f'Linear Probe Learning Curves - {model_type.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add final scores annotation
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        plt.text(0.02, 0.98, 
                f'Final Training: {final_train:.3f} ± {train_std[-1]:.3f}\\n'
                f'Final Validation: {final_val:.3f} ± {val_std[-1]:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'linear_probe_learning_curve_{model_type}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Linear probe learning curve saved to: {plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating linear probe learning curve: {e}")
        import traceback
        traceback.print_exc()


def linear_probe_evaluation(train_embeddings, train_labels, test_embeddings, test_labels, 
                           model_type: str, class_names, metrics_dir: str, is_multilabel: bool = False,
                           use_pca: bool = True, pca_components: int = 64, random_seed: int = 42,
                           create_plots: bool = True):
    """
    Perform linear probing evaluation on extracted embeddings
    
    Args:
        train_embeddings: Training embeddings array
        train_labels: Training labels array
        test_embeddings: Test embeddings array  
        test_labels: Test labels array
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial')
        class_names: List of class names
        metrics_dir: Directory to save metrics
        is_multilabel: Whether this is multilabel classification
        use_pca: Whether to apply PCA for dimensionality reduction
        pca_components: Number of PCA components to keep
        random_seed: Random seed for reproducibility
        create_plots: Whether to create training plots
        
    Returns:
        dict: Linear probing metrics
    """
    try:
        logger.info(f"Starting linear probing evaluation for {model_type.upper()}...")
        
        # Check if we have enough samples
        if len(train_embeddings) == 0 or len(test_embeddings) == 0:
            logger.warning(f"Insufficient data for linear probing: train={len(train_embeddings)}, test={len(test_embeddings)}")
            return {}
        
        # Create training plots if requested
        if create_plots:
            plots_dir = os.path.join(os.path.dirname(metrics_dir), 'plots')
            plot_linear_probe_training(
                train_embeddings, train_labels, test_embeddings, test_labels,
                model_type, plots_dir, is_multilabel, random_seed
            )
        
        # Standardize embeddings
        logger.info("Standardizing embeddings...")
        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        test_embeddings_scaled = scaler.transform(test_embeddings)
        
        # Apply PCA if requested for fair comparison across models
        if use_pca and train_embeddings_scaled.shape[1] > pca_components:
            logger.info(f"Applying PCA to reduce dimensionality to {pca_components} components...")
            pca = PCA(n_components=pca_components, random_state=random_seed)
            train_embeddings_scaled = pca.fit_transform(train_embeddings_scaled)
            test_embeddings_scaled = pca.transform(test_embeddings_scaled)
            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Configure classifier based on task type
        if is_multilabel:
            # Multilabel classification
            logger.info("Configuring multilabel linear probe...")
            base_classifier = LogisticRegression(
                random_state=random_seed,
                max_iter=1000,
                solver='liblinear'
            )
            classifier = MultiOutputClassifier(base_classifier)
            
            # Fit classifier
            classifier.fit(train_embeddings_scaled, train_labels)
            
            # Predictions
            test_pred = classifier.predict(test_embeddings_scaled)
            test_pred_proba = np.array([est.predict_proba(test_embeddings_scaled)[:, 1] 
                                      for est in classifier.estimators_]).T
            
            # Calculate metrics for each class
            metrics = {}
            
            # Overall metrics
            accuracy = accuracy_score(test_labels, test_pred)
            metrics['overall_accuracy'] = accuracy
            
            # Per-class metrics
            class_metrics = {}
            for i, class_name in enumerate(class_names):
                class_true = test_labels[:, i]
                class_pred = test_pred[:, i]
                class_proba = test_pred_proba[:, i]
                
                # Basic metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    class_true, class_pred, average='binary', zero_division=0
                )
                
                # AUC if possible
                try:
                    auc = roc_auc_score(class_true, class_proba)
                except:
                    auc = 0.0
                
                class_metrics[class_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'auc': float(auc)
                }
            
            metrics['per_class_metrics'] = class_metrics
            
            # Macro averages
            macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
            macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
            macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
            macro_auc = np.mean([m['auc'] for m in class_metrics.values()])
            
            metrics['macro_precision'] = float(macro_precision)
            metrics['macro_recall'] = float(macro_recall)
            metrics['macro_f1'] = float(macro_f1)
            metrics['macro_auc'] = float(macro_auc)
            
        else:
            # Binary/multiclass classification
            logger.info("Configuring binary/multiclass linear probe...")
            classifier = LogisticRegression(
                random_state=random_seed,
                max_iter=1000,
                solver='liblinear',
                multi_class='ovr' if len(class_names) > 2 else 'auto'
            )
            
            # Use labels directly for binary/multiclass classification
            train_labels_single = train_labels
            test_labels_single = test_labels
            
            # Fit classifier
            classifier.fit(train_embeddings_scaled, train_labels_single)
            
            # Predictions
            test_pred = classifier.predict(test_embeddings_scaled)
            test_pred_proba = classifier.predict_proba(test_embeddings_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels_single, test_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels_single, test_pred, average='macro', zero_division=0
            )
            
            # AUC for binary classification
            if len(class_names) == 2:
                try:
                    auc = roc_auc_score(test_labels_single, test_pred_proba[:, 1])
                except:
                    auc = 0.0
            else:
                try:
                    auc = roc_auc_score(test_labels_single, test_pred_proba, multi_class='ovr')
                except:
                    auc = 0.0
            
            metrics = {
                'accuracy': float(accuracy),
                'macro_precision': float(precision),
                'macro_recall': float(recall),
                'macro_f1': float(f1),
                'macro_auc': float(auc)
            }
            
            # Per-class metrics for multiclass
            if len(class_names) > 2:
                class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                    test_labels_single, test_pred, average=None, zero_division=0
                )
                
                class_metrics = {}
                for i, class_name in enumerate(class_names):
                    class_metrics[class_name] = {
                        'precision': float(class_precision[i]),
                        'recall': float(class_recall[i]),
                        'f1': float(class_f1[i])
                    }
                
                metrics['per_class_metrics'] = class_metrics
        
        # Cross-validation for robustness
        logger.info("Performing cross-validation...")
        cv_folds = min(5, max(2, len(train_embeddings_scaled) // 10))  # Ensure at least 2 folds
        logger.info(f"Using {cv_folds} CV folds for {len(train_embeddings_scaled)} training samples...")
        
        try:
            cv_scores = cross_val_score(
                classifier, train_embeddings_scaled, 
                train_labels if is_multilabel else train_labels_single,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=1  # Use single job to avoid hanging
            )
            logger.info(f"Cross-validation completed. Scores: {cv_scores}")
        except Exception as cv_error:
            logger.warning(f"Cross-validation failed: {cv_error}. Skipping CV metrics.")
            cv_scores = np.array([0.0])  # Fallback
        
        metrics['cv_mean_accuracy'] = float(cv_scores.mean())
        metrics['cv_std_accuracy'] = float(cv_scores.std())
        logger.info(f"Cross-validation results: mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Add metadata
        logger.info("Adding metadata to metrics...")
        metrics['model_type'] = model_type
        metrics['embedding_dim'] = int(train_embeddings_scaled.shape[1])
        metrics['train_samples'] = int(len(train_embeddings_scaled))
        metrics['test_samples'] = int(len(test_embeddings_scaled))
        metrics['is_multilabel'] = is_multilabel
        metrics['used_pca'] = use_pca
        metrics['pca_components'] = pca_components if use_pca else None
        
        # Save metrics
        logger.info(f"Saving metrics to directory: {metrics_dir}")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, f'linear_probe_{model_type}_metrics.json')
        
        logger.info(f"Writing metrics to file: {metrics_file}")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Linear probe metrics saved to: {metrics_file}")
        accuracy_key = 'accuracy' if 'accuracy' in metrics else 'overall_accuracy'
        accuracy_value = metrics.get(accuracy_key, 0)
        logger.info(f"Linear probe accuracy: {accuracy_value:.3f}")
        logger.info("Linear probing evaluation completed successfully!")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in linear probing evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}





def evaluate_linear_probe(model, train_loader, test_loader, model_type: str, 
                         class_names, metrics_dir: str, device, 
                         is_multilabel: bool = False, use_pca: bool = True, 
                         pca_components: int = 64, random_seed: int = 42,
                         create_plots: bool = True):
    """
    Comprehensive linear probing evaluation
    
    Args:
        model: The trained model
        train_loader: Training DataLoader for linear probing
        test_loader: Test DataLoader for linear probing
        model_type: Type of model ('lstm', 'mamba', 'hybrid_serial')
        class_names: List of class names
        metrics_dir: Directory to save linear probe metrics
        device: Device to run inference on
        is_multilabel: Whether this is a multilabel classification task
        use_pca: Whether to apply PCA for fair comparison across models
        pca_components: Number of PCA components to keep
        random_seed: Random seed for reproducibility
        create_plots: Whether to create training plots
        
    Returns:
        dict: Linear probing metrics
    """
    logger.info(f"Starting linear probing evaluation for {model_type.upper()}...")
    
    # Extract embeddings from both train and test sets for linear probing
    logger.info("Extracting embeddings for linear probing...")
    train_embeddings, train_labels, test_embeddings, test_labels = extract_embeddings_from_dataloaders(
        model, train_loader, test_loader, model_type, device
    )
    
    # Perform linear probing evaluation
    if len(train_embeddings) > 0 and len(test_embeddings) > 0:
        logger.info("Performing linear probing evaluation...")
        probe_metrics = linear_probe_evaluation(
            train_embeddings, train_labels, test_embeddings, test_labels,
            model_type, class_names, metrics_dir, is_multilabel, 
            use_pca, pca_components, random_seed, create_plots
        )
        
        return probe_metrics
    else:
        logger.warning("Insufficient embeddings for linear probing")
        return {}

# ========== Standard Library Imports ==========
import os
import json

# ========== Third-Party Imports ==========
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ========== Local Imports ==========
from evaluation.metrics import calculate_binary_classification_metrics, calculate_multilabel_metrics
from evaluation.plots import plot_multilabel_confusion_matrices
from evaluation.tsne_eval import evaluate_with_tsne, generate_tsne_from_embeddings
from evaluation.linear_prob import evaluate_linear_probe, linear_probe_evaluation
from evaluation.embedding_extractor import extract_embeddings_from_dataloaders, extract_model_embeddings

from utils import get_device
from logger import logger


def evaluate_and_save_metrics(model, criterion, test_loader, config, classes, save_dir, logger, 
                             model_type=None, generate_tsne=False, train_loader=None, 
                             enable_linear_probe=False):
    """
    Evaluate the model and save metrics to disk.
    Args:
        model (torch.nn.Module): Trained model.
        criterion: Loss function.
        test_loader: DataLoader for test set.
        config (dict): Configuration dict.
        classes (list): Class names.
        save_dir (str): Directory to save metrics.
        logger: Logger instance.
        model_type (str): Type of model for t-SNE ('lstm', 'mamba', 'hybrid_serial').
        generate_tsne (bool): Whether to generate t-SNE visualization.
        train_loader: DataLoader for train set (needed for linear probing).
        enable_linear_probe (bool): Whether to perform linear probing evaluation.
    """
    is_multilabel = config['is_multilabel']
    device = get_device()
    
    _, _, all_probs, all_labels, all_preds = evaluate_model(
        model, test_loader, criterion, is_multilabel
    )
    
    # Calculate and store metrics
    calculate_store_metrics(
        all_labels, all_probs, all_preds, save_dir=f"{save_dir}/metrics_results",
        class_names=classes, is_multilabel=is_multilabel
    )
    
    # Extract embeddings once and use for both t-SNE and linear probing
    if (generate_tsne or enable_linear_probe) and model_type:
        logger.info("Extracting embeddings for evaluation...")
        
        # Determine what embeddings we need
        need_train_embeddings = enable_linear_probe and train_loader is not None
        
        if need_train_embeddings:
            # Extract both train and test embeddings for linear probing
            train_embeddings, train_labels, test_embeddings, test_labels = extract_embeddings_from_dataloaders(
                model, train_loader, test_loader, model_type, device
            )
        else:
            # Only extract test embeddings for t-SNE
            test_embeddings, test_labels = extract_model_embeddings(
                model, test_loader, model_type, device
            )
            train_embeddings, train_labels = None, None
    
    # Generate t-SNE visualization from extracted embeddings
    if generate_tsne and model_type and 'test_embeddings' in locals():
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        try:
            logger.info("Generating t-SNE visualization from extracted embeddings...")
            tsne_save_path = os.path.join(plots_dir, f'tsne_{model_type}.png')
            generate_tsne_from_embeddings(
                test_embeddings, test_labels, model_type, classes, 
                tsne_save_path, is_multilabel
            )
        except Exception as e:
            logger.warning(f"Failed to generate t-SNE visualization: {e}")
    elif generate_tsne and not model_type:
        logger.warning("t-SNE generation requested but model_type not provided")
    
    # Perform linear probing evaluation from extracted embeddings
    if enable_linear_probe and model_type and train_loader is not None and 'train_embeddings' in locals():
        metrics_dir = os.path.join(save_dir, 'metrics_results')
        try:
            logger.info("Performing linear probing evaluation from extracted embeddings...")
            probe_metrics = linear_probe_evaluation(
                train_embeddings, train_labels, test_embeddings, test_labels,
                model_type, classes, metrics_dir, is_multilabel, 
                use_pca=True, pca_components=64, random_seed=42
            )
            
            if probe_metrics:
                logger.info(f"Linear probe evaluation completed. Key metrics:")
                if is_multilabel:
                    logger.info(f"  Overall accuracy: {probe_metrics.get('overall_accuracy', 0):.3f}")
                    logger.info(f"  Macro F1: {probe_metrics.get('macro_f1', 0):.3f}")
                else:
                    logger.info(f"  Accuracy: {probe_metrics.get('accuracy', 0):.3f}")
                    logger.info(f"  Macro F1: {probe_metrics.get('macro_f1', 0):.3f}")
                logger.info(f"  Cross-validation accuracy: {probe_metrics.get('cv_mean_accuracy', 0):.3f} ± {probe_metrics.get('cv_std_accuracy', 0):.3f}")
                
        except Exception as e:
            logger.warning(f"Failed to perform linear probing evaluation: {e}")


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    criterion,
    is_multilabel
) -> tuple:
    """
    Evaluate the model on the test set and return (avg_test_loss, test_acc, all_probs, all_labels, all_preds).
    Args:
        model (torch.nn.Module): Trained model.
        test_loader: DataLoader for test set.
        criterion: Loss function.
        is_multilabel (bool): Whether the task is multi-label.
    Returns:
        tuple: (avg_test_loss, test_acc, all_probs, all_labels, all_preds)
    """
    import time

    device = get_device()
    model.eval()

    # Timing variables
    total_inference_time = 0.0
    total_data_transfer_time = 0.0
    batch_times = []

    all_probs = []
    all_labels = []
    all_preds = []
    test_loss = 0.0
    correct, total = 0, 0

    logger.info(f"Starting evaluation on {len(test_loader)} batches")
    eval_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_start = time.time()

            # Data transfer timing
            transfer_start = time.time()
            inputs, labels = batch[0].to(device), batch[1].to(device)
            transfer_time = time.time() - transfer_start
            total_data_transfer_time += transfer_time

            # Inference timing
            inference_start = time.time()
            outputs = model(inputs)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            if is_multilabel:
                labels_for_loss = labels.float()
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                correct += (preds == labels.cpu().numpy().astype(int)).all(axis=1).sum()
            else:
                labels_for_loss = labels.long()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                correct += (preds == labels.cpu().numpy()).sum()
            labels = labels.cpu().numpy()
            loss = criterion(outputs, labels_for_loss)
            test_loss += loss.item()
            all_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
            total += len(labels)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Log progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                avg_batch_time = sum(batch_times[-20:]) / min(20, len(batch_times))
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches | Avg batch time: {avg_batch_time:.4f}s")

    total_eval_time = time.time() - eval_start

    if total == 0:
        logger.warning("Test set is empty. No evaluation performed.")
        return 0.0, 0.0, np.array([]), np.array([]), np.array([])

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    # Calculate timing statistics
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_inference_per_batch = total_inference_time / len(test_loader) if len(test_loader) > 0 else 0
    avg_transfer_per_batch = total_data_transfer_time / len(test_loader) if len(test_loader) > 0 else 0
    samples_per_second = total / total_eval_time if total_eval_time > 0 else 0

    # Log comprehensive timing results
    logger.info(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Inference timing - Total: {total_inference_time:.3f}s | Avg per batch: {avg_inference_per_batch:.4f}s | Data transfer: {total_data_transfer_time:.3f}s")
    logger.info(f"Performance - {samples_per_second:.1f} samples/sec | {avg_batch_time:.4f}s/batch | {total} samples processed")
    return avg_test_loss, test_acc, all_probs, all_labels, all_preds


def calculate_store_metrics(
    y_true,
    y_prob,
    y_pred,
    save_dir: str = "metrics_results",
    class_names=None,
    is_multilabel: bool = False,
) -> None:
    """
    Evaluate model predictions and save metrics, including classification report.
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        y_pred: Predicted labels.
        save_dir (str): Directory to save metrics.
        class_names (list, optional): Class names.
        is_multilabel (bool): Whether the task is multi-label.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if is_multilabel:
        logger.info("[Evaluation] Running Multi-Label Classification Evaluation...")
        metrics = calculate_multilabel_metrics(
            y_true_multi=y_true,
            y_pred_multi=y_pred,
            y_prob_multi=y_prob,
            class_names=class_names
        )
        logger.info("\n[Multi-Label Classification Results]")
        logger.info(f"Macro Precision : {metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall    : {metrics['macro_recall']:.4f}")
        logger.info(f"Macro F1 Score  : {metrics['macro_f1']:.4f}")
        logger.info(f"Macro ROC AUC   : {metrics['macro_roc_auc']:.4f}")
        plot_multilabel_confusion_matrices(
            np.array(metrics['multilabel_confusion_matrix']),
            class_names,
            os.path.join(save_dir, 'multilabel_confusion_matrices.png')
        )
    else:
        logger.info("[Evaluation] Running Binary Classification Evaluation...")
        # For binary classification, extract positive class probabilities for AUC calculation
        y_prob_positive_class = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
        
        metrics = calculate_binary_classification_metrics(
            y_true_binary=y_true,
            y_pred_binary=y_pred,
            y_prob_binary=y_prob_positive_class,  # Use positive class probabilities
            class_names=class_names
        )
    
        # Save confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close(fig)
        logger.info(f"Confusion matrix plot saved at {cm_path}")

        logger.info("\n[Binary Classification Results]")
        logger.info(f"Accuracy      : {metrics['binary_accuracy']:.4f}")
        logger.info(f"Precision     : {metrics['binary_precision']:.4f}")
        logger.info(f"Recall        : {metrics['binary_recall']:.4f}")
        logger.info(f"F1 Score      : {metrics['binary_f1']:.4f}")
        logger.info(f"ROC AUC       : {metrics['binary_auc']:.4f}")

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"[Evaluation] Metrics saved at {save_dir}")

"""
Evaluation metrics for NeuroVoice models.

Includes accuracy, ROC-AUC, F1 score, precision, recall, and confusion matrix.
"""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate accuracy."""
    return accuracy_score(y_true, y_pred)


def calculate_roc_auc(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    average: str = "macro",
) -> float:
    """Calculate ROC-AUC score."""
    try:
        return roc_auc_score(y_true, y_pred_proba, average=average)
    except ValueError:
        # Handle case where only one class is present
        return 0.0


def calculate_f1_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    average: str = "macro",
) -> float:
    """Calculate F1 score."""
    return f1_score(y_true, y_pred, average=average)


def calculate_precision(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    average: str = "macro",
) -> float:
    """Calculate precision."""
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    average: str = "macro",
) -> float:
    """Calculate recall (sensitivity)."""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def calculate_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_pred: Predicted labels (binary)
        y_true: True labels
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': calculate_accuracy(y_pred, y_true),
        'f1_score': calculate_f1_score(y_pred, y_true),
        'precision': calculate_precision(y_pred, y_true),
        'recall': calculate_recall(y_pred, y_true),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = calculate_roc_auc(y_pred_proba, y_true)
    
    return metrics


def get_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_pred: np.ndarray, y_true: np.ndarray) -> str:
    """Get detailed classification report."""
    return classification_report(y_true, y_pred)


def convert_logits_to_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> tuple:
    """
    Convert model logits to binary predictions and probabilities.
    
    Args:
        logits: Model output logits (batch, num_classes)
        threshold: Classification threshold for binary classification
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    # Get probabilities
    if logits.shape[1] == 1:
        # Binary classification (single output)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= threshold).long()
    else:
        # Multi-class or binary with two outputs
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    return preds.cpu().numpy(), probs.cpu().numpy()


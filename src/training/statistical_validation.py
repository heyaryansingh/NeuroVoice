"""
Statistical validation utilities for model evaluation.

Includes bootstrap confidence intervals, paired t-tests, and cross-validation.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.model_selection import StratifiedKFold


def bootstrap_confidence_intervals(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        predictions: Model predictions
        labels: True labels
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with confidence intervals for each metric
    """
    np.random.seed(random_seed)
    
    n_samples = len(predictions)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    from src.training.metrics import (
        calculate_accuracy,
        calculate_roc_auc,
        calculate_f1_score,
        calculate_precision,
        calculate_recall,
    )
    
    # Bootstrap samples
    bootstrap_accuracies = []
    bootstrap_f1_scores = []
    bootstrap_precisions = []
    bootstrap_recalls = []
    
    # For ROC-AUC, we need probabilities
    # Assuming predictions are binary, convert to probabilities
    predictions_proba = np.zeros((len(predictions), 2))
    predictions_proba[np.arange(len(predictions)), predictions] = 1.0
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]
        boot_proba = predictions_proba[indices]
        
        try:
            accuracy = calculate_accuracy(boot_preds, boot_labels)
            f1 = calculate_f1_score(boot_preds, boot_labels)
            precision = calculate_precision(boot_preds, boot_labels)
            recall = calculate_recall(boot_preds, boot_labels)
            
            bootstrap_accuracies.append(accuracy)
            bootstrap_f1_scores.append(f1)
            bootstrap_precisions.append(precision)
            bootstrap_recalls.append(recall)
        except Exception:
            continue
    
    def get_ci(values):
        if len(values) == 0:
            return (0.0, 0.0)
        return (
            np.percentile(values, lower_percentile),
            np.percentile(values, upper_percentile)
        )
    
    return {
        'accuracy': get_ci(bootstrap_accuracies),
        'f1_score': get_ci(bootstrap_f1_scores),
        'precision': get_ci(bootstrap_precisions),
        'recall': get_ci(bootstrap_recalls),
    }


def paired_t_test(
    metric1: np.ndarray,
    metric2: np.ndarray,
    metric_name: str = "metric",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Perform paired t-test between two sets of metrics.
    
    Useful for comparing two models on the same dataset.
    
    Args:
        metric1: Metrics from first model (e.g., accuracy across folds)
        metric2: Metrics from second model
        metric_name: Name of metric being compared
        alpha: Significance level
    
    Returns:
        Dictionary with t-statistic, p-value, and conclusion
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(metric1, metric2)
    
    # Determine significance
    is_significant = p_value < alpha
    conclusion = "significant" if is_significant else "not significant"
    
    # Calculate mean difference
    mean_diff = np.mean(metric1) - np.mean(metric2)
    
    return {
        'metric': metric_name,
        'mean_diff': float(mean_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'alpha': alpha,
        'is_significant': is_significant,
        'conclusion': conclusion,
    }


def stratified_cross_validation(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        model_fn: Function that takes (X_train, y_train, X_val, y_val) and returns metrics dict
        X: Features
        y: Labels
        n_splits: Number of folds (default: 5)
        random_seed: Random seed
    
    Returns:
        Dictionary with metrics arrays (one value per fold)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    all_metrics = {
        'accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and evaluate model
        metrics = model_fn(X_train, y_train, X_val, y_val)
        
        for key in all_metrics.keys():
            if key in metrics:
                all_metrics[key].append(metrics[key])
    
    # Convert to numpy arrays
    return {k: np.array(v) for k, v in all_metrics.items()}


def compare_models(
    metrics_dict1: Dict[str, np.ndarray],
    metrics_dict2: Dict[str, np.ndarray],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two models using paired t-tests on all metrics.
    
    Args:
        metrics_dict1: Metrics from first model (arrays of values)
        metrics_dict2: Metrics from second model
        model1_name: Name of first model
        model2_name: Name of second model
        alpha: Significance level
    
    Returns:
        Dictionary with comparison results for each metric
    """
    results = {}
    
    # Find common metrics
    common_metrics = set(metrics_dict1.keys()) & set(metrics_dict2.keys())
    
    for metric_name in common_metrics:
        metric1 = metrics_dict1[metric_name]
        metric2 = metrics_dict2[metric_name]
        
        if len(metric1) == len(metric2):
            results[metric_name] = paired_t_test(
                metric1,
                metric2,
                metric_name=metric_name,
                alpha=alpha,
            )
    
    return results


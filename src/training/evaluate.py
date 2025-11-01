"""
Evaluation script for NeuroVoice models.

Evaluates trained models on test sets and generates metrics and visualizations.
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DEVICE_CONFIG, EVALUATION_CONFIG, METRICS_DIR, SPLITS_DIR, DATA_DIR
from src.models.fusion_model import FusionModel
from src.data_loaders import get_multimodal_dataloader
from src.training.metrics import (
    calculate_metrics,
    convert_logits_to_predictions,
    get_confusion_matrix,
    get_classification_report,
)
from src.training.statistical_validation import (
    bootstrap_confidence_intervals,
    paired_t_test,
)
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve
import pandas as pd


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    disease: str,
    save_dir: Path = None,
    bootstrap_ci: bool = False,
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        disease: Disease name for saving results
        save_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    save_dir = save_dir or METRICS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for batch in pbar:
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            disease_idx = batch['disease'].to(device)
            
            # Forward pass
            outputs = model(audio, video, disease_idx)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Convert to predictions
            preds, probs = convert_logits_to_predictions(
                logits,
                threshold=EVALUATION_CONFIG["threshold"],
            )
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1] if probs.shape[1] > 1 else probs.flatten())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = calculate_metrics(all_preds, all_labels, all_probs)
    
    # Bootstrap confidence intervals if requested
    if bootstrap_ci:
        try:
            ci_results = bootstrap_confidence_intervals(all_preds, all_labels)
            metrics['confidence_intervals'] = ci_results
            print("\nBootstrap Confidence Intervals (95%):")
            for metric_name, (lower, upper) in ci_results.items():
                print(f"  {metric_name}: [{lower:.4f}, {upper:.4f}]")
        except Exception as e:
            print(f"Warning: Could not compute bootstrap CIs: {e}")
    
    # Confusion matrix
    cm = get_confusion_matrix(all_preds, all_labels)
    
    # Classification report
    report = get_classification_report(all_preds, all_labels)
    
    print(f"\n=== Evaluation Results for {disease} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Save results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(save_dir / f"{disease}_test_metrics.csv", index=False)
    
    # Save confusion matrix
    if EVALUATION_CONFIG["save_confusion_matrix"]:
        plot_confusion_matrix(
            cm,
            save_path=save_dir / f"{disease}_confusion_matrix.png",
            class_names=["Healthy", "Disease"],
        )
    
    # Save ROC curve
    if EVALUATION_CONFIG["save_roc_curve"]:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plot_roc_curve(
            fpr, tpr,
            auc=metrics.get('roc_auc', 0),
            save_path=save_dir / f"{disease}_roc_curve.png",
        )
    
    return metrics


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model: Model architecture
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate NeuroVoice model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--disease", type=str, required=True, choices=["alzheimer", "parkinson", "depression"])
    parser.add_argument("--test_data", type=str, default=str(SPLITS_DIR), help="Path to test data CSV")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--bootstrap_ci", action="store_true", help="Compute bootstrap confidence intervals")
    parser.add_argument("--audio_feature_type", type=str, default="wav2vec")
    parser.add_argument("--video_feature_type", type=str, default="landmarks")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    test_csv = Path(args.test_data) / f"{args.disease}_test.csv"
    if not test_csv.exists():
        # Try default location
        test_csv = SPLITS_DIR / f"{args.disease}_test.csv"
    
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found. Expected: {test_csv}\n"
            "Run scripts/create_labels.py and scripts/split_data.py first."
        )
    
    test_df = pd.read_csv(test_csv)
    print(f"Loaded {len(test_df)} test samples")
    
    # Create test loader
    test_loader = get_multimodal_dataloader(
        audio_dir=DATA_DIR,
        video_dir=DATA_DIR,
        labels_df=test_df,
        audio_feature_type=args.audio_feature_type,
        video_feature_type=args.video_feature_type,
        batch_size=16,
        shuffle=False,
        num_workers=4,
    )
    
    # Load model
    model = FusionModel(
        audio_input_dim=768,
        video_input_dim=None,
    )
    model = load_model_checkpoint(model, Path(args.model_path), device)
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        disease=args.disease,
        bootstrap_ci=args.bootstrap_ci,
    )
    
    print(f"\n=== Final Results for {args.disease} ===")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()


"""
Tests for model training components.

Tests loss functions, optimizers, metrics, and training utilities.
"""

import pytest
import torch
import numpy as np

from src.training.losses import WeightedBCELoss, FocalLoss, MultitaskLoss
from src.training.metrics import (
    calculate_accuracy,
    calculate_roc_auc,
    calculate_f1_score,
    convert_logits_to_predictions,
)
from src.training.optimizer import get_optimizer, get_scheduler
from src.models.fusion_model import FusionModel


def test_weighted_bce_loss():
    """Test weighted binary cross-entropy loss."""
    criterion = WeightedBCELoss(pos_weight=2.0)
    
    batch_size = 8
    num_classes = 2
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    loss = criterion(logits, targets)
    assert loss.item() > 0
    assert isinstance(loss, torch.Tensor)


def test_focal_loss():
    """Test focal loss."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    batch_size = 8
    num_classes = 2
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    loss = criterion(logits, targets)
    assert loss.item() > 0


def test_multitask_loss():
    """Test multi-task loss."""
    classification_loss = WeightedBCELoss()
    criterion = MultitaskLoss(
        classification_loss=classification_loss,
        auxiliary_weight=0.1,
    )
    
    batch_size = 8
    
    logits = torch.randn(batch_size, 2)
    targets = torch.randint(0, 2, (batch_size,))
    aux_logits = torch.randn(batch_size, 7)  # 7 emotions
    aux_targets = torch.randint(0, 7, (batch_size,))
    
    loss_dict = criterion(logits, targets, aux_logits, aux_targets)
    
    assert 'total_loss' in loss_dict
    assert 'classification_loss' in loss_dict
    assert 'auxiliary_loss' in loss_dict
    assert loss_dict['total_loss'].item() > 0


def test_metrics():
    """Test evaluation metrics."""
    # Create dummy predictions and labels
    y_pred = np.array([0, 1, 1, 0, 1, 0])
    y_true = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.2, 0.8, 0.6, 0.3, 0.9, 0.4])
    
    accuracy = calculate_accuracy(y_pred, y_true)
    assert 0 <= accuracy <= 1
    
    roc_auc = calculate_roc_auc(y_proba, y_true)
    assert 0 <= roc_auc <= 1
    
    f1 = calculate_f1_score(y_pred, y_true)
    assert 0 <= f1 <= 1


def test_convert_logits_to_predictions():
    """Test logit to prediction conversion."""
    batch_size = 8
    num_classes = 2
    
    logits = torch.randn(batch_size, num_classes)
    preds, probs = convert_logits_to_predictions(logits)
    
    assert len(preds) == batch_size
    assert len(probs) == batch_size
    assert all(p in [0, 1] for p in preds)
    assert all(0 <= p <= 1 for prob_arr in probs for p in prob_arr)


def test_optimizer():
    """Test optimizer creation."""
    model = FusionModel(audio_input_dim=768, video_input_dim=None)
    
    optimizer = get_optimizer(model, optimizer_type="adam", learning_rate=1e-4)
    assert optimizer is not None
    
    optimizer_adamw = get_optimizer(model, optimizer_type="adamw")
    assert optimizer_adamw is not None
    
    optimizer_sgd = get_optimizer(model, optimizer_type="sgd")
    assert optimizer_sgd is not None


def test_scheduler():
    """Test learning rate scheduler creation."""
    model = FusionModel(audio_input_dim=768, video_input_dim=None)
    optimizer = get_optimizer(model)
    
    scheduler = get_scheduler(optimizer, scheduler_type="cosine", num_epochs=50)
    assert scheduler is not None
    
    scheduler_step = get_scheduler(optimizer, scheduler_type="step", num_epochs=50)
    assert scheduler_step is not None


if __name__ == "__main__":
    pytest.main([__file__])


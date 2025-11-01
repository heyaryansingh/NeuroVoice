"""
Optimizer and scheduler utilities for training.
"""

from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR,
)

from src.config import OPTIMIZER_CONFIG, SCHEDULER_CONFIG


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = None,
    learning_rate: float = None,
    weight_decay: float = None,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Get optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_type or OPTIMIZER_CONFIG["type"]
    learning_rate = learning_rate or OPTIMIZER_CONFIG["lr"]
    weight_decay = weight_decay or OPTIMIZER_CONFIG["weight_decay"]
    
    # Get optimizer parameters
    params = model.parameters()
    
    if optimizer_type.lower() == "adam":
        betas = kwargs.get("betas", OPTIMIZER_CONFIG["betas"])
        optimizer = optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif optimizer_type.lower() == "adamw":
        betas = kwargs.get("betas", OPTIMIZER_CONFIG["betas"])
        optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif optimizer_type.lower() == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", False)
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = None,
    num_epochs: int = None,
    warmup_epochs: int = None,
    **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', None)
        num_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler instance (or None if scheduler_type is None)
    """
    scheduler_type = scheduler_type or SCHEDULER_CONFIG["type"]
    
    if scheduler_type is None:
        return None
    
    num_epochs = num_epochs or SCHEDULER_CONFIG["T_max"]
    warmup_epochs = warmup_epochs or SCHEDULER_CONFIG["warmup_epochs"]
    
    if scheduler_type.lower() == "cosine":
        T_max = kwargs.get("T_max", num_epochs)
        eta_min = kwargs.get("eta_min", SCHEDULER_CONFIG["eta_min"])
        
        # Warmup + cosine annealing
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    return (1 + np.cos(np.pi * (epoch - warmup_epochs) / (T_max - warmup_epochs))) / 2
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
    
    elif scheduler_type.lower() == "step":
        step_size = kwargs.get("step_size", num_epochs // 3)
        gamma = kwargs.get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type.lower() == "plateau":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 5)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


"""
Gradient analysis and diagnostics for training monitoring.

Provides utilities for recording and visualizing gradient statistics,
including layer-wise norms, parameter variance, and gradient noise scale.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def compute_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> Dict[str, float]:
    """
    Compute gradient norms for each layer.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm (2.0 for L2, 1.0 for L1)
    
    Returns:
        Dictionary mapping parameter names to gradient norms
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(norm_type).item()
            grad_norms[name] = grad_norm
        else:
            grad_norms[name] = 0.0
    return grad_norms


def compute_parameter_variance(model: nn.Module) -> Dict[str, float]:
    """
    Compute parameter variance for each layer.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping parameter names to variances
    """
    variances = {}
    for name, param in model.named_parameters():
        if param.data.numel() > 0:
            variances[name] = param.data.var().item()
    return variances


def compute_gradient_noise_scale(grad_norms: Dict[str, float], batch_size: int) -> float:
    """
    Compute gradient noise scale (LeCun-style).
    
    Args:
        grad_norms: Dictionary of gradient norms
        batch_size: Training batch size
    
    Returns:
        Gradient noise scale value
    """
    total_grad_norm = np.sqrt(sum(v**2 for v in grad_norms.values()))
    noise_scale = total_grad_norm / np.sqrt(batch_size)
    return noise_scale


def log_gradients(
    model: nn.Module,
    step: int,
    writer: Optional[SummaryWriter] = None,
    batch_size: int = 16,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Log gradient statistics to TensorBoard and/or file.
    
    Args:
        model: PyTorch model
        step: Training step number
        writer: TensorBoard SummaryWriter (optional)
        batch_size: Batch size for noise scale calculation
        log_to_file: Whether to save to file
        log_dir: Directory to save logs
    
    Returns:
        Dictionary of gradient statistics
    """
    grad_norms = compute_gradient_norm(model)
    param_vars = compute_parameter_variance(model)
    noise_scale = compute_gradient_noise_scale(grad_norms, batch_size)
    
    # Aggregate statistics
    stats = {
        'step': step,
        'mean_grad_norm': float(np.mean(list(grad_norms.values()))),
        'max_grad_norm': float(np.max(list(grad_norms.values()))),
        'min_grad_norm': float(np.min(list(grad_norms.values()))),
        'median_grad_norm': float(np.median(list(grad_norms.values()))),
        'std_grad_norm': float(np.std(list(grad_norms.values()))),
        'mean_param_var': float(np.mean(list(param_vars.values()))),
        'gradient_noise_scale': float(noise_scale),
        'total_params': sum(p.numel() for p in model.parameters()),
        'params_with_grad': sum(1 for p in model.parameters() if p.grad is not None),
        'zero_grad_params': sum(1 for p in model.parameters() if p.grad is None),
    }
    
    # Log to TensorBoard
    if writer:
        # Log individual layer norms
        for name, norm in grad_norms.items():
            writer.add_scalar(f'Gradients/Norm/{name.replace(".", "/")}', norm, step)
        
        # Log aggregated statistics
        writer.add_scalar('Gradients/Stats/MeanNorm', stats['mean_grad_norm'], step)
        writer.add_scalar('Gradients/Stats/MaxNorm', stats['max_grad_norm'], step)
        writer.add_scalar('Gradients/Stats/MinNorm', stats['min_grad_norm'], step)
        writer.add_scalar('Gradients/Stats/MedianNorm', stats['median_grad_norm'], step)
        writer.add_scalar('Gradients/Stats/StdNorm', stats['std_grad_norm'], step)
        writer.add_scalar('Gradients/Stats/NoiseScale', stats['gradient_noise_scale'], step)
        writer.add_scalar('Gradients/Stats/MeanParamVar', stats['mean_param_var'], step)
        writer.add_scalar('Gradients/Stats/ParamsWithGrad', stats['params_with_grad'], step)
        writer.add_scalar('Gradients/Stats/ZeroGradParams', stats['zero_grad_params'], step)
        
        # Log histogram of gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/Hist/{name.replace(".", "/")}', param.grad, step)
    
    # Log to file
    if log_to_file and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "gradient_stats.json"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = []
        
        all_stats.append(stats)
        with open(log_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
    
    return stats


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradients and return the total norm before clipping.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm
    
    Returns:
        Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type,
    )
    return total_norm.item()


def get_layer_gradient_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Get detailed gradient statistics per layer.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping layer names to their gradient statistics
    """
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_stats[name] = {
                'grad_norm_l2': param.grad.data.norm(2.0).item(),
                'grad_norm_l1': param.grad.data.norm(1.0).item(),
                'grad_max': param.grad.data.max().item(),
                'grad_min': param.grad.data.min().item(),
                'grad_mean': param.grad.data.mean().item(),
                'grad_std': param.grad.data.std().item(),
                'param_norm_l2': param.data.norm(2.0).item(),
                'param_mean': param.data.mean().item(),
                'param_std': param.data.std().item(),
            }
        else:
            layer_stats[name] = {
                'grad_norm_l2': 0.0,
                'grad_norm_l1': 0.0,
                'grad_max': 0.0,
                'grad_min': 0.0,
                'grad_mean': 0.0,
                'grad_std': 0.0,
                'param_norm_l2': param.data.norm(2.0).item(),
                'param_mean': param.data.mean().item(),
                'param_std': param.data.std().item(),
            }
    
    return layer_stats


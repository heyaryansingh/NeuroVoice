"""
Experiment management and tracking for NeuroVoice.

Provides version control for experiments, hyperparameters, and results.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ExperimentManager:
    """
    Manage experiment tracking, versioning, and result storage.
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: Path = None,
        create_timestamp: bool = True,
    ):
        """
        Initialize experiment manager.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments (default: outputs/experiments)
            create_timestamp: Whether to append timestamp to experiment name
        """
        if base_dir is None:
            from src.config import OUTPUTS_DIR
            base_dir = OUTPUTS_DIR / "experiments"
        
        if create_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "metrics").mkdir(exist_ok=True)
        (self.experiment_dir / "configs").mkdir(exist_ok=True)
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        
        self.hyperparameters = {}
        self.metrics_history = []
        self.config = {}
    
    def save_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Save hyperparameters to JSON file.
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        self.hyperparameters = hyperparameters
        hyperparams_path = self.experiment_dir / "hyperparameters.json"
        
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparameters, f, indent=2)
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        config_path = self.experiment_dir / "configs" / "config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Log metrics to history.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Epoch number (optional)
            step: Step number (optional)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }
        
        if epoch is not None:
            log_entry['epoch'] = epoch
        if step is not None:
            log_entry['step'] = step
        
        self.metrics_history.append(log_entry)
        
        # Save to file
        metrics_path = self.experiment_dir / "metrics" / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            epoch: Current epoch
            metrics: Current metrics (optional)
            is_best: Whether this is the best model so far
            additional_info: Additional info to save (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'hyperparameters': self.hyperparameters,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.experiment_dir / "checkpoints" / "best_model.pt"
            torch.save(checkpoint, best_path)
            shutil.copy(best_path, self.experiment_dir / "best_model.pt")
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (optional, uses best if None)
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            load_best: Whether to load best model
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.experiment_dir / "best_model.pt"
            else:
                # Load most recent checkpoint
                checkpoints = sorted(self.experiment_dir.glob("checkpoints/checkpoint_epoch_*.pt"))
                if checkpoints:
                    checkpoint_path = checkpoints[-1]
                else:
                    raise FileNotFoundError("No checkpoints found")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def save_gradient_stats(self, gradient_stats: Dict[str, float], step: int):
        """
        Save gradient statistics.
        
        Args:
            gradient_stats: Dictionary of gradient statistics
            step: Training step
        """
        grad_path = self.experiment_dir / "metrics" / "gradient_stats.json"
        
        if grad_path.exists():
            with open(grad_path, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = []
        
        all_stats.append({'step': step, **gradient_stats})
        
        with open(grad_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.
        
        Returns:
            Dictionary with experiment summary
        """
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_dir': str(self.experiment_dir),
            'hyperparameters': self.hyperparameters,
            'num_metrics_logs': len(self.metrics_history),
            'checkpoints': [str(p) for p in (self.experiment_dir / "checkpoints").glob("*.pt")],
        }
        
        if self.metrics_history:
            # Get latest metrics
            summary['latest_metrics'] = self.metrics_history[-1]['metrics']
        
        return summary
    
    def save_summary(self):
        """Save experiment summary to file."""
        summary = self.get_summary()
        summary_path = self.experiment_dir / "summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


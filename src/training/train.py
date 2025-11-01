"""
Main training script for NeuroVoice models.

Implements the training loop with validation, checkpointing, and logging.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DATASET_CONFIGS,
    DEVICE_CONFIG,
    TRAINING_CONFIG,
    get_model_path,
    get_log_path,
    MODELS_DIR,
    LOGS_DIR,
    DATA_DIR,
    SPLITS_DIR,
)
from src.data_loaders import get_multimodal_dataloader
from src.models.fusion_model import FusionModel, EarlyFusionModel, LateFusionModel
from src.training.losses import WeightedBCELoss, FocalLoss, MultitaskLoss, ContrastiveLoss
from src.training.metrics import calculate_metrics, convert_logits_to_predictions
from src.training.optimizer import get_optimizer, get_scheduler
from src.training.gradient_analysis import log_gradients, clip_gradients
from src.training.experiment_manager import ExperimentManager
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------
# Training / validation epoch functions (unchanged from your base)
# ---------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_accumulation: int = 1,
    is_sam: bool = False,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # For SAM optimizer
    if is_sam:
        from src.training.advanced_optimizers import SAM
    
    optimizer.zero_grad()
    accumulation_steps = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Move to device (handle optional missing keys gracefully)
        audio = batch.get('audio')
        video = batch.get('video')
        labels = batch['label']

        if audio is not None:
            audio = audio.to(device)
        if video is not None:
            video = video.to(device)
        labels = labels.to(device)
        disease_idx = batch.get('disease')
        if disease_idx is not None:
            disease_idx = disease_idx.to(device)

        # Forward pass
        outputs = model(audio, video, disease_idx) if (audio is not None or video is not None) else model(None, None, disease_idx)
        
        # Calculate loss
        if isinstance(outputs, dict):
            logits = outputs['logits']
            if isinstance(criterion, MultitaskLoss):
                loss_dict = criterion(
                    logits=logits.squeeze(1) if len(logits.shape) > 2 else logits,
                    targets=labels,
                    auxiliary_logits=outputs.get('emotion_logits'),
                )
                loss = loss_dict['total_loss']
            else:
                loss = criterion(logits.squeeze(1) if len(logits.shape) > 2 else logits, labels)
        else:
            logits = outputs
            loss = criterion(logits, labels)
        
        # For SAM: need closure function
        if is_sam:
            def closure():
                optimizer.zero_grad()
                outputs_sam = model(audio, video, disease_idx) if (audio is not None or video is not None) else model(None, None, disease_idx)
                
                if isinstance(outputs_sam, dict):
                    logits_sam = outputs_sam['logits']
                    if isinstance(criterion, MultitaskLoss):
                        loss_dict = criterion(
                            logits=logits_sam.squeeze(1) if len(logits_sam.shape) > 2 else logits_sam,
                            targets=labels,
                            auxiliary_logits=outputs_sam.get('emotion_logits'),
                        )
                        loss_sam = loss_dict['total_loss']
                    else:
                        loss_sam = criterion(logits_sam.squeeze(1) if len(logits_sam.shape) > 2 else logits_sam, labels)
                else:
                    loss_sam = criterion(logits_sam, labels)
                
                loss_sam = loss_sam / gradient_accumulation
                loss_sam.backward()
                return loss_sam
            
            loss = optimizer.step(closure)
            actual_loss = loss.item() * gradient_accumulation if hasattr(loss, 'item') else float(loss) * gradient_accumulation
        else:
            loss = loss / gradient_accumulation
            loss.backward()
            accumulation_steps += 1
            
            if accumulation_steps % gradient_accumulation == 0:
                # Clip gradients
                clip_gradients(model, max_norm=TRAINING_CONFIG.get("gradient_clip", 1.0))
                optimizer.step()
                optimizer.zero_grad()
                accumulation_steps = 0
            
            actual_loss = loss.item() * gradient_accumulation

        # Accumulate loss
        total_loss += actual_loss
        preds, _ = convert_logits_to_predictions(logits)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Log gradients if writer provided
        if writer and batch_idx % 100 == 0:
            step = global_step * len(dataloader) + batch_idx
            log_gradients(model, step, writer, batch_size=dataloader.batch_size, log_to_file=False)

        # Update progress bar
        pbar.set_postfix({'loss': actual_loss})

    # Calculate epoch metrics
    avg_loss = total_loss / max(1, len(dataloader))
    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
    metrics['loss'] = avg_loss

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            # Move to device (handle optional missing keys gracefully)
            audio = batch.get('audio')
            video = batch.get('video')
            labels = batch['label']

            if audio is not None:
                audio = audio.to(device)
            if video is not None:
                video = video.to(device)
            labels = labels.to(device)
            disease_idx = batch.get('disease')
            if disease_idx is not None:
                disease_idx = disease_idx.to(device)

            # Forward pass
            outputs = model(audio, video, disease_idx) if (audio is not None or video is not None) else model(None, None, disease_idx)

            # Calculate loss
            if isinstance(outputs, dict):
                logits = outputs['logits']
                if isinstance(criterion, MultitaskLoss):
                    loss_dict = criterion(
                        logits=logits.squeeze(1) if len(logits.shape) > 2 else logits,
                        targets=labels,
                        auxiliary_logits=outputs.get('emotion_logits'),
                    )
                    loss = loss_dict['total_loss']
                else:
                    loss = criterion(logits.squeeze(1) if len(logits.shape) > 2 else logits, labels)
            else:
                logits = outputs
                loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            preds, probs = convert_logits_to_predictions(logits)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Determine probability column for positive class
            probs_arr = np.asarray(probs)
            if probs_arr.ndim == 1:
                all_probs.extend(probs_arr.tolist())
            else:
                # assume probs shape: (B, C) -> take class 1 probability
                all_probs.extend(probs_arr[:, 1].tolist())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

    # Calculate epoch metrics
    avg_loss = total_loss / max(1, len(dataloader))
    metrics = calculate_metrics(
        np.array(all_preds),
        np.array(all_labels),
        y_pred_proba=np.array(all_probs) if len(all_probs) > 0 else None,
    )
    metrics['loss'] = avg_loss

    return metrics


# ---------------------------------------------------------------------
# Main training loop that performs checkpointing + logging
# ---------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    num_epochs: int,
    disease: str,
    save_dir: Path = None,
    logger=None,
    gradient_accumulation: int = 1,
    is_sam: bool = False,
    writer: Optional[SummaryWriter] = None,
    exp_manager: Optional[ExperimentManager] = None,
) -> Dict:
    save_dir = save_dir or MODELS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_val_auc = 0.0
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': [],
    }

    logger = logger or setup_logger(get_log_path(disease))
    logger.info(f"Starting training for {disease}")

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            gradient_accumulation=gradient_accumulation,
            is_sam=is_sam,
            writer=writer,
            global_step=global_step,
        )
        global_step += 1

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/Train', train_metrics.get('accuracy', 0), epoch)
            writer.add_scalar('Accuracy/Val', val_metrics.get('accuracy', 0), epoch)
            if 'roc_auc' in val_metrics:
                writer.add_scalar('Metrics/ROC-AUC', val_metrics['roc_auc'], epoch)

        # Update scheduler
        if scheduler is not None:
            try:
                # Some schedulers (ReduceLROnPlateau) expect a metric
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            except Exception:
                scheduler.step()

        # Log metrics
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics.get('accuracy', 0):.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics.get('accuracy', 0):.4f}, "
            f"AUC: {val_metrics.get('roc_auc', 0):.4f}"
        )

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics.get('accuracy', 0))
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics.get('accuracy', 0))
        history['val_roc_auc'].append(val_metrics.get('roc_auc', 0))

        # Save best model (by val loss)
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_val_auc = val_metrics.get('roc_auc', 0)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint via experiment manager or directly
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics.get('roc_auc', 0),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        
        if exp_manager:
            exp_manager.log_metrics({**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}, epoch=epoch)
            exp_manager.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, is_best=is_best)
        else:
            # Direct save
            torch.save(checkpoint_data, save_dir / f"{disease}_best_model.pt" if is_best else save_dir / f"{disease}_model_epoch_{epoch}.pt")
            if is_best:
                logger.info(f"Saved best model (Val Loss: {best_val_loss:.4f})")

        # Early stopping
        if patience_counter >= TRAINING_CONFIG.get("early_stopping_patience", 10):
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Periodic checkpoint
        if epoch % TRAINING_CONFIG.get("save_every_n_epochs", 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f"{disease}_model_epoch_{epoch}.pt")

    logger.info(f"Training completed. Best Val AUC: {best_val_auc:.4f}")

    return history


# ---------------------------------------------------------------------
# CLI / main: load data, build model, train
# ---------------------------------------------------------------------
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train NeuroVoice model")
    parser.add_argument("--disease", type=str, required=True, choices=["alzheimer", "parkinson", "depression"])
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG.get("num_epochs", 50))
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG.get("batch_size", 16))
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG.get("learning_rate", 1e-4))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio_feature_type", type=str, default="wav2vec", choices=["wav2vec", "mfcc", "spectrogram"])
    parser.add_argument("--video_feature_type", type=str, default="landmarks", choices=["frames", "landmarks", "both", "none"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--splits_dir", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--fusion", type=str, default="crossmodal", choices=["early", "late", "crossmodal"],
                        help="Fusion strategy")
    parser.add_argument("--analyze_gradients", action="store_true", help="Log gradient statistics")
    parser.add_argument("--use_sam", action="store_true", help="Use SAM optimizer")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="SAM rho parameter")
    parser.add_argument("--use_lookahead", action="store_true", help="Use Lookahead optimizer")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for tracking")
    parser.add_argument("--use_tensorboard", action="store_true", help="Log to TensorBoard")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # Logger
    logger = setup_logger(Path(args.save_dir) / f"{args.disease}.log")
    logger.info(f"Using device: {device}")
    logger.info(f"Starting with args: {args}")

    # Build CSV paths
    splits_dir = Path(args.splits_dir)
    data_dir = Path(args.data_dir)
    train_csv = splits_dir / f"{args.disease}_train.csv"
    val_csv = splits_dir / f"{args.disease}_val.csv"

    if not train_csv.exists() or not val_csv.exists():
        msg = (
            f"Train/val split files not found for {args.disease}.\n"
            f"Expected: {train_csv} and {val_csv}\n"
            "Run scripts/create_labels.py and scripts/split_data.py to generate splits."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Read dataframes
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Validate that the CSVs contain required columns
    required_cols = {'file_path', 'label', 'disease'}
    for df, name in [(train_df, "train"), (val_df, "val")]:
        if not required_cols.issubset(set(df.columns)):
            msg = f"{name}_csv missing required columns {required_cols}. Found columns: {list(df.columns)}"
            logger.error(msg)
            raise ValueError(msg)

    # Build dataloaders via helper
    logger.info("Building dataloaders...")
    train_loader = get_multimodal_dataloader(
        audio_dir=data_dir,
        video_dir=data_dir,
        labels_df=train_df,
        audio_feature_type=args.audio_feature_type,
        video_feature_type=args.video_feature_type,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = get_multimodal_dataloader(
        audio_dir=data_dir,
        video_dir=data_dir,
        labels_df=val_df,
        audio_feature_type=args.audio_feature_type,
        video_feature_type=args.video_feature_type,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Build model based on fusion type
    logger.info(f"Building model with fusion type: {args.fusion}...")
    audio_input_dim = TRAINING_CONFIG.get("audio_input_dim", 768)
    video_input_dim = None  # Use None for raw video frames
    
    if args.fusion == "early":
        model = EarlyFusionModel(
            audio_input_dim=audio_input_dim,
            video_input_dim=video_input_dim,
            hidden_dim=512,
            num_classes=2,
        ).to(device)
    elif args.fusion == "late":
        model = LateFusionModel(
            audio_input_dim=audio_input_dim,
            video_input_dim=video_input_dim,
            num_classes=2,
            fusion_strategy="learned",
        ).to(device)
    else:  # crossmodal (default)
        model = FusionModel(
            audio_input_dim=audio_input_dim,
            video_input_dim=video_input_dim,
            num_classes=2,
        ).to(device)

    # Optionally resume
    start_epoch = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            logger.info(f"Resumed model weights from {ckpt_path}")
            if 'optimizer_state_dict' in ckpt:
                # We will load optimizer state below after optimizer creation
                resume_optimizer_state = ckpt['optimizer_state_dict']
                start_epoch = ckpt.get('epoch', 1) + 1
            else:
                resume_optimizer_state = None
        else:
            logger.warning(f"Resume checkpoint not found at {ckpt_path}. Continuing from scratch.")
            resume_optimizer_state = None
    else:
        resume_optimizer_state = None

    # Loss
    # Choose loss according to config or fallback
    loss_name = TRAINING_CONFIG.get("loss", "weighted_bce")
    if loss_name == "focal":
        criterion = FocalLoss()
    else:
        criterion = WeightedBCELoss(pos_weight=TRAINING_CONFIG.get("pos_weight", 1.0))

    # Optimizer & scheduler with advanced options
    base_optimizer = get_optimizer(model, learning_rate=args.lr)
    
    # Apply SAM if requested
    is_sam = False
    if args.use_sam:
        from src.training.advanced_optimizers import SAM
        optimizer = SAM(base_optimizer, rho=args.sam_rho)
        is_sam = True
        logger.info(f"Using SAM optimizer with rho={args.sam_rho}")
    else:
        optimizer = base_optimizer
    
    # Apply Lookahead if requested
    if args.use_lookahead:
        from src.training.advanced_optimizers import Lookahead
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
        logger.info("Using Lookahead optimizer wrapper")
    
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            logger.info("Loaded optimizer state from checkpoint.")
        except Exception:
            logger.warning("Failed to load optimizer state (state dict mismatch). Starting with fresh optimizer.")

    scheduler = get_scheduler(optimizer, num_epochs=args.epochs)

    # Experiment tracking
    exp_manager = None
    if args.experiment_name:
        exp_manager = ExperimentManager(
            experiment_name=args.experiment_name or f"{args.disease}_{args.fusion}",
        )
        exp_manager.save_hyperparameters(vars(args))
        logger.info(f"Experiment tracking enabled: {exp_manager.experiment_name}")
    
    # TensorBoard writer
    writer = None
    if args.use_tensorboard:
        tb_dir = LOGS_DIR / "tensorboard" / f"{args.disease}_{args.experiment_name or 'default'}"
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logging to {tb_dir}")

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        disease=args.disease,
        save_dir=Path(args.save_dir),
        logger=logger,
        gradient_accumulation=args.gradient_accumulation,
        is_sam=is_sam,
        writer=writer,
        exp_manager=exp_manager,
    )

    # Save final history to CSV
    history_df = pd.DataFrame(history)
    hist_path = Path(args.save_dir) / f"{args.disease}_training_history.csv"
    history_df.to_csv(hist_path, index=False)
    logger.info(f"Saved training history to {hist_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

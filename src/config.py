"""
Configuration file for NeuroVoice project.

Contains all paths, hyperparameters, dataset URLs, and model configurations.
"""

import os
from pathlib import Path
from typing import Dict, List

# ==================== Project Paths ====================

# Root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Dataset-specific directories
DAIC_WOZ_DIR = DATA_DIR / "daic_woz"
PARKINSON_DIR = DATA_DIR / "parkinson_tsi"
DEMENTIABANK_DIR = DATA_DIR / "dementiabank"
FACES_DIR = DATA_DIR / "faces"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# ==================== Dataset URLs and Info ====================

DATASET_URLS = {
    "daic_woz": {
        "url": "https://dcapswoz.ict.usc.edu/",
        "type": "manual",  # Requires manual registration
        "description": "Depression dataset with audio, video, and transcripts"
    },
    "parkinson": {
        "url": "https://archive.ics.uci.edu/ml/datasets/parkinsons",
        "type": "auto",  # Can be downloaded automatically
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    },
    "dementiabank": {
        "url": "https://dementia.talkbank.org/access/English/Pitt.html",
        "type": "manual",  # Requires manual registration
        "description": "Alzheimer's speech corpus"
    },
    "fer2013": {
        "url": "https://www.kaggle.com/datasets/msambare/fer2013",
        "type": "kaggle",
        "kaggle_dataset": "msambare/fer2013"
    }
}

# ==================== Audio Processing Config ====================

AUDIO_CONFIG = {
    "sample_rate": 16000,  # Hz
    "n_mfcc": 13,
    "n_mels": 128,
    "hop_length": 512,
    "n_fft": 2048,
    "fmin": 0,
    "fmax": 8000,
    "segment_length": 3.0,  # seconds
    "overlap": 0.5,  # overlap ratio for segmentation
}

# Wav2Vec2 model
WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"
WAV2VEC2_EMBEDDING_DIM = 768

# ==================== Video Processing Config ====================

VIDEO_CONFIG = {
    "fps": 30,
    "frame_size": (224, 224),
    "face_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "max_faces": 1,
    "num_landmarks": 468,  # MediaPipe face mesh
}

# Emotion classes (FER2013)
EMOTION_CLASSES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

NUM_EMOTIONS = len(EMOTION_CLASSES)

# ==================== Model Architecture Config ====================

# Audio model
AUDIO_MODEL_CONFIG = {
    "input_dim": WAV2VEC2_EMBEDDING_DIM,  # or MFCC dim if using MFCC
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "pooling": "attention",  # "mean", "max", "attention"
}

# Video model
VIDEO_MODEL_CONFIG = {
    "backbone": "resnet50",  # "resnet50", "resnet18", "vit"
    "pretrained": True,
    "embedding_dim": 512,
    "num_frames": 30,  # frames per sequence
    "temporal_pooling": "attention",
}

# Fusion model
FUSION_MODEL_CONFIG = {
    "audio_dim": AUDIO_MODEL_CONFIG["hidden_dim"] * (2 if AUDIO_MODEL_CONFIG["bidirectional"] else 1),
    "video_dim": VIDEO_MODEL_CONFIG["embedding_dim"],
    "fusion_dim": 512,
    "num_attention_heads": 8,
    "num_fusion_layers": 2,
    "dropout": 0.3,
}

# Classification head
NUM_DISEASES = 3  # Alzheimer's, Parkinson's, Depression
DISEASE_NAMES = ["alzheimer", "parkinson", "depression"]
NUM_CLASSES = 2  # Binary classification (disease vs. healthy) per disease

# ==================== Training Config ====================

TRAINING_CONFIG = {
    "batch_size": 16,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "early_stopping_patience": 10,
    "save_every_n_epochs": 5,
    "eval_every_n_epochs": 1,
}

# Optimizer
OPTIMIZER_CONFIG = {
    "type": "adamw",  # "adam", "adamw", "sgd"
    "lr": TRAINING_CONFIG["learning_rate"],
    "weight_decay": TRAINING_CONFIG["weight_decay"],
    "betas": (0.9, 0.999),
}

# Scheduler
SCHEDULER_CONFIG = {
    "type": "cosine",  # "cosine", "step", "plateau"
    "warmup_epochs": 5,
    "T_max": TRAINING_CONFIG["num_epochs"],
    "eta_min": 1e-6,
}

# ==================== Loss Functions ====================

LOSS_CONFIG = {
    "classification_loss": "bce",  # "bce", "focal", "weighted_bce"
    "auxiliary_loss": True,  # Emotion prediction auxiliary task
    "auxiliary_weight": 0.1,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "class_weights": None,  # Will be computed from data if None
}

# ==================== Data Split Config ====================

SPLIT_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    "stratify": True,  # Stratify by disease labels
}

# ==================== Feature Extraction Config ====================

FEATURE_CONFIG = {
    "audio_features": ["mfcc", "wav2vec"],  # "mfcc", "wav2vec", "spectral"
    "video_features": ["landmarks", "emotion"],  # "landmarks", "emotion", "embeddings"
    "normalize": True,
    "save_features": True,
    "feature_dir": PROCESSED_DATA_DIR / "features",
}

# ==================== Device Config ====================

DEVICE_CONFIG = {
    "use_cuda": True,
    "cuda_device": 0,
    "num_workers": 4,  # DataLoader workers
    "pin_memory": True,
}

# ==================== Logging Config ====================

LOGGING_CONFIG = {
    "log_dir": LOGS_DIR,
    "log_level": "INFO",
    "use_tensorboard": True,
    "tensorboard_dir": LOGS_DIR / "tensorboard",
    "save_plots": True,
}

# ==================== Evaluation Config ====================

EVALUATION_CONFIG = {
    "metrics": ["accuracy", "roc_auc", "f1_score", "precision", "recall"],
    "save_confusion_matrix": True,
    "save_roc_curve": True,
    "threshold": 0.5,  # Binary classification threshold
}

# ==================== Explainability Config ====================

EXPLAINABILITY_CONFIG = {
    "use_gradcam": True,
    "use_attention": True,
    "use_saliency": True,
    "num_samples": 10,  # Number of samples to visualize
    "save_dir": VISUALIZATIONS_DIR / "explainability",
}

# ==================== Utility Functions ====================

def get_model_path(disease: str, epoch: int = None) -> Path:
    """Get path to saved model checkpoint."""
    if epoch is not None:
        return MODELS_DIR / f"{disease}_model_epoch_{epoch}.pt"
    return MODELS_DIR / f"{disease}_best_model.pt"

def get_log_path(disease: str) -> Path:
    """Get path to training log file."""
    return LOGS_DIR / f"{disease}_training.log"

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
        DAIC_WOZ_DIR, PARKINSON_DIR, DEMENTIABANK_DIR, FACES_DIR,
        OUTPUTS_DIR, MODELS_DIR, LOGS_DIR, METRICS_DIR, VISUALIZATIONS_DIR,
        FEATURE_CONFIG["feature_dir"],
        EXPLAINABILITY_CONFIG["save_dir"],
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Create directories on import
ensure_directories()

# ==================== Dataset-specific Configs ====================

DATASET_CONFIGS = {
    "alzheimer": {
        "dataset_name": "dementiabank",
        "data_dir": DEMENTIABANK_DIR,
        "classes": ["control", "alzheimer"],
    },
    "parkinson": {
        "dataset_name": "parkinson",
        "data_dir": PARKINSON_DIR,
        "classes": ["control", "parkinson"],
    },
    "depression": {
        "dataset_name": "daic_woz",
        "data_dir": DAIC_WOZ_DIR,
        "classes": ["control", "depression"],
        "threshold": 10,  # PHQ-8 threshold for depression
    },
}


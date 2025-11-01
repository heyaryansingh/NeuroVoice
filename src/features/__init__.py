"""
Feature extraction modules for NeuroVoice project.
"""

from .audio_features import (
    extract_mfcc,
    extract_wav2vec_embeddings,
    extract_spectral_features,
)
from .video_features import (
    extract_facial_landmarks,
    extract_emotion_embedding,
    extract_facial_features,
)
from .fusion_features import (
    concatenate_features,
    attention_fusion,
    gated_fusion,
)

__all__ = [
    "extract_mfcc",
    "extract_wav2vec_embeddings",
    "extract_spectral_features",
    "extract_facial_landmarks",
    "extract_emotion_embedding",
    "extract_facial_features",
    "concatenate_features",
    "attention_fusion",
    "gated_fusion",
]


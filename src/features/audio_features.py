"""
Audio feature extraction utilities.

Extracts MFCC, wav2vec embeddings, and spectral features from audio files.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from src.config import AUDIO_CONFIG, WAV2VEC2_MODEL


def extract_mfcc(
    wav_path: Union[str, Path],
    sample_rate: int = None,
    n_mfcc: int = None,
) -> np.ndarray:
    """
    Extract MFCC features from audio file.
    
    Args:
        wav_path: Path to audio file (.wav)
        sample_rate: Target sample rate (default from config)
        n_mfcc: Number of MFCC coefficients (default from config)
    
    Returns:
        MFCC features array of shape (time_frames, n_mfcc)
    """
    sample_rate = sample_rate or AUDIO_CONFIG["sample_rate"]
    n_mfcc = n_mfcc or AUDIO_CONFIG["n_mfcc"]
    
    # Load audio
    waveform, orig_sr = torchaudio.load(str(wav_path))
    
    # Resample if necessary
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": AUDIO_CONFIG["n_fft"],
            "n_mels": AUDIO_CONFIG["n_mels"],
            "hop_length": AUDIO_CONFIG["hop_length"],
        }
    )
    mfcc = mfcc_transform(waveform)
    
    return mfcc.squeeze(0).numpy()  # Remove channel dimension and convert to numpy


def extract_wav2vec_embeddings(
    wav_path: Union[str, Path],
    sample_rate: int = None,
    device: str = None,
) -> np.ndarray:
    """
    Extract wav2vec2 embeddings from audio file.
    
    Uses facebook/wav2vec2-base-960h model to extract transformer embeddings.
    
    Args:
        wav_path: Path to audio file (.wav)
        sample_rate: Target sample rate (default from config)
        device: Device to run model on ('cuda' or 'cpu')
    
    Returns:
        Wav2vec2 embeddings array of shape (embedding_dim,)
        (averaged over time dimension)
    """
    sample_rate = sample_rate or AUDIO_CONFIG["sample_rate"]
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load processor and model
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL)
    model = model.to(device)
    model.eval()
    
    # Load audio
    waveform, orig_sr = torchaudio.load(str(wav_path))
    
    # Resample if necessary
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Process audio
    with torch.no_grad():
        inputs = processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        
        # Average pool over time dimension
        embeddings = embeddings.mean(dim=1).squeeze(0)
        
        # Move back to CPU
        if device == "cuda":
            embeddings = embeddings.cpu()
    
    return embeddings.numpy()


def extract_spectral_features(
    wav_path: Union[str, Path],
    sample_rate: int = None,
) -> dict:
    """
    Extract spectral features from audio file.
    
    Extracts: spectral centroid, spectral rolloff, zero crossing rate, etc.
    
    Args:
        wav_path: Path to audio file (.wav)
        sample_rate: Target sample rate (default from config)
    
    Returns:
        Dictionary with spectral features:
            - 'spectral_centroid': Spectral centroid over time
            - 'spectral_rolloff': Spectral rolloff over time
            - 'zero_crossing_rate': Zero crossing rate
            - 'chroma': Chroma features
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa required for spectral features. Install with: pip install librosa"
        )
    
    sample_rate = sample_rate or AUDIO_CONFIG["sample_rate"]
    
    # Load audio
    y, sr = librosa.load(str(wav_path), sr=sample_rate)
    
    # Extract features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    chroma = librosa.feature.chroma(y=y, sr=sr)
    
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate,
        'chroma': chroma,
    }


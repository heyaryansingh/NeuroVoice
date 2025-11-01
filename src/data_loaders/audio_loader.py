"""
Audio data loader for speech datasets.

Loads and preprocesses audio files for training and evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from src.config import AUDIO_CONFIG, WAV2VEC2_EMBEDDING_DIM


class AudioDataset(Dataset):
    """
    Dataset class for audio files.
    
    Loads audio files and their corresponding labels for disease classification.
    Supports MFCC features and wav2vec2 embeddings.
    
    Args:
        data_dir: Directory containing audio files
        labels_df: DataFrame with columns ['file_path', 'label', 'disease']
        feature_type: Type of features to extract ('mfcc', 'wav2vec', 'raw')
        transform: Optional transform to apply to audio
        sample_rate: Target sample rate for audio (default from config)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        labels_df: pd.DataFrame,
        feature_type: str = "wav2vec",
        transform: Optional[callable] = None,
        sample_rate: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.feature_type = feature_type
        self.transform = transform
        self.sample_rate = sample_rate or AUDIO_CONFIG["sample_rate"]
        
        # Validate feature type
        assert feature_type in ["mfcc", "wav2vec", "raw"], \
            f"feature_type must be one of ['mfcc', 'wav2vec', 'raw'], got {feature_type}"
        
        # Load wav2vec2 model if needed
        if feature_type == "wav2vec":
            try:
                from transformers import Wav2Vec2Processor, Wav2Vec2Model
                self.processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self.wav2vec_model.eval()  # Set to eval mode for feature extraction
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.wav2vec_model = self.wav2vec_model.cuda()
            except ImportError:
                raise ImportError(
                    "transformers library required for wav2vec features. "
                    "Install with: pip install transformers"
                )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single audio sample.
        
        Returns:
            Dictionary with keys:
                - 'audio': Audio features (MFCC, wav2vec embeddings, or raw waveform)
                - 'label': Binary label (0=healthy, 1=disease)
                - 'disease': Disease type index
                - 'file_path': Path to original audio file
        """
        row = self.labels_df.iloc[idx]
        file_path = self.data_dir / row['file_path']
        
        # Load audio
        waveform, orig_sr = torchaudio.load(str(file_path))
        
        # Resample if necessary
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract features based on type
        if self.feature_type == "raw":
            features = waveform.squeeze(0)  # Remove channel dimension
        
        elif self.feature_type == "mfcc":
            features = self._extract_mfcc(waveform)
        
        elif self.feature_type == "wav2vec":
            features = self._extract_wav2vec(waveform)
        
        # Apply transforms if provided
        if self.transform:
            features = self.transform(features)
        
        # Get label and disease index
        label = int(row['label'])
        disease = row.get('disease', 'alzheimer')
        disease_idx = ['alzheimer', 'parkinson', 'depression'].index(disease)
        
        return {
            'audio': features,
            'label': torch.tensor(label, dtype=torch.long),
            'disease': torch.tensor(disease_idx, dtype=torch.long),
            'file_path': str(file_path),
        }
    
    def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from waveform."""
        # Create MFCC transform
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=AUDIO_CONFIG["n_mfcc"],
            melkwargs={
                "n_fft": AUDIO_CONFIG["n_fft"],
                "n_mels": AUDIO_CONFIG["n_mels"],
                "hop_length": AUDIO_CONFIG["hop_length"],
            }
        )
        mfcc = mfcc_transform(waveform)
        return mfcc.squeeze(0)  # Remove channel dimension
    
    def _extract_wav2vec(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract wav2vec2 embeddings from waveform."""
        with torch.no_grad():
            # Process audio for wav2vec2
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Extract embeddings
            outputs = self.wav2vec_model(**inputs)
            embeddings = outputs.last_hidden_state
            
            # Average pool over time dimension
            embeddings = embeddings.mean(dim=1).squeeze(0)
            
            # Move back to CPU
            if torch.cuda.is_available():
                embeddings = embeddings.cpu()
            
            return embeddings


def get_audio_dataloader(
    data_dir: Union[str, Path],
    labels_df: pd.DataFrame,
    feature_type: str = "wav2vec",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for audio data.
    
    Args:
        data_dir: Directory containing audio files
        labels_df: DataFrame with audio file paths and labels
        feature_type: Type of features ('mfcc', 'wav2vec', 'raw')
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        transform: Optional transform to apply
    
    Returns:
        DataLoader instance
    """
    dataset = AudioDataset(
        data_dir=data_dir,
        labels_df=labels_df,
        feature_type=feature_type,
        transform=transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_audio_fn,
    )
    
    return dataloader


def _collate_audio_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for audio batch.
    Handles variable-length audio sequences by padding.
    """
    # Separate components
    audios = [item['audio'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    diseases = torch.stack([item['disease'] for item in batch])
    file_paths = [item['file_path'] for item in batch]
    
    # Pad audio sequences if needed (for raw/mfcc features)
    if len(audios[0].shape) == 2:  # Time x Features
        max_len = max(a.shape[0] for a in audios)
        padded_audios = []
        for audio in audios:
            if audio.shape[0] < max_len:
                padding = torch.zeros(max_len - audio.shape[0], audio.shape[1])
                audio = torch.cat([audio, padding], dim=0)
            padded_audios.append(audio)
        audios = torch.stack(padded_audios)
    else:
        audios = torch.stack(audios)
    
    return {
        'audio': audios,
        'label': labels,
        'disease': diseases,
        'file_path': file_paths,
    }


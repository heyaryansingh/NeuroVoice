"""
Multimodal data loader combining audio and video data.

Synchronizes and loads paired audio-video samples for multimodal training.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_loaders.audio_loader import AudioDataset
from src.data_loaders.video_loader import VideoDataset


class MultimodalDataset(Dataset):
    """
    Dataset class for multimodal (audio + video) data.
    
    Combines audio and video datasets to provide synchronized multimodal samples.
    
    Args:
        audio_dir: Directory containing audio files
        video_dir: Directory containing video files
        labels_df: DataFrame with columns ['audio_path', 'video_path', 'label', 'disease']
        audio_feature_type: Type of audio features ('mfcc', 'wav2vec', 'raw')
        video_feature_type: Type of video features ('landmarks', 'emotion', 'both')
        audio_transform: Optional transform for audio
        video_transform: Optional transform for video
    """
    
    def __init__(
        self,
        audio_dir: Union[str, Path],
        video_dir: Union[str, Path],
        labels_df: pd.DataFrame,
        audio_feature_type: str = "wav2vec",
        video_feature_type: str = "both",
        audio_transform: Optional[callable] = None,
        video_transform: Optional[callable] = None,
    ):
        self.audio_dir = Path(audio_dir)
        self.video_dir = Path(video_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        
        # Handle both formats: new format (audio_path, video_path) and old format (file_path)
        if 'audio_path' in labels_df.columns and 'video_path' in labels_df.columns:
            # New format with separate audio/video paths
            audio_labels = labels_df[['audio_path', 'label', 'disease']].copy()
            audio_labels.rename(columns={'audio_path': 'file_path'}, inplace=True)
            
            video_labels = labels_df[['video_path', 'label', 'disease']].copy()
            video_labels.rename(columns={'video_path': 'file_path'}, inplace=True)
        elif 'file_path' in labels_df.columns:
            # Old format with single file_path - use for both audio and video
            audio_labels = labels_df[['file_path', 'label', 'disease']].copy()
            video_labels = labels_df[['file_path', 'label', 'disease']].copy()
        else:
            raise ValueError(
                "labels_df must contain either ('audio_path', 'video_path') or 'file_path' columns. "
                f"Found columns: {list(labels_df.columns)}"
            )
        
        self.audio_dataset = AudioDataset(
            data_dir=audio_dir,
            labels_df=audio_labels,
            feature_type=audio_feature_type,
            transform=audio_transform,
        )
        
        self.video_dataset = VideoDataset(
            data_dir=video_dir,
            labels_df=video_labels,
            feature_type=video_feature_type,
            transform=video_transform,
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single multimodal sample.
        
        Returns:
            Dictionary with keys:
                - 'audio': Audio features
                - 'video': Video features
                - 'label': Binary label (0=healthy, 1=disease)
                - 'disease': Disease type index
                - 'audio_path': Path to audio file
                - 'video_path': Path to video file
        """
        audio_sample = self.audio_dataset[idx]
        video_sample = self.video_dataset[idx]
        
        return {
            'audio': audio_sample['audio'],
            'video': video_sample['video'],
            'label': audio_sample['label'],  # Labels should match
            'disease': audio_sample['disease'],
            'audio_path': audio_sample['file_path'],
            'video_path': video_sample['file_path'],
        }


def get_multimodal_dataloader(
    audio_dir: Union[str, Path],
    video_dir: Union[str, Path],
    labels_df: pd.DataFrame,
    audio_feature_type: str = "wav2vec",
    video_feature_type: str = "both",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    audio_transform: Optional[callable] = None,
    video_transform: Optional[callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for multimodal data.
    
    Args:
        audio_dir: Directory containing audio files
        video_dir: Directory containing video files
        labels_df: DataFrame with audio and video file paths and labels
        audio_feature_type: Type of audio features
        video_feature_type: Type of video features
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        audio_transform: Optional transform for audio
        video_transform: Optional transform for video
    
    Returns:
        DataLoader instance
    """
    dataset = MultimodalDataset(
        audio_dir=audio_dir,
        video_dir=video_dir,
        labels_df=labels_df,
        audio_feature_type=audio_feature_type,
        video_feature_type=video_feature_type,
        audio_transform=audio_transform,
        video_transform=video_transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_multimodal_fn,
    )
    
    return dataloader


def _collate_multimodal_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multimodal batch.
    Handles both audio and video sequences.
    """
    audios = []
    videos = []
    labels = []
    diseases = []
    audio_paths = []
    video_paths = []
    
    for item in batch:
        audios.append(item['audio'])
        videos.append(item['video'])
        labels.append(item['label'])
        diseases.append(item['disease'])
        audio_paths.append(item['audio_path'])
        video_paths.append(item['video_path'])
    
    # Stack tensors
    if isinstance(audios[0], torch.Tensor):
        # Handle variable-length sequences
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
    
    videos = torch.stack(videos)
    labels = torch.stack(labels)
    diseases = torch.stack(diseases)
    
    return {
        'audio': audios,
        'video': videos,
        'label': labels,
        'disease': diseases,
        'audio_path': audio_paths,
        'video_path': video_paths,
    }


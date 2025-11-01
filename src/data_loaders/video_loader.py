"""
Video data loader for facial expression datasets.

Loads and preprocesses video files and extracts facial features.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import VIDEO_CONFIG, NUM_EMOTIONS


class VideoDataset(Dataset):
    """
    Dataset class for video files with facial expressions.
    
    Loads video files, extracts facial landmarks and emotion embeddings.
    
    Args:
        data_dir: Directory containing video files
        labels_df: DataFrame with columns ['file_path', 'label', 'disease']
        feature_type: Type of features ('landmarks', 'emotion', 'both')
        max_frames: Maximum number of frames to extract per video
        transform: Optional transform to apply to frames
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        labels_df: pd.DataFrame,
        feature_type: str = "both",
        max_frames: int = None,
        transform: Optional[callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.feature_type = feature_type
        self.max_frames = max_frames or VIDEO_CONFIG["num_frames"]
        self.transform = transform
        
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=VIDEO_CONFIG["max_faces"],
            min_detection_confidence=VIDEO_CONFIG["face_detection_confidence"],
            min_tracking_confidence=VIDEO_CONFIG["min_tracking_confidence"],
        )
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load emotion classifier if needed
        self.emotion_model = None
        self.emotion_cache = {}  # Cache for emotion embeddings
        if "emotion" in feature_type:
            try:
                from src.features.video_features import load_emotion_model
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.emotion_model = load_emotion_model(device=device)
                print(f"Loaded emotion model on {device}")
            except FileNotFoundError:
                print("Warning: Emotion model not found. Emotion features will be zeros.")
                print("Train emotion model with: python scripts/train_emotion_model.py")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single video sample.
        
        Returns:
            Dictionary with keys:
                - 'video': Video features (landmarks, emotion embeddings, or both)
                - 'label': Binary label (0=healthy, 1=disease)
                - 'disease': Disease type index
                - 'file_path': Path to original video file
        """
        row = self.labels_df.iloc[idx]
        file_path = self.data_dir / row['file_path']
        
        # Extract features from video
        features = self._extract_features(str(file_path))
        
        # Apply transforms if provided
        if self.transform:
            features = self.transform(features)
        
        # Get label and disease index
        label = int(row['label'])
        disease = row.get('disease', 'alzheimer')
        disease_idx = ['alzheimer', 'parkinson', 'depression'].index(disease)
        
        return {
            'video': features,
            'label': torch.tensor(label, dtype=torch.long),
            'disease': torch.tensor(disease_idx, dtype=torch.long),
            'file_path': str(file_path),
        }
    
    def _extract_features(self, video_path: str) -> torch.Tensor:
        """
        Extract features from video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tensor of extracted features
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        features_list = []
        frame_idx = 0
        frame_interval = max(1, frame_count // self.max_frames)
        
        while cap.isOpened() and len(features_list) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at intervals
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract features based on type
                frame_features = []
                
                if "landmarks" in self.feature_type:
                    landmarks = self._extract_landmarks(frame_rgb)
                    frame_features.append(landmarks)
                
                if "emotion" in self.feature_type:
                    # Use frame index as cache key
                    cache_key = f"{str(file_path)}_{frame_idx}"
                    emotion = self._extract_emotion(frame_rgb, cache_key=cache_key)
                    frame_features.append(emotion)
                
                if frame_features:
                    # Concatenate features
                    combined = np.concatenate(frame_features) if len(frame_features) > 1 else frame_features[0]
                    features_list.append(combined)
            
            frame_idx += 1
        
        cap.release()
        
        # Pad or truncate to max_frames
        if len(features_list) < self.max_frames:
            # Pad with last frame
            last_feature = features_list[-1] if features_list else np.zeros(VIDEO_CONFIG["num_landmarks"] * 3)
            while len(features_list) < self.max_frames:
                features_list.append(last_feature)
        else:
            features_list = features_list[:self.max_frames]
        
        # Convert to tensor
        features = torch.tensor(np.stack(features_list), dtype=torch.float32)
        return features
    
    def _extract_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract facial landmarks using MediaPipe.
        
        Args:
            frame: RGB image frame
        
        Returns:
            Flattened landmarks array
        """
        results = self.face_mesh.process(frame)
        
        if results.multi_face_landmarks:
            # Get first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract x, y, z coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        else:
            # No face detected, return zeros
            return np.zeros(VIDEO_CONFIG["num_landmarks"] * 3)
    
    def _extract_emotion(self, frame: np.ndarray, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Extract emotion embedding from frame.
        
        Args:
            frame: RGB image frame
            cache_key: Optional cache key for this frame
        
        Returns:
            Emotion embedding
        """
        # Check cache first
        if cache_key and cache_key in self.emotion_cache:
            return self.emotion_cache[cache_key]
        
        if self.emotion_model is not None:
            from src.features.video_features import extract_emotion_embedding
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Extract emotion embedding
            emotion_embedding = extract_emotion_embedding(
                frame,
                model=self.emotion_model,
                device=device,
                return_logits=False,  # Return probabilities
            )
            
            # Cache result
            if cache_key:
                self.emotion_cache[cache_key] = emotion_embedding
            
            return emotion_embedding
        else:
            # Return zeros if model not available
            return np.zeros(NUM_EMOTIONS)


def get_video_dataloader(
    data_dir: Union[str, Path],
    labels_df: pd.DataFrame,
    feature_type: str = "both",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_frames: int = None,
    transform: Optional[callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for video data.
    
    Args:
        data_dir: Directory containing video files
        labels_df: DataFrame with video file paths and labels
        feature_type: Type of features ('landmarks', 'emotion', 'both')
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        max_frames: Maximum frames per video
        transform: Optional transform to apply
    
    Returns:
        DataLoader instance
    """
    dataset = VideoDataset(
        data_dir=data_dir,
        labels_df=labels_df,
        feature_type=feature_type,
        max_frames=max_frames,
        transform=transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_video_fn,
    )
    
    return dataloader


def _collate_video_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for video batch.
    Handles variable-length video sequences by padding.
    """
    videos = torch.stack([item['video'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    diseases = torch.stack([item['disease'] for item in batch])
    file_paths = [item['file_path'] for item in batch]
    
    return {
        'video': videos,
        'label': labels,
        'disease': diseases,
        'file_path': file_paths,
    }


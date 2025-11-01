"""
Tests for video processing pipeline.

Tests video feature extraction, data loading, and preprocessing.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.features.video_features import extract_facial_landmarks
from src.data_loaders.video_loader import VideoDataset
from src.models.video_model import VideoModel


def test_facial_landmarks_extraction():
    """Test facial landmark extraction."""
    # TODO: Create a dummy video file or use existing test file
    assert callable(extract_facial_landmarks)


def test_video_model_resnet():
    """Test VideoModel with ResNet backbone."""
    model = VideoModel(
        backbone="resnet18",
        pretrained=False,
        embedding_dim=512,
        num_frames=30,
    )
    
    # Test with raw video frames
    batch_size = 2
    num_frames = 30
    channels = 3
    height, width = 224, 224
    
    dummy_input = torch.randn(batch_size, num_frames, channels, height, width)
    output = model(dummy_input)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == 512  # embedding_dim


def test_video_model_landmarks():
    """Test VideoModel with facial landmarks input."""
    model = VideoModel(
        input_dim=468 * 3,  # 468 landmarks * 3 coordinates
        backbone="cnn",
        embedding_dim=256,
        num_frames=30,
    )
    
    batch_size = 2
    num_frames = 30
    num_features = 468 * 3
    
    dummy_input = torch.randn(batch_size, num_frames, num_features)
    output = model(dummy_input)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == 256  # embedding_dim


def test_video_model_temporal_pooling():
    """Test different temporal pooling strategies."""
    for pooling in ["mean", "max", "attention"]:
        model = VideoModel(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=256,
            temporal_pooling=pooling,
        )
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, 30, 3, 224, 224)
        output = model(dummy_input)
        
        assert output.shape == (batch_size, 256)


@pytest.mark.skip(reason="Requires actual video files")
def test_video_dataset():
    """Test VideoDataset loading."""
    # TODO: Implement when test data is available
    pass


if __name__ == "__main__":
    pytest.main([__file__])


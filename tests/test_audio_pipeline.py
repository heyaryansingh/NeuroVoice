"""
Tests for audio processing pipeline.

Tests audio feature extraction, data loading, and preprocessing.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.features.audio_features import extract_mfcc, extract_wav2vec_embeddings
from src.data_loaders.audio_loader import AudioDataset
from src.models.audio_model import AudioModel


def test_mfcc_extraction():
    """Test MFCC feature extraction."""
    # TODO: Create a dummy audio file or use existing test file
    # For now, just test that function exists and can be imported
    assert callable(extract_mfcc)


def test_wav2vec_extraction():
    """Test wav2vec2 embedding extraction."""
    # TODO: Create a dummy audio file or use existing test file
    assert callable(extract_wav2vec_embeddings)


def test_audio_model():
    """Test AudioModel initialization and forward pass."""
    model = AudioModel(
        input_dim=768,
        hidden_dim=256,
        num_layers=2,
        model_type="lstm",
    )
    
    # Test forward pass with dummy input
    batch_size = 4
    seq_len = 100
    dummy_input = torch.randn(batch_size, seq_len, 768)
    
    output = model(dummy_input)
    assert output.shape[0] == batch_size
    assert len(output.shape) == 2  # (batch, features)


def test_audio_model_wav2vec_input():
    """Test AudioModel with wav2vec embeddings (2D input)."""
    model = AudioModel(
        input_dim=768,
        hidden_dim=256,
        model_type="lstm",
    )
    
    # Wav2vec embeddings are 2D (batch, features)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 768)
    
    output = model(dummy_input)
    assert output.shape[0] == batch_size


def test_audio_model_cnn():
    """Test CNN-based AudioModel."""
    model = AudioModel(
        input_dim=13,  # MFCC features
        hidden_dim=128,
        num_layers=3,
        model_type="cnn",
    )
    
    batch_size = 4
    seq_len = 100
    dummy_input = torch.randn(batch_size, seq_len, 13)
    
    output = model(dummy_input)
    assert output.shape[0] == batch_size


@pytest.mark.skip(reason="Requires actual audio files")
def test_audio_dataset():
    """Test AudioDataset loading."""
    # TODO: Implement when test data is available
    pass


if __name__ == "__main__":
    pytest.main([__file__])


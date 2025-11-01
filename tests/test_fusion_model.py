"""
Tests for multimodal fusion model.

Tests fusion model architecture, attention mechanisms, and forward pass.
"""

import pytest
import torch

from src.models.fusion_model import FusionModel, EarlyFusionModel, LateFusionModel
from src.models.attention_modules import CrossModalAttentionBlock, SelfAttentionBlock


def test_fusion_model_forward():
    """Test FusionModel forward pass."""
    model = FusionModel(
        audio_input_dim=768,  # wav2vec2 dimension
        video_input_dim=None,  # Raw frames
        fusion_dim=512,
    )
    
    batch_size = 4
    num_frames = 30
    
    # Dummy inputs
    audio_input = torch.randn(batch_size, 768)  # wav2vec embeddings
    video_input = torch.randn(batch_size, num_frames, 3, 224, 224)  # Video frames
    disease_idx = torch.randint(0, 3, (batch_size,))  # Disease indices
    
    output = model(audio_input, video_input, disease_idx)
    
    assert isinstance(output, dict)
    assert 'logits' in output
    assert 'audio_features' in output
    assert 'video_features' in output
    assert 'fused_features' in output
    
    # Check logits shape
    logits = output['logits']
    assert logits.shape[0] == batch_size


def test_early_fusion_model():
    """Test EarlyFusionModel."""
    model = EarlyFusionModel(
        audio_input_dim=768,
        video_input_dim=None,
        hidden_dim=512,
    )
    
    batch_size = 4
    audio_input = torch.randn(batch_size, 768)
    video_input = torch.randn(batch_size, 30, 3, 224, 224)
    
    output = model(audio_input, video_input)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 2  # Binary classification


def test_late_fusion_model():
    """Test LateFusionModel."""
    model = LateFusionModel(
        audio_input_dim=768,
        video_input_dim=None,
        num_classes=2,
    )
    
    batch_size = 4
    audio_input = torch.randn(batch_size, 768)
    video_input = torch.randn(batch_size, 30, 3, 224, 224)
    
    output = model(audio_input, video_input)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 2


def test_cross_modal_attention():
    """Test CrossModalAttentionBlock."""
    attention = CrossModalAttentionBlock(
        audio_dim=256,
        video_dim=512,
        hidden_dim=512,
        num_heads=8,
    )
    
    batch_size = 4
    audio_features = torch.randn(batch_size, 256)
    video_features = torch.randn(batch_size, 512)
    
    attended_audio, attended_video = attention(audio_features, video_features)
    
    assert attended_audio.shape == (batch_size, 512)
    assert attended_video.shape == (batch_size, 512)


def test_self_attention():
    """Test SelfAttentionBlock."""
    attention = SelfAttentionBlock(
        input_dim=256,
        hidden_dim=256,
        num_heads=8,
    )
    
    batch_size = 4
    seq_len = 100
    input_features = torch.randn(batch_size, seq_len, 256)
    
    output = attention(input_features)
    
    assert output.shape == (batch_size, seq_len, 256)


def test_fusion_model_auxiliary_task():
    """Test FusionModel with auxiliary emotion prediction."""
    model = FusionModel(
        audio_input_dim=768,
        video_input_dim=None,
        use_auxiliary=True,
    )
    
    batch_size = 4
    audio_input = torch.randn(batch_size, 768)
    video_input = torch.randn(batch_size, 30, 3, 224, 224)
    disease_idx = torch.randint(0, 3, (batch_size,))
    
    output = model(audio_input, video_input, disease_idx)
    
    assert 'emotion_logits' in output
    emotion_logits = output['emotion_logits']
    assert emotion_logits.shape[0] == batch_size
    assert emotion_logits.shape[1] == 7  # 7 emotion classes


if __name__ == "__main__":
    pytest.main([__file__])


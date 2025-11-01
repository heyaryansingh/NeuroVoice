"""
Feature fusion utilities for combining audio and visual features.

Implements various fusion strategies: concatenation, attention-based, and gated fusion.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def concatenate_features(
    audio_features: torch.Tensor,
    video_features: torch.Tensor,
) -> torch.Tensor:
    """
    Simple concatenation fusion of audio and video features.
    
    Args:
        audio_features: Audio feature tensor of shape (batch, audio_dim)
        video_features: Video feature tensor of shape (batch, video_dim)
    
    Returns:
        Concatenated features of shape (batch, audio_dim + video_dim)
    """
    return torch.cat([audio_features, video_features], dim=1)


def attention_fusion(
    audio_features: torch.Tensor,
    video_features: torch.Tensor,
    hidden_dim: int = 512,
    num_heads: int = 8,
) -> torch.Tensor:
    """
    Attention-based fusion of audio and video features.
    
    Uses cross-modal attention to learn interactions between modalities.
    
    Args:
        audio_features: Audio feature tensor of shape (batch, audio_dim)
        video_features: Video feature tensor of shape (batch, video_dim)
        hidden_dim: Hidden dimension for attention
        num_heads: Number of attention heads
    
    Returns:
        Fused features of shape (batch, hidden_dim)
    """
    batch_size = audio_features.shape[0]
    
    # Project features to common dimension
    audio_dim = audio_features.shape[1]
    video_dim = video_features.shape[1]
    
    audio_proj = nn.Linear(audio_dim, hidden_dim)
    video_proj = nn.Linear(video_dim, hidden_dim)
    
    # Project features
    audio_proj_features = audio_proj(audio_features)  # (batch, hidden_dim)
    video_proj_features = video_proj(video_features)  # (batch, hidden_dim)
    
    # Create query, key, value for attention
    # Audio attends to video
    Q_audio = nn.Linear(hidden_dim, hidden_dim)(audio_proj_features)
    K_video = nn.Linear(hidden_dim, hidden_dim)(video_proj_features)
    V_video = nn.Linear(hidden_dim, hidden_dim)(video_proj_features)
    
    # Reshape for multi-head attention
    head_dim = hidden_dim // num_heads
    Q_audio = Q_audio.view(batch_size, num_heads, head_dim)
    K_video = K_video.view(batch_size, num_heads, head_dim)
    V_video = V_video.view(batch_size, num_heads, head_dim)
    
    # Scaled dot-product attention
    scores = torch.matmul(Q_audio, K_video.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    attended_video = torch.matmul(attn_weights, V_video)
    
    # Reshape back
    attended_video = attended_video.view(batch_size, hidden_dim)
    
    # Combine with original audio features
    fused = audio_proj_features + attended_video
    
    return fused


def gated_fusion(
    audio_features: torch.Tensor,
    video_features: torch.Tensor,
    hidden_dim: int = 512,
) -> torch.Tensor:
    """
    Gated fusion mechanism for combining audio and video features.
    
    Uses learned gates to control the contribution of each modality.
    
    Args:
        audio_features: Audio feature tensor of shape (batch, audio_dim)
        video_features: Video feature tensor of shape (batch, video_dim)
        hidden_dim: Hidden dimension for fusion
    
    Returns:
        Fused features of shape (batch, hidden_dim)
    """
    audio_dim = audio_features.shape[1]
    video_dim = video_features.shape[1]
    
    # Project features
    audio_proj = nn.Linear(audio_dim, hidden_dim)
    video_proj = nn.Linear(video_dim, hidden_dim)
    
    audio_proj_features = audio_proj(audio_features)
    video_proj_features = video_proj(video_features)
    
    # Learn gates
    gate_audio = nn.Linear(hidden_dim, hidden_dim)
    gate_video = nn.Linear(hidden_dim, hidden_dim)
    
    g_audio = torch.sigmoid(gate_audio(audio_proj_features))
    g_video = torch.sigmoid(gate_video(video_proj_features))
    
    # Gated combination
    fused = g_audio * audio_proj_features + g_video * video_proj_features
    
    return fused


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module for learning interactions between audio and video.
    
    Implements bidirectional attention where audio can attend to video and vice versa.
    """
    
    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(CrossModalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Projections
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # Attention layers
        self.audio_to_video_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.video_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of cross-modal attention.
        
        Args:
            audio_features: Audio features (batch, audio_dim)
            video_features: Video features (batch, video_dim)
        
        Returns:
            Fused features (batch, hidden_dim)
        """
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        video_proj = self.video_proj(video_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Audio attends to video
        audio_attended, _ = self.audio_to_video_attn(
            query=audio_proj,
            key=video_proj,
            value=video_proj,
        )
        
        # Video attends to audio
        video_attended, _ = self.video_to_audio_attn(
            query=video_proj,
            key=audio_proj,
            value=audio_proj,
        )
        
        # Concatenate and project
        audio_attended = audio_attended.squeeze(1)  # (batch, hidden_dim)
        video_attended = video_attended.squeeze(1)  # (batch, hidden_dim)
        
        fused = torch.cat([audio_attended, video_attended], dim=1)
        fused = self.output_proj(fused)
        fused = self.dropout(fused)
        
        return fused


"""
Attention modules for multimodal fusion.

Implements cross-modal attention mechanisms for combining audio and video features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal attention block for learning interactions between audio and video.
    
    Implements bidirectional attention where each modality can attend to the other.
    """
    
    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(CrossModalAttentionBlock, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        
        # Projections
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # Multi-head attention
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
        
        # Layer normalization
        self.norm_audio = nn.LayerNorm(hidden_dim)
        self.norm_video = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ff_audio = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.ff_video = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.norm_ff_audio = nn.LayerNorm(hidden_dim)
        self.norm_ff_video = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> tuple:
        """
        Forward pass of cross-modal attention.
        
        Args:
            audio_features: Audio features (batch, audio_dim)
            video_features: Video features (batch, video_dim)
        
        Returns:
            Tuple of (attended_audio, attended_video) both of shape (batch, hidden_dim)
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
        audio_attended = audio_attended.squeeze(1)  # (batch, hidden_dim)
        audio_attended = self.norm_audio(audio_proj.squeeze(1) + audio_attended)
        
        # Video attends to audio
        video_attended, _ = self.video_to_audio_attn(
            query=video_proj,
            key=audio_proj,
            value=audio_proj,
        )
        video_attended = video_attended.squeeze(1)  # (batch, hidden_dim)
        video_attended = self.norm_video(video_proj.squeeze(1) + video_attended)
        
        # Feed-forward
        audio_out = self.norm_ff_audio(audio_attended + self.ff_audio(audio_attended))
        video_out = self.norm_ff_video(video_attended + self.ff_video(video_attended))
        
        return audio_out, video_out


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for single modality.
    
    Useful for processing sequences within a modality.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(SelfAttentionBlock, self).__init__()
        
        self.hidden_dim = hidden_dim or input_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        
        # Projection if input_dim != hidden_dim
        if input_dim != self.hidden_dim:
            self.proj = nn.Linear(input_dim, self.hidden_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        
        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        x_proj = self.proj(x)
        
        # Self-attention
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        attn_out = self.norm(x_proj + attn_out)
        
        # Feed-forward
        out = self.norm_ff(attn_out + self.ff(attn_out))
        
        return out


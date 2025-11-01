"""
Video processing models for facial expression analysis.

Implements ResNet, Vision Transformer, and CNN models for video feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import VIDEO_MODEL_CONFIG, NUM_EMOTIONS


class VideoModel(nn.Module):
    """
    Video model for processing facial expression features.
    
    Supports facial landmarks, emotion embeddings, or raw video frames.
    Uses ResNet, CNN, or Vision Transformer as backbone.
    
    Args:
        input_dim: Dimension of input features (for landmarks/embeddings)
        backbone: Backbone architecture ('resnet50', 'resnet18', 'vit', 'cnn')
        pretrained: Whether to use pretrained weights
        embedding_dim: Output embedding dimension
        num_frames: Number of frames per sequence
        temporal_pooling: Temporal pooling strategy ('mean', 'max', 'attention', 'lstm')
    """
    
    def __init__(
        self,
        input_dim: int = None,
        backbone: str = None,
        pretrained: bool = None,
        embedding_dim: int = None,
        num_frames: int = None,
        temporal_pooling: str = None,
    ):
        super(VideoModel, self).__init__()
        
        # Use config defaults
        self.backbone_type = backbone or VIDEO_MODEL_CONFIG["backbone"]
        self.pretrained = pretrained if pretrained is not None else VIDEO_MODEL_CONFIG["pretrained"]
        self.embedding_dim = embedding_dim or VIDEO_MODEL_CONFIG["embedding_dim"]
        self.num_frames = num_frames or VIDEO_MODEL_CONFIG["num_frames"]
        self.temporal_pooling = temporal_pooling or VIDEO_MODEL_CONFIG["temporal_pooling"]
        
        # Input dimension depends on input type
        # If None, assume raw video frames (3 channels, 224x224)
        self.input_dim = input_dim
        
        if self.backbone_type.startswith("resnet"):
            self._build_resnet()
        elif self.backbone_type == "vit":
            self._build_vit()
        elif self.backbone_type == "cnn":
            self._build_cnn()
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_type}")
        
        # Temporal modeling
        if self.temporal_pooling == "lstm":
            self.temporal_lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.embedding_dim,
                num_layers=1,
                batch_first=True,
            )
        elif self.temporal_pooling == "attention":
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, 1),
            )
    
    def _build_resnet(self):
        """Build ResNet backbone."""
        if self.backbone_type == "resnet50":
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.backbone_type == "resnet18":
            resnet = models.resnet18(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unknown ResNet variant: {self.backbone_type}")
        
        # Remove final classifier
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet
        resnet_dim = resnet.fc.in_features
        
        # Project to embedding dimension
        self.projection = nn.Linear(resnet_dim, self.embedding_dim)
        
        # Input is raw frames if input_dim not specified
        if self.input_dim is None:
            self.input_dim = 3  # RGB channels
    
    def _build_vit(self):
        """Build Vision Transformer backbone."""
        try:
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained(
                "google/vit-base-patch16-224" if self.pretrained else None
            )
            vit_dim = self.backbone.config.hidden_size
            self.projection = nn.Linear(vit_dim, self.embedding_dim)
            self.input_dim = 3  # RGB channels
        except ImportError:
            raise ImportError(
                "transformers library required for ViT. Install with: pip install transformers"
            )
    
    def _build_cnn(self):
        """Build CNN backbone for landmarks/embeddings."""
        if self.input_dim is None:
            raise ValueError("input_dim must be specified for CNN backbone")
        
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.projection = nn.Linear(256, self.embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape:
               - (batch, num_frames, channels, height, width) for raw frames
               - (batch, num_frames, features) for landmarks/embeddings
        
        Returns:
            Output tensor of shape (batch, embedding_dim)
        """
        batch_size = x.shape[0]
        
        # Process each frame
        if len(x.shape) == 5:  # Raw video frames (batch, frames, C, H, W)
            # Reshape to (batch * frames, C, H, W)
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
            
            # Extract features
            if self.backbone_type == "vit":
                # ViT expects different input format
                features = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token
            else:
                features = self.backbone(x)  # (batch * frames, features)
                features = features.view(batch_size * self.num_frames, -1)
            
            # Project to embedding dimension
            features = self.projection(features)
            
            # Reshape back to (batch, frames, embedding_dim)
            features = features.view(batch_size, self.num_frames, self.embedding_dim)
        
        elif len(x.shape) == 3:  # Landmarks/embeddings (batch, frames, features)
            # Extract features per frame
            features = []
            for i in range(self.num_frames):
                frame_features = self.backbone(x[:, i, :])
                frame_features = self.projection(frame_features)
                features.append(frame_features)
            
            # Stack frames
            features = torch.stack(features, dim=1)  # (batch, frames, embedding_dim)
        
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Temporal pooling
        if self.temporal_pooling == "mean":
            out = features.mean(dim=1)
        elif self.temporal_pooling == "max":
            out = features.max(dim=1)[0]
        elif self.temporal_pooling == "attention":
            # Attention pooling
            attn_weights = self.temporal_attention(features)  # (batch, frames, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            out = (features * attn_weights).sum(dim=1)
        elif self.temporal_pooling == "lstm":
            lstm_out, (h_n, c_n) = self.temporal_lstm(features)
            out = h_n[-1]  # Use last hidden state
        else:
            # Use last frame
            out = features[:, -1, :]
        
        return out


class VideoClassifier(nn.Module):
    """
    Complete video classifier with feature extraction and classification.
    
    Combines VideoModel with a classification head.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        **video_model_kwargs,
    ):
        super(VideoClassifier, self).__init__()
        
        self.video_model = VideoModel(**video_model_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(self.video_model.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input video features
        
        Returns:
            Classification logits
        """
        features = self.video_model(x)
        logits = self.classifier(features)
        return logits


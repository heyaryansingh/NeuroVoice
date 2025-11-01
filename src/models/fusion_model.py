"""
Multimodal fusion model combining audio and video features.

Implements various fusion strategies: early fusion, late fusion, and attention-based fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import FUSION_MODEL_CONFIG, NUM_CLASSES, NUM_DISEASES
from src.models.audio_model import AudioModel
from src.models.video_model import VideoModel
from src.models.attention_modules import CrossModalAttentionBlock


class FusionModel(nn.Module):
    """
    Multimodal fusion model for disease classification.
    
    Combines audio and video features using cross-modal attention and fusion layers.
    
    Args:
        audio_input_dim: Input dimension for audio features
        video_input_dim: Input dimension for video features (None for raw frames)
        audio_model_type: Type of audio model ('lstm', 'cnn', 'transformer')
        video_backbone: Type of video backbone ('resnet50', 'resnet18', 'vit', 'cnn')
        fusion_dim: Dimension of fused features
        num_classes: Number of output classes (default: 2 for binary classification)
        num_diseases: Number of diseases to classify (for multi-task learning)
        use_auxiliary: Whether to use auxiliary emotion prediction task
    """
    
    def __init__(
        self,
        audio_input_dim: int = None,
        video_input_dim: int = None,
        audio_model_type: str = "lstm",
        video_backbone: str = "resnet50",
        fusion_dim: int = None,
        num_classes: int = NUM_CLASSES,
        num_diseases: int = NUM_DISEASES,
        use_auxiliary: bool = True,
    ):
        super(FusionModel, self).__init__()
        
        self.fusion_dim = fusion_dim or FUSION_MODEL_CONFIG["fusion_dim"]
        self.num_classes = num_classes
        self.num_diseases = num_diseases
        self.use_auxiliary = use_auxiliary
        
        # Audio model
        self.audio_model = AudioModel(
            input_dim=audio_input_dim,
            model_type=audio_model_type,
        )
        audio_output_dim = self.audio_model.output_dim
        
        # Video model
        self.video_model = VideoModel(
            input_dim=video_input_dim,
            backbone=video_backbone,
        )
        video_output_dim = self.video_model.embedding_dim
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttentionBlock(
            audio_dim=audio_output_dim,
            video_dim=video_output_dim,
            hidden_dim=self.fusion_dim,
            num_heads=FUSION_MODEL_CONFIG["num_attention_heads"],
            dropout=FUSION_MODEL_CONFIG["dropout"],
        )
        
        # Fusion layers
        fusion_layers = []
        for i in range(FUSION_MODEL_CONFIG["num_fusion_layers"]):
            fusion_layers.extend([
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(FUSION_MODEL_CONFIG["dropout"]),
            ])
        self.fusion_layers = nn.Sequential(*fusion_layers[:-1])  # Remove last dropout
        
        # Classification heads
        # Main disease classification (binary per disease)
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
            for _ in range(num_diseases)
        ])
        
        # Auxiliary emotion classification (optional)
        if use_auxiliary:
            from src.config import NUM_EMOTIONS
            self.emotion_classifier = nn.Sequential(
                nn.Linear(video_output_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, NUM_EMOTIONS),
            )
    
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        disease_idx: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            audio: Audio features (batch, ...)
            video: Video features (batch, ...)
            disease_idx: Disease index for multi-task learning (batch,)
                        If None, returns predictions for all diseases
        
        Returns:
            Dictionary with keys:
                - 'logits': Classification logits
                - 'audio_features': Processed audio features
                - 'video_features': Processed video features
                - 'fused_features': Fused multimodal features
                - 'emotion_logits': Emotion classification logits (if use_auxiliary)
        """
        # Extract features from each modality
        audio_features = self.audio_model(audio)  # (batch, audio_dim)
        video_features = self.video_model(video)  # (batch, video_dim)
        
        # Cross-modal attention
        attended_audio, attended_video = self.cross_modal_attention(
            audio_features,
            video_features,
        )
        
        # Concatenate attended features
        fused = torch.cat([attended_audio, attended_video], dim=1)  # (batch, fusion_dim * 2)
        
        # Project to fusion dimension
        if fused.shape[1] != self.fusion_dim:
            fusion_proj = nn.Linear(fused.shape[1], self.fusion_dim).to(fused.device)
            fused = fusion_proj(fused)
        
        # Apply fusion layers
        fused = self.fusion_layers(fused)  # (batch, fusion_dim)
        
        # Classification
        if disease_idx is not None:
            # Multi-task: use specific classifier for each disease
            logits_list = []
            for i, idx in enumerate(disease_idx):
                logits_list.append(self.classifier[idx.item()](fused[i:i+1]))
            logits = torch.cat(logits_list, dim=0)
        else:
            # Return logits for all diseases
            logits_list = [classifier(fused) for classifier in self.classifier]
            logits = torch.stack(logits_list, dim=1)  # (batch, num_diseases, num_classes)
        
        result = {
            'logits': logits,
            'audio_features': audio_features,
            'video_features': video_features,
            'fused_features': fused,
        }
        
        # Auxiliary emotion prediction
        if self.use_auxiliary:
            emotion_logits = self.emotion_classifier(video_features)
            result['emotion_logits'] = emotion_logits
        
        return result


class EarlyFusionModel(nn.Module):
    """
    Early fusion model that concatenates features before processing.
    
    Simpler alternative to attention-based fusion.
    """
    
    def __init__(
        self,
        audio_input_dim: int,
        video_input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
    ):
        super(EarlyFusionModel, self).__init__()
        
        self.audio_model = AudioModel(input_dim=audio_input_dim)
        self.video_model = VideoModel(input_dim=video_input_dim)
        
        audio_dim = self.audio_model.output_dim
        video_dim = self.video_model.embedding_dim
        combined_dim = audio_dim + video_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        audio_features = self.audio_model(audio)
        video_features = self.video_model(video)
        
        # Concatenate
        combined = torch.cat([audio_features, video_features], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        return logits


class LateFusionModel(nn.Module):
    """
    Late fusion model that processes modalities separately and combines predictions.
    """
    
    def __init__(
        self,
        audio_input_dim: int,
        video_input_dim: int,
        num_classes: int = 2,
        fusion_strategy: str = "weighted_average",
    ):
        super(LateFusionModel, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        
        # Individual classifiers
        self.audio_model = AudioModel(input_dim=audio_input_dim)
        self.video_model = VideoModel(input_dim=video_input_dim)
        
        audio_dim = self.audio_model.output_dim
        video_dim = self.video_model.embedding_dim
        
        self.audio_classifier = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        self.video_classifier = nn.Sequential(
            nn.Linear(video_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        # Learnable fusion weights
        if fusion_strategy == "learned":
            self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
    
    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        audio_features = self.audio_model(audio)
        video_features = self.video_model(video)
        
        # Get predictions from each modality
        audio_logits = self.audio_classifier(audio_features)
        video_logits = self.video_classifier(video_features)
        
        # Fuse predictions
        if self.fusion_strategy == "average":
            logits = (audio_logits + video_logits) / 2
        elif self.fusion_strategy == "weighted_average":
            weights = F.softmax(self.fusion_weights, dim=0)
            logits = weights[0] * audio_logits + weights[1] * video_logits
        else:
            # Concatenate and use additional classifier
            combined = torch.cat([audio_logits, video_logits], dim=1)
            logits = self.fusion_classifier(combined)
        
        return logits


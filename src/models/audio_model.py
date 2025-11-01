"""
Audio processing models for speech analysis.

Implements LSTM, CNN, and transformer-based models for audio feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import AUDIO_MODEL_CONFIG, WAV2VEC2_EMBEDDING_DIM


class AudioModel(nn.Module):
    """
    Audio model for processing speech features.
    
    Supports MFCC, wav2vec embeddings, or raw waveforms.
    Uses LSTM or CNN for temporal modeling.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Hidden dimension for LSTM/CNN
        num_layers: Number of LSTM/CNN layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        pooling: Pooling strategy ('mean', 'max', 'attention')
        model_type: Type of model ('lstm', 'cnn', 'transformer')
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = None,
        bidirectional: bool = None,
        pooling: str = None,
        model_type: str = "lstm",
    ):
        super(AudioModel, self).__init__()
        
        # Use config defaults if not specified
        self.input_dim = input_dim or AUDIO_MODEL_CONFIG["input_dim"]
        self.hidden_dim = hidden_dim or AUDIO_MODEL_CONFIG["hidden_dim"]
        self.num_layers = num_layers or AUDIO_MODEL_CONFIG["num_layers"]
        self.dropout = dropout if dropout is not None else AUDIO_MODEL_CONFIG["dropout"]
        self.bidirectional = bidirectional if bidirectional is not None else AUDIO_MODEL_CONFIG["bidirectional"]
        self.pooling = pooling or AUDIO_MODEL_CONFIG["pooling"]
        self.model_type = model_type
        
        if model_type == "lstm":
            self._build_lstm()
        elif model_type == "cnn":
            self._build_cnn()
        elif model_type == "transformer":
            self._build_transformer()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Pooling layer
        if self.pooling == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 1),
            )
    
    def _build_lstm(self):
        """Build LSTM-based model."""
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        
        # Output dimension
        self.output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
    
    def _build_cnn(self):
        """Build CNN-based model."""
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.input_dim if i == 0 else self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            for i in range(self.num_layers)
        ])
        
        self.output_dim = self.hidden_dim
    
    def _build_transformer(self):
        """Build transformer-based model."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )
        self.output_dim = self.input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
               or (batch, input_dim) for wav2vec embeddings
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # Handle wav2vec embeddings (no sequence dimension)
        if len(x.shape) == 2:
            # Add sequence dimension
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        if self.model_type == "lstm":
            # LSTM forward
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Pooling
            if self.pooling == "mean":
                out = lstm_out.mean(dim=1)
            elif self.pooling == "max":
                out = lstm_out.max(dim=1)[0]
            elif self.pooling == "attention":
                # Attention pooling
                attn_weights = self.attention_pool(lstm_out)  # (batch, seq_len, 1)
                attn_weights = F.softmax(attn_weights, dim=1)
                out = (lstm_out * attn_weights).sum(dim=1)
            else:
                # Use last hidden state
                if self.bidirectional:
                    out = torch.cat([h_n[-2], h_n[-1]], dim=1)
                else:
                    out = h_n[-1]
        
        elif self.model_type == "cnn":
            # Transpose for CNN: (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)
            
            # CNN forward
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            
            # Global average pooling
            x = x.mean(dim=2)  # (batch, features)
            out = x
        
        elif self.model_type == "transformer":
            # Transformer forward
            transformer_out = self.transformer(x)
            
            # Pooling
            if self.pooling == "mean":
                out = transformer_out.mean(dim=1)
            elif self.pooling == "max":
                out = transformer_out.max(dim=1)[0]
            elif self.pooling == "attention":
                attn_weights = self.attention_pool(transformer_out)
                attn_weights = F.softmax(attn_weights, dim=1)
                out = (transformer_out * attn_weights).sum(dim=1)
            else:
                # Use first token (CLS equivalent)
                out = transformer_out[:, 0, :]
        
        return out


class AudioClassifier(nn.Module):
    """
    Complete audio classifier with feature extraction and classification.
    
    Combines AudioModel with a classification head.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        **audio_model_kwargs,
    ):
        super(AudioClassifier, self).__init__()
        
        self.audio_model = AudioModel(**audio_model_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(self.audio_model.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input audio features
        
        Returns:
            Classification logits
        """
        features = self.audio_model(x)
        logits = self.classifier(features)
        return logits


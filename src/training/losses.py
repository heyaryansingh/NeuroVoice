"""
Loss functions for training NeuroVoice models.

Includes weighted BCE, focal loss, and multi-task loss combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    
    Useful for handling class imbalance.
    
    Args:
        pos_weight: Weight for positive class (default: 1.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
        
        Returns:
            Loss value
        """
        # Convert targets to one-hot if needed
        if len(targets.shape) == 1:
            targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        
        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor(self.pos_weight).to(logits.device),
            reduction=self.reduction,
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
        
        Returns:
            Loss value
        """
        # Convert targets to one-hot if needed
        if len(targets.shape) == 1:
            targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        else:
            targets_one_hot = targets
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute cross-entropy
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction='none')
        
        # Compute p_t
        p_t = probs * targets_one_hot + (1 - probs) * (1 - targets_one_hot)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MultitaskLoss(nn.Module):
    """
    Multi-task loss combining classification and auxiliary tasks.
    
    Args:
        classification_loss: Main classification loss function
        auxiliary_loss: Auxiliary task loss function (e.g., emotion prediction)
        auxiliary_weight: Weight for auxiliary loss (default: 0.1)
    """
    
    def __init__(
        self,
        classification_loss: nn.Module,
        auxiliary_loss: nn.Module = None,
        auxiliary_weight: float = 0.1,
    ):
        super(MultitaskLoss, self).__init__()
        self.classification_loss = classification_loss
        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_weight = auxiliary_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        auxiliary_logits: torch.Tensor = None,
        auxiliary_targets: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            logits: Classification logits (batch, num_classes)
            targets: Classification targets (batch,)
            auxiliary_logits: Auxiliary task logits (optional)
            auxiliary_targets: Auxiliary task targets (optional)
        
        Returns:
            Dictionary with:
                - 'total_loss': Combined loss
                - 'classification_loss': Main loss
                - 'auxiliary_loss': Auxiliary loss (if provided)
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, targets)
        
        result = {
            'classification_loss': cls_loss,
            'total_loss': cls_loss,
        }
        
        # Auxiliary loss (if provided)
        if auxiliary_logits is not None and auxiliary_targets is not None and self.auxiliary_loss is not None:
            aux_loss = self.auxiliary_loss(auxiliary_logits, auxiliary_targets)
            result['auxiliary_loss'] = aux_loss
            result['total_loss'] = cls_loss + self.auxiliary_weight * aux_loss
        
        return result


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (InfoNCE) for representation learning.
    
    Reference: "Representation Learning with Contrastive Predictive Coding" (Oord et al., 2018)
    
    Args:
        temperature: Temperature parameter for scaling (default: 0.07)
    """
    
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (batch, embedding_dim)
            embeddings2: Second set of embeddings (batch, embedding_dim)
                         Should be paired with embeddings1 (positive pairs)
        
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings1.shape[0]
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute similarity matrix
        # Shape: (batch, batch)
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Create labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size).to(embeddings1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for multimodal contrastive learning.
    
    More flexible version that supports negative samples and different modalities.
    
    Args:
        temperature: Temperature parameter for scaling (default: 0.07)
    """
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings (batch, dim)
            positive_key: Positive key embeddings (batch, dim)
            negative_keys: Negative key embeddings (batch, num_negatives, dim) (optional)
        
        Returns:
            InfoNCE loss
        """
        batch_size = query.shape[0]
        
        # Normalize
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)
        
        # Positive similarity
        pos_sim = (query * positive_key).sum(dim=1, keepdim=True) / self.temperature  # (batch, 1)
        
        if negative_keys is not None:
            # Normalize negative keys
            negative_keys = F.normalize(negative_keys, p=2, dim=2)
            
            # Compute similarity with negative keys
            # query: (batch, dim), negative_keys: (batch, num_neg, dim)
            neg_sim = torch.bmm(
                query.unsqueeze(1),  # (batch, 1, dim)
                negative_keys.transpose(1, 2)  # (batch, dim, num_neg)
            ) / self.temperature  # (batch, 1, num_neg)
            
            # Concatenate positive and negatives
            logits = torch.cat([pos_sim, neg_sim.squeeze(1)], dim=1)  # (batch, 1 + num_neg)
        else:
            # Use all other samples in batch as negatives
            all_keys = torch.cat([positive_key.unsqueeze(1), 
                                  query.unsqueeze(1)], dim=1)
            all_keys = F.normalize(all_keys, p=2, dim=2)
            
            # Compute similarity with all keys
            sim_matrix = torch.bmm(
                query.unsqueeze(1),  # (batch, 1, dim)
                all_keys.transpose(1, 2)  # (batch, dim, 2)
            ) / self.temperature  # (batch, 1, 2)
            
            logits = sim_matrix.squeeze(1)  # (batch, 2)
        
        # Labels: first element is always positive (index 0)
        labels = torch.zeros(batch_size, dtype=torch.long).to(query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class AuxiliaryEmotionLoss(nn.Module):
    """
    Auxiliary emotion classification loss.
    
    Used as part of multitask learning for emotion prediction.
    
    Args:
        num_emotions: Number of emotion classes (default: 7)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, num_emotions: int = 7, reduction: str = "mean"):
        super(AuxiliaryEmotionLoss, self).__init__()
        self.num_emotions = num_emotions
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        emotion_logits: torch.Tensor,
        emotion_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute emotion classification loss.
        
        Args:
            emotion_logits: Emotion predictions (batch, num_emotions)
            emotion_targets: Emotion labels (batch,)
        
        Returns:
            Loss value
        """
        return self.criterion(emotion_logits, emotion_targets)


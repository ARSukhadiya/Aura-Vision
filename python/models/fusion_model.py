"""
Multimodal Fusion Model

This module implements the fusion mechanism that combines speech and vision
features to make the final emotion prediction, handling conflicting signals
intelligently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for multimodal features."""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, speech_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """Apply attention fusion between speech and vision features."""
        batch_size = speech_features.shape[0]
        
        # Create queries, keys, and values
        Q = self.query(speech_features).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(vision_features).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(vision_features).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_features = torch.matmul(attention_weights, V)
        attended_features = attended_features.view(batch_size, self.feature_dim)
        
        # Output projection
        fused_features = self.output(attended_features)
        
        return fused_features


class ConfidenceWeightedFusion(nn.Module):
    """Confidence-weighted fusion for handling conflicting signals."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Confidence estimation networks
        self.speech_confidence = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.vision_confidence = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.speech_transform = nn.Linear(feature_dim, feature_dim)
        self.vision_transform = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, speech_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """Apply confidence-weighted fusion."""
        # Estimate confidence for each modality
        speech_conf = self.speech_confidence(speech_features)
        vision_conf = self.vision_confidence(vision_features)
        
        # Transform features
        speech_transformed = self.speech_transform(speech_features)
        vision_transformed = self.vision_transform(vision_features)
        
        # Weighted combination
        total_conf = speech_conf + vision_conf + 1e-8  # Avoid division by zero
        weighted_speech = speech_transformed * speech_conf / total_conf
        weighted_vision = vision_transformed * vision_conf / total_conf
        
        fused_features = weighted_speech + weighted_vision
        
        return fused_features


class EmotionFusionModel(nn.Module):
    """
    Multimodal Emotion Fusion Model that intelligently combines speech and vision.
    """
    
    def __init__(
        self,
        speech_feature_dim: int = 512,
        vision_feature_dim: int = 512,
        num_emotions: int = 7,
        fusion_method: str = "attention",
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.speech_feature_dim = speech_feature_dim
        self.vision_feature_dim = vision_feature_dim
        self.num_emotions = num_emotions
        self.fusion_method = fusion_method
        
        # Feature alignment (if dimensions differ)
        if speech_feature_dim != vision_feature_dim:
            self.speech_projection = nn.Linear(speech_feature_dim, vision_feature_dim)
            self.vision_projection = nn.Linear(vision_feature_dim, vision_feature_dim)
            self.aligned_dim = vision_feature_dim
        else:
            self.aligned_dim = speech_feature_dim
        
        # Fusion mechanism
        if fusion_method == "attention":
            self.fusion = AttentionFusion(self.aligned_dim)
        elif fusion_method == "confidence":
            self.fusion = ConfidenceWeightedFusion(self.aligned_dim)
        elif fusion_method == "concat":
            self.fusion = None  # Simple concatenation
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Emotion classifier
        if fusion_method == "concat":
            input_dim = speech_feature_dim + vision_feature_dim
        else:
            input_dim = self.aligned_dim
            
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )
        
        # Emotion mapping
        self.emotion_map = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral"
        }
        
        # Emotion emojis
        self.emotion_emojis = {
            "angry": "😠",
            "disgust": "🤢",
            "fear": "😨",
            "happy": "😊",
            "sad": "😢",
            "surprise": "😲",
            "neutral": "😐"
        }
        
        # Conflict resolution weights
        self.conflict_resolver = nn.Sequential(
            nn.Linear(num_emotions * 2, 128),  # speech + vision probabilities
            nn.ReLU(),
            nn.Linear(128, num_emotions),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self, 
        speech_features: torch.Tensor, 
        vision_features: torch.Tensor,
        speech_probs: Optional[torch.Tensor] = None,
        vision_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multimodal emotion fusion.
        
        Args:
            speech_features: Speech features tensor
            vision_features: Vision features tensor
            speech_probs: Optional speech emotion probabilities
            vision_probs: Optional vision emotion probabilities
            
        Returns:
            Fused emotion logits
        """
        # Align feature dimensions if needed
        if hasattr(self, 'speech_projection'):
            speech_aligned = self.speech_projection(speech_features)
            vision_aligned = self.vision_projection(vision_features)
        else:
            speech_aligned = speech_features
            vision_aligned = vision_features
        
        # Apply fusion
        if self.fusion_method == "concat":
            fused_features = torch.cat([speech_features, vision_features], dim=1)
        else:
            fused_features = self.fusion(speech_aligned, vision_aligned)
        
        # Classify emotions
        emotion_logits = self.classifier(fused_features)
        
        # Apply conflict resolution if probabilities are provided
        if speech_probs is not None and vision_probs is not None:
            conflict_input = torch.cat([speech_probs, vision_probs], dim=1)
            conflict_weights = self.conflict_resolver(conflict_input)
            emotion_logits = emotion_logits * conflict_weights
        
        return emotion_logits
    
    def predict_emotion(
        self, 
        speech_features: torch.Tensor, 
        vision_features: torch.Tensor,
        speech_probs: Optional[torch.Tensor] = None,
        vision_probs: Optional[torch.Tensor] = None
    ) -> str:
        """Predict emotion from fused features."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(speech_features, vision_features, speech_probs, vision_probs)
            probabilities = F.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            return self.emotion_map[predicted_idx]
    
    def get_emotion_probabilities(
        self, 
        speech_features: torch.Tensor, 
        vision_features: torch.Tensor,
        speech_probs: Optional[torch.Tensor] = None,
        vision_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Get emotion probabilities from fused features."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(speech_features, vision_features, speech_probs, vision_probs)
            probabilities = F.softmax(logits, dim=1)
            
            return {
                emotion: probabilities[0, idx].item()
                for idx, emotion in self.emotion_map.items()
            }
    
    def detect_conflict(
        self, 
        speech_probs: torch.Tensor, 
        vision_probs: torch.Tensor,
        threshold: float = 0.3
    ) -> bool:
        """Detect if there's a conflict between speech and vision predictions."""
        speech_pred = torch.argmax(speech_probs, dim=1)
        vision_pred = torch.argmax(vision_probs, dim=1)
        
        # Check if predictions differ
        prediction_conflict = speech_pred != vision_pred
        
        # Check confidence levels
        speech_conf = torch.max(speech_probs, dim=1)[0]
        vision_conf = torch.max(vision_probs, dim=1)[0]
        confidence_conflict = torch.abs(speech_conf - vision_conf) > threshold
        
        return prediction_conflict.item() or confidence_conflict.item()
    
    def resolve_conflict(
        self, 
        speech_probs: torch.Tensor, 
        vision_probs: torch.Tensor
    ) -> torch.Tensor:
        """Resolve conflicts between speech and vision predictions."""
        conflict_input = torch.cat([speech_probs, vision_probs], dim=1)
        resolved_probs = self.conflict_resolver(conflict_input)
        return resolved_probs


class TemporalFusion(nn.Module):
    """Temporal fusion for handling sequences of predictions."""
    
    def __init__(self, feature_dim: int, sequence_length: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply temporal fusion to sequence of features."""
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Self-attention
        attended_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        fused_features = attended_out[:, -1, :]
        
        return fused_features

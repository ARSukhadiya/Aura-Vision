"""
Speech Emotion Recognition Model

This module implements speech emotion recognition using Whisper for transcription 
and audio features for emotion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel


class AudioFeatureExtractor:
    """Extract audio features for emotion recognition."""
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features."""
        features = {}
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        features['mfcc'] = mfcc
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        features['spectral_centroids'] = spectral_centroids
        features['spectral_rolloff'] = spectral_rolloff
        features['spectral_bandwidth'] = spectral_bandwidth
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        features['pitch'] = pitches
        features['magnitudes'] = magnitudes
        
        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        features['rms'] = rms
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr'] = zcr
        
        return features


class SpeechEmotionModel(nn.Module):
    """
    Speech Emotion Recognition Model combining Whisper features and audio features.
    """
    
    def __init__(
        self,
        num_emotions: int = 7,
        whisper_model_name: str = "base",
        hidden_size: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.hidden_size = hidden_size
        
        # Load Whisper model
        self.whisper_model = whisper.load_model(whisper_model_name)
        
        # Audio feature extractor
        self.audio_extractor = AudioFeatureExtractor()
        
        # Whisper feature processing
        self.whisper_projection = nn.Linear(768, hidden_size)  # Whisper base has 768 dims
        
        # Audio feature processing
        self.audio_encoder = nn.Sequential(
            nn.Linear(13 + 3 + 1 + 1, 256),  # mfcc + spectral + rms + zcr
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotions)
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
        
    def forward(self, audio: torch.Tensor, text: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass for speech emotion recognition.
        
        Args:
            audio: Audio tensor of shape (batch_size, audio_length)
            text: Optional text transcription
            
        Returns:
            Emotion logits of shape (batch_size, num_emotions)
        """
        batch_size = audio.shape[0]
        
        # Process audio features
        audio_features = self._extract_audio_features(audio)
        audio_encoded = self.audio_encoder(audio_features)
        
        # Process Whisper features
        if text is not None:
            whisper_features = self._extract_whisper_features(text)
            whisper_encoded = self.whisper_projection(whisper_features)
        else:
            # Use audio-only features if no text
            whisper_encoded = torch.zeros(batch_size, self.hidden_size, device=audio.device)
        
        # Fusion
        combined = torch.cat([audio_encoded, whisper_encoded], dim=1)
        emotion_logits = self.fusion_layer(combined)
        
        return emotion_logits
    
    def _extract_audio_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract and process audio features."""
        features_list = []
        
        for i in range(audio.shape[0]):
            audio_np = audio[i].cpu().numpy()
            features = self.audio_extractor.extract_features(audio_np)
            
            # Aggregate features
            mfcc_mean = np.mean(features['mfcc'], axis=1)
            spectral_mean = np.array([
                np.mean(features['spectral_centroids']),
                np.mean(features['spectral_rolloff']),
                np.mean(features['spectral_bandwidth'])
            ])
            rms_mean = np.mean(features['rms'])
            zcr_mean = np.mean(features['zcr'])
            
            combined = np.concatenate([mfcc_mean, spectral_mean, [rms_mean], [zcr_mean]])
            features_list.append(combined)
        
        return torch.tensor(features_list, dtype=torch.float32, device=audio.device)
    
    def _extract_whisper_features(self, text: str) -> torch.Tensor:
        """Extract Whisper features from text."""
        # For now, use a simple approach - in practice, you'd use Whisper's encoder
        # This is a placeholder for the actual Whisper feature extraction
        return torch.randn(1, 768)  # Placeholder
    
    def predict_emotion(self, audio: torch.Tensor, text: Optional[str] = None) -> str:
        """Predict emotion from audio and optional text."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio, text)
            probabilities = F.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            return self.emotion_map[predicted_idx]
    
    def get_emotion_probabilities(self, audio: torch.Tensor, text: Optional[str] = None) -> Dict[str, float]:
        """Get emotion probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio, text)
            probabilities = F.softmax(logits, dim=1)
            
            return {
                emotion: probabilities[0, idx].item()
                for idx, emotion in self.emotion_map.items()
            }


class WhisperFeatureExtractor:
    """Extract features from Whisper model."""
    
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract Whisper features from audio."""
        # This would implement the actual Whisper feature extraction
        # For now, return a placeholder
        return torch.randn(1, 768)

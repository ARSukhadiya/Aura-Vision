"""
Main Multimodal Emotion Recognition Model

This module orchestrates the complete multimodal emotion recognition pipeline,
combining speech and vision models with intelligent fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2

from .speech_model import SpeechEmotionModel
from .vision_model import VisionEmotionModel
from .fusion_model import EmotionFusionModel, TemporalFusion


class MultimodalEmotionModel(nn.Module):
    """
    Complete multimodal emotion recognition model that combines speech and vision.
    """
    
    def __init__(
        self,
        num_emotions: int = 7,
        speech_model_config: Optional[Dict] = None,
        vision_model_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        use_temporal_fusion: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.device = device
        self.use_temporal_fusion = use_temporal_fusion
        
        # Default configurations
        speech_config = speech_model_config or {
            "whisper_model_name": "base",
            "hidden_size": 512,
            "dropout": 0.3
        }
        
        vision_config = vision_model_config or {
            "input_size": 224,
            "backbone": "mobilenet",
            "use_landmarks": True,
            "dropout": 0.3
        }
        
        fusion_config = fusion_config or {
            "speech_feature_dim": 512,
            "vision_feature_dim": 512,
            "fusion_method": "attention",
            "dropout": 0.3
        }
        
        # Initialize individual models
        self.speech_model = SpeechEmotionModel(
            num_emotions=num_emotions,
            **speech_config
        ).to(device)
        
        self.vision_model = VisionEmotionModel(
            num_emotions=num_emotions,
            **vision_config
        ).to(device)
        
        self.fusion_model = EmotionFusionModel(
            num_emotions=num_emotions,
            **fusion_config
        ).to(device)
        
        # Temporal fusion for sequence processing
        if use_temporal_fusion:
            self.temporal_fusion = TemporalFusion(
                feature_dim=fusion_config["speech_feature_dim"],
                sequence_length=10
            ).to(device)
        
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
        
        # Prediction history for temporal fusion
        self.prediction_history = []
        self.max_history_length = 10
        
    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        landmarks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal emotion recognition.
        
        Args:
            audio: Audio tensor
            image: Image tensor
            text: Optional text transcription
            landmarks: Optional facial landmarks
            
        Returns:
            Dictionary containing predictions and features
        """
        results = {}
        
        # Speech processing
        if audio is not None:
            speech_logits = self.speech_model(audio, text)
            speech_probs = F.softmax(speech_logits, dim=1)
            results['speech_logits'] = speech_logits
            results['speech_probs'] = speech_probs
            results['speech_prediction'] = torch.argmax(speech_probs, dim=1)
        
        # Vision processing
        if image is not None:
            vision_logits = self.vision_model(image, landmarks)
            vision_probs = F.softmax(vision_logits, dim=1)
            results['vision_logits'] = vision_logits
            results['vision_probs'] = vision_probs
            results['vision_prediction'] = torch.argmax(vision_probs, dim=1)
        
        # Multimodal fusion
        if audio is not None and image is not None:
            # Extract features from individual models
            speech_features = self.speech_model.audio_encoder(
                self.speech_model._extract_audio_features(audio)
            )
            vision_features = self.vision_model.backbone(image)
            
            # Apply fusion
            fusion_logits = self.fusion_model(
                speech_features, 
                vision_features,
                speech_probs if 'speech_probs' in results else None,
                vision_probs if 'vision_probs' in results else None
            )
            fusion_probs = F.softmax(fusion_logits, dim=1)
            
            results['fusion_logits'] = fusion_logits
            results['fusion_probs'] = fusion_probs
            results['fusion_prediction'] = torch.argmax(fusion_probs, dim=1)
            
            # Update prediction history
            self._update_prediction_history(fusion_probs)
            
            # Apply temporal fusion if enabled
            if self.use_temporal_fusion and len(self.prediction_history) >= 3:
                temporal_features = torch.stack(self.prediction_history[-10:])
                temporal_fused = self.temporal_fusion(temporal_features.unsqueeze(0))
                results['temporal_features'] = temporal_fused
        
        return results
    
    def predict_emotion(
        self,
        audio: Optional[Union[torch.Tensor, np.ndarray]] = None,
        image: Optional[Union[torch.Tensor, np.ndarray]] = None,
        text: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Predict emotion from audio and/or image.
        
        Args:
            audio: Audio data
            image: Image data
            text: Optional text transcription
            
        Returns:
            Dictionary with emotion predictions
        """
        self.eval()
        with torch.no_grad():
            # Preprocess inputs
            audio_tensor = self._preprocess_audio(audio) if audio is not None else None
            image_tensor = self._preprocess_image(image) if image is not None else None
            
            # Get model outputs
            results = self.forward(audio_tensor, image_tensor, text)
            
            # Extract predictions
            predictions = {}
            
            if 'speech_prediction' in results:
                speech_emotion = self.emotion_map[results['speech_prediction'].item()]
                predictions['speech'] = speech_emotion
                predictions['speech_emoji'] = self.emotion_emojis[speech_emotion]
            
            if 'vision_prediction' in results:
                vision_emotion = self.emotion_map[results['vision_prediction'].item()]
                predictions['vision'] = vision_emotion
                predictions['vision_emoji'] = self.emotion_emojis[vision_emotion]
            
            if 'fusion_prediction' in results:
                fusion_emotion = self.emotion_map[results['fusion_prediction'].item()]
                predictions['fusion'] = fusion_emotion
                predictions['fusion_emoji'] = self.emotion_emojis[fusion_emotion]
                
                # Check for conflicts
                if 'speech_prediction' in results and 'vision_prediction' in results:
                    conflict = self.fusion_model.detect_conflict(
                        results['speech_probs'], 
                        results['vision_probs']
                    )
                    predictions['conflict_detected'] = conflict
            
            return predictions
    
    def get_emotion_probabilities(
        self,
        audio: Optional[Union[torch.Tensor, np.ndarray]] = None,
        image: Optional[Union[torch.Tensor, np.ndarray]] = None,
        text: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get emotion probabilities from all modalities.
        
        Args:
            audio: Audio data
            image: Image data
            text: Optional text transcription
            
        Returns:
            Dictionary with emotion probabilities for each modality
        """
        self.eval()
        with torch.no_grad():
            # Preprocess inputs
            audio_tensor = self._preprocess_audio(audio) if audio is not None else None
            image_tensor = self._preprocess_image(image) if image is not None else None
            
            # Get model outputs
            results = self.forward(audio_tensor, image_tensor, text)
            
            # Extract probabilities
            probabilities = {}
            
            if 'speech_probs' in results:
                speech_probs = results['speech_probs'][0]
                probabilities['speech'] = {
                    emotion: speech_probs[idx].item()
                    for idx, emotion in self.emotion_map.items()
                }
            
            if 'vision_probs' in results:
                vision_probs = results['vision_probs'][0]
                probabilities['vision'] = {
                    emotion: vision_probs[idx].item()
                    for idx, emotion in self.emotion_map.items()
                }
            
            if 'fusion_probs' in results:
                fusion_probs = results['fusion_probs'][0]
                probabilities['fusion'] = {
                    emotion: fusion_probs[idx].item()
                    for idx, emotion in self.emotion_map.items()
                }
            
            return probabilities
    
    def _preprocess_audio(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess audio for model input."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        return audio.to(self.device)
    
    def _preprocess_image(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def _update_prediction_history(self, probs: torch.Tensor):
        """Update prediction history for temporal fusion."""
        self.prediction_history.append(probs[0].detach().cpu())
        
        if len(self.prediction_history) > self.max_history_length:
            self.prediction_history.pop(0)
    
    def clear_history(self):
        """Clear prediction history."""
        self.prediction_history.clear()
    
    def get_model_summary(self) -> Dict[str, int]:
        """Get model parameter counts."""
        return {
            'speech_model': sum(p.numel() for p in self.speech_model.parameters()),
            'vision_model': sum(p.numel() for p in self.vision_model.parameters()),
            'fusion_model': sum(p.numel() for p in self.fusion_model.parameters()),
            'temporal_fusion': sum(p.numel() for p in self.temporal_fusion.parameters()) if self.use_temporal_fusion else 0,
            'total': sum(p.numel() for p in self.parameters())
        }
    
    def optimize_for_mobile(self):
        """Optimize model for mobile deployment."""
        # Convert to evaluation mode
        self.eval()
        
        # Quantize models
        self.speech_model = torch.quantization.quantize_dynamic(
            self.speech_model, {nn.Linear}, dtype=torch.qint8
        )
        
        self.vision_model = torch.quantization.quantize_dynamic(
            self.vision_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        self.fusion_model = torch.quantization.quantize_dynamic(
            self.fusion_model, {nn.Linear}, dtype=torch.qint8
        )
        
        if self.use_temporal_fusion:
            self.temporal_fusion = torch.quantization.quantize_dynamic(
                self.temporal_fusion, {nn.Linear}, dtype=torch.qint8
            )
    
    def export_to_coreml(self, output_path: str):
        """Export model to Core ML format for iOS deployment."""
        try:
            import coremltools as ct
            
            # Create a dummy input for tracing
            dummy_audio = torch.randn(1, 16000)  # 1 second of audio
            dummy_image = torch.randn(1, 3, 224, 224)
            
            # Trace the model
            traced_model = torch.jit.trace(
                self, 
                (dummy_audio, dummy_image, None, None)
            )
            
            # Convert to Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(name="audio", shape=dummy_audio.shape),
                    ct.TensorType(name="image", shape=dummy_image.shape)
                ],
                outputs=[
                    ct.TensorType(name="emotion_logits"),
                    ct.TensorType(name="emotion_probs")
                ]
            )
            
            # Save the model
            coreml_model.save(output_path)
            print(f"Model exported to {output_path}")
            
        except ImportError:
            print("Core ML Tools not available. Install with: pip install coremltools")
        except Exception as e:
            print(f"Error exporting to Core ML: {e}")


class RealTimeEmotionProcessor:
    """Real-time emotion processing for live audio/video streams."""
    
    def __init__(self, model: MultimodalEmotionModel, buffer_size: int = 30):
        self.model = model
        self.buffer_size = buffer_size
        self.audio_buffer = []
        self.video_buffer = []
        self.emotion_history = []
        
    def process_frame(
        self, 
        audio_frame: Optional[np.ndarray] = None,
        video_frame: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """Process a single frame of audio/video data."""
        # Add to buffers
        if audio_frame is not None:
            self.audio_buffer.append(audio_frame)
        if video_frame is not None:
            self.video_buffer.append(video_frame)
        
        # Maintain buffer size
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer.pop(0)
        if len(self.video_buffer) > self.buffer_size:
            self.video_buffer.pop(0)
        
        # Process if we have enough data
        if len(self.audio_buffer) >= 5 or len(self.video_buffer) >= 5:
            # Combine audio frames
            if self.audio_buffer:
                combined_audio = np.concatenate(self.audio_buffer)
                audio_tensor = torch.from_numpy(combined_audio).float()
            else:
                audio_tensor = None
            
            # Use latest video frame
            video_tensor = None
            if self.video_buffer:
                latest_frame = self.video_buffer[-1]
                video_tensor = torch.from_numpy(latest_frame).permute(2, 0, 1).float() / 255.0
                video_tensor = video_tensor.unsqueeze(0)
            
            # Get prediction
            predictions = self.model.predict_emotion(audio_tensor, video_tensor)
            
            # Update history
            if 'fusion' in predictions:
                self.emotion_history.append(predictions['fusion'])
                if len(self.emotion_history) > 50:
                    self.emotion_history.pop(0)
            
            return predictions
        
        return {"status": "buffering"}
    
    def get_emotion_trend(self) -> str:
        """Get the most common emotion in recent history."""
        if not self.emotion_history:
            return "neutral"
        
        from collections import Counter
        emotion_counts = Counter(self.emotion_history[-20:])  # Last 20 predictions
        return emotion_counts.most_common(1)[0][0]

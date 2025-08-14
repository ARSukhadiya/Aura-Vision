"""
Vision-based Emotion Recognition Model

This module implements facial expression recognition for emotion classification
using deep learning models optimized for mobile deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import mediapipe as mp


class FaceDetector:
    """Face detection using MediaPipe for real-time processing."""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop face from image."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding
            padding = 0.2
            x = max(0, int(x - width * padding))
            y = max(0, int(y - height * padding))
            width = min(w - x, int(width * (1 + 2 * padding)))
            height = min(h - y, int(height * (1 + 2 * padding)))
            
            face_crop = image[y:y+height, x:x+width]
            return face_crop
        
        return None
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Convert landmarks to numpy array
            landmarks_array = np.array([
                [landmark.x, landmark.y, landmark.z] 
                for landmark in landmarks.landmark
            ])
            return landmarks_array
        
        return None


class VisionEmotionModel(nn.Module):
    """
    Vision-based Emotion Recognition Model using facial expression analysis.
    """
    
    def __init__(
        self,
        num_emotions: int = 7,
        input_size: int = 224,
        backbone: str = "mobilenet",
        use_landmarks: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.input_size = input_size
        self.use_landmarks = use_landmarks
        
        # Face detector
        self.face_detector = FaceDetector()
        
        # Backbone network
        if backbone == "mobilenet":
            self.backbone = self._create_mobilenet_backbone()
        elif backbone == "efficientnet":
            self.backbone = self._create_efficientnet_backbone()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Landmark processing (if enabled)
        if use_landmarks:
            self.landmark_encoder = nn.Sequential(
                nn.Linear(468 * 3, 512),  # MediaPipe has 468 landmarks with 3D coordinates
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Emotion classifier
        backbone_features = 1280 if backbone == "mobilenet" else 1280  # EfficientNet-B0
        landmark_features = 256 if use_landmarks else 0
        
        self.classifier = nn.Sequential(
            nn.Linear(backbone_features + landmark_features, 512),
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
    
    def _create_mobilenet_backbone(self) -> nn.Module:
        """Create MobileNet backbone for feature extraction."""
        # Simplified MobileNet-like architecture for mobile deployment
        return nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Depthwise separable convolutions
            self._make_dw_separable(32, 64, stride=1),
            self._make_dw_separable(64, 128, stride=2),
            self._make_dw_separable(128, 128, stride=1),
            self._make_dw_separable(128, 256, stride=2),
            self._make_dw_separable(256, 256, stride=1),
            self._make_dw_separable(256, 512, stride=2),
            
            # Final layers
            self._make_dw_separable(512, 512, stride=1),
            self._make_dw_separable(512, 512, stride=1),
            self._make_dw_separable(512, 512, stride=1),
            self._make_dw_separable(512, 512, stride=1),
            self._make_dw_separable(512, 512, stride=1),
            
            self._make_dw_separable(512, 1024, stride=2),
            self._make_dw_separable(1024, 1024, stride=1),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _make_dw_separable(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """Create depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _create_efficientnet_backbone(self) -> nn.Module:
        """Create EfficientNet backbone for feature extraction."""
        # Simplified EfficientNet-like architecture
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # MBConv blocks (simplified)
            self._make_mbconv(32, 16, stride=1, expand_ratio=1),
            self._make_mbconv(16, 24, stride=2, expand_ratio=6),
            self._make_mbconv(24, 24, stride=1, expand_ratio=6),
            self._make_mbconv(24, 40, stride=2, expand_ratio=6),
            self._make_mbconv(40, 40, stride=1, expand_ratio=6),
            self._make_mbconv(40, 80, stride=2, expand_ratio=6),
            self._make_mbconv(80, 80, stride=1, expand_ratio=6),
            self._make_mbconv(80, 80, stride=1, expand_ratio=6),
            self._make_mbconv(80, 112, stride=1, expand_ratio=6),
            self._make_mbconv(112, 112, stride=1, expand_ratio=6),
            self._make_mbconv(112, 192, stride=2, expand_ratio=6),
            self._make_mbconv(192, 192, stride=1, expand_ratio=6),
            self._make_mbconv(192, 192, stride=1, expand_ratio=6),
            self._make_mbconv(192, 320, stride=1, expand_ratio=6),
            
            # Final conv
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _make_mbconv(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> nn.Sequential:
        """Create MBConv block for EfficientNet."""
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride,
                     padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, image: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for vision emotion recognition.
        
        Args:
            image: Image tensor of shape (batch_size, 3, height, width)
            landmarks: Optional landmark tensor of shape (batch_size, num_landmarks, 3)
            
        Returns:
            Emotion logits of shape (batch_size, num_emotions)
        """
        # Extract features from image
        image_features = self.backbone(image)
        
        # Process landmarks if provided
        if self.use_landmarks and landmarks is not None:
            batch_size = landmarks.shape[0]
            landmarks_flat = landmarks.view(batch_size, -1)
            landmark_features = self.landmark_encoder(landmarks_flat)
            combined_features = torch.cat([image_features, landmark_features], dim=1)
        else:
            combined_features = image_features
        
        # Classify emotions
        emotion_logits = self.classifier(combined_features)
        
        return emotion_logits
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Preprocess image for model input."""
        # Detect and crop face
        face_crop = self.face_detector.detect_face(image)
        if face_crop is None:
            return None, None
        
        # Resize to model input size
        face_resized = cv2.resize(face_crop, (self.input_size, self.input_size))
        
        # Normalize and convert to tensor
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Extract landmarks
        landmarks = self.face_detector.extract_landmarks(face_crop)
        if landmarks is not None:
            landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0)
        else:
            landmarks_tensor = None
        
        return face_tensor, landmarks_tensor
    
    def predict_emotion(self, image: np.ndarray) -> str:
        """Predict emotion from image."""
        self.eval()
        with torch.no_grad():
            face_tensor, landmarks_tensor = self.preprocess_image(image)
            if face_tensor is None:
                return "no_face"
            
            logits = self.forward(face_tensor, landmarks_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            return self.emotion_map[predicted_idx]
    
    def get_emotion_probabilities(self, image: np.ndarray) -> Dict[str, float]:
        """Get emotion probabilities from image."""
        self.eval()
        with torch.no_grad():
            face_tensor, landmarks_tensor = self.preprocess_image(image)
            if face_tensor is None:
                return {"no_face": 1.0}
            
            logits = self.forward(face_tensor, landmarks_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            return {
                emotion: probabilities[0, idx].item()
                for idx, emotion in self.emotion_map.items()
            }
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji for emotion."""
        return self.emotion_emojis.get(emotion, "❓")

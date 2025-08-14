"""
Models package for Aura-Vision multimodal emotion recognition.
"""

from .multimodal_model import MultimodalEmotionModel
from .speech_model import SpeechEmotionModel
from .vision_model import VisionEmotionModel
from .fusion_model import EmotionFusionModel

__all__ = [
    'MultimodalEmotionModel',
    'SpeechEmotionModel', 
    'VisionEmotionModel',
    'EmotionFusionModel'
]

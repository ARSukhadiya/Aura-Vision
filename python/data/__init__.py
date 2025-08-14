"""
Data processing package for Aura-Vision multimodal emotion recognition.
"""

from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .dataset import EmotionDataset
from .augmentation import AudioAugmentation, VideoAugmentation

__all__ = [
    'AudioProcessor',
    'VideoProcessor',
    'EmotionDataset',
    'AudioAugmentation',
    'VideoAugmentation'
]

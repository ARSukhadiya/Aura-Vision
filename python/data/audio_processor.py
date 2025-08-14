"""
Audio Processing Utilities

This module provides audio processing functionality for speech emotion recognition,
including feature extraction, preprocessing, and augmentation.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import torch
import torchaudio
from scipy import signal
import matplotlib.pyplot as plt


class AudioProcessor:
    """Audio processing utilities for emotion recognition."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample to target sample rate."""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return np.array([])
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        return mfcc
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive spectral features."""
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Spectral flatness
        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            y=audio, hop_length=self.hop_length
        )[0]
        
        return features
    
    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract pitch-related features."""
        features = {}
        
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        features['pitches'] = pitches
        features['magnitudes'] = magnitudes
        
        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        features['f0'] = f0
        features['voiced_flag'] = voiced_flag
        features['voiced_probs'] = voiced_probs
        
        return features
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract energy-related features."""
        features = {}
        
        # RMS energy
        features['rms'] = librosa.feature.rms(
            y=audio, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        # Energy entropy
        features['energy_entropy'] = librosa.feature.spectral_flatness(
            y=audio, hop_length=self.hop_length
        )[0]
        
        return features
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all audio features."""
        features = {}
        
        # MFCC
        features['mfcc'] = self.extract_mfcc(audio)
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)
        
        # Pitch features
        pitch_features = self.extract_pitch_features(audio)
        features.update(pitch_features)
        
        # Energy features
        energy_features = self.extract_energy_features(audio)
        features.update(energy_features)
        
        return features
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to zero mean and unit variance."""
        return (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
    
    def trim_silence(self, audio: np.ndarray, threshold_db: float = -60) -> np.ndarray:
        """Remove silence from beginning and end of audio."""
        return librosa.effects.trim(audio, top_db=abs(threshold_db))[0]
    
    def segment_audio(self, audio: np.ndarray, segment_length: float = 3.0) -> List[np.ndarray]:
        """Segment audio into fixed-length chunks."""
        segment_samples = int(segment_length * self.sample_rate)
        segments = []
        
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) == segment_samples:  # Only include complete segments
                segments.append(segment)
        
        return segments
    
    def pad_audio(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad audio to target length."""
        if len(audio) >= target_length:
            return audio[:target_length]
        else:
            padding = target_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple noise reduction using spectral gating."""
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first few frames
        noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        
        # Apply spectral gating
        threshold = 2.0 * noise_estimate
        mask = magnitude > threshold
        magnitude_filtered = magnitude * mask
        
        # Reconstruct audio
        stft_filtered = magnitude_filtered * np.exp(1j * phase)
        audio_filtered = librosa.istft(stft_filtered, hop_length=self.hop_length)
        
        return audio_filtered
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=128
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    def extract_chromagram(self, audio: np.ndarray) -> np.ndarray:
        """Extract chromagram features."""
        return librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
    
    def get_feature_statistics(self, features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute statistics for extracted features."""
        stats = {}
        
        for feature_name, feature_data in features.items():
            if feature_data.size > 0:
                stats[feature_name] = {
                    'mean': np.mean(feature_data),
                    'std': np.std(feature_data),
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'median': np.median(feature_data)
                }
        
        return stats
    
    def visualize_features(self, audio: np.ndarray, features: Dict[str, np.ndarray], save_path: Optional[str] = None):
        """Visualize extracted audio features."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('Audio Features Visualization', fontsize=16)
        
        # Waveform
        axes[0, 0].plot(audio)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        
        # MFCC
        if 'mfcc' in features:
            im = axes[0, 1].imshow(features['mfcc'], aspect='auto', origin='lower')
            axes[0, 1].set_title('MFCC')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('MFCC Coefficient')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Mel spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        im = axes[1, 0].imshow(mel_spec, aspect='auto', origin='lower')
        axes[1, 0].set_title('Mel Spectrogram')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Mel Frequency')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Spectral centroid
        if 'spectral_centroid' in features:
            axes[1, 1].plot(features['spectral_centroid'])
            axes[1, 1].set_title('Spectral Centroid')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Frequency (Hz)')
        
        # RMS energy
        if 'rms' in features:
            axes[2, 0].plot(features['rms'])
            axes[2, 0].set_title('RMS Energy')
            axes[2, 0].set_xlabel('Frame')
            axes[2, 0].set_ylabel('Energy')
        
        # Zero crossing rate
        if 'zcr' in features:
            axes[2, 1].plot(features['zcr'])
            axes[2, 1].set_title('Zero Crossing Rate')
            axes[2, 1].set_xlabel('Frame')
            axes[2, 1].set_ylabel('Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class AudioAugmentation:
    """Audio augmentation techniques for training data enhancement."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to audio."""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def time_shift(self, audio: np.ndarray, shift_factor: float = 0.1) -> np.ndarray:
        """Apply time shifting to audio."""
        shift_samples = int(shift_factor * len(audio))
        return np.roll(audio, shift_samples)
    
    def pitch_shift(self, audio: np.ndarray, steps: int = 2) -> np.ndarray:
        """Apply pitch shifting to audio."""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
    
    def speed_change(self, audio: np.ndarray, speed_factor: float = 1.2) -> np.ndarray:
        """Change audio speed."""
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def add_reverb(self, audio: np.ndarray, room_scale: float = 0.1) -> np.ndarray:
        """Add reverb effect to audio."""
        # Simple reverb simulation
        delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
        decay = 0.5
        
        delayed = np.pad(audio, (delay_samples, 0), mode='constant')
        reverb = decay * delayed[:len(audio)]
        
        return audio + room_scale * reverb
    
    def apply_bandpass_filter(self, audio: np.ndarray, low_freq: float = 300, high_freq: float = 3000) -> np.ndarray:
        """Apply bandpass filter to audio."""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)
    
    def random_augment(self, audio: np.ndarray, augmentation_prob: float = 0.5) -> np.ndarray:
        """Apply random augmentation to audio."""
        augmented = audio.copy()
        
        if np.random.random() < augmentation_prob:
            augmented = self.add_noise(augmented, noise_level=np.random.uniform(0.005, 0.02))
        
        if np.random.random() < augmentation_prob:
            augmented = self.time_shift(augmented, shift_factor=np.random.uniform(-0.1, 0.1))
        
        if np.random.random() < augmentation_prob:
            steps = np.random.uniform(-2, 2)
            augmented = self.pitch_shift(augmented, steps=int(steps))
        
        if np.random.random() < augmentation_prob:
            speed_factor = np.random.uniform(0.9, 1.1)
            augmented = self.speed_change(augmented, speed_factor)
        
        return augmented

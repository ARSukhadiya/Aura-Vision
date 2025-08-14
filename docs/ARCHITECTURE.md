# Aura-Vision Architecture Documentation

## Overview

Aura-Vision is a multimodal emotion recognition system that combines speech and vision analysis to provide real-time emotional feedback. The system is designed for on-device processing to ensure privacy and low latency.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Aura-Vision System                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   iOS App       │    │  Python ML      │                │
│  │   (Swift)       │    │  Pipeline       │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           │                       │                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Core ML Model                          │    │
│  │         (Exported from Python)                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Python ML Pipeline

### Components

#### 1. Speech Emotion Recognition (`speech_model.py`)
- **Audio Feature Extraction**: MFCC, spectral features, pitch, energy
- **Whisper Integration**: Speech-to-text and feature extraction
- **Neural Network**: Audio feature processing and emotion classification

#### 2. Vision Emotion Recognition (`vision_model.py`)
- **Face Detection**: MediaPipe-based face detection and landmark extraction
- **Backbone Networks**: MobileNet/EfficientNet for feature extraction
- **Facial Expression Analysis**: Landmark-based emotion classification

#### 3. Multimodal Fusion (`fusion_model.py`)
- **Attention Fusion**: Cross-modal attention mechanism
- **Confidence Weighting**: Dynamic weighting based on modality confidence
- **Conflict Resolution**: Handling conflicting signals between modalities

#### 4. Main Model (`multimodal_model.py`)
- **Orchestration**: Coordinates all components
- **Real-time Processing**: Buffered processing for live streams
- **Model Optimization**: Quantization and pruning for mobile deployment

### Data Flow

```
Audio Input → Audio Features → Speech Model → Speech Emotion
     ↓              ↓              ↓              ↓
Video Input → Face Detection → Vision Model → Vision Emotion
     ↓              ↓              ↓              ↓
              Multimodal Fusion → Final Emotion
```

## iOS Application

### Components

#### 1. Managers
- **EmotionRecognitionManager**: Orchestrates emotion recognition
- **CameraManager**: Handles video capture and face detection
- **AudioManager**: Manages audio capture and speech recognition

#### 2. Views
- **EmotionDisplayView**: Shows current emotion with confidence
- **CameraPreviewView**: Live camera feed with face detection overlay
- **TranscriptionView**: Real-time speech transcription

#### 3. Core ML Integration
- **Model Loading**: Loads trained Core ML model
- **Inference**: Real-time emotion prediction
- **Result Processing**: Processes and displays results

### Data Flow

```
Camera → CameraManager → Image Buffer → Core ML Model
  ↓           ↓              ↓              ↓
Audio → AudioManager → Audio Buffer → Emotion Result
  ↓           ↓              ↓              ↓
        UI Updates ← EmotionManager ← Processing
```

## Model Architecture

### Speech Model
```
Input Audio (16kHz) → MFCC Features → Audio Encoder → Speech Emotion
     ↓                    ↓              ↓              ↓
Whisper Features → Feature Fusion → Classification → Confidence
```

### Vision Model
```
Input Image (224x224) → Face Detection → Face Crop → Backbone Network
     ↓                    ↓              ↓              ↓
Landmarks → Landmark Encoder → Feature Fusion → Vision Emotion
```

### Fusion Model
```
Speech Features → Attention/Confidence → Fusion Layer → Final Emotion
     ↓                    ↓              ↓              ↓
Vision Features → Weighting → Classification → Confidence
```

## Performance Considerations

### Optimization Strategies

1. **Model Quantization**
   - 8-bit quantization for weights
   - Reduced model size and inference time
   - Minimal accuracy loss

2. **Model Pruning**
   - Remove unnecessary connections
   - Sparse model architecture
   - Faster inference

3. **Efficient Architectures**
   - MobileNet/EfficientNet backbones
   - Depthwise separable convolutions
   - Reduced parameter count

### Real-time Processing

1. **Buffering Strategy**
   - 3-second audio buffers
   - Frame-by-frame video processing
   - Sliding window approach

2. **Parallel Processing**
   - Separate queues for audio/video
   - Background processing
   - UI updates on main thread

3. **Memory Management**
   - Efficient buffer management
   - Automatic cleanup
   - Memory pooling

## Privacy and Security

### On-Device Processing
- All processing happens locally
- No data sent to external servers
- User privacy maintained

### Data Handling
- No persistent storage of audio/video
- Temporary buffers only
- Automatic cleanup

### Permissions
- Camera access for face detection
- Microphone access for speech recognition
- Clear usage descriptions

## Deployment Pipeline

### Training Pipeline
```
Data Collection → Preprocessing → Model Training → Evaluation
     ↓              ↓              ↓              ↓
Augmentation → Feature Extraction → Optimization → Validation
```

### Export Pipeline
```
Trained Model → Core ML Conversion → iOS Integration → Testing
     ↓              ↓              ↓              ↓
Quantization → Model Optimization → App Bundle → Deployment
```

## Future Enhancements

### Planned Features
1. **Temporal Analysis**: Long-term emotion trends
2. **Context Awareness**: Environmental context integration
3. **Personalization**: User-specific model adaptation
4. **Accessibility**: Enhanced accessibility features

### Technical Improvements
1. **Advanced Fusion**: Transformer-based fusion
2. **Real-time Adaptation**: Online learning capabilities
3. **Multi-language Support**: Internationalization
4. **Cloud Integration**: Optional cloud processing

## Development Guidelines

### Code Organization
- Modular architecture
- Clear separation of concerns
- Comprehensive documentation
- Unit and integration tests

### Performance Standards
- < 100ms inference time
- < 50MB model size
- < 10% CPU usage
- Smooth 30 FPS video

### Quality Assurance
- Automated testing
- Performance benchmarking
- User experience testing
- Accessibility compliance

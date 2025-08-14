# Aura-Vision
On-Device Multimodal Emotion Recognition for Enhanced Accessibility

## Project Overview
Aura-Vision is an interactive mobile application that provides real-time feedback on the emotional sentiment of conversations. This tool assists individuals who may have difficulty perceiving social cues by analyzing both speech (content and tone) and facial expressions (visual cues). The core innovation is running the model efficiently on-device to ensure user privacy and low latency.

## Key Features
- **Real-time Multimodal Analysis**: Combines speech recognition and facial expression analysis
- **On-Device Processing**: Ensures privacy and low latency
- **Accessibility Focus**: Designed for users with social cue perception difficulties
- **Interactive Feedback**: Real-time emotion display with emojis and text labels

## Architecture

### Python ML Pipeline
- **Data Preprocessing**: Audio and video preprocessing for model training
- **Model Training**: Multimodal fusion of speech and facial expression models
- **Model Optimization**: Quantization and pruning for mobile deployment
- **Core ML Conversion**: Export trained models for iOS integration

### iOS Swift Application
- **Real-time Capture**: Audio from microphone and video from camera
- **Model Integration**: Core ML integration for on-device inference
- **User Interface**: Live camera feed with real-time transcription and emotion feedback
- **Accessibility Features**: Designed for users with diverse needs

## Technology Stack

### Machine Learning
- **Python**: PyTorch/TensorFlow for model development
- **Speech Recognition**: Whisper for audio processing
- **Facial Expression Recognition**: FER (Facial Expression Recognition) models
- **Model Fusion**: Custom multimodal fusion algorithms
- **Optimization**: Model quantization and pruning techniques

### Mobile Development
- **Swift**: Native iOS application development
- **Core ML**: On-device machine learning inference
- **AVFoundation**: Audio and video capture
- **UIKit/SwiftUI**: User interface development

## Project Structure
```
Aura-Vision/
├── python/                    # Python ML pipeline
│   ├── models/               # Model definitions and training
│   ├── data/                 # Data preprocessing and augmentation
│   ├── utils/                # Utility functions
│   └── notebooks/            # Jupyter notebooks for experimentation
├── ios/                      # iOS Swift application
│   ├── AuraVision/          # Main iOS app
│   ├── AuraVisionTests/     # Unit tests
│   └── AuraVisionUITests/   # UI tests
├── models/                   # Trained model files
├── data/                     # Dataset storage
├── docs/                     # Documentation
└── scripts/                  # Build and deployment scripts
```

## Getting Started

### Prerequisites
- Python 3.8+
- Xcode 14+
- iOS 16.0+
- PyTorch/TensorFlow
- Core ML Tools

### Installation
1. Clone the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Open the iOS project in Xcode
4. Build and run the application

## Development Roadmap
- [ ] Speech recognition model integration
- [ ] Facial expression recognition model
- [ ] Multimodal fusion algorithm
- [ ] Model optimization for mobile
- [ ] iOS application development
- [ ] Real-time processing pipeline
- [ ] User interface design
- [ ] Accessibility features
- [ ] Testing and validation

## Contributing
This project is designed to enhance accessibility and social interaction. We welcome contributions that improve the user experience, model accuracy, or performance.

## License
[License information to be added]

## Contact
[Contact information to be added]

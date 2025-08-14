#!/usr/bin/env python3
"""
Main Training Script for Aura-Vision Multimodal Emotion Recognition

This script trains the complete multimodal emotion recognition model
combining speech and vision features with intelligent fusion.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_model import MultimodalEmotionModel
from utils.training_utils import TrainingUtils
from data.audio_processor import AudioProcessor
from data.video_processor import VideoProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Aura-Vision Multimodal Emotion Recognition Model')
    
    # Model configuration
    parser.add_argument('--num-emotions', type=int, default=7, help='Number of emotion classes')
    parser.add_argument('--speech-hidden-size', type=int, default=512, help='Speech model hidden size')
    parser.add_argument('--vision-input-size', type=int, default=224, help='Vision model input size')
    parser.add_argument('--fusion-method', type=str, default='attention', 
                       choices=['attention', 'confidence', 'concat'], help='Fusion method')
    parser.add_argument('--use-temporal-fusion', action='store_true', help='Use temporal fusion')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    
    # Data configuration
    parser.add_argument('--train-data-dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val-data-dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--audio-sample-rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--audio-segment-length', type=float, default=3.0, help='Audio segment length in seconds')
    
    # Loss configuration
    parser.add_argument('--speech-weight', type=float, default=1.0, help='Speech loss weight')
    parser.add_argument('--vision-weight', type=float, default=1.0, help='Vision loss weight')
    parser.add_argument('--fusion-weight', type=float, default=1.0, help='Fusion loss weight')
    parser.add_argument('--consistency-weight', type=float, default=0.1, help='Consistency loss weight')
    parser.add_argument('--use-focal-loss', action='store_true', help='Use focal loss')
    
    # Output configuration
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--save-history', action='store_true', help='Save training history')
    parser.add_argument('--plot-history', action='store_true', help='Plot training history')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


def create_model(args):
    """Create the multimodal emotion recognition model."""
    print("Creating multimodal emotion recognition model...")
    
    # Model configurations
    speech_config = {
        "whisper_model_name": "base",
        "hidden_size": args.speech_hidden_size,
        "dropout": 0.3
    }
    
    vision_config = {
        "input_size": args.vision_input_size,
        "backbone": "mobilenet",
        "use_landmarks": True,
        "dropout": 0.3
    }
    
    fusion_config = {
        "speech_feature_dim": args.speech_hidden_size,
        "vision_feature_dim": 1280,  # MobileNet output size
        "fusion_method": args.fusion_method,
        "dropout": 0.3
    }
    
    # Create model
    model = MultimodalEmotionModel(
        num_emotions=args.num_emotions,
        speech_model_config=speech_config,
        vision_model_config=vision_config,
        fusion_config=fusion_config,
        use_temporal_fusion=args.use_temporal_fusion,
        device=args.device
    )
    
    # Print model summary
    model_summary = model.get_model_summary()
    print(f"Model Summary:")
    for component, params in model_summary.items():
        print(f"  {component}: {params:,} parameters")
    print(f"  Total: {model_summary['total']:,} parameters")
    
    return model


def create_data_loaders(args):
    """Create data loaders for training and validation."""
    print("Creating data loaders...")
    
    # TODO: Implement actual dataset classes
    # For now, create dummy data loaders
    # In practice, you would implement EmotionDataset classes
    
    # Dummy dataset for demonstration
    class DummyDataset:
        def __init__(self, size=1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Generate dummy data
            audio = torch.randn(1, int(args.audio_sample_rate * args.audio_segment_length))
            image = torch.randn(3, args.vision_input_size, args.vision_input_size)
            target = torch.randint(0, args.num_emotions, (1,)).item()
            text = "dummy text"
            
            return {
                'audio': audio,
                'image': image,
                'target': target,
                'text': text
            }
    
    # Create datasets
    train_dataset = DummyDataset(size=1000)
    val_dataset = DummyDataset(size=200)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Created data loaders:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


def save_config(args, save_dir):
    """Save training configuration."""
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {args.device}")
    if args.device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create save directory
    if args.experiment_name:
        save_dir = os.path.join(args.save_dir, args.experiment_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.save_dir, f"experiment_{timestamp}")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Save configuration
    save_config(args, save_dir)
    
    # Create model
    model = create_model(args)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    
    # Create training utilities
    training_utils = TrainingUtils(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        training_utils.load_checkpoint(args.resume)
    
    # Train the model
    print("\nStarting training...")
    print("=" * 60)
    
    history = training_utils.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=save_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    print("\nTraining completed!")
    print("=" * 60)
    
    # Save training history
    if args.save_history:
        history_path = os.path.join(save_dir, 'training_history.json')
        training_utils.save_training_history(history_path)
    
    # Plot training history
    if args.plot_history:
        plot_path = os.path.join(save_dir, 'training_history.png')
        training_utils.plot_training_history(plot_path)
    
    # Optimize model for mobile deployment
    print("\nOptimizing model for mobile deployment...")
    model.optimize_for_mobile()
    
    # Export to Core ML
    coreml_path = os.path.join(save_dir, 'aura_vision_model.mlmodel')
    model.export_to_coreml(coreml_path)
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved to: {save_dir}")
    print(f"Core ML model: {coreml_path}")


if __name__ == "__main__":
    main()

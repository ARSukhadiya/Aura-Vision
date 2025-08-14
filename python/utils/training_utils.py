"""
Training Utilities for Multimodal Emotion Recognition

This module provides training utilities, loss functions, and optimization
strategies for the Aura-Vision multimodal emotion recognition model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in emotion recognition."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultimodalLoss(nn.Module):
    """Combined loss function for multimodal emotion recognition."""
    
    def __init__(
        self,
        speech_weight: float = 1.0,
        vision_weight: float = 1.0,
        fusion_weight: float = 1.0,
        consistency_weight: float = 0.1,
        use_focal: bool = True
    ):
        super().__init__()
        self.speech_weight = speech_weight
        self.vision_weight = vision_weight
        self.fusion_weight = fusion_weight
        self.consistency_weight = consistency_weight
        
        if use_focal:
            self.speech_loss = FocalLoss()
            self.vision_loss = FocalLoss()
            self.fusion_loss = FocalLoss()
        else:
            self.speech_loss = nn.CrossEntropyLoss()
            self.vision_loss = nn.CrossEntropyLoss()
            self.fusion_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        speech_logits: torch.Tensor,
        vision_logits: torch.Tensor,
        fusion_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multimodal loss."""
        # Individual modality losses
        speech_loss = self.speech_loss(speech_logits, targets)
        vision_loss = self.vision_loss(vision_logits, targets)
        fusion_loss = self.fusion_loss(fusion_logits, targets)
        
        # Consistency loss (encourage agreement between modalities)
        speech_probs = torch.softmax(speech_logits, dim=1)
        vision_probs = torch.softmax(vision_logits, dim=1)
        consistency_loss = nn.functional.mse_loss(speech_probs, vision_probs)
        
        # Total loss
        total_loss = (
            self.speech_weight * speech_loss +
            self.vision_weight * vision_loss +
            self.fusion_weight * fusion_loss +
            self.consistency_weight * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'speech_loss': speech_loss,
            'vision_loss': vision_loss,
            'fusion_loss': fusion_loss,
            'consistency_loss': consistency_loss
        }


class TrainingUtils:
    """Training utilities for multimodal emotion recognition."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = MultimodalLoss()
        
        # Training history
        self.train_history = {
            'loss': [],
            'speech_loss': [],
            'vision_loss': [],
            'fusion_loss': [],
            'consistency_loss': [],
            'accuracy': []
        }
        
        self.val_history = {
            'loss': [],
            'speech_loss': [],
            'vision_loss': [],
            'fusion_loss': [],
            'consistency_loss': [],
            'accuracy': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Loss components
        speech_losses = []
        vision_losses = []
        fusion_losses = []
        consistency_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract data
            audio = batch['audio'].to(self.device)
            image = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            text = batch.get('text', None)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(audio, image, text)
            
            # Compute loss
            loss_dict = self.criterion(
                outputs['speech_logits'],
                outputs['vision_logits'],
                outputs['fusion_logits'],
                targets
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute accuracy
            fusion_preds = torch.argmax(outputs['fusion_logits'], dim=1)
            accuracy = (fusion_preds == targets).float().mean().item()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            speech_losses.append(loss_dict['speech_loss'].item())
            vision_losses.append(loss_dict['vision_loss'].item())
            fusion_losses.append(loss_dict['fusion_loss'].item())
            consistency_losses.append(loss_dict['consistency_loss'].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy:.4f}"
            })
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_speech_loss = np.mean(speech_losses)
        avg_vision_loss = np.mean(vision_losses)
        avg_fusion_loss = np.mean(fusion_losses)
        avg_consistency_loss = np.mean(consistency_losses)
        
        # Update history
        self.train_history['loss'].append(avg_loss)
        self.train_history['accuracy'].append(avg_accuracy)
        self.train_history['speech_loss'].append(avg_speech_loss)
        self.train_history['vision_loss'].append(avg_vision_loss)
        self.train_history['fusion_loss'].append(avg_fusion_loss)
        self.train_history['consistency_loss'].append(avg_consistency_loss)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'speech_loss': avg_speech_loss,
            'vision_loss': avg_vision_loss,
            'fusion_loss': avg_fusion_loss,
            'consistency_loss': avg_consistency_loss
        }
    
    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Loss components
        speech_losses = []
        vision_losses = []
        fusion_losses = []
        consistency_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Extract data
                audio = batch['audio'].to(self.device)
                image = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)
                text = batch.get('text', None)
                
                # Forward pass
                outputs = self.model(audio, image, text)
                
                # Compute loss
                loss_dict = self.criterion(
                    outputs['speech_logits'],
                    outputs['vision_logits'],
                    outputs['fusion_logits'],
                    targets
                )
                
                loss = loss_dict['total_loss']
                
                # Compute accuracy
                fusion_preds = torch.argmax(outputs['fusion_logits'], dim=1)
                accuracy = (fusion_preds == targets).float().mean().item()
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                speech_losses.append(loss_dict['speech_loss'].item())
                vision_losses.append(loss_dict['vision_loss'].item())
                fusion_losses.append(loss_dict['fusion_loss'].item())
                consistency_losses.append(loss_dict['consistency_loss'].item())
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_speech_loss = np.mean(speech_losses)
        avg_vision_loss = np.mean(vision_losses)
        avg_fusion_loss = np.mean(fusion_losses)
        avg_consistency_loss = np.mean(consistency_losses)
        
        # Update history
        self.val_history['loss'].append(avg_loss)
        self.val_history['accuracy'].append(avg_accuracy)
        self.val_history['speech_loss'].append(avg_speech_loss)
        self.val_history['vision_loss'].append(avg_vision_loss)
        self.val_history['fusion_loss'].append(avg_fusion_loss)
        self.val_history['consistency_loss'].append(avg_consistency_loss)
        
        # Update learning rate scheduler
        self.scheduler.step(avg_loss)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'speech_loss': avg_speech_loss,
            'vision_loss': avg_vision_loss,
            'fusion_loss': avg_fusion_loss,
            'consistency_loss': avg_consistency_loss
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str = "checkpoints",
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Complete training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        if 'val_history' in checkpoint:
            self.val_history = checkpoint['val_history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss plots
        axes[0, 0].plot(self.train_history['loss'], label='Train')
        axes[0, 0].plot(self.val_history['loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_history['accuracy'], label='Train')
        axes[0, 1].plot(self.val_history['accuracy'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Speech loss
        axes[0, 2].plot(self.train_history['speech_loss'], label='Train')
        axes[0, 2].plot(self.val_history['speech_loss'], label='Validation')
        axes[0, 2].set_title('Speech Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Vision loss
        axes[1, 0].plot(self.train_history['vision_loss'], label='Train')
        axes[1, 0].plot(self.val_history['vision_loss'], label='Validation')
        axes[1, 0].set_title('Vision Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Fusion loss
        axes[1, 1].plot(self.train_history['fusion_loss'], label='Train')
        axes[1, 1].plot(self.val_history['fusion_loss'], label='Validation')
        axes[1, 1].set_title('Fusion Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Consistency loss
        axes[1, 2].plot(self.train_history['consistency_loss'], label='Train')
        axes[1, 2].plot(self.val_history['consistency_loss'], label='Validation')
        axes[1, 2].set_title('Consistency Loss')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def save_training_history(self, save_path: str):
        """Save training history to JSON file."""
        history = {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {save_path}")

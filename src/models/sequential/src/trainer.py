# Sequential Recommendation Model Training and Evaluation
# Specialized trainer for sequential models with proper handling of variable-length sequences
# Includes comprehensive evaluation metrics for next-item prediction and ranking tasks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path
import json
import math
import sys
import os

# Add parent directory to path for W&B imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wandb_config import WandbManager
from wandb_training_integration import WandbTrainingLogger

from .model import (
    SequentialRecommender, AttentionalSequentialRecommender,
    HierarchicalSequentialRecommender, SessionBasedRecommender
)


class SequentialTrainer:
    """
    Specialized trainer for sequential recommendation models with proper sequence handling.
    
    Handles training and evaluation of various sequential architectures including:
    - RNN-based models (LSTM/GRU)
    - Attention-based models (SASRec-style)
    - Hierarchical models (short/long-term)
    - Session-based models
    
    Includes proper handling of variable-length sequences, causal masking for attention,
    and specialized metrics for sequential recommendation evaluation.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5, 
                 wandb_manager: Optional[WandbManager] = None):
        """
        Args:
            model: Sequential recommendation model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            wandb_manager: Optional W&B manager for logging
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # W&B integration
        self.wandb_manager = wandb_manager
        self.training_logger = None
        if wandb_manager:
            self.training_logger = WandbTrainingLogger(wandb_manager, 'sequential')
        
        # Initialize optimizer with Adam (works well for sequential models)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay  # L2 regularization
        )
        
        # Use cross-entropy loss for next-item prediction task
        # ignore_index=0 ensures padding tokens don't contribute to loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_hit_rates = []
        self.learning_rates = []
        
        self.logger = logging.getLogger(__name__)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with proper handling of different model types"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        batch_losses = []
        batch_accuracies = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different model architectures with their specific input requirements
            if 'short_sequence' in batch:  # Hierarchical model needs both short and long sequences
                short_sequences = batch['short_sequence'].to(self.device)
                long_sequences = batch['long_sequence'].to(self.device)
                short_lengths = batch['short_length'].to(self.device)
                long_lengths = batch['long_length'].to(self.device)
                targets = batch['target'].to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(short_sequences, long_sequences, short_lengths, long_lengths)
                loss = self.criterion(logits, targets)
                
            else:  # Standard sequential model
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with model-specific handling
                if isinstance(self.model, AttentionalSequentialRecommender):
                    # Create causal mask for self-attention (prevents looking at future items)
                    seq_len = sequences.size(1)
                    mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))  # Lower triangular mask
                    logits = self.model(sequences, mask)
                    # Extract prediction from last valid position for each sequence
                    batch_size = logits.size(0)
                    final_logits = logits[range(batch_size), lengths - 1]  # Use actual sequence lengths
                    loss = self.criterion(final_logits, targets)
                    
                elif isinstance(self.model, SessionBasedRecommender):
                    logits = self.model(sequences, lengths)
                    loss = self.criterion(logits, targets)
                    
                else:  # SequentialRecommender
                    logits = self.model(sequences, lengths)
                    # Use the last position's prediction for each sequence
                    batch_size = logits.size(0)
                    final_logits = logits[range(batch_size), lengths - 1]
                    loss = self.criterion(final_logits, targets)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping prevents exploding gradients in RNNs
            # Essential for stable training of sequential models
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            batch_losses.append(batch_loss)
            
            # Calculate batch accuracy for logging
            if 'short_sequence' in batch:
                _, predicted = torch.max(logits, 1)
                accuracy = (predicted == targets).float().mean().item()
            else:
                if isinstance(self.model, AttentionalSequentialRecommender):
                    _, predicted = torch.max(final_logits, 1)
                    accuracy = (predicted == targets).float().mean().item()
                else:
                    _, predicted = torch.max(final_logits, 1)
                    accuracy = (predicted == targets).float().mean().item()
            
            batch_accuracies.append(accuracy)
            
            # Log batch metrics to W&B
            if self.training_logger and batch_idx % 50 == 0:  # Log every 50 batches
                additional_metrics = {
                    'accuracy': accuracy,
                    'batch_loss': batch_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                self.training_logger.log_batch(
                    epoch, batch_idx, len(train_loader),
                    batch_loss, targets.size(0), additional_metrics
                )
        
        # Store batch accuracies for epoch-level logging
        self.batch_accuracies = batch_accuracies
        return total_loss / num_batches
    
    def evaluate(self, val_loader: DataLoader, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Evaluate model with comprehensive sequential recommendation metrics
        
        Computes standard metrics for next-item prediction task:
        - Accuracy (top-1 prediction)
        - Hit Rate@K (whether target is in top-K predictions)
        - Loss (cross-entropy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # For hit rate calculation
        hit_counts = {k: 0 for k in k_values}
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if 'short_sequence' in batch:  # Hierarchical model
                    short_sequences = batch['short_sequence'].to(self.device)
                    long_sequences = batch['long_sequence'].to(self.device)
                    short_lengths = batch['short_length'].to(self.device)
                    long_lengths = batch['long_length'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    logits = self.model(short_sequences, long_sequences, short_lengths, long_lengths)
                    loss = self.criterion(logits, targets)
                    predictions = logits
                    
                else:  # Standard sequential model
                    sequences = batch['sequence'].to(self.device)
                    targets = batch['target'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    
                    if isinstance(self.model, AttentionalSequentialRecommender):
                        seq_len = sequences.size(1)
                        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                        logits = self.model(sequences, mask)
                        batch_size = logits.size(0)
                        predictions = logits[range(batch_size), lengths - 1]
                        loss = self.criterion(predictions, targets)
                        
                    elif isinstance(self.model, SessionBasedRecommender):
                        predictions = self.model(sequences, lengths)
                        loss = self.criterion(predictions, targets)
                        
                    else:  # SequentialRecommender
                        logits = self.model(sequences, lengths)
                        batch_size = logits.size(0)
                        predictions = logits[range(batch_size), lengths - 1]
                        loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted_items = torch.max(predictions, 1)
                correct_predictions += (predicted_items == targets).sum().item()
                total_predictions += targets.size(0)
                
                # Calculate hit rates (whether target item appears in top-K predictions)
                _, top_k_items = torch.topk(predictions, max(k_values), dim=1)
                
                for k in k_values:
                    # Check if target item is in top-k predictions for each sample
                    hits = (top_k_items[:, :k] == targets.unsqueeze(1)).any(dim=1).sum().item()
                    hit_counts[k] += hits
                
                total_samples += targets.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        hit_rates = {f'hit_rate@{k}': hit_counts[k] / total_samples for k in k_values}
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            **hit_rates
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, patience: int = 10, save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and model checkpointing.
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Log epoch start to W&B
            if self.training_logger:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_logger.log_epoch_start(epoch, epochs, current_lr)
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            if 'hit_rate@10' in val_metrics:
                self.val_hit_rates.append(val_metrics['hit_rate@10'])
            
            epoch_time = time.time() - start_time
            
            # Log epoch end to W&B
            if self.training_logger:
                train_metrics = {'accuracy': val_metrics['accuracy']}  # We'll use val accuracy as approx
                val_metrics_formatted = {k: v for k, v in val_metrics.items() if k != 'loss'}
                self.training_logger.log_epoch_end(
                    epoch, train_loss, val_metrics['loss'], 
                    train_metrics, val_metrics_formatted
                )
            
            # Log detailed validation metrics to W&B
            if self.wandb_manager:
                # Calculate additional training metrics
                train_accuracy = sum(batch_accuracies) / len(batch_accuracies) if hasattr(self, 'batch_accuracies') else 0
                
                # Get model parameters stats
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                # Calculate gradient norm
                grad_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = math.sqrt(grad_norm)
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                else:
                    memory_allocated = 0
                    memory_cached = 0
                
                detailed_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time_sec': epoch_time,
                    'gradient_norm': grad_norm,
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'gpu_memory_allocated_gb': memory_allocated,
                    'gpu_memory_cached_gb': memory_cached,
                    'batches_per_epoch': len(train_loader),
                    'samples_per_epoch': len(train_loader.dataset),
                    'avg_batch_time_ms': (epoch_time / len(train_loader)) * 1000,
                    'patience_counter': self.patience_counter,
                    'best_val_loss': self.best_val_loss
                }
                
                # Add all validation metrics
                for k, v in val_metrics.items():
                    detailed_metrics[f'val_{k}'] = v
                
                # Add perplexity if using cross-entropy loss
                detailed_metrics['train_perplexity'] = math.exp(min(train_loss, 10))  # Cap to prevent overflow
                detailed_metrics['val_perplexity'] = math.exp(min(val_metrics['loss'], 10))
                
                self.wandb_manager.log_metrics(detailed_metrics, commit=True)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Hit@10: {val_metrics.get('hit_rate@10', 0):.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping and model checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save best model
                if save_dir:
                    self._save_checkpoint(save_path / 'best_model.pt', epoch, val_metrics)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Restored best model weights")
        
        # Save final training history
        if save_dir:
            self._save_training_history(save_path / 'training_history.json')
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_accuracy': self.val_accuracies,
            'val_hit_rates': self.val_hit_rates,
            'learning_rate': self.learning_rates
        }
    
    def _save_checkpoint(self, filepath: Path, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'model_config': {
                'model_class': self.model.__class__.__name__,
                'num_items': getattr(self.model, 'num_items', None),
                'embedding_dim': getattr(self.model, 'embedding_dim', None),
                'hidden_dim': getattr(self.model, 'hidden_dim', None),
            }
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def _save_training_history(self, filepath: Path):
        """Save training history as JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_hit_rates': self.val_hit_rates,
            'best_val_loss': self.best_val_loss
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved training history to {filepath}")


class SequentialEvaluator:
    """
    Comprehensive evaluator for sequential recommendation models with advanced metrics.
    
    Provides detailed evaluation including:
    - Next-item prediction metrics (Hit Rate, NDCG, MRR)
    - Multi-step sequence prediction accuracy
    - User behavior pattern analysis
    
    Used for thorough model assessment beyond basic training metrics.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_next_item_prediction(self, test_loader: DataLoader, 
                                    k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Comprehensive evaluation of next-item prediction with ranking metrics.
        
        Computes multiple ranking-based metrics:
        - Hit Rate@K: Proportion of cases where target is in top-K
        - NDCG@K: Normalized Discounted Cumulative Gain (position-aware)
        - MRR: Mean Reciprocal Rank (harmonic mean of ranks)
        - Accuracy: Simple top-1 prediction accuracy
        """
        metrics = {f'hit_rate@{k}': 0 for k in k_values}
        metrics.update({f'ndcg@{k}': 0 for k in k_values})
        metrics['mrr'] = 0
        metrics['accuracy'] = 0
        
        total_samples = 0
        correct_predictions = 0
        mrr_sum = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if 'short_sequence' in batch:  # Hierarchical model
                    short_sequences = batch['short_sequence'].to(self.device)
                    long_sequences = batch['long_sequence'].to(self.device)
                    short_lengths = batch['short_length'].to(self.device)
                    long_lengths = batch['long_length'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    predictions = self.model(short_sequences, long_sequences, short_lengths, long_lengths)
                    
                else:  # Standard sequential model
                    sequences = batch['sequence'].to(self.device)
                    targets = batch['target'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    
                    if isinstance(self.model, AttentionalSequentialRecommender):
                        seq_len = sequences.size(1)
                        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                        logits = self.model(sequences, mask)
                        batch_size = logits.size(0)
                        predictions = logits[range(batch_size), lengths - 1]
                        
                    elif isinstance(self.model, SessionBasedRecommender):
                        predictions = self.model(sequences, lengths)
                        
                    else:  # SequentialRecommender
                        logits = self.model(sequences, lengths)
                        batch_size = logits.size(0)
                        predictions = logits[range(batch_size), lengths - 1]
                
                # Calculate metrics for this batch
                batch_size = predictions.size(0)
                total_samples += batch_size
                
                # Top-k predictions
                _, top_k_items = torch.topk(predictions, max(k_values), dim=1)
                
                # Accuracy (top-1)
                top_1_predictions = top_k_items[:, 0]
                correct_predictions += (top_1_predictions == targets).sum().item()
                
                # Hit rates and NDCG
                for k in k_values:
                    # Hit rate
                    hits = (top_k_items[:, :k] == targets.unsqueeze(1)).any(dim=1)
                    metrics[f'hit_rate@{k}'] += hits.sum().item()
                    
                    # NDCG (Normalized Discounted Cumulative Gain)
                    # Rewards correct predictions with higher weight for better positions
                    for i in range(batch_size):
                        target_item = targets[i].item()
                        top_k_list = top_k_items[i, :k].cpu().numpy()
                        
                        if target_item in top_k_list:
                            position = np.where(top_k_list == target_item)[0][0]
                            # NDCG formula: 1 / log2(position + 2) - discounts lower positions
                            ndcg_score = 1.0 / math.log2(position + 2)  # +2 because log2(1) = 0
                            metrics[f'ndcg@{k}'] += ndcg_score
                
                # MRR (Mean Reciprocal Rank) - harmonic mean of prediction ranks
                # Higher weight for predictions ranked higher in the list
                for i in range(batch_size):
                    target_item = targets[i].item()
                    top_k_list = top_k_items[i].cpu().numpy()
                    
                    if target_item in top_k_list:
                        position = np.where(top_k_list == target_item)[0][0]
                        mrr_sum += 1.0 / (position + 1)  # Reciprocal of rank (1-indexed)
        
        # Normalize metrics
        for k in k_values:
            metrics[f'hit_rate@{k}'] /= total_samples
            metrics[f'ndcg@{k}'] /= total_samples
        
        metrics['accuracy'] = correct_predictions / total_samples
        metrics['mrr'] = mrr_sum / total_samples
        
        return metrics
    
    def evaluate_sequence_prediction(self, test_loader: DataLoader, 
                                   future_steps: int = 5) -> Dict[str, float]:
        """
        Evaluate multi-step sequence prediction capability.
        
        Tests the model's ability to predict multiple future items in sequence,
        which is useful for understanding how well the model captures longer-term
        sequential patterns beyond just the immediate next item.
        
        Args:
            future_steps: Number of future items to predict in sequence
        """
        total_samples = 0
        step_accuracies = {i: 0 for i in range(1, future_steps + 1)}
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                batch_size = sequences.size(0)
                
                for sample_idx in range(batch_size):
                    seq = sequences[sample_idx]
                    seq_len = lengths[sample_idx].item()
                    
                    if seq_len < future_steps + 2:  # Need enough history
                        continue
                    
                    # Use first part of sequence as input, predict the rest
                    input_len = seq_len - future_steps
                    input_seq = seq[:input_len].unsqueeze(0)
                    
                    # Predict next items step by step (autoregressive generation)
                    current_seq = input_seq.clone()  # Start with initial sequence
                    
                    for step in range(1, future_steps + 1):
                        # Get prediction for next item
                        if isinstance(self.model, AttentionalSequentialRecommender):
                            curr_len = current_seq.size(1)
                            mask = torch.tril(torch.ones(curr_len, curr_len, device=self.device))
                            logits = self.model(current_seq, mask)
                            predictions = logits[0, -1]  # Last position
                        else:
                            curr_lengths = torch.tensor([current_seq.size(1)], device=self.device)
                            logits = self.model(current_seq, curr_lengths)
                            predictions = logits[0, -1]  # Last position
                        
                        # Get predicted item
                        predicted_item = torch.argmax(predictions).item()
                        
                        # Check if prediction matches actual next item
                        actual_item = seq[input_len + step - 1].item()
                        if predicted_item == actual_item:
                            step_accuracies[step] += 1
                        
                        # Add predicted item to sequence for next prediction
                        next_item = torch.tensor([[predicted_item]], device=self.device)
                        current_seq = torch.cat([current_seq, next_item], dim=1)
                    
                    total_samples += 1
        
        # Normalize accuracies
        if total_samples > 0:
            for step in range(1, future_steps + 1):
                step_accuracies[step] /= total_samples
        
        return {f'step_{step}_accuracy': acc for step, acc in step_accuracies.items()}
    
    def analyze_user_behavior(self, test_loader: DataLoader, user_sequences: Dict) -> Dict[str, float]:
        """
        Analyze how well the model captures different user behavior patterns.
        """
        # Categorize users by activity level
        sequence_lengths = [len(seq) for seq in user_sequences.values()]
        length_percentiles = np.percentile(sequence_lengths, [33, 66])
        
        low_activity_users = set()
        medium_activity_users = set()
        high_activity_users = set()
        
        for user_id, seq in user_sequences.items():
            if len(seq) <= length_percentiles[0]:
                low_activity_users.add(user_id)
            elif len(seq) <= length_percentiles[1]:
                medium_activity_users.add(user_id)
            else:
                high_activity_users.add(user_id)
        
        # This is a simplified analysis - in practice, you'd need user IDs in the test set
        # to properly categorize and evaluate different user types
        
        return {
            'low_activity_threshold': length_percentiles[0],
            'high_activity_threshold': length_percentiles[1],
            'num_low_activity': len(low_activity_users),
            'num_medium_activity': len(medium_activity_users),
            'num_high_activity': len(high_activity_users)
        }
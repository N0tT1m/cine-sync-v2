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

from .model import (
    SequentialRecommender, AttentionalSequentialRecommender,
    HierarchicalSequentialRecommender, SessionBasedRecommender
)


class SequentialTrainer:
    """
    Trainer class for sequential recommendation models with specialized evaluation metrics.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Args:
            model: Sequential recommendation model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Use cross-entropy loss for next-item prediction
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_hit_rates = []
        
        self.logger = logging.getLogger(__name__)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            if 'short_sequence' in batch:  # Hierarchical model
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
                
                # Forward pass
                if isinstance(self.model, AttentionalSequentialRecommender):
                    # Create causal mask for self-attention
                    seq_len = sequences.size(1)
                    mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                    logits = self.model(sequences, mask)
                    # Use the last position's prediction
                    batch_size = logits.size(0)
                    final_logits = logits[range(batch_size), lengths - 1]
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
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for RNNs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader: DataLoader, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Evaluate model on validation data"""
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
                
                # Calculate hit rates
                _, top_k_items = torch.topk(predictions, max(k_values), dim=1)
                
                for k in k_values:
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
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            if 'hit_rate@10' in val_metrics:
                self.val_hit_rates.append(val_metrics['hit_rate@10'])
            
            epoch_time = time.time() - start_time
            
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
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_hit_rates': self.val_hit_rates
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
    Comprehensive evaluator for sequential recommendation models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_next_item_prediction(self, test_loader: DataLoader, 
                                    k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate next-item prediction performance.
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
                    
                    # NDCG
                    for i in range(batch_size):
                        target_item = targets[i].item()
                        top_k_list = top_k_items[i, :k].cpu().numpy()
                        
                        if target_item in top_k_list:
                            position = np.where(top_k_list == target_item)[0][0]
                            ndcg_score = 1.0 / math.log2(position + 2)  # +2 because log2(1) = 0
                            metrics[f'ndcg@{k}'] += ndcg_score
                
                # MRR (Mean Reciprocal Rank)
                for i in range(batch_size):
                    target_item = targets[i].item()
                    top_k_list = top_k_items[i].cpu().numpy()
                    
                    if target_item in top_k_list:
                        position = np.where(top_k_list == target_item)[0][0]
                        mrr_sum += 1.0 / (position + 1)
        
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
        Evaluate multi-step sequence prediction accuracy.
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
                    
                    # Use first part of sequence as input
                    input_len = seq_len - future_steps
                    input_seq = seq[:input_len].unsqueeze(0)
                    
                    # Predict next items step by step
                    current_seq = input_seq.clone()
                    
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
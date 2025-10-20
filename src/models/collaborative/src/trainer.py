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

from .model import NeuralCollaborativeFiltering, SimpleNCF, DeepNCF

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class NCFTrainer:
    """
    Trainer class for Neural Collaborative Filtering models with comprehensive evaluation metrics.
    
    This trainer provides a complete training pipeline for NCF models including:
    - Training loop with batch processing and gradient updates
    - Validation monitoring with multiple metrics (RMSE, MAE, R²)
    - Early stopping to prevent overfitting
    - Model checkpointing and state management
    - Comprehensive logging and progress tracking
    
    The trainer supports different NCF variants and automatically handles
    models with or without additional features (e.g., genre embeddings).
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5, checkpoint_dir: Optional[str] = None):
        """
        Initialize NCF trainer with model and training configuration.
        
        Args:
            model: NCF model to train (SimpleNCF, NeuralCollaborativeFiltering, etc.)
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization weight for preventing overfitting
            checkpoint_dir: Directory to auto-load checkpoints from
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize Adam optimizer with L2 regularization
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,          # Learning rate for gradient updates
            weight_decay=weight_decay  # L2 penalty for regularization
        )
        # Use MSE loss for rating prediction (regression task)
        self.criterion = nn.MSELoss()
        
        # Training history for monitoring and plotting
        self.train_losses = []    # Training loss per epoch
        self.val_losses = []      # Validation loss per epoch
        self.val_rmses = []       # Validation RMSE per epoch
        self.val_maes = []        # Validation MAE per epoch
        
        self.logger = logging.getLogger(__name__)
        
        # Best model tracking for early stopping and checkpointing
        self.best_val_loss = float('inf')  # Best validation loss seen
        self.best_model_state = None       # Best model state dict
        self.patience_counter = 0          # Epochs without improvement
        
        # Auto-load most recent checkpoint if available
        if checkpoint_dir:
            self._auto_load_checkpoint(checkpoint_dir)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch
        
        Executes one complete pass through the training data with:
        - Forward pass through the model
        - Loss computation and backpropagation
        - Parameter updates with optimizer
        - Automatic handling of different model architectures
        
        Args:
            train_loader: DataLoader with training batches
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch data to appropriate device (GPU/CPU)
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            ratings = batch['rating'].to(self.device)
            
            # Standard PyTorch training step
            self.optimizer.zero_grad()  # Clear gradients from previous step
            
            # Adaptive forward pass - handle models with/without genre features
            if 'genre_id' in batch and hasattr(self.model, 'genre_embedding'):
                # Enhanced model with content-based features
                genre_ids = batch['genre_id'].to(self.device)
                predictions = self.model(user_ids, item_ids, genre_ids)
            else:
                # Standard collaborative filtering model
                predictions = self.model(user_ids, item_ids)
            
            # Compute loss
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data
        
        Computes comprehensive evaluation metrics without updating model parameters:
        - MSE Loss: Mean squared error for optimization
        - RMSE: Root mean squared error (interpretable rating scale)
        - MAE: Mean absolute error (robust to outliers)
        - R²: Coefficient of determination (explained variance)
        
        Args:
            val_loader: DataLoader with validation batches
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch in val_loader:
                # Move validation data to device
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # Forward pass (same adaptive logic as training)
                if 'genre_id' in batch and hasattr(self.model, 'genre_embedding'):
                    genre_ids = batch['genre_id'].to(self.device)
                    predictions = self.model(user_ids, item_ids, genre_ids)
                else:
                    predictions = self.model(user_ids, item_ids)
                
                # Compute loss
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                predictions_list.extend(predictions.cpu().numpy())
                targets_list.extend(ratings.cpu().numpy())
        
        # Calculate metrics
        predictions_arr = np.array(predictions_list)
        targets_arr = np.array(targets_list)
        
        # Convert normalized ratings back to original scale for interpretable metrics
        # Assumes 0-1 normalized range maps to 0.5-5.0 MovieLens rating scale
        predictions_scaled = predictions_arr * 4.5 + 0.5
        targets_scaled = targets_arr * 4.5 + 0.5
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'rmse': np.sqrt(np.mean((predictions_scaled - targets_scaled) ** 2)),
            'mae': np.mean(np.abs(predictions_scaled - targets_scaled)),
            'r2': self._calculate_r2(targets_scaled, predictions_scaled)
        }
        
        return metrics
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score (coefficient of determination)
        
        R² measures the proportion of variance in the target variable
        that's predictable from the input features. Higher values (closer to 1)
        indicate better model performance.
        
        Args:
            y_true: Ground truth ratings
            y_pred: Predicted ratings
            
        Returns:
            float: R-squared score (can be negative for very poor fits)
        """
        ss_res = np.sum((y_true - y_pred) ** 2)      # Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        return 1 - (ss_res / ss_tot)  # R² = 1 - (SSres / SStot)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, patience: int = 10, save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and model checkpointing.
        
        Implements a complete training loop with:
        - Epoch-wise training and validation
        - Early stopping based on validation loss
        - Best model checkpointing
        - Comprehensive metric tracking
        - Progress logging and timing
        
        Args:
            train_loader: Training data loader with user-item-rating batches
            val_loader: Validation data loader for monitoring
            epochs: Maximum number of epochs to train
            patience: Early stopping patience (epochs without improvement)
            save_dir: Directory to save model checkpoints and training history
        
        Returns:
            Dict[str, List[float]]: Training history with losses and metrics
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
            self.val_rmses.append(val_metrics['rmse'])
            self.val_maes.append(val_metrics['mae'])
            
            epoch_time = time.time() - start_time
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"R²: {val_metrics['r2']:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_r2': val_metrics['r2'],
                    'epoch_time': epoch_time,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping mechanism to prevent overfitting
            if val_metrics['loss'] < self.best_val_loss:
                # New best model found - save state and reset patience
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save checkpoint of best model
                if save_dir:
                    self._save_checkpoint(save_path / 'best_model.pt', epoch, val_metrics)
            else:
                # No improvement - increment patience counter
                self.patience_counter += 1
                
                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model weights for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Restored best model weights")
        
        # Save final training history
        if save_dir:
            self._save_training_history(save_path / 'training_history.json')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_rmses': self.val_rmses,
            'val_maes': self.val_maes
        }
    
    def _save_checkpoint(self, filepath: Path, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint with complete training state
        
        Saves all information needed to resume training or deploy the model:
        - Model state dict (learned parameters)
        - Optimizer state dict (for training resumption)
        - Training metrics and epoch information
        - Model configuration for reconstruction
        
        Args:
            filepath: Path where checkpoint should be saved
            epoch: Current training epoch
            metrics: Current validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),      # Learned parameters
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,                               # Current performance
            'model_config': {                                 # Model architecture info
                'model_class': self.model.__class__.__name__,
                'num_users': getattr(self.model, 'num_users', None),
                'num_items': getattr(self.model, 'num_items', None),
                'embedding_dim': getattr(self.model, 'embedding_dim', None),
            }
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def _save_training_history(self, filepath: Path):
        """Save training history as JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_rmses': self.val_rmses,
            'val_maes': self.val_maes,
            'best_val_loss': self.best_val_loss
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Saved training history to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load model checkpoint and restore training state
        
        Restores the complete training state from a saved checkpoint,
        allowing for training resumption or model deployment.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            Dict: Loaded checkpoint data with metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint
    
    def _auto_load_checkpoint(self, checkpoint_dir: str):
        """Auto-load the most recent checkpoint from directory"""
        import glob
        import os
        
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return
        
        # Find all checkpoint files
        checkpoint_files = list(checkpoint_path.glob("*.pt"))
        if not checkpoint_files:
            return
        
        # Get most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        try:
            self.load_checkpoint(str(latest_checkpoint))
            self.logger.info(f"Auto-loaded checkpoint: {latest_checkpoint}")
        except Exception as e:
            self.logger.warning(f"Failed to auto-load checkpoint {latest_checkpoint}: {e}")


class NCFEvaluator:
    """
    Evaluator for comprehensive testing of NCF models including ranking metrics.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_ranking(self, test_loader: DataLoader, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate ranking metrics including Hit Rate, NDCG, and MRR.
        
        Args:
            test_loader: Test data loader
            k_values: List of k values for top-k metrics
        
        Returns:
            Dictionary with ranking metrics
        """
        user_recommendations = {}
        user_ground_truth = {}
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # Get predictions
                if 'genre_id' in batch and hasattr(self.model, 'genre_embedding'):
                    genre_ids = batch['genre_id'].to(self.device)
                    predictions = self.model(user_ids, item_ids, genre_ids)
                else:
                    predictions = self.model(user_ids, item_ids)
                
                # Group by user
                for user_id, item_id, rating, pred in zip(
                    user_ids.cpu().numpy(), 
                    item_ids.cpu().numpy(),
                    ratings.cpu().numpy(),
                    predictions.cpu().numpy()
                ):
                    if user_id not in user_recommendations:
                        user_recommendations[user_id] = []
                        user_ground_truth[user_id] = []
                    
                    user_recommendations[user_id].append((item_id, pred))
                    if rating > 0.5:  # Positive interaction threshold
                        user_ground_truth[user_id].append(item_id)
        
        # Calculate ranking metrics
        metrics = {}
        for k in k_values:
            hit_rates = []
            ndcgs = []
            mrrs = []
            
            for user_id in user_recommendations:
                # Sort items by predicted rating
                sorted_items = sorted(user_recommendations[user_id], 
                                    key=lambda x: x[1], reverse=True)
                top_k_items = [item_id for item_id, _ in sorted_items[:k]]
                
                # Ground truth positive items
                positive_items = set(user_ground_truth[user_id])
                
                if positive_items:  # Only evaluate users with positive interactions
                    # Hit Rate@k
                    hits = len(set(top_k_items) & positive_items)
                    hit_rate = hits / min(len(positive_items), k)
                    hit_rates.append(hit_rate)
                    
                    # NDCG@k
                    ndcg = self._calculate_ndcg(top_k_items, positive_items, k)
                    ndcgs.append(ndcg)
                    
                    # MRR (Mean Reciprocal Rank)
                    mrr = self._calculate_mrr(top_k_items, positive_items)
                    mrrs.append(mrr)
            
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
            metrics[f'mrr@{k}'] = np.mean(mrrs) if mrrs else 0.0
        
        return metrics
    
    def _calculate_ndcg(self, ranked_items: List[int], positive_items: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        for i, item in enumerate(ranked_items[:k]):
            if item in positive_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Ideal DCG (assuming all positive items are ranked first)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(positive_items), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, ranked_items: List[int], positive_items: set) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, item in enumerate(ranked_items):
            if item in positive_items:
                return 1.0 / (i + 1)
        return 0.0
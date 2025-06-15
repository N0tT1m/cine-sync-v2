# Two-Tower Model Training and Evaluation Framework
# Specialized trainer for dual-encoder architectures with retrieval-focused evaluation
# Includes support for FAISS-based efficient similarity search and ranking metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from pathlib import Path
import json
import faiss  # Facebook AI Similarity Search for efficient vector retrieval
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .model import (
    TwoTowerModel, EnhancedTwoTowerModel, 
    MultiTaskTwoTowerModel, CollaborativeTwoTowerModel
)


class TwoTowerTrainer:
    """
    Specialized trainer for Two-Tower recommendation models with retrieval-focused evaluation.
    
    Handles training of dual-encoder architectures with proper support for:
    - Multiple model variants (basic, enhanced, multi-task, collaborative)
    - Mixed loss functions (MSE for ratings, BCE for clicks)
    - Retrieval-specific metrics (Hit Rate, NDCG, MRR)
    - Multi-task learning with task-specific heads
    - Temperature-scaled similarity learning
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Args:
            model: Two-Tower model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions for different prediction tasks
        self.mse_loss = nn.MSELoss()                  # For rating prediction (regression)
        self.bce_loss = nn.BCEWithLogitsLoss()        # For click prediction (binary classification)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        self.logger = logging.getLogger(__name__)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], predictions: Any) -> torch.Tensor:
        """Compute loss based on model type and predictions"""
        targets = batch['rating'].to(self.device)
        
        if isinstance(predictions, dict):  # Multi-task model
            total_loss = 0.0
            
            # Rating loss
            if 'rating' in predictions:
                rating_loss = self.mse_loss(predictions['rating'].squeeze(), targets)
                total_loss += rating_loss
            
            # Classification losses (e.g., click prediction)
            if 'click' in predictions:
                # Convert ratings to binary labels (rating > 3.0 = click)
                click_labels = (targets > 3.0).float()
                click_loss = self.bce_loss(predictions['click'].squeeze(), click_labels)
                total_loss += click_loss
            
            # Similarity loss (contrastive learning)
            if 'similarity' in predictions:
                # Normalize targets to similarity scale
                sim_targets = (targets - 2.5) / 2.5  # Scale to [-1, 1]
                sim_loss = self.mse_loss(predictions['similarity'], sim_targets)
                total_loss += sim_loss
            
            return total_loss
        
        else:  # Single output model
            if targets.max() <= 1.0:  # Binary classification
                return self.bce_loss(predictions.squeeze(), targets)
            else:  # Rating prediction
                return self.mse_loss(predictions.squeeze(), targets)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device
            if 'user_categorical' in batch:  # Enhanced model
                user_categorical = {k: v.to(self.device) for k, v in batch['user_categorical'].items()}
                user_numerical = batch['user_numerical'].to(self.device)
                item_categorical = {k: v.to(self.device) for k, v in batch['item_categorical'].items()}
                item_numerical = batch['item_numerical'].to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(user_categorical, user_numerical, item_categorical, item_numerical)
                
            elif 'user_id' in batch and hasattr(self.model, 'user_collaborative_embedding'):  # Collaborative model
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids, user_features, item_features)
                
            else:  # Simple model
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(user_features, item_features)
            
            # Compute loss
            loss = self._compute_loss(batch, predictions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0.0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass (similar to training but no gradients)
                if 'user_categorical' in batch:
                    user_categorical = {k: v.to(self.device) for k, v in batch['user_categorical'].items()}
                    user_numerical = batch['user_numerical'].to(self.device)
                    item_categorical = {k: v.to(self.device) for k, v in batch['item_categorical'].items()}
                    item_numerical = batch['item_numerical'].to(self.device)
                    
                    predictions = self.model(user_categorical, user_numerical, item_categorical, item_numerical)
                    
                elif 'user_id' in batch and hasattr(self.model, 'user_collaborative_embedding'):
                    user_ids = batch['user_id'].to(self.device)
                    item_ids = batch['item_id'].to(self.device)
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    
                    predictions = self.model(user_ids, item_ids, user_features, item_features)
                    
                else:
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    
                    predictions = self.model(user_features, item_features)
                
                # Compute loss
                loss = self._compute_loss(batch, predictions)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                targets = batch['rating'].to(self.device)
                
                if isinstance(predictions, dict):
                    # Use rating predictions if available, otherwise similarity
                    if 'rating' in predictions:
                        preds = predictions['rating'].squeeze()
                    else:
                        preds = predictions['similarity']
                else:
                    preds = predictions.squeeze()
                
                predictions_list.extend(preds.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
        
        # Calculate metrics
        predictions_arr = np.array(predictions_list)
        targets_arr = np.array(targets_list)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'rmse': np.sqrt(np.mean((predictions_arr - targets_arr) ** 2)),
            'mae': np.mean(np.abs(predictions_arr - targets_arr))
        }
        
        # Add classification metrics if binary task
        if targets_arr.max() <= 1.0:
            try:
                auc_score = roc_auc_score(targets_arr, predictions_arr)
                metrics['auc'] = auc_score
                
                # Precision-Recall AUC
                precision, recall, _ = precision_recall_curve(targets_arr, predictions_arr)
                pr_auc = auc(recall, precision)
                metrics['pr_auc'] = pr_auc
            except ValueError:
                # Handle case where all targets are same class
                metrics['auc'] = 0.5
                metrics['pr_auc'] = 0.5
        
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
            self.val_metrics.append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Log progress
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"{metrics_str} | "
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
            'val_metrics': self.val_metrics
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
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        self.logger.info(f"Saved training history to {filepath}")


class TwoTowerEvaluator:
    """
    Comprehensive evaluator for Two-Tower models with retrieval and ranking metrics.
    
    Specializes in evaluation metrics relevant to recommendation systems:
    - Retrieval metrics (Hit Rate, NDCG, MRR)
    - Efficient similarity search using FAISS
    - Embedding quality analysis
    - Large-scale evaluation capabilities
    
    This evaluator is designed for production-like evaluation scenarios
    where efficiency and scalability are important.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_retrieval(self, test_loader: DataLoader, k_values: List[int] = [5, 10, 20, 50],
                          build_index: bool = True) -> Dict[str, float]:
        """
        Evaluate retrieval performance using efficient similarity search.
        
        Args:
            test_loader: Test data loader
            k_values: List of k values for top-k metrics
            build_index: Whether to build FAISS index for efficient search
        
        Returns:
            Dictionary with retrieval metrics
        """
        self.model.eval()
        
        # Collect all user and item embeddings
        user_embeddings = []
        item_embeddings = []
        user_ids = []
        item_ids = []
        ratings = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Extract embeddings based on model type
                if 'user_categorical' in batch:
                    user_categorical = {k: v.to(self.device) for k, v in batch['user_categorical'].items()}
                    user_numerical = batch['user_numerical'].to(self.device)
                    item_categorical = {k: v.to(self.device) for k, v in batch['item_categorical'].items()}
                    item_numerical = batch['item_numerical'].to(self.device)
                    
                    user_emb = self.model.encode_users(user_categorical, user_numerical)
                    item_emb = self.model.encode_items(item_categorical, item_numerical)
                    
                elif 'user_id' in batch and hasattr(self.model, 'user_collaborative_embedding'):  # Collaborative model
                    user_id_batch = batch['user_id'].to(self.device)
                    item_id_batch = batch['item_id'].to(self.device)
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    
                    user_emb = self.model.encode_users(user_id_batch, user_features)
                    item_emb = self.model.encode_items(item_id_batch, item_features)
                    
                else:  # Basic model
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    
                    user_emb = self.model.encode_users(user_features)
                    item_emb = self.model.encode_items(item_features)
                
                user_embeddings.append(user_emb.cpu().numpy())
                item_embeddings.append(item_emb.cpu().numpy())
                
                if 'user_id' in batch:
                    user_ids.extend(batch['user_id'].cpu().numpy())
                    item_ids.extend(batch['item_id'].cpu().numpy())
                
                ratings.extend(batch['rating'].cpu().numpy())
        
        # Concatenate all embeddings
        user_embeddings = np.vstack(user_embeddings)
        item_embeddings = np.vstack(item_embeddings)
        ratings = np.array(ratings)
        
        # Build user-item ground truth mapping
        user_item_ratings = {}
        for i, rating in enumerate(ratings):
            if len(user_ids) > 0:
                user_id = user_ids[i]
                item_id = item_ids[i]
            else:
                user_id = i  # Use index as user_id if not available
                item_id = i  # Use index as item_id if not available
            
            if user_id not in user_item_ratings:
                user_item_ratings[user_id] = {}
            user_item_ratings[user_id][item_id] = rating
        
        # Calculate retrieval metrics
        metrics = {}
        
        if build_index and len(item_embeddings) > 1000:
            # Use FAISS for efficient similarity search
            metrics.update(self._evaluate_with_faiss(
                user_embeddings, item_embeddings, user_item_ratings, k_values
            ))
        else:
            # Use direct similarity computation
            metrics.update(self._evaluate_direct_similarity(
                user_embeddings, item_embeddings, user_item_ratings, k_values
            ))
        
        return metrics
    
    def _evaluate_with_faiss(self, user_embeddings: np.ndarray, item_embeddings: np.ndarray,
                           user_item_ratings: Dict, k_values: List[int]) -> Dict[str, float]:
        """Evaluate using FAISS index for efficient search"""
        # Build FAISS index
        embedding_dim = item_embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity for normalized vectors)
        index.add(item_embeddings.astype(np.float32))
        
        metrics = {f'hit_rate@{k}': 0 for k in k_values}
        metrics.update({f'ndcg@{k}': 0 for k in k_values})
        metrics['mrr'] = 0
        
        total_users = len(set(user_item_ratings.keys()))
        
        for user_id in user_item_ratings:
            user_idx = user_id if isinstance(user_id, int) and user_id < len(user_embeddings) else 0
            user_emb = user_embeddings[user_idx:user_idx+1].astype(np.float32)
            
            # Search for top-k items
            max_k = max(k_values)
            similarities, item_indices = index.search(user_emb, max_k)
            
            # Get ground truth positive items (rating > threshold)
            positive_items = set()
            for item_id, rating in user_item_ratings[user_id].items():
                if rating > 3.0:  # Positive threshold
                    positive_items.add(item_id)
            
            if len(positive_items) == 0:
                continue
            
            # Calculate metrics for this user
            for k in k_values:
                top_k_items = set(item_indices[0][:k])
                
                # Hit rate
                hits = len(top_k_items & positive_items)
                metrics[f'hit_rate@{k}'] += hits / min(len(positive_items), k)
                
                # NDCG
                ndcg_score = self._calculate_ndcg(item_indices[0][:k], positive_items, k)
                metrics[f'ndcg@{k}'] += ndcg_score
            
            # MRR
            mrr_score = self._calculate_mrr(item_indices[0], positive_items)
            metrics['mrr'] += mrr_score
        
        # Normalize by number of users
        for key in metrics:
            metrics[key] /= total_users
        
        return metrics
    
    def _evaluate_direct_similarity(self, user_embeddings: np.ndarray, item_embeddings: np.ndarray,
                                  user_item_ratings: Dict, k_values: List[int]) -> Dict[str, float]:
        """Evaluate using direct similarity computation"""
        metrics = {f'hit_rate@{k}': 0 for k in k_values}
        metrics.update({f'ndcg@{k}': 0 for k in k_values})
        metrics['mrr'] = 0
        
        total_users = len(set(user_item_ratings.keys()))
        
        # Compute similarity matrix
        similarity_matrix = np.dot(user_embeddings, item_embeddings.T)
        
        for user_id in user_item_ratings:
            user_idx = user_id if isinstance(user_id, int) and user_id < len(user_embeddings) else 0
            user_similarities = similarity_matrix[user_idx]
            
            # Get top-k items
            max_k = max(k_values)
            top_k_indices = np.argsort(user_similarities)[::-1][:max_k]
            
            # Get ground truth positive items
            positive_items = set()
            for item_id, rating in user_item_ratings[user_id].items():
                if rating > 3.0:
                    positive_items.add(item_id)
            
            if len(positive_items) == 0:
                continue
            
            # Calculate metrics
            for k in k_values:
                top_k_items = set(top_k_indices[:k])
                
                # Hit rate
                hits = len(top_k_items & positive_items)
                metrics[f'hit_rate@{k}'] += hits / min(len(positive_items), k)
                
                # NDCG
                ndcg_score = self._calculate_ndcg(top_k_indices[:k], positive_items, k)
                metrics[f'ndcg@{k}'] += ndcg_score
            
            # MRR
            mrr_score = self._calculate_mrr(top_k_indices, positive_items)
            metrics['mrr'] += mrr_score
        
        # Normalize by number of users
        for key in metrics:
            metrics[key] /= total_users
        
        return metrics
    
    def _calculate_ndcg(self, ranked_items: np.ndarray, positive_items: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        for i, item in enumerate(ranked_items[:k]):
            if item in positive_items:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(positive_items), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, ranked_items: np.ndarray, positive_items: set) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, item in enumerate(ranked_items):
            if item in positive_items:
                return 1.0 / (i + 1)
        return 0.0
    
    def analyze_embeddings(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Analyze learned embeddings for insights.
        
        Returns:
            Dictionary with embedding analysis results
        """
        self.model.eval()
        
        user_embeddings = []
        item_embeddings = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Extract embeddings (similar to retrieval evaluation)
                if 'user_categorical' in batch:
                    user_categorical = {k: v.to(self.device) for k, v in batch['user_categorical'].items()}
                    user_numerical = batch['user_numerical'].to(self.device)
                    item_categorical = {k: v.to(self.device) for k, v in batch['item_categorical'].items()}
                    item_numerical = batch['item_numerical'].to(self.device)
                    
                    user_emb = self.model.encode_users(user_categorical, user_numerical)
                    item_emb = self.model.encode_items(item_categorical, item_numerical)
                    
                else:
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    
                    user_emb = self.model.encode_users(user_features)
                    item_emb = self.model.encode_items(item_features)
                
                user_embeddings.append(user_emb.cpu().numpy())
                item_embeddings.append(item_emb.cpu().numpy())
        
        user_embeddings = np.vstack(user_embeddings)
        item_embeddings = np.vstack(item_embeddings)
        
        # Analyze embedding properties
        analysis = {
            'user_embedding_stats': {
                'mean_norm': np.mean(np.linalg.norm(user_embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(user_embeddings, axis=1)),
                'mean_value': np.mean(user_embeddings),
                'std_value': np.std(user_embeddings)
            },
            'item_embedding_stats': {
                'mean_norm': np.mean(np.linalg.norm(item_embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(item_embeddings, axis=1)),
                'mean_value': np.mean(item_embeddings),
                'std_value': np.std(item_embeddings)
            }
        }
        
        # Compute pairwise similarities
        user_similarities = np.dot(user_embeddings, user_embeddings.T)
        item_similarities = np.dot(item_embeddings, item_embeddings.T)
        
        analysis['user_similarity_stats'] = {
            'mean': np.mean(user_similarities),
            'std': np.std(user_similarities),
            'min': np.min(user_similarities),
            'max': np.max(user_similarities)
        }
        
        analysis['item_similarity_stats'] = {
            'mean': np.mean(item_similarities),
            'std': np.std(item_similarities),
            'min': np.min(item_similarities),
            'max': np.max(item_similarities)
        }
        
        return analysis
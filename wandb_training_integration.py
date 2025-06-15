#!/usr/bin/env python3
"""
Wandb Training Integration for CineSync v2 Models
Enhanced training scripts with comprehensive wandb logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

from wandb_config import WandbManager, init_wandb_for_training

logger = logging.getLogger(__name__)


class WandbTrainingLogger:
    """Enhanced training logger with wandb integration"""
    
    def __init__(self, wandb_manager: WandbManager, model_name: str):
        self.wandb_manager = wandb_manager
        self.model_name = model_name
        self.epoch_start_time = None
        self.batch_times = []
        
    def log_epoch_start(self, epoch: int, total_epochs: int, lr: float = None):
        """Log epoch start information"""
        self.epoch_start_time = time.time()
        self.batch_times = []
        
        log_dict = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'epoch_progress': epoch / total_epochs
        }
        
        if lr is not None:
            log_dict['learning_rate'] = lr
            
        self.wandb_manager.log_metrics(log_dict, commit=False)
        logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def log_batch(self, epoch: int, batch_idx: int, total_batches: int, 
                  loss: float, batch_size: int, additional_metrics: Dict[str, float] = None):
        """Log batch-level metrics"""
        batch_start = time.time()
        
        log_dict = {
            'epoch': epoch,
            'batch': batch_idx,
            'batch_progress': batch_idx / total_batches,
            'train_loss_batch': loss,
            'batch_size': batch_size
        }
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                log_dict[f'train_{key}_batch'] = value
        
        # Add timing info
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times[-10:])  # Last 10 batches
            log_dict['avg_batch_time_sec'] = avg_batch_time
            log_dict['estimated_epoch_time_min'] = (avg_batch_time * total_batches) / 60
        
        self.wandb_manager.log_metrics(log_dict, step=epoch * total_batches + batch_idx, commit=False)
        
        self.batch_times.append(time.time() - batch_start)
    
    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float = None, 
                     train_metrics: Dict[str, float] = None, val_metrics: Dict[str, float] = None):
        """Log epoch summary metrics"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        log_dict = {
            'epoch': epoch,
            'train_loss_epoch': train_loss,
            'epoch_time_sec': epoch_time,
            'epoch_time_min': epoch_time / 60
        }
        
        if val_loss is not None:
            log_dict['val_loss_epoch'] = val_loss
        
        # Add training metrics
        if train_metrics:
            for key, value in train_metrics.items():
                log_dict[f'train_{key}_epoch'] = value
        
        # Add validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                log_dict[f'val_{key}_epoch'] = value
        
        self.wandb_manager.log_metrics(log_dict, step=epoch, commit=True)
        
        logger.info(f"Epoch {epoch} completed - Train Loss: {train_loss:.4f}" + 
                   (f", Val Loss: {val_loss:.4f}" if val_loss else ""))


def train_with_wandb(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    config: Dict[str, Any], model_name: str = "model") -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Enhanced training function with comprehensive wandb logging
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        model_name: Name for the model
        
    Returns:
        Trained model and training history
    """
    
    # Initialize wandb
    wandb_manager = init_wandb_for_training(model_name, config)
    
    try:
        # Log model architecture
        wandb_manager.log_model_architecture(model, model_name)
        
        # Log dataset information
        dataset_info = {
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'batch_size': train_loader.batch_size,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader)
        }
        wandb_manager.log_dataset_info(dataset_info)
        
        # Log hyperparameters
        wandb_manager.log_hyperparameters(config)
        
        # Setup training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training logger
        training_logger = WandbTrainingLogger(wandb_manager, model_name)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': []
        }
        
        # Training loop
        epochs = config.get('epochs', 20)
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = config.get('early_stopping_patience', 10)
        
        for epoch in range(epochs):
            # Log epoch start
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch_start(epoch, epochs, current_lr)
            
            # Training phase
            model.train()
            train_losses = []
            train_rmses = []
            
            for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
                batch_start_time = time.time()
                
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device).float()
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if config.get('gradient_clipping', False):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate metrics
                batch_loss = loss.item()
                batch_rmse = torch.sqrt(loss).item()
                
                train_losses.append(batch_loss)
                train_rmses.append(batch_rmse)
                
                # Log batch metrics
                if batch_idx % config.get('log_interval', 100) == 0:
                    additional_metrics = {
                        'rmse': batch_rmse,
                        'gradient_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
                    }
                    
                    training_logger.log_batch(
                        epoch, batch_idx, len(train_loader), 
                        batch_loss, len(user_ids), additional_metrics
                    )
            
            # Validation phase
            model.eval()
            val_losses = []
            val_rmses = []
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in val_loader:
                    user_ids = user_ids.to(device)
                    item_ids = item_ids.to(device)
                    ratings = ratings.to(device).float()
                    
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    
                    val_losses.append(loss.item())
                    val_rmses.append(torch.sqrt(loss).item())
            
            # Calculate epoch metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_rmse = np.mean(train_rmses)
            val_rmse = np.mean(val_rmses)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            history['learning_rate'].append(current_lr)
            
            # Log epoch summary
            train_metrics = {'rmse': train_rmse}
            val_metrics = {'rmse': val_rmse}
            training_logger.log_epoch_end(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = f"models/best_{model_name}_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, best_model_path)
                
                # Save as wandb artifact
                wandb_manager.save_model_artifact(
                    best_model_path, 
                    model_name,
                    metadata={
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_rmse': val_rmse,
                        'train_loss': train_loss,
                        'train_rmse': train_rmse
                    }
                )
                
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Log final model comparison
        final_metrics = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_rmse': history['train_rmse'][-1],
            'final_val_rmse': history['val_rmse'][-1],
            'best_val_loss': best_val_loss,
            'total_epochs': len(history['train_loss']),
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
        
        wandb_manager.log_metrics({f'final/{k}': v for k, v in final_metrics.items()})
        
        return model, history
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Always finish wandb run
        wandb_manager.finish()


def train_ensemble_with_wandb(models: Dict[str, nn.Module], train_loader: DataLoader, 
                            val_loader: DataLoader, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train multiple models and create ensemble with wandb tracking
    
    Args:
        models: Dictionary of models to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        
    Returns:
        Dictionary containing trained models and ensemble results
    """
    
    # Initialize wandb for ensemble training
    wandb_manager = init_wandb_for_training("ensemble", config)
    
    try:
        ensemble_results = {}
        model_performances = {}
        
        # Train each model individually
        for model_name, model in models.items():
            logger.info(f"Training {model_name} model...")
            
            # Create model-specific config
            model_config = config.copy()
            model_config['model_name'] = model_name
            
            # Train model
            trained_model, history = train_with_wandb(
                model, train_loader, val_loader, model_config, model_name
            )
            
            ensemble_results[model_name] = {
                'model': trained_model,
                'history': history,
                'final_val_loss': history['val_loss'][-1],
                'final_val_rmse': history['val_rmse'][-1]
            }
            
            # Store performance for comparison
            model_performances[model_name] = {
                'val_loss': history['val_loss'][-1],
                'val_rmse': history['val_rmse'][-1],
                'train_loss': history['train_loss'][-1],
                'train_rmse': history['train_rmse'][-1],
                'epochs_trained': len(history['train_loss']),
                'parameters': sum(p.numel() for p in trained_model.parameters())
            }
        
        # Create model comparison table
        wandb_manager.create_model_comparison_table(model_performances)
        
        # Calculate ensemble weights based on validation performance
        val_losses = [result['final_val_loss'] for result in ensemble_results.values()]
        ensemble_weights = calculate_ensemble_weights(val_losses)
        
        # Log ensemble configuration
        ensemble_config = {
            'ensemble_weights': {name: weight for name, weight in zip(models.keys(), ensemble_weights)},
            'ensemble_val_loss': np.average(val_losses, weights=ensemble_weights),
            'best_single_model': min(model_performances.keys(), key=lambda k: model_performances[k]['val_loss'])
        }
        
        wandb_manager.log_metrics({f'ensemble/{k}': v for k, v in ensemble_config.items()})
        
        ensemble_results['ensemble_config'] = ensemble_config
        ensemble_results['model_performances'] = model_performances
        
        return ensemble_results
        
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        raise
    finally:
        wandb_manager.finish()


def calculate_ensemble_weights(val_losses: List[float], method: str = 'inverse_loss') -> List[float]:
    """
    Calculate ensemble weights based on validation performance
    
    Args:
        val_losses: List of validation losses for each model
        method: Method for calculating weights
        
    Returns:
        List of normalized weights
    """
    val_losses = np.array(val_losses)
    
    if method == 'inverse_loss':
        # Weight inversely proportional to loss
        weights = 1.0 / (val_losses + 1e-8)
    elif method == 'softmax':
        # Softmax of negative losses
        weights = np.exp(-val_losses)
    elif method == 'uniform':
        # Equal weights
        weights = np.ones(len(val_losses))
    else:
        raise ValueError(f"Unknown ensemble weight method: {method}")
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    return weights.tolist()


def create_training_sweep(model_type: str, train_function: callable, 
                        sweep_config: Dict[str, Any] = None) -> str:
    """
    Create and run hyperparameter sweep with wandb
    
    Args:
        model_type: Type of model to sweep
        train_function: Training function to use
        sweep_config: Custom sweep configuration
        
    Returns:
        Sweep ID
    """
    import wandb
    from wandb_config import create_sweep_config
    
    # Use provided config or create default
    if sweep_config is None:
        sweep_config = create_sweep_config(model_type)
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=f"cinesync-v2-{model_type}-sweep"
    )
    
    logger.info(f"Created sweep {sweep_id} for {model_type}")
    
    return sweep_id


# Example usage functions for each model type
def train_ncf_with_wandb(config: Dict[str, Any] = None):
    """Train NCF model with wandb integration"""
    from neural_collaborative_filtering.src.model import NeuralCollaborativeFiltering
    from neural_collaborative_filtering.src.data_loader import create_data_loaders
    
    # Default config
    default_config = {
        'embedding_dim': 64,
        'hidden_layers': [128, 64],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 20,
        'early_stopping_patience': 10
    }
    
    if config:
        default_config.update(config)
    
    # Create data loaders
    train_loader, val_loader, num_users, num_items = create_data_loaders(
        batch_size=default_config['batch_size']
    )
    
    # Create model
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=default_config['embedding_dim'],
        hidden_layers=default_config['hidden_layers'],
        dropout=default_config['dropout']
    )
    
    # Train with wandb
    return train_with_wandb(model, train_loader, val_loader, default_config, "ncf")


def train_two_tower_with_wandb(config: Dict[str, Any] = None):
    """Train Two-Tower model with wandb integration"""
    from two_tower_model.src.model import TwoTowerModel
    from two_tower_model.src.data_loader import create_data_loaders
    
    default_config = {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10
    }
    
    if config:
        default_config.update(config)
    
    # Create data loaders
    train_loader, val_loader, num_users, num_items = create_data_loaders(
        batch_size=default_config['batch_size']
    )
    
    # Create model
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=default_config['embedding_dim'],
        hidden_dim=default_config['hidden_dim'],
        num_heads=default_config['num_heads'],
        dropout=default_config['dropout']
    )
    
    # Train with wandb
    return train_with_wandb(model, train_loader, val_loader, default_config, "two_tower")


if __name__ == "__main__":
    # Example: Train NCF model with wandb
    logger.info("Starting NCF training with wandb integration...")
    model, history = train_ncf_with_wandb()
    logger.info("Training completed!")
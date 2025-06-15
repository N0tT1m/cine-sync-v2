#!/usr/bin/env python3
"""
Enhanced Training Script for CineSync v2
Trains models using all available datasets with the enhanced data loader.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import pickle
import logging
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import wandb

# Add current directory to path
sys.path.append('.')
sys.path.append('./hybrid_recommendation')

from enhanced_data_loader import EnhancedDatasetLoader, create_data_loaders
from hybrid_recommendation.models.hybrid_recommender import HybridRecommenderModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """Enhanced trainer that uses all available datasets
    
    This trainer orchestrates the training process for the CineSync recommendation system
    using multiple datasets (MovieLens, Netflix, TMDB, anime, etc.) to create a robust
    hybrid recommendation model.
    
    Key Features:
    - Multi-dataset training with unified preprocessing
    - Mixed precision training for RTX 4090 optimization
    - Early stopping with patience-based convergence
    - WandB integration for experiment tracking
    - Memory-efficient batch processing
    - Automatic model checkpointing
    """
    
    def __init__(self, config):
        """Initialize the enhanced trainer with configuration
        
        Args:
            config: Configuration object containing training hyperparameters,
                   system settings, and experiment tracking options
        """
        self.config = config
        # Setup compute device - prefer CUDA for faster training
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
        # Initialize gradient scaler for mixed precision training (RTX 4090 optimization)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Initialize enhanced data loader that handles all dataset formats
        self.data_loader = EnhancedDatasetLoader()
        
        # Initialize Weights & Biases for experiment tracking if enabled
        if config.use_wandb:
            wandb.init(
                project="cinesync-enhanced",
                config=vars(config),  # Log all training parameters
                tags=["enhanced", "multi-dataset", "hybrid"]
            )
    
    def load_and_prepare_data(self):
        """Load and prepare all datasets for training
        
        This method orchestrates the entire data pipeline:
        1. Loads all available datasets (MovieLens, Netflix, TMDB, anime, etc.)
        2. Creates unified movie and TV show datasets
        3. Preprocesses data with proper encoding and scaling
        4. Creates PyTorch DataLoaders for efficient batch processing
        
        Returns:
            dict: Contains train_loader, test_loader, metadata, and dataset info
        """
        logger.info("Loading all available datasets...")
        
        # Load all available datasets from different sources
        datasets = self.data_loader.load_all_datasets()
        
        # Log dataset loading summary for debugging and monitoring
        logger.info(f"Loaded {len(datasets)} datasets:")
        for name, df in datasets.items():
            logger.info(f"  - {name}: {len(df)} rows")
        
        # Combine heterogeneous datasets into unified movie and TV formats
        unified_movies, unified_tv = self.data_loader.create_unified_dataset(datasets)
        
        # Apply preprocessing: encoding, scaling, train/test split
        training_data = self.data_loader.prepare_training_data(unified_movies, unified_tv)
        
        # Create data loaders
        train_loader, test_loader = create_data_loaders(
            training_data['train_data'], 
            training_data['test_data'],
            batch_size=self.config.batch_size
        )
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'metadata': training_data['metadata'],
            'datasets_info': {name: len(df) for name, df in datasets.items()}
        }
    
    def create_model(self, metadata):
        """Create the hybrid recommendation model
        
        Initializes a neural collaborative filtering model with embeddings for users and items,
        plus additional features for content-based recommendations.
        
        Args:
            metadata (dict): Contains num_users, num_items, and encoders
        
        Returns:
            torch.nn.Module: The initialized hybrid recommendation model
        """
        model = HybridRecommenderModel(
            num_users=metadata['num_users'],
            num_items=metadata['num_items'],
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Count model parameters for memory estimation and debugging
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
        
        if self.config.use_wandb:
            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params
            })
        
        return model
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train the model for one complete epoch
        
        Processes all training batches with:
        - Mixed precision training for speed (if GPU available)
        - Gradient accumulation for large effective batch sizes
        - Regular logging for monitoring progress
        
        Args:
            model: The recommendation model to train
            train_loader: DataLoader with training batches
            optimizer: Adam optimizer for parameter updates
            criterion: Loss function (MSE for rating prediction)
            epoch: Current epoch number for logging
        
        Returns:
            float: Average training loss for the epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            users = batch['user'].to(self.device)
            items = batch['item'].to(self.device)
            ratings = batch['rating'].to(self.device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training if available (RTX 4090 optimization)
            if self.scaler:
                with autocast():  # Automatic mixed precision for faster training
                    predictions = model(users, items).squeeze()
                    loss = criterion(predictions, ratings)
                
                # Scale gradients to prevent underflow in FP16
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard FP32 training for CPUs or older GPUs
                predictions = model(users, items).squeeze()
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 1000 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/epoch": epoch,
                        "train/batch": batch_idx
                    })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, model, test_loader, criterion):
        """Evaluate the model on validation/test data
        
        Computes validation loss and Mean Absolute Error (MAE) without updating
        model parameters. Uses the same mixed precision setup as training.
        
        Args:
            model: The trained model to evaluate
            test_loader: DataLoader with validation/test batches
            criterion: Loss function for computing validation loss
        
        Returns:
            tuple: (average_loss, average_mae) for the validation set
        """
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                users = batch['user'].to(self.device)
                items = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # Use same precision as training for consistency
                if self.scaler:
                    with autocast():  # Mixed precision inference
                        predictions = model(users, items).squeeze()
                        loss = criterion(predictions, ratings)
                else:
                    # Standard precision inference
                    predictions = model(users, items).squeeze()
                    loss = criterion(predictions, ratings)
                
                # Calculate Mean Absolute Error for interpretable metric
                mae = torch.mean(torch.abs(predictions - ratings))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(self):
        """Main training loop with early stopping and model checkpointing
        
        Orchestrates the complete training process:
        1. Data loading and preprocessing
        2. Model initialization and optimizer setup
        3. Training loop with validation and early stopping
        4. Model checkpointing and final evaluation
        
        Returns:
            tuple: (trained_model, metadata) for inference
        """
        logger.info("Starting enhanced training with all datasets...")
        
        # Load and prepare data
        data_info = self.load_and_prepare_data()
        train_loader = data_info['train_loader']
        test_loader = data_info['test_loader']
        metadata = data_info['metadata']
        
        # Log dataset information
        logger.info("Dataset composition:")
        for source, count in data_info['datasets_info'].items():
            logger.info(f"  - {source}: {count:,} records")
        
        if self.config.use_wandb:
            wandb.log({f"datasets/{k}": v for k, v in data_info['datasets_info'].items()})
            wandb.log({
                "data/num_users": metadata['num_users'],
                "data/num_items": metadata['num_items'],
                "data/train_samples": len(train_loader.dataset),
                "data/test_samples": len(test_loader.dataset)
            })
        
        # Create model
        model = self.create_model(metadata)
        
        # Setup optimizer with L2 regularization to prevent overfitting
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        # Use MSE loss for rating prediction (regression task)
        criterion = nn.MSELoss()
        # Learning rate scheduler reduces LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        # Initialize training loop variables for early stopping
        best_val_loss = float('inf')  # Track best validation performance
        best_model_state = None       # Store best model weights
        patience_counter = 0          # Count epochs without improvement
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Evaluation
            val_loss, val_mae = self.evaluate(model, test_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Val MAE: {val_mae:.6f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.8f}")
            
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mae": val_mae,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch_time": epoch_time
                })
            
            # Early stopping mechanism to prevent overfitting
            if val_loss < best_val_loss:
                # New best model found - save it and reset patience
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model checkpoint
                self.save_model(model, metadata, f"enhanced_best_model.pt")
                logger.info(f"  New best model saved (Val Loss: {val_loss:.6f})")
            else:
                # No improvement - increment patience counter
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Memory cleanup to prevent GPU memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache
            gc.collect()  # Python garbage collection
        
        # Load best model for final evaluation
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_val_loss, final_val_mae = self.evaluate(model, test_loader, criterion)
        logger.info(f"Final validation - Loss: {final_val_loss:.6f}, MAE: {final_val_mae:.6f}")
        
        if self.config.use_wandb:
            wandb.log({
                "final/val_loss": final_val_loss,
                "final/val_mae": final_val_mae,
                "final/best_val_loss": best_val_loss
            })
        
        # Save final model
        self.save_model(model, metadata, f"enhanced_final_model.pt")
        
        return model, metadata
    
    def save_model(self, model, metadata, filename):
        """Save the trained model with all necessary metadata for inference
        
        Saves both model weights and all preprocessing information needed
        to make predictions on new data.
        
        Args:
            model: The trained PyTorch model
            metadata: Dictionary with encoders, scalers, and data info
            filename: Name for the saved model file
        """
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        save_path = models_dir / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_users': metadata['num_users'],
                'num_items': metadata['num_items'],
                'embedding_dim': self.config.embedding_dim,
                'hidden_dim': self.config.hidden_dim,
                'dropout_rate': self.config.dropout_rate
            },
            'metadata': metadata,
            'training_config': vars(self.config)
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")

def parse_args():
    """Parse command line arguments for training configuration
    
    Supports all hyperparameters and system settings needed for training:
    - Model architecture parameters (embedding_dim, hidden_dim, etc.)
    - Training parameters (epochs, batch_size, learning_rate, etc.)
    - System parameters (CUDA usage, mixed precision, etc.)
    - Experiment tracking (WandB integration)
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Enhanced CineSync Training")
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Use CUDA if available')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use Weights & Biases logging')
    parser.add_argument('--use_mixed_precision', action='store_true', default=False, help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    return parser.parse_args()

def main():
    """Main training function that orchestrates the entire process
    
    Entry point for the enhanced training script. Handles:
    - Argument parsing and validation
    - Trainer initialization and configuration
    - Training execution with error handling
    - Final results logging and cleanup
    """
    args = parse_args()
    
    logger.info("CineSync v2 Enhanced Training")
    logger.info("=" * 50)
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() and args.use_cuda else 'CPU'}")
    logger.info(f"Embedding dim: {args.embedding_dim}")
    logger.info(f"Hidden dim: {args.hidden_dim}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 50)
    
    # Create trainer
    trainer = EnhancedTrainer(args)
    
    try:
        # Start training
        model, metadata = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final model saved with {metadata['num_users']} users and {metadata['num_items']} items")
        
        # Log final statistics
        logger.info("Dataset sources used:")
        for source in metadata.get('sources', []):
            logger.info(f"  - {source}")
        
        logger.info("Content types:")
        for content_type in metadata.get('content_types', []):
            logger.info(f"  - {content_type}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.use_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
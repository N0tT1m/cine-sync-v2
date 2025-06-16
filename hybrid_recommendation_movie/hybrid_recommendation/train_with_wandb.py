#!/usr/bin/env python3
"""
Enhanced Hybrid Recommender Training with Full Wandb Integration
Production-ready training script with comprehensive monitoring
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import pickle
import logging
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path


class MovieDataset(Dataset):
    """PyTorch Dataset for movie ratings"""
    
    def __init__(self, ratings_df):
        """Initialize dataset with ratings dataframe
        
        Args:
            ratings_df: DataFrame with columns ['user_idx', 'movie_idx', 'rating']
        """
        self.user_ids = torch.tensor(ratings_df['user_idx'].values, dtype=torch.long)
        self.movie_ids = torch.tensor(ratings_df['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


class HybridRecommenderModel(nn.Module):
    """Hybrid recommender model combining collaborative filtering and content features"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128, dropout=0.2):
        super(HybridRecommenderModel, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        """Forward pass"""
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Get biases
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        
        # MLP component
        mlp_input = torch.cat([user_emb, item_emb], dim=-1)
        mlp_output = self.mlp(mlp_input).squeeze()
        
        # Combine components
        output = self.global_bias + user_bias + item_bias + mlp_output
        
        return output

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import wandb utilities
from wandb_config import init_wandb_for_training, WandbManager
from wandb_training_integration import train_with_wandb, WandbTrainingLogger

# Import existing modules
from config import load_config
from models.movie_recommender import MovieHybridRecommender, MovieRecommendationSystem
from utils import DatabaseManager, load_ratings_data, load_movies_data

logger = logging.getLogger(__name__)


def setup_logging():
    """Minimal logging setup - Wandb handles most logging"""
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Hybrid Recommender with Wandb')
    
    # Data arguments
    # Try multiple possible paths for the data files
    possible_ratings_paths = [
        str(Path(__file__).parent.parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'ratings.csv'),
        '../../movies/cinesync/ml-32m/ratings.csv',
        'movies/cinesync/ml-32m/ratings.csv',
        '/Users/timmy/workspace/ai-apps/cine-sync-v2/movies/cinesync/ml-32m/ratings.csv'
    ]
    possible_movies_paths = [
        str(Path(__file__).parent.parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'movies.csv'),
        '../../movies/cinesync/ml-32m/movies.csv', 
        'movies/cinesync/ml-32m/movies.csv',
        '/Users/timmy/workspace/ai-apps/cine-sync-v2/movies/cinesync/ml-32m/movies.csv'
    ]
    
    # Find first existing path
    default_ratings_path = None
    for path in possible_ratings_paths:
        if os.path.exists(path):
            default_ratings_path = path
            break
    
    default_movies_path = None
    for path in possible_movies_paths:
        if os.path.exists(path):
            default_movies_path = path
            break
    
    parser.add_argument('--ratings-path', type=str, default=default_ratings_path,
                       help='Path to ratings CSV file')
    parser.add_argument('--movies-path', type=str, default=default_movies_path,
                       help='Path to movies CSV file')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--gradient-clipping', action='store_true',
                       help='Enable gradient clipping')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='cinesync-v2-hybrid',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', 
                       default=['hybrid', 'baseline', 'production'],
                       help='Wandb tags')
    parser.add_argument('--wandb-notes', type=str, default=None,
                       help='Wandb run notes')
    parser.add_argument('--wandb-offline', action='store_true',
                       help='Run wandb in offline mode')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def load_ratings_chunked(filepath, chunk_size=100000):
    """Load large ratings file in chunks to prevent memory issues"""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def prepare_data(ratings_path, movies_path, test_size=0.2, val_size=0.1):
    """
    Prepare data for training with proper ID mappings and chunked loading
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    # Load data with chunked loading for large files
    file_size = os.path.getsize(ratings_path) / (1024 * 1024)  # Size in MB
    
    if file_size > 500:  # Use chunked loading for files > 500MB
        ratings_df = load_ratings_chunked(ratings_path, chunk_size=50000)
    else:
        ratings_df = pd.read_csv(ratings_path)
        
    movies_df = pd.read_csv(movies_path)
    
    # Create ID mappings to prevent out-of-bounds errors
    unique_user_ids = sorted(ratings_df['userId'].unique())
    unique_movie_ids = sorted(ratings_df['movieId'].unique())
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
    
    # Map to contiguous indices
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_to_idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_to_idx)
    
    # Remove any unmapped entries
    ratings_df = ratings_df.dropna(subset=['user_idx', 'movie_idx'])
    
    # Convert to integers
    ratings_df['user_idx'] = ratings_df['user_idx'].astype(int)
    ratings_df['movie_idx'] = ratings_df['movie_idx'].astype(int)
    
    # Split data
    train_val_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), random_state=42)
    
    # Create datasets
    train_dataset = MovieDataset(train_df)
    val_dataset = MovieDataset(val_df)
    test_dataset = MovieDataset(test_df)
    
    # Create optimized data loaders for Ryzen 3900X (12C/24T)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=512,  # Larger batch for better GPU utilization
        shuffle=True, 
        num_workers=12,  # Use physical cores (3900X has 12 cores)
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Avoid recreation overhead
        prefetch_factor=2  # Pipeline optimization
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1024,  # Larger batch for validation (no gradient computation)
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1024, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Metadata
    metadata = {
        'num_users': len(unique_user_ids),
        'num_movies': len(unique_movie_ids),
        'num_ratings': len(ratings_df),
        'rating_range': (ratings_df['rating'].min(), ratings_df['rating'].max()),
        'sparsity': 1 - (len(ratings_df) / (len(unique_user_ids) * len(unique_movie_ids))),
        'user_id_to_idx': user_id_to_idx,
        'movie_id_to_idx': movie_id_to_idx,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }
    
    return train_loader, val_loader, test_loader, metadata


def create_hybrid_model(metadata, embedding_dim=64, hidden_dim=128, dropout=0.2):
    """Create hybrid recommender model"""
    
    model = HybridRecommenderModel(
        num_users=metadata['num_users'],
        num_items=metadata['num_movies'],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    return model


def train_hybrid_with_wandb(args):
    """Main training function with wandb integration"""
    
    # Prepare training configuration
    config = {
        'model_type': 'hybrid',
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'gradient_clipping': args.gradient_clipping,
        'ratings_path': args.ratings_path,
        'movies_path': args.movies_path
    }
    
    # Initialize wandb with enhanced monitoring
    wandb_manager = init_wandb_for_training(
        'hybrid', 
        config,
        resume=False
    )
    
    # Enable detailed system monitoring
    import wandb
    
    # Add custom metrics for bottleneck detection
    wandb.define_metric("performance/data_loading_time")
    wandb.define_metric("performance/forward_pass_time") 
    wandb.define_metric("performance/backward_pass_time")
    wandb.define_metric("performance/batch_processing_time")
    wandb.define_metric("performance/gpu_memory_allocated")
    wandb.define_metric("performance/gpu_memory_cached")
    
    # Override wandb config if provided
    if args.wandb_project:
        wandb_manager.config.project = args.wandb_project
    if args.wandb_entity:
        wandb_manager.config.entity = args.wandb_entity
    if args.wandb_name:
        wandb_manager.config.name = args.wandb_name
    if args.wandb_tags:
        wandb_manager.config.tags = args.wandb_tags
    if args.wandb_notes:
        wandb_manager.config.notes = args.wandb_notes
    if args.wandb_offline:
        wandb_manager.config.mode = 'offline'
    
    try:
        # Prepare data
        train_loader, val_loader, test_loader, metadata = prepare_data(
            args.ratings_path, args.movies_path
        )
        
        # Log dataset information
        wandb_manager.log_dataset_info(metadata)
        
        # Create model
        model = create_hybrid_model(
            metadata, args.embedding_dim, args.hidden_dim, args.dropout
        )
        
        # Log model architecture and enable model watching
        wandb_manager.log_model_architecture(model, 'hybrid')
        # Don't use wandb.watch to avoid step conflicts with manual logging
        # wandb.watch(model, log="all", log_freq=500)  # Watch gradients and parameters
        
        # Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        model = model.to(device)
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Mixed precision training
        scaler = GradScaler('cuda') if device.type == 'cuda' else None
        
        # Training logger
        training_logger = WandbTrainingLogger(wandb_manager, 'hybrid')
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Log epoch start
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch_start(epoch, args.epochs, current_lr)
            
            # Training phase
            model.train()
            train_loss_sum = 0.0
            train_rmse_sum = 0.0
            batch_count = 0
            
            for batch_idx, (user_ids, movie_ids, ratings) in enumerate(train_loader):
                batch_start_time = time.time()
                
                # Data loading time (time to get to this point)
                data_load_time = time.time() - batch_start_time
                
                user_ids = user_ids.to(device, non_blocking=True)
                movie_ids = movie_ids.to(device, non_blocking=True)
                ratings = ratings.to(device, non_blocking=True).float()
                
                optimizer.zero_grad()
                
                # Forward pass timing
                forward_start = time.time()
                if scaler:
                    with autocast('cuda'):
                        predictions = model(user_ids, movie_ids)
                        loss = criterion(predictions, ratings)
                else:
                    predictions = model(user_ids, movie_ids)
                    loss = criterion(predictions, ratings)
                forward_time = time.time() - forward_start
                
                # Backward pass timing  
                backward_start = time.time()
                if scaler:
                    scaler.scale(loss).backward()
                    
                    if args.gradient_clipping:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    
                    if args.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                backward_time = time.time() - backward_start
                
                # Total batch time
                total_batch_time = time.time() - batch_start_time
                
                # Accumulate metrics without .item() calls (keep on GPU)
                train_loss_sum += loss.detach()
                train_rmse_sum += torch.sqrt(loss.detach())
                batch_count += 1
                
                # Log performance metrics every 100 batches
                if batch_idx % 100 == 0:
                    # GPU memory metrics
                    if device.type == 'cuda':
                        gpu_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                        gpu_cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
                    else:
                        gpu_allocated = gpu_cached = 0
                    
                    # Log performance metrics through training logger to maintain step consistency
                    performance_metrics = {
                        "performance/data_loading_time": data_load_time * 1000,  # ms
                        "performance/forward_pass_time": forward_time * 1000,   # ms  
                        "performance/backward_pass_time": backward_time * 1000, # ms
                        "performance/batch_processing_time": total_batch_time * 1000, # ms
                        "performance/gpu_memory_allocated": gpu_allocated,
                        "performance/gpu_memory_cached": gpu_cached,
                    }
                    
                    # Use training logger for consistent step tracking
                    training_logger.log_batch(
                        epoch, batch_idx, len(train_loader),
                        loss.item(), len(user_ids), performance_metrics
                    )
            
            # Validation phase
            model.eval()
            val_losses = []
            val_rmses = []
            
            with torch.no_grad():
                for user_ids, movie_ids, ratings in val_loader:
                    user_ids = user_ids.to(device)
                    movie_ids = movie_ids.to(device)
                    ratings = ratings.to(device).float()
                    
                    predictions = model(user_ids, movie_ids)
                    loss = criterion(predictions, ratings)
                    
                    val_losses.append(loss.item())
                    val_rmses.append(torch.sqrt(loss).item())
            
            # Calculate epoch metrics (convert to CPU only once)
            train_loss = (train_loss_sum / batch_count).item()
            val_loss = np.mean(val_losses)
            train_rmse = (train_rmse_sum / batch_count).item()
            val_rmse = np.mean(val_rmses)
            
            # Log epoch summary
            train_metrics = {'rmse': train_rmse}
            val_metrics = {'rmse': val_rmse}
            training_logger.log_epoch_end(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Memory cleanup every few epochs
            if epoch % 5 == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = "models/best_hybrid_model.pt"
                os.makedirs("models", exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'metadata': metadata,
                    'config': config
                }, best_model_path)
                
                # Save metadata
                with open("models/hybrid_metadata.pkl", 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Save as wandb artifact
                wandb_manager.save_model_artifact(
                    best_model_path,
                    'hybrid',
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
                if patience_counter >= args.early_stopping_patience:
                    break
        
        # Test evaluation
        model.eval()
        test_losses = []
        test_rmses = []
        
        with torch.no_grad():
            for user_ids, movie_ids, ratings in test_loader:
                user_ids = user_ids.to(device)
                movie_ids = movie_ids.to(device)
                ratings = ratings.to(device).float()
                
                predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                
                test_losses.append(loss.item())
                test_rmses.append(torch.sqrt(loss).item())
        
        test_loss = np.mean(test_losses)
        test_rmse = np.mean(test_rmses)
        
        # Log final metrics
        final_metrics = {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        }
        
        # Use training logger's current step for final metrics to maintain consistency
        wandb_manager.log_metrics({f'final/{k}': v for k, v in final_metrics.items()}, 
                                step=training_logger.get_current_step())
        
        return model, final_metrics
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        # Save checkpoint if model exists
        if 'model' in locals():
            checkpoint_path = f"checkpoint_interrupted_epoch_{epoch if 'epoch' in locals() else 0}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
                'epoch': epoch if 'epoch' in locals() else 0,
                'best_val_loss': best_val_loss if 'best_val_loss' in locals() else float('inf')
            }, checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
        raise
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        # Always finish wandb run, but handle potential errors during cleanup
        try:
            wandb_manager.finish()
        except Exception as cleanup_error:
            print(f"Warning: Error during wandb cleanup: {cleanup_error}")


def main():
    """Main function"""
    setup_logging()
    args = parse_args()
    
    # Check if datasets exist
    if not os.path.exists(args.ratings_path):
        print(f"Error: Ratings file not found: {args.ratings_path}")
        return
    
    if not os.path.exists(args.movies_path):
        print(f"Error: Movies file not found: {args.movies_path}")
        return
    
    # Train model
    model, metrics = train_hybrid_with_wandb(args)


if __name__ == "__main__":
    main()
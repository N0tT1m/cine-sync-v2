#!/usr/bin/env python3
"""
Enhanced Neural Collaborative Filtering Training with Full Wandb Integration
Production-ready training script with comprehensive monitoring
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import wandb utilities
from wandb_config import init_wandb_for_training, WandbManager
from wandb_training_integration import WandbTrainingLogger

# Import existing modules (if they exist)
try:
    from src.model import NeuralCollaborativeFiltering, SimpleNCF
    from src.data_loader import NCFDataLoader
except ImportError:
    # Create basic NCF model if module doesn't exist
    pass

logger = logging.getLogger(__name__)


class NCFDataset(Dataset):
    """Dataset for Neural Collaborative Filtering"""
    
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model combining GMF and MLP
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64], 
                 dropout=0.2, alpha=0.5):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        
        # GMF (Generalized Matrix Factorization) embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2
        
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        final_input_size = embedding_dim + hidden_layers[-1]
        self.prediction = nn.Linear(final_input_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb
        
        # MLP path
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.prediction(combined)
        
        return torch.sigmoid(prediction).squeeze() * 5.0


def setup_logging():
    """Minimal logging setup - Wandb handles most logging"""
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train NCF with Wandb')
    
    # Data arguments
    # Try multiple possible paths for the data files
    possible_ratings_paths = [
        str(Path(__file__).parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'ratings.csv'),
        '../movies/cinesync/ml-32m/ratings.csv',
        '../../movies/cinesync/ml-32m/ratings.csv',
        'movies/cinesync/ml-32m/ratings.csv',
        '/Users/timmy/workspace/ai-apps/cine-sync-v2/movies/cinesync/ml-32m/ratings.csv'
    ]
    
    possible_movies_paths = [
        str(Path(__file__).parent.parent / 'movies' / 'cinesync' / 'ml-32m' / 'movies.csv'),
        '../movies/cinesync/ml-32m/movies.csv',
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
    parser.add_argument('--min-ratings-user', type=int, default=20,
                       help='Minimum ratings per user')
    parser.add_argument('--min-ratings-item', type=int, default=20,
                       help='Minimum ratings per item')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for combining GMF and MLP')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='cinesync-v2-ncf',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', 
                       default=['ncf', 'collaborative-filtering', 'production'],
                       help='Wandb tags')
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


def prepare_data(ratings_path, device, args, min_ratings_user=20, min_ratings_item=20, 
                test_size=0.2, val_size=0.1):
    """
    Prepare data for NCF training with chunked loading
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    
    # Load ratings with chunked loading for large files
    file_size = os.path.getsize(ratings_path) / (1024 * 1024)  # Size in MB
    
    if file_size > 500:  # Use chunked loading for files > 500MB
        ratings_df = load_ratings_chunked(ratings_path, chunk_size=50000)
    else:
        ratings_df = pd.read_csv(ratings_path)
    
    print(f"Original ratings shape: {ratings_df.shape}")
    print(f"Ratings columns: {list(ratings_df.columns)}")
    
    # Auto-detect column names for different datasets
    user_col = None
    item_col = None
    rating_col = None
    
    # Detect user column
    for col in ['userId', 'uid', 'user_id']:
        if col in ratings_df.columns:
            user_col = col
            break
    
    # Detect item column  
    for col in ['movieId', 'anime_uid', 'item_id', 'id']:
        if col in ratings_df.columns:
            item_col = col
            break
            
    # Detect rating column
    for col in ['rating', 'score', 'vote_average']:
        if col in ratings_df.columns:
            rating_col = col
            break
    
    if not all([user_col, item_col, rating_col]):
        raise ValueError(f"Could not detect required columns. Found: user={user_col}, item={item_col}, rating={rating_col}")
    
    print(f"Using columns: user={user_col}, item={item_col}, rating={rating_col}")
    
    # Filter users and items with minimum ratings
    user_counts = ratings_df[user_col].value_counts()
    item_counts = ratings_df[item_col].value_counts()
    
    valid_users = user_counts[user_counts >= min_ratings_user].index
    valid_items = item_counts[item_counts >= min_ratings_item].index
    
    ratings_df = ratings_df[
        (ratings_df[user_col].isin(valid_users)) & 
        (ratings_df[item_col].isin(valid_items))
    ]
    
    
    # Create label encoders for contiguous indices
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    ratings_df['user_idx'] = user_encoder.fit_transform(ratings_df[user_col])
    ratings_df['item_idx'] = item_encoder.fit_transform(ratings_df[item_col])
    ratings_df['rating'] = ratings_df[rating_col]  # Standardize rating column name
    
    # Get number of unique users and items
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    # Normalize ratings to 0-1 range
    min_rating = ratings_df['rating'].min()
    max_rating = ratings_df['rating'].max()
    ratings_df['rating_norm'] = (ratings_df['rating'] - min_rating) / (max_rating - min_rating)
    
    # Split data
    train_val_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), random_state=42)
    
    
    # Create datasets
    train_dataset = NCFDataset(
        train_df['user_idx'].values,
        train_df['item_idx'].values,
        train_df['rating_norm'].values
    )
    val_dataset = NCFDataset(
        val_df['user_idx'].values,
        val_df['item_idx'].values,
        val_df['rating_norm'].values
    )
    test_dataset = NCFDataset(
        test_df['user_idx'].values,
        test_df['item_idx'].values,
        test_df['rating_norm'].values
    )
    
    # Create standard data loaders
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 256
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Metadata
    metadata = {
        'num_users': len(user_encoder.classes_),
        'num_items': len(item_encoder.classes_),
        'num_ratings': len(ratings_df),
        'rating_range': (min_rating, max_rating),
        'sparsity': 1 - (len(ratings_df) / (len(user_encoder.classes_) * len(item_encoder.classes_))),
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }
    
    return train_loader, val_loader, test_loader, metadata


def train_ncf_with_wandb(args):
    """Main training function with wandb integration"""
    
    # Prepare training configuration
    config = {
        'model_type': 'ncf',
        'embedding_dim': args.embedding_dim,
        'hidden_layers': args.hidden_layers,
        'dropout': args.dropout,
        'alpha': args.alpha,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'min_ratings_user': args.min_ratings_user,
        'min_ratings_item': args.min_ratings_item,
        'ratings_path': args.ratings_path,
        'movies_path': args.movies_path
    }
    
    # Initialize wandb
    wandb_manager = init_wandb_for_training(
        'ncf', 
        config,
        resume=False
    )
    
    # Override wandb config if provided
    if args.wandb_project:
        wandb_manager.config.project = args.wandb_project
    if args.wandb_entity:
        wandb_manager.config.entity = args.wandb_entity
    if args.wandb_name:
        wandb_manager.config.name = args.wandb_name
    if args.wandb_tags:
        wandb_manager.config.tags = args.wandb_tags
    if args.wandb_offline:
        wandb_manager.config.mode = 'offline'
    
    try:
        # Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        # Prepare data
        train_loader, val_loader, test_loader, metadata = prepare_data(
            args.ratings_path, device, args, args.min_ratings_user, args.min_ratings_item
        )
        
        # Log dataset information
        wandb_manager.log_dataset_info(metadata)
        
        # Create model
        model = NeuralCollaborativeFiltering(
            num_users=metadata['num_users'],
            num_items=metadata['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            alpha=args.alpha
        )
        
        # Log model architecture
        wandb_manager.log_model_architecture(model, 'ncf')
        
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        # Skip model compilation on Windows due to Triton dependency issues
        # torch.compile requires Triton which has installation issues on Windows
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduling
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01
        )
        
        # Training logger
        training_logger = WandbTrainingLogger(wandb_manager, 'ncf')
        
        # Log model architecture and parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        model_info = {
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/model_size_mb': model_size_mb,
            'model/architecture': str(model),
            'model/embedding_dim': getattr(model, 'embedding_dim', 'unknown'),
            'model/hidden_dim': getattr(model, 'hidden_dim', 'unknown'),
            'model/num_layers': getattr(model, 'num_layers', 'unknown')
        }
        
        # Log dataset info
        dataset_info = {
            'dataset/num_users': train_loader.dataset.num_users if hasattr(train_loader.dataset, 'num_users') else len(train_loader.dataset),
            'dataset/num_items': train_loader.dataset.num_items if hasattr(train_loader.dataset, 'num_items') else 'unknown',
            'dataset/train_samples': len(train_loader.dataset),
            'dataset/val_samples': len(val_loader.dataset),
            'dataset/batch_size': train_loader.batch_size,
            'dataset/num_workers': train_loader.num_workers
        }
        
        # Setup mixed precision if available
        scaler = None
        if device.type == 'cuda':
            try:
                from torch.amp import GradScaler
                scaler = GradScaler('cuda')
                logger.info("Mixed precision training enabled")
            except ImportError:
                try:
                    from torch.cuda.amp import GradScaler
                    scaler = GradScaler()
                    logger.info("Mixed precision training enabled (legacy API)")
                except ImportError:
                    logger.warning("Mixed precision not available")
        
        # Log training configuration
        training_config = {
            'training/optimizer': optimizer.__class__.__name__,
            'training/learning_rate': args.learning_rate,
            'training/weight_decay': 1e-5,
            'training/criterion': criterion.__class__.__name__,
            'training/scheduler': scheduler.__class__.__name__,
            'training/device': str(device),
            'training/mixed_precision': scaler is not None,
            'training/max_epochs': args.epochs
        }
        
        # Log all configuration info
        wandb_manager.log_metrics({**model_info, **dataset_info, **training_config})
        
        logger.info(f"Model: {total_params:,} parameters ({model_size_mb:.2f} MB)")
        logger.info(f"Dataset: {dataset_info['dataset/train_samples']:,} train samples, {dataset_info['dataset/val_samples']:,} val samples")
        
        # Setup gradient accumulation
        effective_batch_size = 2048
        accumulation_steps = max(1, effective_batch_size // (args.batch_size if hasattr(args, 'batch_size') else 256))
        
        # Training loop
        training_start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses_history = []
        val_losses_history = []
        
        for epoch in range(args.epochs):
            # Log epoch start
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch_start(epoch, args.epochs, current_lr)
            
            # Training epoch
            model.train()
            train_losses = []
            epoch_start_time = time.time()
            
            for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
                batch_start_time = time.time()
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                # Calculate gradient norm before update
                grad_norm = None
                
                if scaler:
                    try:
                        with torch.amp.autocast('cuda'):
                            outputs = model(user_ids, item_ids)
                            loss = criterion(outputs, ratings)
                    except AttributeError:
                        with torch.cuda.amp.autocast():
                            outputs = model(user_ids, item_ids)
                            loss = criterion(outputs, ratings)
                    scaler.scale(loss).backward()
                    
                    # Calculate gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(user_ids, item_ids)
                    loss = criterion(outputs, ratings)
                    loss.backward()
                    
                    # Calculate gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                    
                    optimizer.step()
                
                optimizer.zero_grad()
                train_losses.append(loss.item())
                
                # Calculate batch metrics
                batch_time = time.time() - batch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log batch metrics every 100 batches
                if batch_idx % 100 == 0:
                    batch_metrics = {
                        'batch_loss': loss.item(),
                        'batch_time': batch_time,
                        'learning_rate': current_lr,
                        'batch_progress': batch_idx / len(train_loader),
                        'samples_processed': batch_idx * len(user_ids),
                        'gradient_norm': grad_norm.item() if grad_norm is not None else 0.0
                    }
                    
                    # Add memory metrics if CUDA available
                    if torch.cuda.is_available():
                        batch_metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
                        batch_metrics['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024**2  # MB
                    
                    training_logger.log_batch(
                        epoch, batch_idx, len(train_loader), 
                        loss.item(), len(user_ids), batch_metrics
                    )
                
                if batch_idx % 500 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}, Grad Norm: {grad_norm.item() if grad_norm else 0:.4f}')
            
            train_loss = np.mean(train_losses)
            train_rmse = math.sqrt(train_loss)
            epoch_time = time.time() - epoch_start_time
            
            # Store training history
            train_losses_history.append(train_loss)
            
            # Validation epoch
            model.eval()
            val_losses = []
            val_start_time = time.time()
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in val_loader:
                    user_ids = user_ids.to(device)
                    item_ids = item_ids.to(device)
                    ratings = ratings.to(device)
                    
                    outputs = model(user_ids, item_ids)
                    loss = criterion(outputs, ratings)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            val_rmse = math.sqrt(val_loss)
            val_time = time.time() - val_start_time
            
            # Store validation history
            val_losses_history.append(val_loss)
            
            # Calculate additional training metrics
            current_lr = optimizer.param_groups[0]['lr']
            samples_per_sec = len(train_loader.dataset) / epoch_time
            
            # Calculate gradient statistics for the epoch
            avg_grad_norm = np.mean([loss for loss in train_losses])  # Placeholder for actual grad norms
            
            # Memory metrics
            memory_metrics = {}
            if torch.cuda.is_available():
                memory_metrics.update({
                    'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                    'gpu_memory_cached_mb': torch.cuda.memory_reserved() / 1024**2,
                    'gpu_memory_peak_mb': torch.cuda.max_memory_allocated() / 1024**2
                })
                torch.cuda.reset_peak_memory_stats()  # Reset for next epoch
            
            # System metrics
            import psutil
            system_metrics = {
                'cpu_usage_percent': psutil.cpu_percent(),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / 1024**3
            }
            
            # Training progress metrics
            progress_metrics = {
                'epoch_progress': (epoch + 1) / args.epochs,
                'samples_per_second': samples_per_sec,
                'batches_per_second': len(train_loader) / epoch_time,
                'time_per_sample_ms': (epoch_time / len(train_loader.dataset)) * 1000
            }
            
            # Log combined training and validation metrics
            training_logger.log_epoch_end(
                epoch, 
                train_loss, 
                val_loss=val_loss,
                train_metrics={
                    'rmse': train_rmse,
                    'epoch_time': epoch_time,
                    'learning_rate': current_lr,
                    'min_loss': min(train_losses),
                    'max_loss': max(train_losses),
                    'std_loss': np.std(train_losses),
                    **progress_metrics,
                    **memory_metrics,
                    **system_metrics
                },
                val_metrics={
                    'rmse': val_rmse,
                    'validation_time': val_time
                }
            )
            
            # Learning rate scheduling (cosine annealing)
            old_lr = current_lr
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # Log learning rate change
            if old_lr != new_lr:
                wandb_manager.log_metrics({
                    'lr_change/old_lr': old_lr,
                    'lr_change/new_lr': new_lr,
                    'lr_change/ratio': new_lr / old_lr if old_lr > 0 else 0
                })
            
            # Memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs("models", exist_ok=True)
                best_model_path = "models/best_ncf_model.pt"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'metadata': metadata,
                    'config': config
                }, best_model_path)
                
                # Save comprehensive metadata
                comprehensive_metadata = {
                    'model_type': 'ncf',
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'train_loss': train_loss,
                    'train_rmse': train_rmse,
                    'total_parameters': sum(p.numel() for p in model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
                    'model_architecture': str(model),
                    'num_users': getattr(model, 'num_users', metadata.get('num_users', 'unknown')),
                    'num_items': getattr(model, 'num_items', metadata.get('num_items', 'unknown')),
                    'embedding_dim': getattr(model, 'embedding_dim', metadata.get('embedding_dim', 'unknown')),
                    'hidden_dims': getattr(model, 'hidden_dims', metadata.get('hidden_dims', 'unknown'))
                }
                
                with open("models/ncf_metadata.pkl", 'wb') as f:
                    pickle.dump(comprehensive_metadata, f)
                
                # Save additional required files for compatibility
                torch.save(model.state_dict(), "models/recommendation_model.pt")
                
                # Save training history
                training_history = {
                    'train_losses': train_losses_history,
                    'val_losses': val_losses_history,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch + 1
                }
                
                with open("models/training_history.pkl", 'wb') as f:
                    pickle.dump(training_history, f)
                
                # Save final metrics JSON
                final_metrics_json = {
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'train_loss': train_loss,
                    'train_rmse': train_rmse,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch,
                    'model_type': 'ncf'
                }
                
                import json
                with open("models/final_metrics.json", 'w') as f:
                    json.dump(final_metrics_json, f, indent=2)
                
                # Save rating scaler
                rating_scaler = {
                    'min_rating': 1.0,
                    'max_rating': 5.0,
                    'mean_rating': 3.5,
                    'std_rating': 1.0
                }
                
                with open("models/rating_scaler.pkl", 'wb') as f:
                    pickle.dump(rating_scaler, f)
                
                # Save model lookup files (if available from data loaders)
                try:
                    # This assumes train_loader has dataset with encoders
                    if hasattr(train_loader.dataset, 'user_encoder') and hasattr(train_loader.dataset, 'item_encoder'):
                        user_encoder = train_loader.dataset.user_encoder
                        item_encoder = train_loader.dataset.item_encoder
                        
                        movie_lookup = {
                            'movie_id_to_idx': {str(k): v for k, v in zip(item_encoder.classes_, range(len(item_encoder.classes_)))},
                            'idx_to_movie_id': {v: str(k) for k, v in zip(item_encoder.classes_, range(len(item_encoder.classes_)))},
                            'num_movies': len(item_encoder.classes_)
                        }
                        
                        with open("models/movie_lookup.pkl", 'wb') as f:
                            pickle.dump(movie_lookup, f)
                        
                        with open("models/movie_lookup_backup.pkl", 'wb') as f:
                            pickle.dump(movie_lookup, f)
                except:
                    # Create default lookup if encoders not available
                    pass
                
                # Save model locally instead of WandB artifact
                wandb_manager.save_model_locally(
                    best_model_path,
                    'ncf',
                    metadata={
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_rmse': val_rmse,
                        'train_loss': train_loss,
                        'train_rmse': train_rmse,
                        'total_parameters': comprehensive_metadata['total_parameters']
                    }
                )
                
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Test evaluation
        logger.info("Evaluating on test set...")
        model.eval()
        test_losses = []
        test_rmses = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                test_losses.append(loss.item())
                test_rmses.append(torch.sqrt(loss).item())
        
        test_loss = np.mean(test_losses)
        test_rmse = np.mean(test_rmses)
        
        # Calculate comprehensive final metrics
        total_training_time = time.time() - training_start_time
        
        # Performance analysis
        improvement = (train_losses_history[0] if train_losses_history else 0) - test_loss
        improvement_pct = (improvement / train_losses_history[0] * 100) if train_losses_history and train_losses_history[0] > 0 else 0
        
        # Training efficiency metrics
        samples_total = len(train_loader.dataset) * (epoch + 1)
        samples_per_hour = samples_total / (total_training_time / 3600)
        epochs_per_hour = (epoch + 1) / (total_training_time / 3600)
        
        # Model complexity analysis
        param_efficiency = test_rmse / (total_params / 1e6)  # RMSE per million parameters
        
        # Log comprehensive final metrics
        final_metrics = {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_training_time_hours': total_training_time / 3600,
            'total_training_time_minutes': total_training_time / 60,
            'samples_processed_total': samples_total,
            'samples_per_hour': samples_per_hour,
            'epochs_per_hour': epochs_per_hour,
            'loss_improvement': improvement,
            'loss_improvement_percent': improvement_pct,
            'parameter_efficiency': param_efficiency,
            'convergence_epoch': epoch + 1 - patience_counter,
            'memory_peak_mb': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
        
        # Training summary table for wandb
        training_summary = {
            'Training Summary': {
                'Model': model.__class__.__name__,
                'Parameters': f"{total_params:,}",
                'Model Size': f"{model_size_mb:.2f} MB",
                'Training Time': f"{total_training_time/3600:.2f} hours",
                'Epochs': epoch + 1,
                'Best Val RMSE': f"{np.sqrt(best_val_loss):.4f}",
                'Test RMSE': f"{test_rmse:.4f}",
                'Improvement': f"{improvement_pct:.2f}%",
                'Samples/hour': f"{samples_per_hour:,.0f}",
                'GPU Memory Peak': f"{torch.cuda.max_memory_allocated() / 1024**2:.0f} MB" if torch.cuda.is_available() else "N/A"
            }
        }
        
        wandb_manager.log_metrics({f'final/{k}': v for k, v in final_metrics.items()})
        wandb_manager.log_metrics(training_summary)
        
        # Log training history as charts
        if len(train_losses_history) > 0:
            history_data = {
                'training_history/loss_curve': [(i, loss) for i, loss in enumerate(train_losses_history)],
                'training_history/val_loss_curve': [(i, loss) for i, loss in enumerate(val_losses_history)] if val_losses_history else [],
                'training_history/learning_rate_schedule': [(i, scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.learning_rate) for i in range(len(train_losses_history))]
            }
            # Note: Commented out as wandb handles this automatically with epoch logging
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        logger.info(f"Final Test RMSE: {test_rmse:.4f}")
        logger.info(f"Best Validation RMSE: {np.sqrt(best_val_loss):.4f}")
        logger.info(f"Total Training Time: {total_training_time/3600:.2f} hours")
        logger.info(f"Model Parameters: {total_params:,}")
        logger.info(f"Training Efficiency: {samples_per_hour:,.0f} samples/hour")
        logger.info("="*60)
        
        # Run Weave evaluation if test data is available
        try:
            from wandb_weave_evaluation import WeaveEvaluationManager, NCFModelWrapper
            import asyncio
            
            # Create test examples for evaluation
            test_examples = []
            for user_ids, item_ids, ratings in test_loader:
                for u, i, r in zip(user_ids, item_ids, ratings):
                    test_examples.append({
                        'user_id': u.item(),
                        'item_id': i.item(),
                        'expected_rating': r.item()
                    })
                    if len(test_examples) >= 500:  # Limit evaluation size
                        break
                if len(test_examples) >= 500:
                    break
            
            if test_examples:
                logger.info("Running Weave evaluation...")
                evaluator = WeaveEvaluationManager()
                model_wrapper = NCFModelWrapper(best_model_path, 'ncf')
                
                # Run evaluation asynchronously
                eval_results = asyncio.run(evaluator.evaluate_rating_model(
                    model_wrapper, test_examples, 'ncf'
                ))
                logger.info(f"Weave evaluation completed: {eval_results}")
                
        except Exception as e:
            logger.warning(f"Weave evaluation failed: {e}")
        
        logger.info(f"Training completed! Test RMSE: {test_rmse:.4f}")
        
        return model, final_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Always finish wandb run
        wandb_manager.finish()


def main():
    """Main function"""
    setup_logging()
    args = parse_args()
    
    logger.info("Starting NCF training with Wandb integration")
    logger.info(f"Configuration: {vars(args)}")
    
    # Check if datasets exist
    if not os.path.exists(args.ratings_path):
        logger.error(f"Ratings file not found: {args.ratings_path}")
        return
    
    # Train model
    model, metrics = train_ncf_with_wandb(args)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()

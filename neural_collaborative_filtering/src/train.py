#!/usr/bin/env python3
"""
Enhanced Training script for Neural Collaborative Filtering models with Full Wandb Integration
Production-ready training script with comprehensive monitoring
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from pathlib import Path
import sys
import os
import time
import math
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import wandb utilities
from wandb_config import init_wandb_for_training, WandbManager
from wandb_training_integration import WandbTrainingLogger

from src.model import NeuralCollaborativeFiltering, SimpleNCF, DeepNCF
from src.data_loader import NCFDataLoader
from src.trainer import NCFTrainer, NCFEvaluator
from memory_config import MemoryOptimizer, apply_memory_optimizations
from performance_config import PerformanceOptimizer, apply_rtx4090_optimizations


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for NCF training
    
    Creates loggers that output to both console and file for comprehensive
    monitoring of the neural collaborative filtering training process.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ncf_training.log')
        ]
    )


def parse_args():
    """Parse command line arguments for NCF training
    
    Comprehensive argument parsing for Neural Collaborative Filtering training
    including data preprocessing options, model architecture parameters,
    training hyperparameters, and experiment tracking configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train Neural Collaborative Filtering models')
    
    # Data preprocessing arguments
    parser.add_argument('--ratings-path', type=str, default='../../movies/cinesync/ml-32m/ratings.csv',
                       help='Path to ratings CSV file')
    parser.add_argument('--movies-path', type=str, default='../../movies/cinesync/ml-32m/movies.csv',
                       help='Path to movies CSV file')
    parser.add_argument('--min-ratings-user', type=int, default=20,
                       help='Minimum ratings per user (for data sparsity filtering)')
    parser.add_argument('--min-ratings-item', type=int, default=20,
                       help='Minimum ratings per item (for cold start mitigation)')
    
    # Model architecture arguments
    parser.add_argument('--model-type', type=str, default='ncf', 
                       choices=['ncf', 'simple', 'deep'],
                       help='Type of NCF model: ncf=full NCF, simple=basic CF, deep=enhanced')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension for user/item representations')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer sizes for MLP component (decreasing order)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for regularization (0.0-1.0)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16384,
                       help='Training batch size (optimized for RTX 4090 24GB)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for Adam optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='L2 regularization weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=24,
                       help='Number of data loader workers (maximized for Ryzen 9 3900X 24 threads)')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save trained models')
    
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='cinesync-v2-ncf-basic',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity')
    parser.add_argument('--wandb-name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', 
                       default=['ncf', 'basic', 'collaborative-filtering'],
                       help='Wandb tags')
    parser.add_argument('--wandb-offline', action='store_true',
                       help='Run wandb in offline mode')
    
    # Evaluation configuration
    parser.add_argument('--evaluate-ranking', action='store_true',
                       help='Evaluate ranking metrics (NDCG, Hit Rate, MRR) on test set')
    parser.add_argument('--top-k', type=int, nargs='+', default=[5, 10, 20],
                       help='Top-k values for ranking evaluation metrics')
    
    return parser.parse_args()


def create_model(model_type: str, config: dict, args) -> torch.nn.Module:
    """Create NCF model based on type and configuration
    
    Factory function that creates different variants of Neural Collaborative
    Filtering models based on the specified type and configuration.
    
    Args:
        model_type (str): Type of model ('ncf', 'simple', 'deep')
        config (dict): Model configuration with num_users, num_items, etc.
        args: Command line arguments with hyperparameters
        
    Returns:
        torch.nn.Module: Initialized NCF model
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'ncf':
        # Full Neural Collaborative Filtering model (GMF + MLP)
        model = NeuralCollaborativeFiltering(
            num_users=config['num_users'],     # Number of unique users
            num_items=config['num_items'],     # Number of unique items
            embedding_dim=args.embedding_dim,  # Embedding vector size
            hidden_layers=args.hidden_layers,  # MLP architecture
            dropout=args.dropout               # Regularization rate
        )
    elif model_type == 'simple':
        # Simplified NCF model for faster training and inference
        model = SimpleNCF(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_layers[0] if args.hidden_layers else 64,
            dropout=args.dropout
        )
    elif model_type == 'deep':
        # Enhanced NCF model with additional features (e.g., genres)
        model = DeepNCF(
            num_users=config['num_users'],
            num_items=config['num_items'],
            num_genres=config.get('num_genres', 20),  # Content-based features
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Prepare training configuration
    config = {
        'model_type': args.model_type,
        'embedding_dim': args.embedding_dim,
        'hidden_layers': args.hidden_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'min_ratings_user': args.min_ratings_user,
        'min_ratings_item': args.min_ratings_item,
        'ratings_path': args.ratings_path
    }
    
    # Initialize wandb
    wandb_manager = init_wandb_for_training(
        'ncf_basic', 
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
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Apply MAXIMUM performance optimizations for RTX 4090 + Ryzen 9 3900X
    apply_rtx4090_optimizations()
    MemoryOptimizer.setup_performance_training(device)
    
    # Get RTX 4090 beast mode config
    performance_config = PerformanceOptimizer.get_rtx4090_config()
    
    # Override batch size and num_workers if using defaults
    if args.batch_size == 16384:  # Default value
        args.batch_size = performance_config['batch_size']
        logger.info(f"Using RTX 4090 optimized batch size: {args.batch_size}")
    
    if args.num_workers == 24:  # Default value  
        args.num_workers = performance_config['num_workers']
        logger.info(f"Using Ryzen 9 3900X optimized workers: {args.num_workers}")
    
    try:
        # Load and preprocess MovieLens data for NCF training
        logger.info("Loading and preprocessing data...")
        data_loader = NCFDataLoader(
            ratings_path=args.ratings_path,                    # User-item ratings
            movies_path=args.movies_path,                      # Movie metadata
            min_ratings_per_user=args.min_ratings_user,       # Sparsity filtering
            min_ratings_per_item=args.min_ratings_item        # Cold start filtering
        )
        
        # Get data loaders
        train_loader, val_loader, test_loader = data_loader.get_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Get model configuration
        model_config = data_loader.get_model_config()
        logger.info(f"Model config: {model_config}")
        
        # Log dataset information
        wandb_manager.log_dataset_info({
            'num_users': model_config['num_users'],
            'num_items': model_config['num_items'],
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'test_size': len(test_loader.dataset),
            'sparsity': 1 - (len(train_loader.dataset) / (model_config['num_users'] * model_config['num_items']))
        })
        
        # Create model
        model = create_model(args.model_type, model_config, args)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Log model architecture
        wandb_manager.log_model_architecture(model, 'ncf_basic')
        
        model = model.to(device)
        
        # Apply RTX 4090 model optimizations for maximum speed
        model = PerformanceOptimizer.optimize_model_for_speed(model)
        
        # Find optimal batch size for RTX 4090
        if device.type == 'cuda':
            optimal_batch = PerformanceOptimizer.get_optimal_batch_size_rtx4090(model, device)
            if optimal_batch > args.batch_size:
                logger.info(f"🚀 Upgrading batch size from {args.batch_size} to {optimal_batch} for RTX 4090")
                args.batch_size = optimal_batch
        
        # GPU memory optimization
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Optimize CUDA kernels
            torch.cuda.empty_cache()               # Clear cache
        
        # Setup training components manually for full control
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Mixed precision training for maximum speed
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
        # Enable maximum performance settings for RTX 4090
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Optimize for speed
            torch.backends.cudnn.enabled = True
            torch.cuda.empty_cache()
            # Use tensor cores and optimize memory access
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Training logger
        training_logger = WandbTrainingLogger(wandb_manager, 'ncf_basic')
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        model_info = {
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/model_size_mb': model_size_mb,
            'model/architecture': str(model)
        }
        wandb_manager.log_metrics(model_info)
        
        # Training loop with comprehensive logging
        best_val_loss = float('inf')
        patience_counter = 0
        training_start_time = time.time()
        
        for epoch in range(args.epochs):
            # Log epoch start
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch_start(epoch, args.epochs, current_lr)
            
            # Training phase
            model.train()
            train_losses = []
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                batch_start_time = time.time()
                
                # Move data to device
                if isinstance(batch, (list, tuple)):
                    user_ids, item_ids, ratings = batch
                    user_ids = user_ids.to(device)
                    item_ids = item_ids.to(device)
                    ratings = ratings.to(device).float()
                else:
                    # Handle different batch formats
                    batch = batch.to(device)
                    continue
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        predictions = model(user_ids, item_ids)
                        loss = criterion(predictions, ratings)
                    
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping with scaler
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision fallback
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_losses.append(loss.item())
                
                # Log batch metrics every 1000 batches
                if batch_idx % 1000 == 0:
                    batch_time = time.time() - batch_start_time
                    batch_metrics = {
                        'batch_time': batch_time,
                        'learning_rate': current_lr,
                        'gradient_norm': grad_norm.item()
                    }
                    
                    training_logger.log_batch(
                        epoch, batch_idx, len(train_loader),
                        loss.item(), len(user_ids), batch_metrics
                    )
                    
                    if batch_idx % 500 == 0:
                        logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                        
                        # Reduced memory cleanup for RTX 4090 (24GB VRAM)
                        if batch_idx % 5000 == 0:  # Less frequent cleanup
                            MemoryOptimizer.cleanup_memory()
            
            train_loss = np.mean(train_losses)
            train_rmse = math.sqrt(train_loss)
            epoch_time = time.time() - epoch_start_time
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        user_ids, item_ids, ratings = batch
                        user_ids = user_ids.to(device)
                        item_ids = item_ids.to(device)
                        ratings = ratings.to(device).float()
                    else:
                        continue
                    
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            val_rmse = math.sqrt(val_loss)
            
            # Log epoch summary
            train_metrics = {
                'rmse': train_rmse,
                'epoch_time': epoch_time,
                'samples_per_sec': len(train_loader.dataset) / epoch_time
            }
            val_metrics = {'rmse': val_rmse}
            training_logger.log_epoch_end(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs(args.save_dir, exist_ok=True)
                best_model_path = os.path.join(args.save_dir, "best_ncf_basic_model.pt")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'model_config': model_config,
                    'args': vars(args)
                }, best_model_path)
                
                # Save model locally
                wandb_manager.save_model_locally(
                    best_model_path,
                    'ncf_basic',
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
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Test evaluation
        logger.info("Evaluating on test set...")
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    user_ids, item_ids, ratings = batch
                    user_ids = user_ids.to(device)
                    item_ids = item_ids.to(device)
                    ratings = ratings.to(device).float()
                else:
                    continue
                
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                test_losses.append(loss.item())
        
        test_loss = np.mean(test_losses)
        test_rmse = math.sqrt(test_loss)
        
        # Calculate final metrics
        total_training_time = time.time() - training_start_time
        
        final_metrics = {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_training_time_hours': total_training_time / 3600,
            'total_parameters': total_params,
            'model_size_mb': model_size_mb
        }
        
        wandb_manager.log_metrics({f'final/{k}': v for k, v in final_metrics.items()})
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        logger.info(f"Final Test RMSE: {test_rmse:.4f}")
        logger.info(f"Best Validation RMSE: {math.sqrt(best_val_loss):.4f}")
        logger.info(f"Total Training Time: {total_training_time/3600:.2f} hours")
        logger.info("="*60)
        
        # Use original test metrics for backward compatibility
        all_metrics = {'loss': test_loss, 'rmse': test_rmse}
        
        # Save all artifacts needed for model deployment
        save_path = Path(args.save_dir)
        data_loader.save_encoders(save_path / 'encoders.pkl')  # User/item ID mappings
        
        # Save final evaluation metrics for analysis
        import json
        with open(save_path / 'final_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save additional artifacts for compatibility
        print("Saving additional model artifacts...")
        
        # Get ID mappings
        user_encoder = data_loader.user_encoder
        item_encoder = data_loader.item_encoder
        
        # Save ID mappings in expected format
        id_mappings = {
            'user_id_to_idx': {str(user_encoder.classes_[i]): i for i in range(len(user_encoder.classes_))},
            'movie_id_to_idx': {str(item_encoder.classes_[i]): i for i in range(len(item_encoder.classes_))},
            'idx_to_user_id': {i: str(user_encoder.classes_[i]) for i in range(len(user_encoder.classes_))},
            'idx_to_movie_id': {i: str(item_encoder.classes_[i]) for i in range(len(item_encoder.classes_))}
        }
        
        with open(save_path / 'id_mappings.pkl', 'wb') as f:
            pickle.dump(id_mappings, f)
        print("ID mappings saved to id_mappings.pkl")
        
        # Save model metadata
        model_metadata = {
            'num_users': model_config['num_users'],
            'num_items': model_config['num_items'], 
            'embedding_dim': getattr(args, 'embedding_dim', 64),
            'hidden_dims': getattr(args, 'hidden_dims', [128, 64]),
            'dropout': getattr(args, 'dropout', 0.2),
            'final_metrics': all_metrics,
            'model_architecture': str(model),
            'model_type': args.model_type
        }
        
        with open(save_path / 'model_metadata.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
        print("Model metadata saved to model_metadata.pkl")
        
        # Save training history
        with open(save_path / 'training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print("Training history saved to training_history.pkl")
        
        # Create movie lookup file
        movie_lookup = {
            'movie_id_to_idx': id_mappings['movie_id_to_idx'],
            'idx_to_movie_id': id_mappings['idx_to_movie_id'],
            'num_movies': model_config['num_items']
        }
        
        with open(save_path / 'movie_lookup.pkl', 'wb') as f:
            pickle.dump(movie_lookup, f)
        print("Movie lookup saved to movie_lookup.pkl")
        
        # Create backup
        with open(save_path / 'movie_lookup_backup.pkl', 'wb') as f:
            pickle.dump(movie_lookup, f)
        print("Movie lookup backup saved to movie_lookup_backup.pkl")
        
        # Create rating scaler file
        rating_scaler = {
            'min_rating': 1.0,  # Typical MovieLens range
            'max_rating': 5.0,
            'mean_rating': 3.5,
            'std_rating': 1.0
        }
        
        with open(save_path / 'rating_scaler.pkl', 'wb') as f:
            pickle.dump(rating_scaler, f)
        print("Rating scaler saved to rating_scaler.pkl")
        
        # Save main model file (alternative name)
        torch.save(model.state_dict(), save_path / 'recommendation_model.pt')
        print("Model state dict saved to recommendation_model.pt")
        
        logger.info(f"All artifacts saved successfully to {args.save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Always finish wandb run
        wandb_manager.finish()


if __name__ == "__main__":
    main()
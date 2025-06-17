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
    
    # Find first existing path
    default_ratings_path = None
    for path in possible_ratings_paths:
        if os.path.exists(path):
            default_ratings_path = path
            break
    
    parser.add_argument('--ratings-path', type=str, default=default_ratings_path,
                       help='Path to ratings CSV file')
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


def prepare_data(ratings_path, device, min_ratings_user=20, min_ratings_item=20, 
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
    
    # Filter users and items with minimum ratings
    user_counts = ratings_df['userId'].value_counts()
    item_counts = ratings_df['movieId'].value_counts()
    
    valid_users = user_counts[user_counts >= min_ratings_user].index
    valid_items = item_counts[item_counts >= min_ratings_item].index
    
    ratings_df = ratings_df[
        (ratings_df['userId'].isin(valid_users)) & 
        (ratings_df['movieId'].isin(valid_items))
    ]
    
    
    # Create label encoders for contiguous indices
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    ratings_df['user_idx'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['item_idx'] = item_encoder.fit_transform(ratings_df['movieId'])
    
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
    
    # Import training optimizations
    from training_optimizations import TrainingOptimizer, create_ncf_sample_input
    
    # Initialize training optimizer
    training_optimizer = TrainingOptimizer(device)
    
    # Find optimal batch size for this model
    model_temp = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        alpha=args.alpha
    ).to(device)
    
    optimal_batch_size = training_optimizer.find_optimal_batch_size(
        model_temp,
        lambda bs: create_ncf_sample_input(bs, device),
        initial_batch_size=256
    )
    del model_temp  # Clean up temporary model
    
    # Create optimized data loaders
    train_loader = training_optimizer.create_optimized_dataloader(
        train_dataset, 
        batch_size=optimal_batch_size,
        shuffle=True
    )
    val_loader = training_optimizer.create_optimized_dataloader(
        val_dataset, 
        batch_size=optimal_batch_size * 2,  # Larger batch for validation
        shuffle=False
    )
    test_loader = training_optimizer.create_optimized_dataloader(
        test_dataset, 
        batch_size=optimal_batch_size * 2,
        shuffle=False
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
        'ratings_path': args.ratings_path
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
            args.ratings_path, device, args.min_ratings_user, args.min_ratings_item
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
        
        # Compile model for performance optimization (20-30% speedup)
        model = training_optimizer.compile_model(model)
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        
        # Improved learning rate scheduling with cosine annealing
        from training_optimizations import CosineAnnealingWarmup
        scheduler = CosineAnnealingWarmup(
            optimizer, 
            warmup_epochs=5, 
            max_epochs=args.epochs,
            eta_min=args.learning_rate * 0.01
        )
        
        # Training logger
        training_logger = WandbTrainingLogger(wandb_manager, 'ncf')
        
        # Setup optimized training loop with mixed precision and gradient accumulation
        from training_optimizations import OptimizedTrainingLoop
        
        # Calculate gradient accumulation steps for effective batch size
        effective_batch_size = 2048  # Target effective batch size
        accumulation_steps = training_optimizer.setup_gradient_accumulation(
            effective_batch_size, optimal_batch_size
        )
        
        # Create optimized training loop
        optimized_trainer = OptimizedTrainingLoop(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            accumulation_steps=accumulation_steps,
            mixed_precision=device.type == 'cuda'
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Log epoch start
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch_start(epoch, args.epochs, current_lr)
            
            # Optimized training epoch
            train_metrics = optimized_trainer.train_epoch(
                train_loader, epoch, log_frequency=500  # Reduced logging frequency
            )
            
            train_loss = train_metrics['avg_loss']
            train_rmse = math.sqrt(train_loss)
            
            # Log training metrics
            training_logger.log_epoch_end(
                epoch, train_loss, train_rmse, 
                {
                    'epoch_time': train_metrics['epoch_time'],
                    'batches_per_sec': train_metrics['batches_per_sec']
                }
            )
            
            # Optimized validation epoch
            val_metrics = optimized_trainer.validate_epoch(val_loader)
            val_loss = val_metrics['avg_loss']
            val_rmse = math.sqrt(val_loss)
            
            # Log validation metrics
            validation_metrics = {
                'rmse': val_rmse,
                'validation_time': val_metrics['validation_time']
            }
            training_logger.log_validation(epoch, val_loss, validation_metrics)
            
            # Learning rate scheduling (cosine annealing)
            scheduler.step()
            
            # Memory cleanup
            training_optimizer.memory_cleanup(epoch)
            
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
                
                # Save metadata
                with open("models/ncf_metadata.pkl", 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Save model locally instead of WandB artifact
                wandb_manager.save_model_locally(
                    best_model_path,
                    'ncf',
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
        
        # Log final metrics
        final_metrics = {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        }
        
        wandb_manager.log_metrics({f'final/{k}': v for k, v in final_metrics.items()})
        
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
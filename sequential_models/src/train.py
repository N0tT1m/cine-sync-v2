#!/usr/bin/env python3
"""
Comprehensive Training Script for Sequential Recommendation Models

Supports training of multiple sequential architectures:
- Standard RNN-based sequential models (LSTM/GRU)
- Attention-based sequential models (SASRec-style)
- Hierarchical sequential models (short/long-term)
- Session-based recommendation models

Includes comprehensive evaluation, experiment tracking with WandB,
and flexible hyperparameter configuration through command-line arguments.
"""

import argparse
import logging
import torch
import wandb
import pickle
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    SequentialRecommender, AttentionalSequentialRecommender,
    HierarchicalSequentialRecommender, SessionBasedRecommender
)
from src.data_loader import SequentialDataLoader
from src.trainer import SequentialTrainer, SequentialEvaluator

# Import RTX 4090 BEAST MODE optimizations
sys.path.append(str(Path(__file__).parent.parent.parent / 'neural_collaborative_filtering'))
from performance_config import PerformanceOptimizer, apply_rtx4090_optimizations


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for training monitoring
    
    Configures both console and file logging to track training progress,
    model performance, and any issues during the training process.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sequential_training.log')
        ]
    )


def parse_args():
    """Parse comprehensive command line arguments for flexible training configuration
    
    Supports configuration of:
    - Data preprocessing parameters
    - Model architecture hyperparameters  
    - Training optimization settings
    - Evaluation and experiment tracking options
    """
    parser = argparse.ArgumentParser(description='Train Sequential Recommendation models')
    
    # Data preprocessing arguments
    parser.add_argument('--content-type', type=str, default='movies', 
                       choices=['movies', 'tv', 'both'],
                       help='Type of content to train on: movies, tv shows, or both')
    parser.add_argument('--dataset-sources', type=str, nargs='+', 
                       default=['movielens', 'netflix', 'tmdb', 'amazon', 'disney'],
                       help='Dataset sources to use (auto-selects based on content-type)')
    parser.add_argument('--ratings-path', type=str, default=None,
                       help='Path to ratings CSV file (auto-detected if not provided)')
    parser.add_argument('--combine-datasets', action='store_true', default=True,
                       help='Combine multiple datasets for training (enabled by default)')
    parser.add_argument('--min-interactions', type=int, default=20,
                       help='Minimum interactions per user (filter cold users)')
    parser.add_argument('--min-seq-length', type=int, default=5,
                       help='Minimum sequence length to include in training')
    parser.add_argument('--max-seq-length', type=int, default=100,
                       help='Maximum sequence length (BEAST MODE - longer sequences for better patterns)')
    
    # Model architecture arguments
    parser.add_argument('--model-type', type=str, default='sequential',
                       choices=['sequential', 'attention', 'hierarchical', 'session'],
                       help='Type of sequential model architecture')
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Item embedding dimension (BEAST MODE - larger capacity)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension for RNN/attention layers (BEAST MODE - larger networks)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of RNN layers (depth of network)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads (for attention model)')
    parser.add_argument('--num-blocks', type=int, default=2,
                       help='Number of transformer/attention blocks')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for regularization')
    parser.add_argument('--rnn-type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       help='Type of RNN cell (LSTM generally better)')
    
    # Training optimization arguments
    parser.add_argument('--data-type', type=str, default='sequential',
                       choices=['sequential', 'session'],
                       help='Data processing type (sequential vs session-based)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8192,
                       help='Training batch size (RTX 4090 BEAST MODE - massive batches for sequential models)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Adam optimizer learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='L2 regularization weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=24,
                       help='Number of data loader workers (Ryzen 9 3900X BEAST MODE - 24 threads)')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models')
    
    # Experiment tracking and logging
    parser.add_argument('--use-wandb', action='store_true', default=True,
                       help='Enable Weights & Biases experiment tracking (enabled by default)')
    parser.add_argument('--wandb-project', type=str, default='sequential-rec',
                       help='WandB project name for organizing experiments')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name (auto-generated if None)')
    
    # Evaluation
    parser.add_argument('--evaluate-ranking', action='store_true',
                       help='Evaluate ranking metrics')
    parser.add_argument('--top-k', type=int, nargs='+', default=[5, 10, 20],
                       help='Top-k values for evaluation')
    
    return parser.parse_args()


def load_comprehensive_datasets(content_type: str, dataset_sources: list, combine_datasets: bool = True) -> tuple:
    """Load and combine datasets based on content type and sources - identical to NCF version"""
    import glob
    import pandas as pd
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    # Dataset mapping (same as NCF)
    movie_datasets = {
        'movielens': {
            'ratings': '../../movies/cinesync/ml-32m/ratings.csv',
            'content': '../../movies/cinesync/ml-32m/movies.csv',
            'id_col': 'movieId',
            'title_col': 'title'
        },
        'netflix': {
            'content': '../../movies/netflix/netflix_movies.csv',
            'id_col': 'show_id',
            'title_col': 'title'
        },
        'tmdb': {
            'content': '../../movies/tmdb-movies/movies_metadata.csv',
            'id_col': 'id', 
            'title_col': 'title'
        },
        'amazon': {
            'content': '../../movies/amazon/amazon_prime_titles.csv',
            'id_col': 'show_id',
            'title_col': 'title'
        },
        'disney': {
            'content': '../../movies/disney/disney_plus_movies.csv',
            'id_col': 'show_id',
            'title_col': 'title'
        }
    }
    
    tv_datasets = {
        'tmdb': {
            'content': '../../tv/misc/TMDB_tv_dataset_v3.csv',
            'id_col': 'id',
            'title_col': 'name'
        },
        'netflix': {
            'content': '../../tv/netflix/netflix_titles.csv',
            'id_col': 'show_id',
            'title_col': 'title'
        },
        'amazon': {
            'content': '../../tv/amazon/amazon_prime_tv_shows.csv',
            'id_col': 'show_id',
            'title_col': 'title'
        },
        'disney': {
            'content': '../../tv/misc/disney_plus_tv_shows.csv',
            'id_col': 'show_id',
            'title_col': 'title'
        },
        'anime': {
            'content': '../../tv/anime/animes.csv',
            'id_col': 'anime_id',
            'title_col': 'Name'
        },
        'imdb': {
            'content': '../../tv/imdb/*.csv',
            'id_col': 'tconst',
            'title_col': 'primaryTitle'
        }
    }
    
    all_ratings = []
    all_content = []
    
    # Load movie datasets
    if content_type in ['movies', 'both']:
        for source in dataset_sources:
            if source in movie_datasets:
                dataset_info = movie_datasets[source]
                logger.info(f"Loading movie dataset: {source}")
                
                content_path = dataset_info['content']
                if Path(content_path).exists():
                    try:
                        content_df = pd.read_csv(content_path)
                        content_df = content_df.rename(columns={
                            dataset_info['id_col']: 'itemId',
                            dataset_info['title_col']: 'title'
                        })
                        content_df['content_type'] = 'movie'
                        content_df['source'] = source
                        all_content.append(content_df)
                        
                        if 'ratings' in dataset_info:
                            ratings_path = dataset_info['ratings']
                            if Path(ratings_path).exists():
                                ratings_df = pd.read_csv(ratings_path)
                                ratings_df = ratings_df.rename(columns={
                                    dataset_info['id_col']: 'itemId'
                                })
                                ratings_df['content_type'] = 'movie'
                                ratings_df['source'] = source
                                all_ratings.append(ratings_df)
                                
                    except Exception as e:
                        logger.warning(f"Failed to load {source} movie dataset: {e}")
    
    # Load TV datasets  
    if content_type in ['tv', 'both']:
        for source in dataset_sources:
            if source in tv_datasets:
                dataset_info = tv_datasets[source]
                logger.info(f"Loading TV dataset: {source}")
                
                content_path = dataset_info['content']
                if '*' in content_path:
                    content_files = glob.glob(content_path)
                else:
                    content_files = [content_path]
                
                for file_path in content_files:
                    if Path(file_path).exists():
                        try:
                            content_df = pd.read_csv(file_path)
                            content_df = content_df.rename(columns={
                                dataset_info['id_col']: 'itemId',
                                dataset_info['title_col']: 'title'
                            })
                            content_df['content_type'] = 'tv'
                            content_df['source'] = source
                            all_content.append(content_df)
                            
                        except Exception as e:
                            logger.warning(f"Failed to load {file_path}: {e}")
    
    # Combine datasets
    if all_content:
        combined_content = pd.concat(all_content, ignore_index=True)
        combined_content = combined_content.drop_duplicates(subset=['title'], keep='first')
        combined_content['itemId'] = range(1, len(combined_content) + 1)
    else:
        raise ValueError(f"No content datasets found for content_type: {content_type}")
    
    # Handle ratings
    if all_ratings:
        combined_ratings = pd.concat(all_ratings, ignore_index=True)
    else:
        logger.info("Creating synthetic ratings for sequential training...")
        combined_ratings = create_synthetic_ratings_sequential(combined_content)
    
    logger.info(f"Loaded {len(combined_content)} items and {len(combined_ratings)} ratings for sequential training")
    return combined_ratings, combined_content


def create_synthetic_ratings_sequential(content_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic sequential ratings with temporal patterns"""
    import numpy as np
    from datetime import datetime, timedelta
    
    num_synthetic_users = 1000
    avg_ratings_per_user = 80  # More ratings for sequential patterns
    
    synthetic_ratings = []
    base_timestamp = int(datetime(2020, 1, 1).timestamp())
    
    for user_id in range(1, num_synthetic_users + 1):
        # Create realistic sequential viewing patterns
        num_ratings = np.random.poisson(avg_ratings_per_user)
        user_items = content_df.sample(n=min(num_ratings, len(content_df)))
        
        # Sort by some logical order (e.g., release date if available)
        current_timestamp = base_timestamp + np.random.randint(0, 365*24*3600)  # Random start time
        
        for idx, (_, item) in enumerate(user_items.iterrows()):
            # Sequential timestamps (realistic viewing intervals)
            time_gap = np.random.exponential(3*24*3600)  # Average 3 days between views
            current_timestamp += int(time_gap)
            
            # Generate ratings with some sequential correlation
            if idx == 0:
                rating = np.random.normal(3.5, 1.0)
            else:
                # Some correlation with previous rating (viewing patterns)
                prev_rating = synthetic_ratings[-1]['rating']
                rating = np.random.normal(prev_rating, 0.8)
            
            rating = np.clip(rating, 0.5, 5.0)
            
            synthetic_ratings.append({
                'userId': user_id,
                'itemId': item['itemId'],
                'rating': rating,
                'timestamp': current_timestamp
            })
    
    return pd.DataFrame(synthetic_ratings).sort_values(['userId', 'timestamp'])


def create_model(model_type: str, config: dict, args) -> torch.nn.Module:
    """Create sequential recommendation model based on specified architecture type
    
    Factory function that instantiates the appropriate model class with
    hyperparameters from command-line arguments. Each model type has different
    architectural characteristics suited for different recommendation scenarios.
    
    Args:
        model_type: Architecture type ('sequential', 'attention', 'hierarchical', 'session')
        config: Model configuration dict with vocab sizes
        args: Parsed command-line arguments with hyperparameters
    
    Returns:
        Initialized PyTorch model ready for training
    """
    if model_type == 'sequential':
        # Standard RNN-based sequential model (LSTM/GRU)
        # Good baseline for sequential recommendation with temporal patterns
        model = SequentialRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            rnn_type=args.rnn_type
        )
    elif model_type == 'attention':
        # Self-attention based model (SASRec-style)
        # Better at capturing long-range dependencies in sequences
        model = AttentionalSequentialRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            max_seq_len=args.max_seq_length
        )
    elif model_type == 'hierarchical':
        # Hierarchical model with separate short-term and long-term modeling
        # Captures both recent preferences and overall user taste
        model = HierarchicalSequentialRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            short_hidden_dim=args.hidden_dim // 2,  # Smaller for recent items
            long_hidden_dim=args.hidden_dim,        # Larger for long-term patterns
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif model_type == 'session':
        # Session-based model optimized for short interaction sessions
        # Uses GRU and attention for session-level recommendation
        model = SessionBasedRecommender(
            num_items=config['num_items'],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_attention=True  # Attention helps identify important session items
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # 🔥🔥🔥 ACTIVATE SEQUENTIAL MODEL BEAST MODE 🔥🔥🔥
    apply_rtx4090_optimizations()
    
    # Get RTX 4090 performance config
    perf_config = PerformanceOptimizer.get_rtx4090_config()
    
    # Override with BEAST MODE settings if using defaults
    if args.batch_size == 8192:
        # For sequential models, use slightly smaller batch due to sequence memory
        args.batch_size = max(4096, perf_config['batch_size'] // 2)
        logger.info(f"🚀 SEQUENTIAL BEAST MODE batch size: {args.batch_size}")
    
    if args.num_workers == 24:
        args.num_workers = perf_config['num_workers']
        logger.info(f"💪 BEAST MODE workers: {args.num_workers} threads")
    
    # Initialize wandb (enabled by default)
    wandb_manager = None
    # Import W&B utilities
    sys.path.append(str(Path(__file__).parent.parent))
    from wandb_config import init_wandb_for_training
    
    # Initialize W&B manager
    config = vars(args)
    wandb_manager = init_wandb_for_training('sequential', config)
    
    # Also initialize regular wandb for backward compatibility
    wandb.init(
        project=args.wandb_project,
        name=args.experiment_name,
        config=config
    )
    
    try:
        # Load comprehensive datasets for sequential modeling
        logger.info(f"Loading {args.content_type} datasets for sequential modeling from: {args.dataset_sources}")
        ratings_df, content_df = load_comprehensive_datasets(
            content_type=args.content_type,
            dataset_sources=args.dataset_sources,
            combine_datasets=args.combine_datasets
        )
        
        # Create sequential data loader with loaded datasets
        logger.info("Creating sequential data loader...")
        data_loader = SequentialDataLoader(
            ratings_df=ratings_df,                   # Combined ratings with temporal data
            content_df=content_df,                   # Combined content metadata
            min_interactions=args.min_interactions,  # Filter sparse users
            min_seq_length=args.min_seq_length,      # Minimum meaningful sequence
            max_seq_length=args.max_seq_length       # Truncation point for efficiency
        )
        
        # Create PyTorch data loaders with proper sequence handling
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            data_type=args.data_type,    # Sequential vs session-based processing
            batch_size=args.batch_size,  # Larger batches for stable training
            num_workers=args.num_workers # Parallel data loading
        )
        
        # Get model config
        model_config = data_loader.get_model_config()
        logger.info(f"Model config: {model_config}")
        
        # Create model
        model = create_model(args.model_type, model_config, args)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # 🔥 SEQUENTIAL MODEL BEAST MODE OPTIMIZATIONS 🔥
        model = model.to(device)
        model = PerformanceOptimizer.optimize_model_for_speed(model)
        
        # Find optimal batch size for sequential models on RTX 4090
        if device == 'cuda':
            # Sequential models need less batch size due to sequence memory overhead
            optimal_batch = min(8192, PerformanceOptimizer.get_optimal_batch_size_rtx4090(model, torch.device(device)) // 2)
            if optimal_batch > args.batch_size:
                logger.info(f"🚀 Upgrading sequential batch size from {args.batch_size} to {optimal_batch}")
                args.batch_size = optimal_batch
                # Recreate data loaders with new batch size
                train_loader, val_loader, test_loader = data_loader.create_data_loaders(
                    data_type=args.data_type,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
        
        # Create trainer with W&B integration
        trainer = SequentialTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            wandb_manager=wandb_manager  # Pass W&B manager for comprehensive logging
        )
        
        # Train model with early stopping and checkpointing
        logger.info("Starting sequential model training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,  # Early stopping to prevent overfitting
            save_dir=args.save_dir   # Save best model and training history
        )
        
        logger.info(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader, args.top_k)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Comprehensive evaluation with advanced metrics if requested
        if args.evaluate_ranking:
            logger.info("Evaluating advanced ranking and sequence prediction metrics...")
            evaluator = SequentialEvaluator(model, device)
            
            # Next-item prediction with ranking-aware metrics
            ranking_metrics = evaluator.evaluate_next_item_prediction(test_loader, args.top_k)
            logger.info("Ranking Results (NDCG, MRR, Hit Rates):")
            for metric, value in ranking_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Multi-step sequence prediction capability
            seq_metrics = evaluator.evaluate_sequence_prediction(test_loader, future_steps=5)
            logger.info("Multi-step Sequence Prediction Results:")
            for metric, value in seq_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Combine all metrics
            all_metrics = {**test_metrics, **ranking_metrics, **seq_metrics}
        else:
            all_metrics = test_metrics
        
        # Log comprehensive results to Weights & Biases
        # Use automatic stepping to avoid step conflicts
        wandb.log(all_metrics, commit=True)                    # Final test metrics
        
        # Log training curves for visualization
        for epoch in range(len(history['train_loss'])):
            wandb.log({
                'epoch': epoch,
                'train_loss': history['train_loss'][epoch],
                'val_loss': history['val_loss'][epoch] if epoch < len(history['val_loss']) else None,
                'learning_rate': history['learning_rate'][epoch] if epoch < len(history['learning_rate']) else None
            }, commit=False)
        
        # Final commit for training history
        wandb.log({"training_complete": True}, commit=True)
        
        # Save all artifacts for reproducibility and deployment
        save_path = Path(args.save_dir)
        data_loader.save_encoders(save_path / 'encoders.pkl')  # ID mappings for inference
        
        # Save final metrics for comparison and reporting
        import json
        with open(save_path / 'final_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save additional artifacts for compatibility
        print("Saving additional model artifacts...")
        
        # Get encoders from data loader
        user_encoder = data_loader.user_encoder
        item_encoder = data_loader.item_encoder
        
        # Save ID mappings in expected format
        id_mappings = {
            'user_id_to_idx': {str(k): v for k, v in zip(user_encoder.classes_, range(len(user_encoder.classes_)))},
            'movie_id_to_idx': {str(k): v for k, v in zip(item_encoder.classes_, range(len(item_encoder.classes_)))},
            'idx_to_user_id': {v: str(k) for k, v in zip(user_encoder.classes_, range(len(user_encoder.classes_)))},
            'idx_to_movie_id': {v: str(k) for k, v in zip(item_encoder.classes_, range(len(item_encoder.classes_)))}
        }
        
        with open(save_path / 'id_mappings.pkl', 'wb') as f:
            pickle.dump(id_mappings, f)
        print("ID mappings saved to id_mappings.pkl")
        
        # Save model metadata
        model_metadata = {
            'num_users': len(user_encoder.classes_),
            'num_items': len(item_encoder.classes_),
            'embedding_dim': getattr(args, 'embedding_dim', 128),
            'hidden_size': getattr(args, 'hidden_size', 256),
            'num_layers': getattr(args, 'num_layers', 2),
            'max_seq_length': getattr(args, 'max_seq_length', 50),
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
            'num_movies': len(item_encoder.classes_)
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
        wandb.finish(exit_code=1)
        raise
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
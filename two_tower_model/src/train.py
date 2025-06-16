#!/usr/bin/env python3
"""
Comprehensive Training Script for Two-Tower/Dual-Encoder Models

Supports training of multiple two-tower architectures for large-scale recommendation:
- Basic Two-Tower: simple dual-encoder with dense features
- Enhanced Two-Tower: mixed categorical/numerical features with cross-attention
- Multi-Task Two-Tower: joint optimization of multiple objectives
- Collaborative Two-Tower: hybrid content + collaborative filtering

Includes advanced evaluation with retrieval metrics and FAISS-based similarity search.
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
    TwoTowerModel, EnhancedTwoTowerModel, 
    MultiTaskTwoTowerModel, CollaborativeTwoTowerModel
)
from src.data_loader import TwoTowerDataLoader
from src.trainer import TwoTowerTrainer, TwoTowerEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup comprehensive logging for training monitoring and debugging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('two_tower_training.log')
        ]
    )


def parse_args():
    """Parse comprehensive command line arguments for flexible two-tower training
    
    Supports configuration of multiple model types, training parameters,
    evaluation settings, and experiment tracking options.
    """
    parser = argparse.ArgumentParser(description='Train Two-Tower models')
    
    # Data preprocessing arguments
    parser.add_argument('--ratings-path', type=str, default='../../movies/cinesync/ml-32m/ratings.csv',
                       help='Path to ratings CSV file (userId, movieId, rating, timestamp)')
    parser.add_argument('--movies-path', type=str, default='../../movies/cinesync/ml-32m/movies.csv',
                       help='Path to movies CSV file with metadata (movieId, title, genres)')
    parser.add_argument('--min-interactions', type=int, default=20,
                       help='Minimum interactions per user/item (filter sparse entities)')
    
    # Model architecture arguments
    parser.add_argument('--model-type', type=str, default='enhanced',
                       choices=['simple', 'enhanced', 'multitask', 'collaborative'],
                       help='Two-Tower model variant to train')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Final embedding dimension for both towers')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer dimensions for tower networks')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for regularization')
    parser.add_argument('--use-batch-norm', action='store_true', default=True,
                       help='Use batch normalization in tower networks')
    parser.add_argument('--use-cross-attention', action='store_true',
                       help='Enable cross-attention between towers (enhanced model only)')
    
    # Multi-task learning arguments
    parser.add_argument('--task-heads', type=str, nargs='+', default=['rating', 'click'],
                       help='Prediction tasks for multi-task model (rating, click, etc.)')
    
    # Training optimization arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Training batch size (larger for stable two-tower training)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Adam optimizer learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='L2 regularization weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    
    # Data processing and sampling
    parser.add_argument('--negative-sampling', action='store_true', default=True,
                       help='Add negative samples for implicit feedback training')
    parser.add_argument('--neg-ratio', type=float, default=2.0,
                       help='Ratio of negative to positive samples')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models')
    
    # Experiment tracking
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='two-tower-rec',
                       help='Wandb project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Evaluation and metrics
    parser.add_argument('--evaluate-retrieval', action='store_true',
                       help='Enable comprehensive retrieval evaluation (Hit Rate, NDCG, MRR)')
    parser.add_argument('--top-k', type=int, nargs='+', default=[5, 10, 20, 50],
                       help='Top-k values for retrieval evaluation metrics')
    parser.add_argument('--build-index', action='store_true', default=True,
                       help='Use FAISS index for efficient similarity search evaluation')
    
    return parser.parse_args()


def create_model(model_type: str, config: dict, args) -> torch.nn.Module:
    """Factory function to create Two-Tower model based on specified architecture type
    
    Args:
        model_type: Architecture variant ('simple', 'enhanced', 'multitask', 'collaborative')
        config: Model configuration from data loader (feature dimensions, vocab sizes)
        args: Command-line arguments with hyperparameters
    
    Returns:
        Initialized Two-Tower model ready for training
    """
    if model_type == 'simple':
        # Basic Two-Tower model with flattened feature vectors
        # All features are concatenated into dense vectors for each tower
        user_feature_size = config['user_numerical_dim'] + len(config['user_categorical_dims'])
        item_feature_size = config['item_numerical_dim'] + len(config['item_categorical_dims'])
        
        model = TwoTowerModel(
            user_features_dim=user_feature_size,
            item_features_dim=item_feature_size,
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm
        )
        
    elif model_type == 'enhanced':
        # Enhanced model with separate categorical/numerical feature handling
        # Uses embedding layers for categorical features and can include cross-attention
        model = EnhancedTwoTowerModel(
            user_categorical_dims=config['user_categorical_dims'],
            user_numerical_dim=config['user_numerical_dim'],
            item_categorical_dims=config['item_categorical_dims'],
            item_numerical_dim=config['item_numerical_dim'],
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            use_cross_attention=args.use_cross_attention
        )
        
    elif model_type == 'multitask':
        # Multi-task model for joint optimization of multiple objectives
        # Shares tower representations but has task-specific prediction heads
        user_feature_size = config['user_numerical_dim'] + len(config['user_categorical_dims'])
        item_feature_size = config['item_numerical_dim'] + len(config['item_categorical_dims'])
        
        # Create task heads dictionary (each task outputs single value)
        task_heads = {task: 1 for task in args.task_heads}
        
        model = MultiTaskTwoTowerModel(
            user_features_dim=user_feature_size,
            item_features_dim=item_feature_size,
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            task_heads=task_heads,
            dropout=args.dropout
        )
        
    elif model_type == 'collaborative':
        # Hybrid model combining collaborative filtering and content-based features
        # Uses both learned embeddings from interactions and content features
        user_feature_size = config['user_numerical_dim'] + len(config['user_categorical_dims'])
        item_feature_size = config['item_numerical_dim'] + len(config['item_categorical_dims'])
        
        model = CollaborativeTwoTowerModel(
            num_users=config['num_users'],
            num_items=config['num_items'],
            user_features_dim=user_feature_size,
            item_features_dim=item_feature_size,
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
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    try:
        # Load and preprocess data with comprehensive feature engineering
        logger.info("Loading and preprocessing data for two-tower training...")
        data_loader = TwoTowerDataLoader(
            ratings_path=args.ratings_path,
            movies_path=args.movies_path,
            min_interactions=args.min_interactions  # Filter sparse users/items
        )
        
        # Create PyTorch data loaders with model-specific formatting
        train_loader, val_loader, test_loader = data_loader.create_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_type=args.model_type  # Determines feature format (simple vs enhanced)
        )
        
        # Get model config
        model_config = data_loader.get_model_config()
        logger.info(f"Model config: {model_config}")
        
        # Create model
        model = create_model(args.model_type, model_config, args)
        logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create trainer
        trainer = TwoTowerTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Train model with early stopping and checkpointing
        logger.info("Starting two-tower model training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,  # Early stopping for optimal generalization
            save_dir=args.save_dir   # Save best model and training artifacts
        )
        
        logger.info(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Comprehensive retrieval evaluation if requested
        if args.evaluate_retrieval:
            logger.info("Evaluating retrieval performance with FAISS-based similarity search...")
            evaluator = TwoTowerEvaluator(model, device)
            
            # Retrieval metrics: Hit Rate, NDCG, MRR at multiple k values
            retrieval_metrics = evaluator.evaluate_retrieval(
                test_loader, 
                k_values=args.top_k,        # Multiple k values for comprehensive evaluation
                build_index=args.build_index # Use FAISS for efficient search
            )
            
            logger.info("Retrieval Results:")
            for metric, value in retrieval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Analyze learned embedding quality and distribution
            embedding_analysis = evaluator.analyze_embeddings(test_loader)
            logger.info("Embedding Quality Analysis:")
            for category, stats in embedding_analysis.items():
                logger.info(f"  {category}: {stats}")
            
            # Combine all metrics
            all_metrics = {**test_metrics, **retrieval_metrics}
            all_metrics['embedding_analysis'] = embedding_analysis
        else:
            all_metrics = test_metrics
        
        # Log to wandb
        if args.use_wandb:
            # Use automatic stepping to avoid step conflicts
            wandb.log(all_metrics)
            wandb.log({"training_history": history})
        
        # Save all artifacts for reproducibility and deployment
        save_path = Path(args.save_dir)
        data_loader.save_preprocessors(save_path / 'preprocessors.pkl')  # Encoders and scalers
        
        # Save final evaluation metrics for comparison and reporting
        import json
        with open(save_path / 'final_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
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
            'hidden_dims': getattr(args, 'hidden_dims', [256, 128]),
            'dropout': getattr(args, 'dropout', 0.2),
            'final_metrics': all_metrics,
            'model_architecture': str(model),
            'model_type': 'two_tower'
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
        if args.use_wandb:
            wandb.finish(exit_code=1)
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
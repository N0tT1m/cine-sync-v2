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
    parser.add_argument('--ratings-path', type=str, default='../../movies/cinesync/ml-32m/ratings.csv',
                       help='Path to ratings CSV file (userId, movieId, rating, timestamp)')
    parser.add_argument('--min-interactions', type=int, default=20,
                       help='Minimum interactions per user (filter cold users)')
    parser.add_argument('--min-seq-length', type=int, default=5,
                       help='Minimum sequence length to include in training')
    parser.add_argument('--max-seq-length', type=int, default=50,
                       help='Maximum sequence length (truncation point)')
    
    # Model architecture arguments
    parser.add_argument('--model-type', type=str, default='sequential',
                       choices=['sequential', 'attention', 'hierarchical', 'session'],
                       help='Type of sequential model architecture')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Item embedding dimension (larger = more capacity)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for RNN/attention layers')
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
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size (larger = more stable gradients)')
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
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models')
    
    # Experiment tracking and logging
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable Weights & Biases experiment tracking')
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
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    try:
        # Load and preprocess data with temporal ordering
        logger.info("Loading and preprocessing data for sequential modeling...")
        data_loader = SequentialDataLoader(
            ratings_path=args.ratings_path,
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
        
        # Create trainer
        trainer = SequentialTrainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
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
        if args.use_wandb:
            # Use explicit steps to avoid conflicts
            final_step = wandb.run.step if wandb.run and hasattr(wandb.run, 'step') else 0
            wandb.log(all_metrics, step=final_step)                    # Final test metrics
            wandb.log({"training_history": history}, step=final_step + 1)  # Training curves and progress
        
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
        if args.use_wandb:
            wandb.finish(exit_code=1)
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()